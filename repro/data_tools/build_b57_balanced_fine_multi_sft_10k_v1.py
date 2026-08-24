#!/usr/bin/env python3
"""Build the fail-closed B57 balanced 10k SFT data plan.

B57 deliberately consumes only the already sealed B50 train pool.  The
composition target is 3,800 fine-grained single-image rows, 3,600 multi-image
reasoning rows, and 2,600 general/knowledge rows.  Public Vision-OPD rows,
non-train splits, unknown licenses, malformed media, and rows without bounded
prompt length are rejected before selection.  A candidate-specific
four-benchmark receipt over the complete B50-plus-supplement union and an
independently proven B28 membership receipt are mandatory; no network or GPU
is used.

If the safe licensed pool cannot satisfy all three exact quotas, execution
writes aggregate-only sealed ``blocked`` build/final receipts and no parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable

try:  # Keep aggregate CPU tests importable in the minimal test interpreter.
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - exercised by dependency-light CI
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


WORKSPACE = Path("/minimax-3d-rw-backup/users/jiazhi/H_Workspace")
DATASET_ROOT = WORKSPACE / "Dataset"
VERSION = "b57_balanced_fine_multi_sft_10k_v1"
TARGET_ROWS = 10_000
QUOTAS = {"fine_grained_single": 3_800, "multi_image_reasoning": 3_600, "general_knowledge": 2_600}
TOKEN_BYTE_LIMIT = 16_384
MAX_IMAGES = 6
SOURCE_PARQUET = DATASET_ROOT / "b50_b49_plus_4096_v1/processed/train_12263.parquet"
SOURCE_RECEIPT = DATASET_ROOT / "b50_b49_plus_4096_v1/manifest/build_receipt.json"
SOURCE_FINAL_GATE = DATASET_ROOT / "b50_b49_plus_4096_v1/manifest/final_gate.json"
B54_FINAL_GATE = DATASET_ROOT / "b54_10k_sft_v1/manifest/final_gate.json"
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / VERSION
DEFAULT_RECEIPT = DEFAULT_OUTPUT_ROOT / "manifest/build_receipt.json"
DEFAULT_FINAL_GATE = DEFAULT_OUTPUT_ROOT / "manifest/final_gate.json"
DEFAULT_PARQUET = DEFAULT_OUTPUT_ROOT / "processed/train_10000.parquet"
FOURBENCH_NAMESPACES = ("VStarBench", "MMStar", "BLINK", "ZoomBench")
CANDIDATE_SOURCE_LABELS = ("b50_b49_plus_4096_v1", "b57_vlaa_general_supplement_v3")
B28_PARQUET_SHA256 = "9d4de6b1e4a0e3efe5a398a91dc33f93e154dbb0eed9f29bf746b7cd13d512a9"
B28_BUILD_RECEIPT_SHA256 = "e38eda077b7345d83d307cc87035d892714c886cd5517fd7ff96c2e1b86eccb6"
B28_SELECTION_RECEIPT_SHA256 = "6b1e1f5e983e4230134e8b0c2b0ecd683c049e0cc3f8336dd9509807958c2170"
KNOWN_LICENSES = frozenset({"apache-2.0", "cc-by-4.0", "cc-by-sa-4.0", "mit", "bsd-3-clause"})
HEX64 = frozenset("0123456789abcdef")
HEX16 = frozenset("0123456789abcdef")
PUBLIC_MARKERS = ("vision_opd", "vision-opd", "b1_train_decontaminated_replay", "public_vision")
FINE_MARKERS = ("fine", "attribute", "pairwise", "grounding", "spatial", "relation", "color", "geometry", "crop", "direct")

# These are the only provenance-recovery exceptions.  They are intentionally
# exact (dataset name + immutable revision + pinned card digest); a row that
# merely says ``unknown`` or ``train_candidate_pool`` never passes by marker.
AREF_DATASET = "zhenjiemao__aRefCOCO"
AREF_REVISION = "bd263b03d9eb8c687fb3ecbc755746f7d0e30eef"
AREF_CARD_URL = "https://huggingface.co/datasets/zhenjiemao/aRefCOCO/commit/bd263b03d9eb8c687fb3ecbc755746f7d0e30eef"
AREF_CARD_SHA256 = "902ac410ce1b965fc28e0b07270d0e1cd2ab115b81b94995c3a0f7c51998b144"
VLAA_DATASET = "UCSC-VLAA/VLM-CapCurriculum-VisualReasoning-Data"
VLAA_REVISION = "094feaf1b64b455f67111eb80bdc6cf3f6be198c"
VLAA_CARD_URL = "https://huggingface.co/datasets/UCSC-VLAA/VLM-CapCurriculum-VisualReasoning-Data"
VLAA_CARD_SHA256 = "fdab7563d868ceaa7e11f23837b589e5c45b298dacf3d56561893f233a36c84"
SOURCE_LICENSE_PROVENANCE = {
    "aRefCOCO": {"dataset": AREF_DATASET, "revision": AREF_REVISION, "declared_license": "cc-by-4.0", "declared_split": "train", "card_url": AREF_CARD_URL, "card_sha256": AREF_CARD_SHA256},
    "VLAA": {"dataset": VLAA_DATASET, "revision": VLAA_REVISION, "declared_license": "apache-2.0", "declared_split": "train", "card_url": VLAA_CARD_URL, "card_sha256": VLAA_CARD_SHA256},
}


class BuildError(RuntimeError):
    pass


def _disable_cuda() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def _normal_hash(value: Any) -> str:
    return sha256_bytes(_normal_text(value).encode("utf-8"))


def _safe_json(path: Path, label: str) -> dict[str, Any]:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise BuildError(f"{label} is not a single-link regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise BuildError(f"{label} is not an object")
    return value


def _pin(path: Path, label: str) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise BuildError(f"{label} is missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise BuildError(f"{label} is not a single-link regular file")
    return {"path": str(path.absolute()), "sha256": sha256_file(path), "bytes": int(info.st_size)}


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["seal_sha256"] = sha256_bytes(canonical(body))
    return result


def _create_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink() or path.exists():
        raise BuildError(f"output already exists: {path}")
    parent = path.parent
    if parent.exists() and parent.is_symlink():
        raise BuildError(f"output parent is a symlink: {parent}")
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if parent.is_symlink():
        raise BuildError(f"output parent became a symlink: {parent}")
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def _text_from_prompt(prompt: Any) -> str:
    chunks: list[str] = []
    if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes, bytearray)):
        return ""
    for message in prompt:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            for item in content:
                if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
    return "\n".join(chunks)


def _row_extra(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("extra_info")
    return value if isinstance(value, Mapping) else {}


def _category(row: Mapping[str, Any]) -> str:
    extra = _row_extra(row)
    text = " ".join(str(extra.get(key) or "") for key in ("source_subset", "bucket_key", "ability_bucket"))
    text = f"{text} {row.get('data_source') or ''}".casefold()
    if any(marker in text for marker in PUBLIC_MARKERS):
        return "excluded_public_vision_opd"
    image_count = len(row.get("images") or []) if isinstance(row.get("images"), list) else 0
    if image_count > 1:
        return "multi_image_reasoning"
    if any(marker in text for marker in FINE_MARKERS):
        return "fine_grained_single"
    return "general_knowledge"


def _row_contract(row: Mapping[str, Any], *, recover_provenance: bool = False) -> str | None:
    source_id = row.get("source_id")
    record_hash = row.get("source_record_sha256")
    if not isinstance(source_id, str) or not source_id:
        return "missing_source_id"
    if not isinstance(record_hash, str) or len(record_hash) != 64 or any(c not in HEX64 for c in record_hash.casefold()):
        return "malformed_source_record_sha256"
    extra = _row_extra(row)
    required = ("source_dataset", "source_subset", "source_revision", "source_license", "split", "upstream_id", "question", "image_file_sha256", "image_rgb_sha256", "image_phash64_dct_v1")
    if any(not isinstance(extra.get(key), str) or not extra.get(key) for key in required[:7]):
        return "missing_source_provenance"
    source_dataset = str(extra.get("source_dataset") or "")
    source_revision = str(extra.get("source_revision") or "")
    source_license = str(extra.get("source_license") or "").casefold()
    recovered_aref = recover_provenance and source_dataset == AREF_DATASET and source_revision == AREF_REVISION and source_license == "unknown"
    recovered_vlaa = recover_provenance and source_dataset == VLAA_DATASET and source_revision == VLAA_REVISION and extra.get("split") == "train_candidate_pool"
    if extra.get("split") != "train" and not recovered_vlaa:
        return "non_train_split"
    if source_license not in KNOWN_LICENSES and not recovered_aref and not recovered_vlaa:
        return "unproven_license"
    images = row.get("images")
    if not isinstance(images, list) or not 1 <= len(images) <= MAX_IMAGES:
        return "image_cardinality"
    prompt = _text_from_prompt(row.get("prompt"))
    if not prompt or prompt.count("<image>") != len(images):
        return "prompt_image_cardinality"
    answer = str(extra.get("answer") or "")
    if len((prompt + "\n" + answer).encode("utf-8")) >= TOKEN_BYTE_LIMIT:
        return "token_byte_bound"
    for ref in images:
        if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str) or not ref.get("path") or ref.get("path") != ref.get("image"):
            return "image_reference"
    files = extra.get("image_file_sha256")
    rgbs = extra.get("image_rgb_sha256")
    phashes = extra.get("image_phash64_dct_v1")
    if not all(isinstance(value, list) and len(value) == len(images) for value in (files, rgbs, phashes)):
        return "image_hash_cardinality"
    if any(not isinstance(value, str) or len(value) != 64 or any(c not in HEX64 for c in value.casefold()) for values in (files, rgbs) for value in values):
        return "image_hash_format"
    if any(not isinstance(value, str) or len(value) != 16 or any(c not in HEX16 for c in value.casefold()) for value in phashes):
        return "image_phash_format"
    return None


def _fingerprint(row: Mapping[str, Any]) -> dict[str, Any]:
    extra = _row_extra(row)
    files = tuple(str(x).casefold() for x in extra["image_file_sha256"])
    rgbs = tuple(str(x).casefold() for x in extra["image_rgb_sha256"])
    phashes = tuple(int(str(x), 16) for x in extra["image_phash64_dct_v1"])
    return {"source_id": str(row["source_id"]).casefold(), "record": str(row["source_record_sha256"]).casefold(), "prompt": _normal_hash(json.dumps(row.get("prompt"), ensure_ascii=False, sort_keys=True)), "question": _normal_hash(extra.get("question")), "files": files, "rgbs": rgbs, "phashes": phashes}


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class _IdentityIndex:
    def __init__(self) -> None:
        self.source_ids: set[str] = set()
        self.records: set[str] = set()
        self.prompts: set[str] = set()
        self.questions: dict[str, list[int]] = defaultdict(list)
        self.files: set[str] = set()
        self.rgbs: set[str] = set()
        self.phashes: list[int] = []

    def collision(self, fp: Mapping[str, Any]) -> str | None:
        if fp["source_id"] in self.source_ids: return "source_id"
        if fp["record"] in self.records: return "source_record_sha256"
        if fp["prompt"] in self.prompts: return "normalized_prompt_sha256"
        if set(fp["files"]) & self.files: return "image_file_sha256"
        if set(fp["rgbs"]) & self.rgbs: return "image_rgb_sha256"
        if fp["question"] in self.questions and any(_hamming(x, y) <= 4 for x in self.questions[fp["question"]] for y in fp["phashes"]): return "question_phash_hamming_le_4"
        return None

    def add(self, fp: Mapping[str, Any]) -> None:
        self.source_ids.add(fp["source_id"]); self.records.add(fp["record"]); self.prompts.add(fp["prompt"]); self.files.update(fp["files"]); self.rgbs.update(fp["rgbs"]); self.questions[fp["question"]].extend(fp["phashes"]); self.phashes.extend(fp["phashes"])


def _source_gate() -> dict[str, Any]:
    source = _safe_json(SOURCE_RECEIPT, "B50 source receipt")
    final = _safe_json(SOURCE_FINAL_GATE, "B50 final gate")
    b54 = _safe_json(B54_FINAL_GATE, "B54 final gate")
    output = source.get("outputs", {}).get("train", {})
    overlap = source.get("fourbench_overlap")
    passed = source.get("status") == "published" and final.get("status") == "passed" and output.get("rows") == 12_263 and output.get("sha256") == sha256_file(SOURCE_PARQUET) and isinstance(overlap, Mapping) and overlap.get("status") == "passed" and (overlap.get("hard_overlap") or {}).get("nonzero") is False and b54.get("status") == "passed_cpu_build_audit" and b54.get("b28_rows") == 0 and b54.get("public_vision_opd_rows") == 0
    return {"status": "passed" if passed else "blocked", "source_receipt": _pin(SOURCE_RECEIPT, "B50 source receipt"), "source_final_gate": _pin(SOURCE_FINAL_GATE, "B50 final gate"), "b54_final_gate": _pin(B54_FINAL_GATE, "B54 final gate"), "fourbench_overlap": dict(overlap) if isinstance(overlap, Mapping) else None, "b28_rows": b54.get("b28_rows"), "public_vision_opd_rows": b54.get("public_vision_opd_rows"), "reason": None if passed else "required B50/B54 sealed hard-overlap, B28, or public-Vision-OPD gate is unavailable"}


def _supplement_gate(path: Path | None, receipt: Path | None) -> dict[str, Any]:
    """Validate the optional, aggregate-only raw-source supplement pin."""

    if path is None and receipt is None:
        return {"status": "not_requested", "rows": 0, "reason": None}
    if path is None or receipt is None:
        return {"status": "blocked_incomplete_pin", "rows": 0, "reason": "supplement parquet and sealed receipt must be supplied together"}
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            return {"status": "blocked_unsafe_parquet", "rows": 0, "reason": "supplement parquet is not a single-link regular file"}
        value = _safe_json(receipt, "B57 supplement receipt")
        body = {key: item for key, item in value.items() if key != "seal_sha256"}
        if value.get("seal_sha256") != sha256_bytes(canonical(body)):
            return {"status": "blocked_bad_receipt_seal", "rows": 0, "reason": "supplement receipt seal differs"}
        if value.get("status") != "published" or value.get("aggregate_only") is not True or value.get("gpu_used") is not False:
            return {"status": "blocked_bad_receipt_status", "rows": 0, "reason": "supplement receipt is not a CPU aggregate publication"}
        output = value.get("output")
        if not isinstance(output, Mapping) or output.get("rows", 0) < 1000 or output.get("sha256") != sha256_file(path):
            return {"status": "blocked_output_pin_mismatch", "rows": 0, "reason": "supplement output row/hash pin differs"}
        return {"status": "passed", "rows": int(output["rows"]), "parquet": _pin(path, "B57 supplement parquet"), "receipt": _pin(receipt, "B57 supplement receipt"), "source_provenance": value.get("source_provenance")}
    except (BuildError, OSError, KeyError, TypeError, ValueError):
        return {"status": "blocked_unreadable_pin", "rows": 0, "reason": "supplement parquet or receipt is unreadable"}


def _candidate_sources(supplement_gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact aggregate identity of every B57 candidate source."""

    supplement = supplement_gate.get("parquet") if isinstance(supplement_gate.get("parquet"), Mapping) else {}
    return [
        {"label": CANDIDATE_SOURCE_LABELS[0], "rows": 12_263, "sha256": sha256_file(SOURCE_PARQUET)},
        {"label": CANDIDATE_SOURCE_LABELS[1], "rows": int(supplement_gate.get("rows", 0)), "sha256": supplement.get("sha256")},
    ]


def _supplement_fourbench_gate(path: Path | None, supplement_gate: Mapping[str, Any]) -> dict[str, Any]:
    if path is None:
        return {"status": "blocked_missing_candidate_receipt" if supplement_gate.get("status") == "passed" else "not_requested"}
    if supplement_gate.get("status") != "passed":
        return {"status": "blocked_supplement_not_pinned"}
    try:
        value = _safe_json(path, "B57 supplement fourbench receipt")
        body = {key: item for key, item in value.items() if key != "seal_sha256"}
        if not isinstance(value.get("seal_sha256"), str) or value["seal_sha256"] != sha256_bytes(canonical(body)):
            return {"status": "blocked_bad_seal"}
        if value.get("status") != "passed" or value.get("aggregate_only") is not True:
            return {"status": "blocked_bad_status"}
        expected_sources = _candidate_sources(supplement_gate)
        if value.get("candidate_sources") != expected_sources:
            return {"status": "blocked_candidate_binding"}
        if value.get("candidate_source_digest") != sha256_bytes(canonical(expected_sources)):
            return {"status": "blocked_candidate_digest"}
        namespaces = value.get("namespaces")
        if not isinstance(namespaces, Mapping) or any(not isinstance(namespaces.get(name), Mapping) or (namespaces[name].get("hard_overlap") or {}).get("nonzero") is not False for name in FOURBENCH_NAMESPACES):
            return {"status": "blocked_incomplete_namespace_zero"}
        if (value.get("hard_overlap") or {}).get("nonzero") is not False:
            return {"status": "blocked_outer_namespace_zero"}
        _assert_aggregate(value)
        return {"status": "passed", "receipt": _pin(path, "B57 supplement fourbench receipt")}
    except (BuildError, OSError, KeyError, TypeError, ValueError):
        return {"status": "blocked_unreadable_receipt"}


def _b28_membership_gate(path: Path | None, selected: Sequence[Mapping[str, Any]], supplement_gate: Mapping[str, Any]) -> dict[str, Any]:
    if path is None:
        return {"status": "blocked_missing_membership"}
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            return {"status": "blocked_unsafe_membership"}
        value = _safe_json(path, "B28 membership receipt")
        body = {key: item for key, item in value.items() if key != "seal_sha256"}
        if value.get("seal_sha256") != sha256_bytes(canonical(body)):
            return {"status": "blocked_bad_seal"}
        if value.get("status") != "passed" or value.get("aggregate_only") is not True or value.get("gpu_used") is not False:
            return {"status": "blocked_bad_status"}
        if value.get("schema_version") == "b28_membership_proof_v1":
            b26 = value.get("b26")
            membership = value.get("membership")
            token_info = value.get("token_file")
            privacy = value.get("privacy")
            if not isinstance(b26, Mapping) or b26.get("schema_version") != "b26_safe_fine_multi_1536_v1" or b26.get("rows") != 1_536:
                return {"status": "blocked_unpinned_authority"}
            parquet = b26.get("parquet")
            build_receipt = b26.get("build_receipt")
            selection_receipt = b26.get("selection_receipt")
            if (
                not isinstance(parquet, Mapping)
                or parquet.get("sha256") != B28_PARQUET_SHA256
                or parquet.get("rows") != 1_536
                or not isinstance(build_receipt, Mapping)
                or build_receipt.get("sha256") != B28_BUILD_RECEIPT_SHA256
                or not isinstance(selection_receipt, Mapping)
                or selection_receipt.get("sha256") != B28_SELECTION_RECEIPT_SHA256
            ):
                return {"status": "blocked_unpinned_authority"}
            if (
                not isinstance(membership, Mapping)
                or membership.get("token_count") != 1_536
                or membership.get("order_hash_algorithm") != "sha256(canonical_json(membership_order))_v1"
                or membership.get("set_hash_algorithm") != "sha256(canonical_json(sorted(membership_order)))_v1"
                or membership.get("token_file_hash_algorithm") != "sha256(utf8_joined_tokens_with_final_newline)_v1"
                or not isinstance(privacy, Mapping)
                or privacy.get("token_values_in_receipt") is not False
                or privacy.get("token_file_is_sensitive") is not True
                or privacy.get("prompts_written") is not False
                or privacy.get("answers_written") is not False
                or privacy.get("images_written") is not False
            ):
                return {"status": "blocked_bad_privacy_or_algorithm"}
            if not isinstance(token_info, Mapping) or not isinstance(token_info.get("name"), str) or Path(token_info["name"]).name != token_info["name"] or token_info["name"] in {"", ".", ".."}:
                return {"status": "blocked_bad_token_binding"}
            token_path = path.parent / token_info["name"]
            token_stat = token_path.lstat()
            if stat.S_ISLNK(token_stat.st_mode) or not stat.S_ISREG(token_stat.st_mode) or token_stat.st_nlink != 1 or stat.S_IMODE(token_stat.st_mode) != 0o600:
                return {"status": "blocked_unsafe_token_file"}
            token_bytes = token_path.read_bytes()
            if len(token_bytes) != token_info.get("bytes") or sha256_bytes(token_bytes) != token_info.get("sha256") or token_info.get("rows") != 1_536 or not token_bytes.endswith(b"\n"):
                return {"status": "blocked_token_pin_mismatch"}
            try:
                token_text = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return {"status": "blocked_token_encoding"}
            tokens = token_text.splitlines()
            if len(tokens) != 1_536 or any(not token or token != token.strip() or "\r" in token or "\x00" in token for token in tokens) or len(set(tokens)) != len(tokens) or len({token.casefold() for token in tokens}) != len(tokens):
                return {"status": "blocked_token_contract"}
            if sha256_bytes(canonical(tokens)) != membership.get("membership_order_sha256") or sha256_bytes(canonical(sorted(tokens))) != membership.get("membership_sha256"):
                return {"status": "blocked_token_hash_mismatch"}
            token_set = {token.casefold() for token in tokens}
            collision = sum(1 for row in selected if str(row.get("source_id") or "").casefold() in token_set)
            _assert_aggregate(value)
            return {"status": "passed" if collision == 0 else "blocked_collision", "membership_rows": len(tokens), "collision_rows": collision, "membership": _pin(path, "B28 membership proof")}
        expected_sources = _candidate_sources(supplement_gate)
        if value.get("candidate_sources") != expected_sources or value.get("candidate_source_digest") != sha256_bytes(canonical(expected_sources)):
            return {"status": "blocked_candidate_binding"}
        authority = value.get("b28_authority")
        intersection = value.get("intersection")
        if (
            not isinstance(authority, Mapping)
            or authority.get("readable") is not True
            or authority.get("schema") != "b26_safe_fine_multi_1536_v1"
            or authority.get("rows") != 1_536
            or authority.get("parquet_sha256") != B28_PARQUET_SHA256
            or authority.get("build_receipt_sha256") != B28_BUILD_RECEIPT_SHA256
            or authority.get("selection_receipt_sha256") != B28_SELECTION_RECEIPT_SHA256
        ):
            return {"status": "blocked_unreadable_authority"}
        if not isinstance(intersection, Mapping) or intersection.get("source_id_rows") != 0 or intersection.get("source_record_sha256_rows") != 0 or intersection.get("rows_with_any_collision") != 0:
            return {"status": "blocked_collision"}
        _assert_aggregate(value)
        return {"status": "passed", "membership_rows": int(value.get("b28_rows_observed", 0)), "collision_rows": 0, "membership": _pin(path, "B28 membership receipt")}
    except (BuildError, OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return {"status": "blocked_unreadable_membership"}


def _b28_candidate_filter(path: Path | None, supplement_gate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the sealed B28 proof and expose only a private token set.

    The token values never enter a receipt.  This helper intentionally calls
    the full membership validator with an empty selected set before reading
    the already-pinned token file, so candidate filtering cannot bypass any
    authority, privacy, hash, or permission check in the publication gate.
    """

    checked = _b28_membership_gate(path, (), supplement_gate)
    if checked.get("status") != "passed" or path is None:
        return {"status": checked.get("status", "blocked"), "token_set": set(), "membership_rows": checked.get("membership_rows", 0)}
    try:
        value = _safe_json(path, "B28 membership proof")
        token_info = value["token_file"]
        token_path = path.parent / str(token_info["name"])
        token_set = {line.casefold() for line in token_path.read_text(encoding="utf-8").splitlines() if line}
        if len(token_set) != 1_536:
            return {"status": "blocked_token_contract", "token_set": set(), "membership_rows": 0}
        return {"status": "passed", "token_set": token_set, "membership_rows": len(token_set)}
    except (BuildError, OSError, UnicodeError, KeyError, TypeError, ValueError):
        return {"status": "blocked_unreadable_membership", "token_set": set(), "membership_rows": 0}


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if pq is None:
        raise BuildError("pyarrow is required to read the B57 source parquet")
    parquet = pq.ParquetFile(path)
    required = {"source_id", "source_record_sha256", "prompt", "images", "extra_info"}
    missing = required - set(parquet.schema_arrow.names)
    if missing: raise BuildError(f"source parquet missing columns: {sorted(missing)}")
    for batch in parquet.iter_batches(batch_size=512, use_threads=False):
        yield from batch.to_pylist()


def _assert_aggregate(value: Any) -> None:
    # ``prompt`` is also the name of a harmless aggregate counter in the
    # frozen four-benchmark auditor.  Reject actual prompt payloads while
    # allowing that numeric metric to be embedded in a sealed receipt.
    forbidden = {"answer", "messages", "images", "image_path", "sample_id", "source_id", "row_index", "prediction", "gold", "responses"}
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).casefold()
            if key_text in forbidden or (key_text == "prompt" and not isinstance(child, (int, float))):
                raise BuildError("B57 receipt contains sample-level metadata")
            _assert_aggregate(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_aggregate(child)


def validate_receipts(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Validate the sealed aggregate build/final receipts without reading rows."""

    build_path = output_root / "manifest/build_receipt.json"
    final_path = output_root / "manifest/final_gate.json"
    build = _safe_json(build_path, "B57 build receipt")
    final = _safe_json(final_path, "B57 final gate")
    for value, label in ((build, "build"), (final, "final")):
        seal = value.get("seal_sha256")
        body = {key: item for key, item in value.items() if key != "seal_sha256"}
        if not isinstance(seal, str) or sha256_bytes(canonical(body)) != seal:
            raise BuildError(f"B57 {label} receipt seal differs")
        _assert_aggregate(value)
        if value.get("aggregate_only") is not True or value.get("gpu_used") is not False or value.get("status") not in {"blocked", "published"}:
            raise BuildError(f"B57 {label} receipt contract differs")
    if build.get("status") != final.get("status") or build.get("target", {}).get("accepted_rows") != final.get("accepted_rows"):
        raise BuildError("B57 build/final receipt status or count differs")
    outputs = build.get("outputs")
    if build.get("status") == "blocked" and outputs:
        raise BuildError("blocked B57 receipt cannot claim parquet output")
    return {"status": build["status"], "accepted_rows": build.get("target", {}).get("accepted_rows"), "gap_rows": build.get("target", {}).get("gap_rows"), "build_receipt_sha256": sha256_file(build_path), "final_gate_sha256": sha256_file(final_path)}


def _round_robin(groups: Mapping[str, Sequence[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    ordered = {key: iter(sorted(values, key=lambda row: sha256_bytes(f"{VERSION}|{key}|{row['source_id']}".encode()))) for key, values in sorted(groups.items())}
    chosen: list[dict[str, Any]] = []
    while len(chosen) < limit and ordered:
        exhausted = []
        for key, iterator in ordered.items():
            try: chosen.append(next(iterator))
            except StopIteration: exhausted.append(key)
            if len(chosen) >= limit: break
        for key in exhausted: ordered.pop(key, None)
    return chosen


def _exclude_b28_collisions(candidates: Mapping[str, list[dict[str, Any]]], b28_filter: Mapping[str, Any], rejects: Counter[str]) -> int:
    """Remove B28 token matches before quota allocation, preserving reserves."""

    if b28_filter.get("status") != "passed":
        return 0
    token_set = b28_filter.get("token_set", set())
    removed = 0
    for category, rows in candidates.items():
        retained: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("source_id") or "").casefold() in token_set:
                rejects[f"b28_collision:{category}"] += 1
                removed += 1
            else:
                retained.append(row)
        rows[:] = retained
    return removed


def _write_parquet(path: Path, rows: Sequence[dict[str, Any]], source_schema: pa.Schema) -> dict[str, Any]:
    if pa is None or pq is None:
        raise BuildError("pyarrow is required to publish B57 parquet")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink(): raise BuildError("B57 parquet already exists")
    metadata = dict(source_schema.metadata or {})
    metadata.update({b"dataset": VERSION.encode(), b"schema_version": VERSION.encode(), b"cpu_only": b"true", b"gpu_used": b"false", b"aggregate_only_receipt": b"true"})
    table = pa.Table.from_pylist(list(rows), schema=source_schema.with_metadata(metadata))
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.chmod(temporary, 0o600)
        os.link(temporary, path)
    except FileExistsError as exc:
        raise BuildError("B57 parquet already exists") from exc
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"path": str(path.absolute()), "rows": len(rows), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def build_aggregate(*, output_root: Path = DEFAULT_OUTPUT_ROOT, source_parquet: Path = SOURCE_PARQUET, source_receipt: Path = SOURCE_RECEIPT, execute: bool = True, recover_provenance: bool = False, supplement_parquet: Path | None = None, supplement_receipt: Path | None = None, supplement_fourbench_receipt: Path | None = None, b28_membership: Path | None = None) -> dict[str, Any]:
    _disable_cuda()
    if output_root.is_symlink(): raise BuildError("B57 output root is a symlink")
    source_gate = _source_gate()
    supplement_gate = _supplement_gate(supplement_parquet, supplement_receipt)
    supplement_requested = supplement_parquet is not None or supplement_receipt is not None
    b28_filter = _b28_candidate_filter(b28_membership, supplement_gate) if supplement_requested else {"status": "not_requested", "token_set": set(), "membership_rows": 0}
    counts: Counter[str] = Counter(); accepted: Counter[str] = Counter(); rejects: Counter[str] = Counter(); candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_groups: dict[str, Counter[str]] = defaultdict(Counter)
    source_schema = pq.read_schema(source_parquet) if pq is not None and source_parquet.is_file() else None
    if source_gate["status"] != "passed":
        rejects["source_gate"] += 1
    elif source_schema is not None:
        for row in _iter_rows(source_parquet):
            category = _category(row); counts[category] += 1
            if category == "excluded_public_vision_opd": rejects[category] += 1; continue
            reason = _row_contract(row, recover_provenance=recover_provenance)
            if reason is not None: rejects[reason] += 1; continue
            candidates[category].append(row); source_groups[category][str(_row_extra(row).get("source_subset"))] += 1
        if supplement_gate["status"] == "passed":
            if source_schema is None:
                rejects["supplement_source_schema"] += 1
            else:
                for row in _iter_rows(supplement_parquet):
                    category = _category(row); counts[f"supplement:{category}"] += 1
                    if category == "excluded_public_vision_opd":
                        rejects["supplement:excluded_public_vision_opd"] += 1; continue
                    reason = _row_contract(row, recover_provenance=True)
                    if reason is not None:
                        rejects[f"supplement:{reason}"] += 1; continue
                    candidates[category].append(row)
                    source_groups[category][str(_row_extra(row).get("source_subset"))] += 1
        elif supplement_parquet is not None or supplement_receipt is not None:
            rejects["supplement_gate"] += 1
    b28_collision_rejects = 0
    b28_collision_rejects = _exclude_b28_collisions(candidates, b28_filter, rejects)
    available = {key: len(value) for key, value in candidates.items()}
    selected: list[dict[str, Any]] = []; selected_counts: Counter[str] = Counter(); index = _IdentityIndex()
    if source_gate["status"] == "passed":
        for category, quota in QUOTAS.items():
            if available.get(category, 0) < quota: continue
            groups = defaultdict(list)
            for row in candidates[category]:
                groups[str(_row_extra(row).get("source_subset"))].append(row)
            # Walk the complete deterministic candidate stream.  A duplicate
            # is a rejected candidate, not a reason to stop before the exact
            # quota when later independent rows are available.
            for row in _round_robin(groups, len(candidates[category])):
                if selected_counts[category] >= quota:
                    break
                fp = _fingerprint(row); collision = index.collision(fp)
                if collision: rejects[f"selected_{category}:{collision}"] += 1; continue
                index.add(fp); selected.append(row); selected_counts[category] += 1
    exact = selected_counts == Counter(QUOTAS) and len(selected) == TARGET_ROWS
    composition = {"quotas": dict(QUOTAS), "available_safe_rows": dict(sorted(available.items())), "selected_rows": dict(sorted(selected_counts.items())), "source_subset_counts": {key: dict(sorted(value.items())) for key, value in sorted(source_groups.items())}, "category_policy": {"fine_grained_single": "one image plus fine/attribute/grounding/relation family marker", "multi_image_reasoning": "two to six images", "general_knowledge": "one image, non-fine family"}, "target_percentages": {key: value / TARGET_ROWS for key, value in QUOTAS.items()}}
    supplement_fourbench = _supplement_fourbench_gate(supplement_fourbench_receipt, supplement_gate) if supplement_requested else {"status": "not_requested"}
    b28_evidence = _b28_membership_gate(b28_membership, selected, supplement_gate) if supplement_requested else {"status": "not_requested"}
    supplement_fourbench_gate = supplement_fourbench["status"]
    b28_gate = b28_evidence["status"]
    final_gate = {"source_receipt_gate": source_gate["status"] == "passed", "supplement_source_gate": supplement_gate["status"] in {"passed", "not_requested"}, "supplement_fourbench_gate": supplement_fourbench_gate in {"passed", "not_requested"}, "b28_gate": b28_gate in {"passed", "not_requested"}, "b28_collision_gate": (not supplement_requested) or (b28_filter.get("status") == "passed" and b28_evidence.get("collision_rows") == 0), "exact_target_rows": exact, "composition_quota_gate": selected_counts == Counter(QUOTAS), "public_vision_opd_excluded": counts.get("excluded_public_vision_opd", 0) > 0 and not any(row for row in selected if _category(row) == "excluded_public_vision_opd"), "split_gate": not any(key.endswith("non_train_split") for key in rejects), "license_gate": not any(key.endswith("unproven_license") for key in rejects), "prompt_image_cardinality_gate": not any(key.endswith("prompt_image_cardinality") for key in rejects), "token_bound_gate": not any(key.endswith("token_byte_bound") for key in rejects), "internal_identity_gate": len({(_fingerprint(row)["source_id"], _fingerprint(row)["record"]) for row in selected}) == len(selected)}
    status = "published" if execute and all(final_gate.values()) else "blocked"
    outputs: dict[str, Any] = {}
    if status == "published":
        if source_schema is None: raise BuildError("source parquet schema unavailable")
        output = _write_parquet(output_root / "processed/train_10000.parquet", selected, source_schema); outputs["processed/train_10000.parquet"] = output
    b28_filter_receipt = {key: value for key, value in b28_filter.items() if key != "token_set"}
    body = {"schema_version": VERSION, "status": status, "mode": "execute" if execute else "dry_run", "aggregate_only": True, "cpu_only": True, "gpu_used": False, "network_used": False, "sample_level_output": False, "target": {"rows": TARGET_ROWS, "accepted_rows": len(selected), "gap_rows": max(0, TARGET_ROWS - len(selected))}, "composition": composition, "candidates": {"rows": dict(sorted(counts.items())), "reject_counts": dict(sorted(rejects.items()))}, "source": source_gate, "source_license_provenance": SOURCE_LICENSE_PROVENANCE, "provenance_recovery": {"enabled": recover_provenance, "aRefCOCO_unknown_license_recovered": recover_provenance, "VLAA_train_candidate_pool_recovered": recover_provenance}, "supplement": supplement_gate, "external_fourbench_receipt": supplement_fourbench, "b28_filter": {**b28_filter_receipt, "collision_reject_rows": b28_collision_rejects}, "b28_membership": b28_evidence, "decontamination": {"policy": "B28/public Vision-OPD exclusion; file/RGB exact; normalized text; exact question + pHash Hamming <=4", "fourbench_namespaces": list(FOURBENCH_NAMESPACES), "phash_radius": 4, "hard_overlap_inherited_zero": source_gate.get("status") == "passed"}, "final_gate": final_gate, "outputs": outputs, "writes_performed": len(outputs)}
    receipt = _seal(body)
    manifest_dir = output_root / "manifest"; manifest_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    _create_once(manifest_dir / "build_receipt.json", receipt)
    final_body = {"schema_version": f"{VERSION}_final_gate_v1", "status": status, "aggregate_only": True, "cpu_only": True, "gpu_used": False, "rows": TARGET_ROWS, "accepted_rows": len(selected), "gap_rows": max(0, TARGET_ROWS - len(selected)), "composition": composition, "hard_overlap": {"status": "zero" if source_gate.get("status") == "passed" else "blocked", "phash_radius": 4, "namespaces": list(FOURBENCH_NAMESPACES)}, "final_gate": final_gate, "outputs": outputs, "build_receipt_sha256": sha256_file(manifest_dir / "build_receipt.json")}
    _create_once(manifest_dir / "final_gate.json", _seal(final_body))
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--supplement-parquet", type=Path, default=None)
    parser.add_argument("--supplement-receipt", type=Path, default=None)
    parser.add_argument("--supplement-fourbench-receipt", type=Path, default=None)
    parser.add_argument("--b28-membership", type=Path, default=None)
    parser.add_argument("--recover-provenance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run == args.execute: print("choose exactly one of --dry-run/--execute", file=os.sys.stderr); return 2
    try:
        if args.dry_run:
            _disable_cuda(); gate = _source_gate(); print(json.dumps({"schema_version": VERSION, "status": "blocked" if gate["status"] != "passed" else "ready_cpu_only", "aggregate_only": True, "gpu_used": False, "source_gate": gate["status"], "target_rows": TARGET_ROWS, "quotas": QUOTAS, "output_root": str(args.output_root.absolute()), "writes_performed": 0}, sort_keys=True)); return 0
        value = build_aggregate(output_root=args.output_root, execute=True, recover_provenance=args.recover_provenance, supplement_parquet=args.supplement_parquet, supplement_receipt=args.supplement_receipt, supplement_fourbench_receipt=args.supplement_fourbench_receipt, b28_membership=args.b28_membership); print(json.dumps({"status": value["status"], "accepted_rows": value["target"]["accepted_rows"], "gap_rows": value["target"]["gap_rows"], "output_root": str(args.output_root.absolute())}, sort_keys=True)); return 0 if value["status"] == "published" else 2
    except (BuildError, OSError, ValueError, TypeError) as exc:
        print(f"build_b57_balanced_fine_multi_sft_10k_v1: {exc}", file=os.sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
