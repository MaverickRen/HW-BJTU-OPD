#!/usr/bin/env python3
"""Merge a veRL LoRA adapter into a sharded Qwen3.5 HF checkpoint.

This is the sharded-checkpoint counterpart of the pinned B24 v3/v4 merger.
It retains v3's exact PEFT loading and in-memory key-universe checks, while
accepting either a single safetensors file or a standards-compliant
``model.safetensors.index.json``.  Both the immutable base and the staged
publication are checked shard-by-shard against their index before an atomic,
create-once publication.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


V3_PATH = Path(__file__).resolve().with_name("merge_b24_lora_to_hf_v3.py")
V3_SHA256 = "bf9bb5a8dd479fb206b77bff1e76d96d253bf83196224e16d9b3773395b58cfa"
SCHEMA_VERSION = "qwen35_sharded_lora_to_hf_v1"
MAX_SHARD_SIZE = "5GB"
MTP_KEYS = (
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if _sha256(V3_PATH) != V3_SHA256:
    raise RuntimeError(f"pinned B24 v3 merger source changed: {V3_PATH}")
_SPEC = importlib.util.spec_from_file_location("qwen35_sharded_b24_v3_pinned", V3_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load pinned B24 v3 merger: {V3_PATH}")
_V3 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V3)
_V1 = _V3._V2._V1

for _name in dir(_V3):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_V3, _name))

SOURCE_PATH = Path(__file__).resolve()
SOURCE_SHA256 = _sha256(SOURCE_PATH)


def _regular(path: Path, label: str) -> None:
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise _V3.MergeError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise _V3.MergeError(f"{label} must be a single-link regular file: {path}")


def _read_index(path: Path) -> dict[str, Any]:
    value = _V1._read_json(path, "safetensors index")
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise _V3.MergeError(f"safetensors index has no weight_map: {path}")
    return value


def _checkpoint_tensor_inventory(root: Path) -> dict[str, Any]:
    """Return and verify the exact tensor-key universe of an HF checkpoint."""

    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    if index_path.exists() or index_path.is_symlink():
        _regular(index_path, "safetensors index")
        if single_path.exists() or single_path.is_symlink():
            raise _V3.MergeError("indexed checkpoint also contains model.safetensors")
        index = _read_index(index_path)
        raw_map = index["weight_map"]
        weight_map: dict[str, str] = {}
        for raw_key, raw_name in raw_map.items():
            if not isinstance(raw_key, str) or not raw_key or not isinstance(raw_name, str):
                raise _V3.MergeError("safetensors index contains a non-string key or shard")
            name = Path(raw_name)
            if name.name != raw_name or not raw_name.endswith(".safetensors"):
                raise _V3.MergeError(f"unsafe safetensors shard name: {raw_name!r}")
            weight_map[raw_key] = raw_name
        shard_names = sorted(set(weight_map.values()))
        discovered = sorted(path.name for path in root.glob("*.safetensors"))
        if discovered != shard_names:
            raise _V3.MergeError(
                f"safetensors index shard set differs: indexed={shard_names}, files={discovered}"
            )
        observed_keys: list[str] = []
        shards: list[dict[str, Any]] = []
        for name in shard_names:
            shard = root / name
            _regular(shard, f"safetensors shard {name}")
            header_keys = _V1._tensor_header_keys(shard)
            indexed_keys = sorted(key for key, value in weight_map.items() if value == name)
            if header_keys != indexed_keys:
                missing = sorted(set(indexed_keys) - set(header_keys))[:8]
                unexpected = sorted(set(header_keys) - set(indexed_keys))[:8]
                raise _V3.MergeError(
                    f"safetensors shard/header index mismatch for {name}: "
                    f"indexed={len(indexed_keys)}, observed={len(header_keys)}, "
                    f"missing={missing}, unexpected={unexpected}"
                )
            observed_keys.extend(header_keys)
            shards.append({"name": name, "bytes": shard.stat().st_size})
        keys = sorted(observed_keys)
        if len(keys) != len(set(keys)) or keys != sorted(weight_map):
            raise _V3.MergeError("sharded safetensors tensor-key universe is duplicated or incomplete")
        return {
            "layout": "sharded",
            "index": {"name": index_path.name, "sha256": _sha256(index_path)},
            "shards": shards,
            "keys": keys,
        }

    _regular(single_path, "model safetensors")
    discovered = sorted(path.name for path in root.glob("*.safetensors"))
    if discovered != [single_path.name]:
        raise _V3.MergeError(f"unindexed checkpoint has unexpected safetensors: {discovered}")
    keys = _V1._tensor_header_keys(single_path)
    if not keys or len(keys) != len(set(keys)):
        raise _V3.MergeError("single-file safetensors tensor-key universe is empty or duplicated")
    return {
        "layout": "single",
        "index": None,
        "shards": [{"name": single_path.name, "bytes": single_path.stat().st_size}],
        "keys": sorted(keys),
    }


def _servable_base_tensor_inventory(base: Path) -> dict[str, Any]:
    """Account for the exact MTP head ignored by the HF generation class.

    The raw Qwen3.5 9B and 27B publications contain the same 15 ``mtp.*``
    tensors.  ``AutoModelForImageTextToText`` intentionally does not expose
    them in ``state_dict``; consequently neither LoRA-SFT nor
    ``save_pretrained`` can preserve them.  No partial or different ignored
    set is accepted.
    """

    inventory = _checkpoint_tensor_inventory(base)
    keys = list(inventory["keys"])
    ignored = sorted(key for key in keys if key.startswith("mtp."))
    if ignored not in ([], list(MTP_KEYS)):
        raise _V3.MergeError(f"base has an unrecognized MTP tensor set: {ignored}")
    retained = sorted(set(keys) - set(ignored))
    if not retained:
        raise _V3.MergeError("base has no serving tensor keys after MTP accounting")
    return {**inventory, "raw_keys": keys, "keys": retained, "ignored_mtp_keys": ignored}


def _base_tensor_keys(base: Path) -> list[str]:
    return list(_servable_base_tensor_inventory(base)["keys"])


def _validate_staged_key_universe(staging: Path, base: Path) -> dict[str, Any]:
    expected_inventory = _servable_base_tensor_inventory(base)
    observed_inventory = _checkpoint_tensor_inventory(staging)
    expected = list(expected_inventory["keys"])
    observed = list(observed_inventory["keys"])
    if observed != expected:
        expected_set, observed_set = set(expected), set(observed)
        raise _V3.MergeError(
            "serialized model changed the exact base state-key universe: "
            f"expected={len(expected)}, observed={len(observed)}, "
            f"missing={sorted(expected_set - observed_set)[:8]}, "
            f"unexpected={sorted(observed_set - expected_set)[:8]}"
        )
    key_digest = hashlib.sha256(_V1._canonical(observed)).hexdigest()
    return {
        "save_original_format": False,
        "max_shard_size": MAX_SHARD_SIZE,
        "base_weight_layout": expected_inventory["layout"],
        "serialized_weight_layout": observed_inventory["layout"],
        "base_shard_count": len(expected_inventory["shards"]),
        "serialized_shard_count": len(observed_inventory["shards"]),
        "base_raw_tensor_key_count": len(expected_inventory["raw_keys"]),
        "base_ignored_mtp_keys": expected_inventory["ignored_mtp_keys"],
        "serialized_tensor_key_count": len(observed),
        "exact_base_key_universe_after_serialization": True,
        "serialized_tensor_keys_sha256": key_digest,
    }


def _copy_non_weight_files(base: Path, target: Path) -> None:
    """Copy HF metadata while excluding every possible checkpoint payload."""

    forbidden_markers = {"merge_complete.json", "manifest.json", "model_manifest.json", "merge_receipt.json"}
    for source in sorted(base.iterdir(), key=lambda item: item.name):
        name = source.name
        if (
            name.endswith(".safetensors")
            or name.endswith(".safetensors.index.json")
            or name.endswith(".bin")
            or name.endswith(".bin.index.json")
            or name in forbidden_markers
            or name.startswith("manifest")
        ):
            continue
        if source.is_symlink() or not source.is_file():
            continue
        shutil.copy2(source, target / name)


def _atomic_publish(
    merged: Any,
    base: Path,
    output: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise _V3.MergeError(f"create-once output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    if staging.exists() or staging.is_symlink():
        raise _V3.MergeError("staging path unexpectedly exists")
    try:
        staging.mkdir(mode=0o755)
        merged.save_pretrained(
            str(staging),
            safe_serialization=True,
            save_original_format=False,
            max_shard_size=MAX_SHARD_SIZE,
        )
        _copy_non_weight_files(base, staging)
        inventory = _V1._assert_no_lora_residue(staging)
        inventory = {**inventory, **_validate_staged_key_universe(staging, base)}
        receipt_body = dict(
            receipt,
            status="passed",
            merger_source={
                "path": str(SOURCE_PATH),
                "sha256": SOURCE_SHA256,
                "pinned_v3_path": str(V3_PATH),
                "pinned_v3_sha256": V3_SHA256,
            },
            output={"path": str(output), **inventory},
        )
        receipt_body["seal_sha256"] = hashlib.sha256(_V1._canonical(receipt_body)).hexdigest()
        (staging / "merge_receipt.json").write_bytes(_V1._canonical(receipt_body) + b"\n")
        _V1._fsync_tree(staging)
        if output.exists() or output.is_symlink():
            raise _V3.MergeError(f"create-once output appeared during merge: {output}")
        _V1._rename_noreplace(staging, output)
        return inventory
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


# v3's function resolves this name in its own module globals.
_V3._base_tensor_keys = _base_tensor_keys
_V1._atomic_publish = _atomic_publish
_V1.SCHEMA_VERSION = SCHEMA_VERSION


def main(argv: Sequence[str] | None = None) -> int:
    return _V1.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
