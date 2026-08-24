#!/usr/bin/env python3
"""Fully validate Vision-OPD-6K and emit deterministic subsets/manifests."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from PIL import Image, ImageFile, ImageOps
from scipy.fft import dctn

from vision_opd_common import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_MEDIA_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_ROOT,
    EXPECTED_RAW_SHA256,
    EXPECTED_RAW_SIZES,
    EXPECTED_ROWS,
    REPO_ID,
    REVISION,
    REVISION_SHORT,
    SEED,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    clean_question,
    iter_jsonl,
    resolve_media_path,
    sha256_bytes,
    sha256_file,
    summarize_numbers,
)


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = False
OPTION_PATTERN = re.compile(r"(?m)^\s*([A-D])\.\s+\S")
OPTION_VALUE_PATTERN = re.compile(r"(?m)^\s*([A-D])\.\s*(.*?)\s*$")
ROLES = {
    "student": "images",
    "teacher": "teacher_images",
    "original": "original_images",
}


def normalize_text(value: Any) -> str:
    """unicode_nfkc_casefold_whitespace_v1, shared with the denylist exporter."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def question_stem(item: dict[str, Any]) -> str:
    """Recover the bare source question, excluding hint, choices, and answer instruction."""
    problem = str(item.get("problem", "")).replace("<image>", "", 1).strip()
    hint_marker = "\n\nOnly focus on the objects inside the red bounding box"
    if hint_marker in problem:
        return problem.split(hint_marker, 1)[0].strip()
    source_question = str((item.get("extra_info") or {}).get("question", problem))
    return re.split(r"\n\s*\n\s*A\.\s+", source_question, maxsplit=1)[0].strip()


def normalized_prompt(item: dict[str, Any]) -> str:
    """Match export_eval_denylist.py's question-plus-options serialization."""
    question = normalize_text(question_stem(item))
    options = {key: value for key, value in OPTION_VALUE_PATTERN.findall(str(item.get("problem", "")))}
    option_parts = [f"{key}:{normalize_text(options[key])}" for key in ("A", "B", "C", "D") if key in options]
    return "\n".join([question, *option_parts])


def phash64(rgb: Image.Image) -> str:
    gray = rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = dctn(np.asarray(gray, dtype=np.float32), type=2, norm="ortho")[:8, :8]
    median = float(np.median(coefficients))
    bits = (coefficients > median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--denylist-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR.parent / "denylist/eval_primary",
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    return parser.parse_args()


def decode_and_hash(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    with Image.open(io.BytesIO(payload)) as image:
        image_format = image.format or "UNKNOWN"
        mode = image.mode
        width, height = image.size
        image.load()
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb_width, rgb_height = rgb.size
        dimension_prefix = rgb_width.to_bytes(8, "big") + rgb_height.to_bytes(8, "big")
        rgb_sha256 = sha256_bytes(dimension_prefix + rgb.tobytes())
        image_phash = phash64(rgb)
    if width <= 0 or height <= 0:
        raise ValueError(f"Non-positive decoded dimensions for {path}: {(width, height)}")
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": digest,
        "format": image_format,
        "mode": mode,
        "width": width,
        "height": height,
        "rgb_width": rgb_width,
        "rgb_height": rgb_height,
        "rgb_sha256": rgb_sha256,
        "phash64_dct_v1": image_phash,
    }


def hash_many(paths: Iterable[Path], workers: int) -> dict[Path, str]:
    ordered = sorted(set(paths))
    results: dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(sha256_file, path): path for path in ordered}
        for future in as_completed(futures):
            path = futures[future]
            results[path] = future.result()
    return results


def write_parquet_atomic(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(
            table,
            temporary_path,
            compression="zstd",
            compression_level=9,
            use_dictionary=True,
            write_statistics=True,
            data_page_version="2.0",
        )
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def stable_subset_indices(rows: list[dict[str, Any]], count: int) -> list[int]:
    """Cross-runtime deterministic seed-42 selection using a SHA-256 ranking."""
    ranked: list[tuple[str, int]] = []
    for index, row in enumerate(rows):
        selection_key = hashlib.sha256(
            f"seed={SEED}|{row['source_id']}".encode("utf-8")
        ).hexdigest()
        ranked.append((selection_key, index))
    return sorted(index for _, index in sorted(ranked)[:count])


def subset_entries(rows: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "row_index": index,
            "source_id": rows[index]["source_id"],
            "source_record_sha256": rows[index]["source_record_sha256"],
        }
        for index in indices
    ]


def load_hash_set(path: Path, expected_hex_length: int) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required evaluation denylist is missing: {path}")
    values = {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    invalid = [value for value in values if len(value) != expected_hex_length or not re.fullmatch(r"[0-9a-f]+", value)]
    if invalid:
        raise ValueError(f"Invalid hashes in {path}: {invalid[:5]}")
    return values


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def write_filtered_outputs(
    table: pa.Table,
    processed_rows: list[dict[str, Any]],
    excluded_indices: set[int],
    output_dir: Path,
) -> dict[str, Any]:
    kept_indices = [index for index in range(table.num_rows) if index not in excluded_indices]
    filtered_table = table.take(pa.array(kept_indices, type=pa.int64()))
    filtered_rows = [processed_rows[index] for index in kept_indices]
    outputs: dict[str, Any] = {}

    train_path = output_dir / "train_decontaminated.parquet"
    write_parquet_atomic(filtered_table, train_path)
    reread_train = pq.read_table(train_path)
    if reread_train.num_rows != filtered_table.num_rows or reread_train.schema != table.schema:
        raise ValueError("train_decontaminated.parquet failed row-count/schema round-trip validation")
    outputs[train_path.name] = {
        "rows": filtered_table.num_rows,
        "bytes": train_path.stat().st_size,
        "sha256": sha256_file(train_path),
    }

    for filename, count in (("smoke_8_decontaminated.parquet", 8), ("pilot_96_decontaminated.parquet", 96)):
        if len(filtered_rows) < count:
            raise ValueError(f"Cannot create {filename}: only {len(filtered_rows)} filtered rows remain")
        local_indices = stable_subset_indices(filtered_rows, count)
        subset_table = filtered_table.take(pa.array(local_indices, type=pa.int64()))
        subset_path = output_dir / filename
        write_parquet_atomic(subset_table, subset_path)
        reread_subset = pq.read_table(subset_path)
        if reread_subset.num_rows != count or reread_subset.schema != table.schema:
            raise ValueError(f"{filename} failed row-count/schema round-trip validation")
        outputs[filename] = {
            "rows": count,
            "seed": SEED,
            "selection_method": "sha256_rank(seed=42,source_id)",
            "bytes": subset_path.stat().st_size,
            "sha256": sha256_file(subset_path),
            "records": subset_entries(filtered_rows, local_indices),
        }
    return outputs


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    media_root = args.media_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest_dir = args.manifest_dir.resolve()
    denylist_dir = args.denylist_dir.resolve()
    workers = max(1, args.workers)
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    raw_jsonl = raw_root / "train.jsonl"
    parquet_path = output_dir / "train.parquet"
    if not raw_jsonl.is_file():
        raise FileNotFoundError(raw_jsonl)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    for relative, expected_size in EXPECTED_RAW_SIZES.items():
        path = raw_root / relative
        check(path.is_file(), f"Missing raw file: {path}")
        if path.is_file():
            check(
                path.stat().st_size == expected_size,
                f"Raw size mismatch for {path}: expected {expected_size}, got {path.stat().st_size}",
            )
    if errors:
        raise ValueError("Raw snapshot verification failed:\n" + "\n".join(errors))

    raw_rows = [item for _, item in iter_jsonl(raw_jsonl)]
    check(len(raw_rows) == EXPECTED_ROWS, f"JSONL rows: expected {EXPECTED_ROWS}, got {len(raw_rows)}")
    table = pq.read_table(parquet_path)
    processed_rows = table.to_pylist()
    check(
        table.num_rows == EXPECTED_ROWS,
        f"Parquet rows: expected {EXPECTED_ROWS}, got {table.num_rows}",
    )
    check(len(processed_rows) == len(raw_rows), "Raw and parquet row counts differ")

    answer_counts: Counter[str] = Counter()
    placeholder_counts: Counter[int] = Counter()
    path_roles: dict[Path, set[str]] = defaultdict(set)
    referenced_paths: dict[str, list[Path]] = {role: [] for role in ROLES}
    raw_original_by_row: list[Path | None] = []
    raw_student_by_row: list[Path | None] = []
    resolved_paths_by_row: list[dict[str, list[Path]]] = []
    bboxes: list[list[int]] = []
    source_ids: set[str] = set()
    source_hashes: list[str] = []

    for index, item in enumerate(raw_rows):
        prefix = f"row {index}"
        required = ("images", "teacher_images", "original_images", "bbox", "problem", "answer", "extra_info")
        missing = [field for field in required if field not in item]
        check(not missing, f"{prefix}: missing fields {missing}")
        if missing:
            raw_original_by_row.append(None)
            raw_student_by_row.append(None)
            resolved_paths_by_row.append({role: [] for role in ROLES})
            bboxes.append([])
            continue

        problem = item["problem"]
        answer = item["answer"]
        check(isinstance(problem, str) and problem, f"{prefix}: problem must be non-empty text")
        check(answer in {"A", "B", "C", "D"}, f"{prefix}: invalid A-D answer {answer!r}")
        if answer in {"A", "B", "C", "D"}:
            answer_counts[answer] += 1
        options = OPTION_PATTERN.findall(problem) if isinstance(problem, str) else []
        check(
            Counter(options) == Counter({"A": 1, "B": 1, "C": 1, "D": 1}),
            f"{prefix}: expected exactly one option each for A-D, got {options}",
        )
        placeholders = problem.count("<image>") if isinstance(problem, str) else 0
        placeholder_counts[placeholders] += 1
        images = item.get("images")
        check(isinstance(images, list), f"{prefix}: images is not a list")
        if isinstance(images, list):
            check(placeholders == len(images), f"{prefix}: {placeholders} placeholders != {len(images)} images")
            check(len(images) == 1, f"{prefix}: expected one student image, got {len(images)}")

        extra = item.get("extra_info")
        check(isinstance(extra, dict), f"{prefix}: extra_info is not an object")
        if isinstance(extra, dict):
            check(extra.get("answer") == answer, f"{prefix}: extra_info.answer differs from answer")

        row_paths: dict[str, list[Path]] = {}
        for role, field in ROLES.items():
            values = item.get(field)
            check(isinstance(values, list), f"{prefix}: {field} is not a list")
            if not isinstance(values, list):
                row_paths[role] = []
                continue
            check(len(values) == 1, f"{prefix}: expected one {field} entry, got {len(values)}")
            resolved_values: list[Path] = []
            for value in values:
                try:
                    resolved = resolve_media_path(media_root, value)
                except (TypeError, ValueError) as error:
                    errors.append(f"{prefix}: invalid {field} path {value!r}: {error}")
                    continue
                check(resolved.is_file(), f"{prefix}: missing {role} image {resolved}")
                if resolved.is_file():
                    resolved_values.append(resolved)
                    referenced_paths[role].append(resolved)
                    path_roles[resolved].add(role)
            row_paths[role] = resolved_values

        raw_student_by_row.append(row_paths.get("student", [None])[0] if row_paths.get("student") else None)
        raw_original_by_row.append(row_paths.get("original", [None])[0] if row_paths.get("original") else None)
        resolved_paths_by_row.append(row_paths)

        bbox = item.get("bbox")
        bbox_valid = (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(value, int) and not isinstance(value, bool) for value in bbox)
        )
        check(bbox_valid, f"{prefix}: bbox must be four integers, got {bbox!r}")
        if bbox_valid:
            check(bbox[0] < bbox[2] and bbox[1] < bbox[3], f"{prefix}: unordered bbox {bbox}")
            bboxes.append(bbox)
        else:
            bboxes.append([])

        if index >= len(processed_rows):
            continue
        processed = processed_rows[index]
        source_record_json = canonical_json(item)
        source_hash = sha256_bytes(source_record_json.encode("utf-8"))
        expected_source_id = f"vision_opd_6k_{index:04d}_{source_hash[:12]}"
        check(processed.get("source_id") == expected_source_id, f"{prefix}: source_id mismatch")
        check(processed.get("source_record_sha256") == source_hash, f"{prefix}: source hash mismatch")
        check(processed.get("data_source") == "zwz_rl_vqa_bbox_teacher", f"{prefix}: data_source mismatch")
        check(processed.get("ability") == "visual_question_answering", f"{prefix}: ability mismatch")
        check(
            processed.get("prompt") == [{"role": "user", "content": problem}],
            f"{prefix}: processed prompt differs from raw problem",
        )
        expected_images = [{"path": str(path), "image": str(path)} for path in row_paths.get("student", [])]
        expected_teacher = [{"path": str(path), "image": str(path)} for path in row_paths.get("teacher", [])]
        expected_original = [{"path": str(path), "image": str(path)} for path in row_paths.get("original", [])]
        check(processed.get("images") == expected_images, f"{prefix}: processed student paths mismatch")
        check(processed.get("bbox_images") == expected_teacher, f"{prefix}: processed teacher paths mismatch")
        reward_model = processed.get("reward_model") or {}
        check(reward_model.get("ground_truth") == answer, f"{prefix}: ground truth mismatch")
        check(reward_model.get("style") == "none", f"{prefix}: reward style mismatch")
        processed_extra = processed.get("extra_info") or {}
        check(processed_extra.get("question") == clean_question(problem), f"{prefix}: cleaned question mismatch")
        check(processed_extra.get("answer") == answer, f"{prefix}: processed answer mismatch")
        check(processed_extra.get("row_index") == index, f"{prefix}: processed row_index mismatch")
        check(processed_extra.get("source_revision") == REVISION, f"{prefix}: revision mismatch")
        check(processed_extra.get("source_record_json") == source_record_json, f"{prefix}: source JSON mismatch")
        check(processed_extra.get("bbox") == bbox, f"{prefix}: processed bbox mismatch")
        check(
            processed_extra.get("original_images") == expected_original,
            f"{prefix}: processed original paths mismatch",
        )
        source_id = processed.get("source_id")
        if isinstance(source_id, str):
            check(source_id not in source_ids, f"{prefix}: duplicate source_id {source_id}")
            source_ids.add(source_id)
        source_hashes.append(source_hash)

    if errors:
        raise ValueError(
            f"Structural validation found {len(errors)} error(s); first 50:\n"
            + "\n".join(errors[:50])
        )

    unique_media_paths = sorted(path_roles)
    print(f"Fully decoding and hashing {len(unique_media_paths):,} unique referenced images", flush=True)
    image_results: dict[Path, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(decode_and_hash, path): path for path in unique_media_paths}
        completed = 0
        for future in as_completed(futures):
            path = futures[future]
            image_results[path] = future.result()
            completed += 1
            if completed % 1_000 == 0 or completed == len(unique_media_paths):
                print(f"Decoded {completed:,}/{len(unique_media_paths):,} images", flush=True)

    for index, bbox in enumerate(bboxes):
        if not bbox or raw_original_by_row[index] is None:
            continue
        original = image_results[raw_original_by_row[index]]
        width, height = original["width"], original["height"]
        check(
            0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height,
            f"row {index}: bbox {bbox} is outside original image {(width, height)}",
        )
        student_path = raw_student_by_row[index]
        if student_path is not None:
            student = image_results[student_path]
            check(
                (student["width"], student["height"]) == (width, height),
                f"row {index}: student/original dimensions differ",
            )

    actual_files: dict[str, set[Path]] = {}
    for role, field in ROLES.items():
        directory = media_root / field
        files = {
            path.resolve()
            for path in directory.rglob("*")
            if path.is_file() and not path.name.endswith(".extracting")
        }
        actual_files[role] = files
        referenced = set(referenced_paths[role])
        check(files == referenced, f"{role}: actual media files differ from referenced files")

    if errors:
        raise ValueError(
            f"Image/bbox validation found {len(errors)} error(s); first 50:\n"
            + "\n".join(errors[:50])
        )

    # Primary-evaluation leakage audit, rule decontam_v2. Generic question-only
    # matches are audited but retained unless backed by a near-identical image.
    question_denylist_path = denylist_dir / "question_sha256.txt"
    prompt_denylist_path = denylist_dir / "prompt_sha256.txt"
    rgb_denylist_path = denylist_dir / "image_rgb_sha256.txt"
    phash_denylist_path = denylist_dir / "image_phash64_dct_v1.txt"
    question_denylist = load_hash_set(question_denylist_path, 64)
    prompt_denylist = load_hash_set(prompt_denylist_path, 64)
    rgb_denylist = load_hash_set(rgb_denylist_path, 64)
    phash_denylist = load_hash_set(phash_denylist_path, 16)

    question_matches: list[dict[str, Any]] = []
    prompt_matches: list[dict[str, Any]] = []
    rgb_matches: list[dict[str, Any]] = []
    phash_candidates: list[dict[str, Any]] = []
    question_exact_indices: set[int] = set()
    prompt_exact_indices: set[int] = set()
    rgb_exact_indices: set[int] = set()
    phash_le4_indices: set[int] = set()
    rgb_matches_by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    phash_matches_by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for index, item in enumerate(raw_rows):
        source_id = processed_rows[index]["source_id"]
        question_hash = sha256_bytes(normalize_text(question_stem(item)).encode("utf-8"))
        prompt_hash = sha256_bytes(normalized_prompt(item).encode("utf-8"))
        if question_hash in question_denylist:
            question_exact_indices.add(index)
            question_matches.append(
                {
                    "row_index": index,
                    "source_id": source_id,
                    "source_record_sha256": processed_rows[index]["source_record_sha256"],
                    "question_sha256": question_hash,
                }
            )
        if prompt_hash in prompt_denylist:
            prompt_exact_indices.add(index)
            prompt_matches.append(
                {
                    "row_index": index,
                    "source_id": source_id,
                    "source_record_sha256": processed_rows[index]["source_record_sha256"],
                    "prompt_sha256": prompt_hash,
                }
            )

        for role, paths in resolved_paths_by_row[index].items():
            for path in paths:
                result = image_results[path]
                rgb_hash = result["rgb_sha256"]
                if rgb_hash in rgb_denylist:
                    rgb_exact_indices.add(index)
                    match = {
                        "row_index": index,
                        "source_id": source_id,
                        "source_record_sha256": processed_rows[index]["source_record_sha256"],
                        "role": role,
                        "media_path": path.relative_to(media_root).as_posix(),
                        "image_rgb_sha256": rgb_hash,
                    }
                    rgb_matches.append(match)
                    rgb_matches_by_row[index].append(match)

                distances = [(hamming_hex(result["phash64_dct_v1"], candidate), candidate) for candidate in phash_denylist]
                best_distance, best_hash = min(distances)
                if best_distance <= 4:
                    phash_le4_indices.add(index)
                    candidate = {
                        "row_index": index,
                        "source_id": source_id,
                        "source_record_sha256": processed_rows[index]["source_record_sha256"],
                        "role": role,
                        "media_path": path.relative_to(media_root).as_posix(),
                        "image_phash64_dct_v1": result["phash64_dct_v1"],
                        "denylist_phash64_dct_v1": best_hash,
                        "hamming_distance": best_distance,
                        "canonical_rgb_exact": rgb_hash in rgb_denylist,
                    }
                    phash_candidates.append(candidate)
                    phash_matches_by_row[index].append(candidate)

    question_plus_phash_indices = question_exact_indices & phash_le4_indices
    excluded_indices = rgb_exact_indices | prompt_exact_indices | question_plus_phash_indices
    question_only_indices = question_exact_indices - excluded_indices
    question_and_rgb_indices = question_exact_indices & rgb_exact_indices
    multi_hard_signal_indices = {
        index
        for index in excluded_indices
        if sum(
            (
                index in rgb_exact_indices,
                index in prompt_exact_indices,
                index in question_plus_phash_indices,
            )
        )
        >= 2
    }
    exclusion_records: list[dict[str, Any]] = []
    for index in sorted(excluded_indices):
        item = raw_rows[index]
        question_hash = sha256_bytes(normalize_text(question_stem(item)).encode("utf-8"))
        prompt_hash = sha256_bytes(normalized_prompt(item).encode("utf-8"))
        reasons = []
        if index in rgb_exact_indices:
            reasons.append("canonical_rgb_exact")
        if index in prompt_exact_indices:
            reasons.append("normalized_full_prompt_exact")
        if index in question_plus_phash_indices:
            reasons.append("question_exact_and_phash_hamming_le_4")
        exclusion_records.append(
            {
                "row_index": index,
                "source_id": processed_rows[index]["source_id"],
                "source_record_sha256": processed_rows[index]["source_record_sha256"],
                "question_sha256": question_hash,
                "prompt_sha256": prompt_hash,
                "question_exact": index in question_exact_indices,
                "prompt_exact": index in prompt_exact_indices,
                "rgb_exact": index in rgb_exact_indices,
                "question_exact_and_phash_hamming_le_4": index in question_plus_phash_indices,
                "filter_reasons": reasons,
                "rgb_matches": [
                    {key: value for key, value in match.items() if key in {"role", "media_path", "image_rgb_sha256"}}
                    for match in rgb_matches_by_row.get(index, [])
                ],
                "phash_matches": [
                    {
                        key: value
                        for key, value in match.items()
                        if key
                        in {
                            "role",
                            "media_path",
                            "image_phash64_dct_v1",
                            "denylist_phash64_dct_v1",
                            "hamming_distance",
                        }
                    }
                    for match in phash_matches_by_row.get(index, [])
                ],
            }
        )

    decontaminated_outputs = write_filtered_outputs(table, processed_rows, excluded_indices, output_dir)
    exclusion_path = manifest_dir / f"vision_opd_6k_{REVISION_SHORT}_eval_overlap_exclusions.jsonl"
    exclusion_payload = "".join(canonical_json(record) + "\n" for record in exclusion_records)
    atomic_write_text(exclusion_path, exclusion_payload)
    print(
        f"Eval denylist decontam_v2 excluded {len(excluded_indices)} rows; "
        "wrote independent decontaminated outputs",
        flush=True,
    )

    # Subsets are created only after every full-dataset structural and decode check passes.
    subset_specs = {"smoke_8.parquet": 8, "pilot_96.parquet": 96}
    subsets: dict[str, dict[str, Any]] = {}
    for filename, count in subset_specs.items():
        indices = stable_subset_indices(processed_rows, count)
        subset_table = table.take(pa.array(indices, type=pa.int64()))
        subset_path = output_dir / filename
        write_parquet_atomic(subset_table, subset_path)
        reread = pq.read_table(subset_path)
        check(reread.num_rows == count, f"{filename}: expected {count} rows, got {reread.num_rows}")
        check(reread.schema == table.schema, f"{filename}: schema differs from train.parquet")
        subsets[filename] = {
            "rows": count,
            "seed": SEED,
            "selection_method": "sha256_rank(seed=42,source_id)",
            "bytes": subset_path.stat().st_size,
            "sha256": sha256_file(subset_path),
            "records": subset_entries(processed_rows, indices),
        }

    if errors:
        raise ValueError("Subset validation failed:\n" + "\n".join(errors))

    print("Hashing pinned raw files", flush=True)
    raw_files = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(raw_root).parts
    )
    raw_hashes = hash_many(raw_files, min(4, workers))
    for relative, expected_digest in EXPECTED_RAW_SHA256.items():
        path = raw_root / relative
        check(
            raw_hashes.get(path) == expected_digest,
            f"Raw SHA-256 mismatch for {path}: expected {expected_digest}, got {raw_hashes.get(path)}",
        )
    if errors:
        raise ValueError("Raw SHA-256 verification failed:\n" + "\n".join(errors))
    processed_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    processed_hashes = hash_many(processed_files, min(4, workers))

    checksum_records: list[tuple[str, str, int]] = []
    for path in raw_files:
        checksum_records.append((f"raw/{path.relative_to(raw_root).as_posix()}", raw_hashes[path], path.stat().st_size))
    for path, result in image_results.items():
        checksum_records.append((f"media/{path.relative_to(media_root).as_posix()}", result["sha256"], result["bytes"]))
    for path in processed_files:
        checksum_records.append(
            (f"processed/{path.relative_to(output_dir).as_posix()}", processed_hashes[path], path.stat().st_size)
        )
    checksum_records.sort()
    checksum_payload = "".join(f"{digest}  {logical_path}\n" for logical_path, digest, _ in checksum_records)
    checksum_path = manifest_dir / f"vision_opd_6k_{REVISION_SHORT}_files.sha256"
    atomic_write_text(checksum_path, checksum_payload)

    role_stats: dict[str, Any] = {}
    for role in ROLES:
        paths = sorted(set(referenced_paths[role]))
        digests = [image_results[path]["sha256"] for path in paths]
        role_stats[role] = {
            "references": len(referenced_paths[role]),
            "unique_paths": len(paths),
            "unique_content_sha256": len(set(digests)),
            "unique_canonical_rgb_sha256": len({image_results[path]["rgb_sha256"] for path in paths}),
            "duplicate_paths_by_content": len(paths) - len(set(digests)),
            "formats": dict(sorted(Counter(image_results[path]["format"] for path in paths).items())),
            "modes": dict(sorted(Counter(image_results[path]["mode"] for path in paths).items())),
            "width": summarize_numbers(image_results[path]["width"] for path in paths),
            "height": summarize_numbers(image_results[path]["height"] for path in paths),
            "bytes": sum(image_results[path]["bytes"] for path in paths),
        }

    manifest = {
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "rows": len(raw_rows),
            "dataset_source_record_sha256": sha256_bytes("\n".join(source_hashes).encode("ascii")),
            "processed_image_struct_compatibility": {
                "schema": {"path": "absolute media path", "image": "same absolute media path"},
                "reason": "Preserve upstream Vision-OPD path while satisfying qwen_vl_utils image/image_url dispatch",
                "target_stack": "verl@11c94ad + qwen-vl-utils==0.0.14",
                "raw_records_modified": False,
            },
        },
        "paths": {
            "raw_root": str(raw_root),
            "media_root": str(media_root),
            "output_dir": str(output_dir),
            "train_parquet": str(parquet_path),
            "eval_denylist_dir": str(denylist_dir),
            "eval_overlap_exclusions": str(exclusion_path) if exclusion_path else None,
        },
        "validation": {
            "jsonl_rows": len(raw_rows),
            "parquet_rows": table.num_rows,
            "parquet_columns": table.num_columns,
            "unique_source_ids": len(source_ids),
            "answer_counts": dict(sorted(answer_counts.items())),
            "image_placeholder_counts": {str(key): value for key, value in sorted(placeholder_counts.items())},
            "all_options_exactly_A_through_D": True,
            "all_answers_in_A_through_D": True,
            "all_paths_exist": True,
            "all_images_fully_decoded": True,
            "decoded_unique_images": len(image_results),
            "all_bboxes_inside_original_images": True,
            "student_dimensions_match_original": True,
        },
        "images": role_stats,
        "bbox": {
            "x1": summarize_numbers(bbox[0] for bbox in bboxes),
            "y1": summarize_numbers(bbox[1] for bbox in bboxes),
            "x2": summarize_numbers(bbox[2] for bbox in bboxes),
            "y2": summarize_numbers(bbox[3] for bbox in bboxes),
            "width": summarize_numbers(bbox[2] - bbox[0] for bbox in bboxes),
            "height": summarize_numbers(bbox[3] - bbox[1] for bbox in bboxes),
        },
        "eval_overlap_audit": {
            "filter_rule_version": "decontam_v2",
            "denylist_normalization": "unicode_nfkc_casefold_whitespace_v1",
            "canonical_rgb": "EXIF transpose; RGB; SHA256(width_u64be || height_u64be || rgb_bytes)",
            "normalized_full_prompt": "normalized_question + newline + A:normalized_option ... D:normalized_option",
            "hard_filter_rule": "exclude if (a) any student/teacher/original canonical-RGB SHA256 exact; OR (b) normalized full-prompt SHA256 exact; OR (c) normalized-question SHA256 exact AND any image pHash Hamming<=4",
            "question_only_policy": "audit and retain",
            "phash_rule": "32x32 grayscale DCT-II ortho; top-left 8x8 > median; Hamming<=4 is report-only unless paired with exact normalized question",
            "denylist_files": {
                "question_sha256": {
                    "path": str(question_denylist_path),
                    "entries": len(question_denylist),
                    "sha256": sha256_file(question_denylist_path),
                },
                "prompt_sha256": {
                    "path": str(prompt_denylist_path),
                    "entries": len(prompt_denylist),
                    "sha256": sha256_file(prompt_denylist_path),
                },
                "image_rgb_sha256": {
                    "path": str(rgb_denylist_path),
                    "entries": len(rgb_denylist),
                    "sha256": sha256_file(rgb_denylist_path),
                },
                "image_phash64_dct_v1": {
                    "path": str(phash_denylist_path),
                    "entries": len(phash_denylist),
                    "sha256": sha256_file(phash_denylist_path),
                },
            },
            "match_strata": {
                "normalized_question_exact_audit_rows": len(question_exact_indices),
                "question_only_retained_rows": len(question_only_indices),
                "normalized_full_prompt_exact_rows": len(prompt_exact_indices),
                "canonical_rgb_exact_rows": len(rgb_exact_indices),
                "question_exact_and_phash_hamming_le_4_rows": len(question_plus_phash_indices),
                "question_and_rgb_exact_rows": len(question_and_rgb_indices),
                "multiple_hard_signals_rows": len(multi_hard_signal_indices),
            },
            "excluded_union_rows": len(excluded_indices),
            "question_matches": question_matches,
            "question_only_retained_source_ids": [
                processed_rows[index]["source_id"] for index in sorted(question_only_indices)
            ],
            "prompt_matches": prompt_matches,
            "rgb_matches": rgb_matches,
            "question_and_rgb_exact_source_ids": [
                processed_rows[index]["source_id"] for index in sorted(question_and_rgb_indices)
            ],
            "multiple_hard_signals_source_ids": [
                processed_rows[index]["source_id"] for index in sorted(multi_hard_signal_indices)
            ],
            "phash_hamming_le_4_candidates": phash_candidates,
            "decontaminated_outputs": decontaminated_outputs,
            "exclusion_manifest": {
                "path": str(exclusion_path),
                "rows": len(exclusion_records),
                "sha256": sha256_file(exclusion_path),
            },
        },
        "subsets": subsets,
        "hash_inventory": {
            "path": str(checksum_path),
            "sha256": sha256_bytes(checksum_payload.encode("utf-8")),
            "files": len(checksum_records),
            "bytes": sum(size for _, _, size in checksum_records),
            "raw_files": len(raw_files),
            "media_files": len(image_results),
            "processed_files": len(processed_files),
        },
    }
    manifest_path = manifest_dir / f"vision_opd_6k_{REVISION_SHORT}_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(f"Validation PASSED for {len(raw_rows):,} rows", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"SHA-256 inventory: {checksum_path}", flush=True)


if __name__ == "__main__":
    main()
