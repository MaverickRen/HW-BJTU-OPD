#!/usr/bin/env python3
"""Materialize the pinned ZoomBench release and export contamination hashes.

This is intentionally separate from VLMEvalKit: ZoomBench is not registered in
the pinned VLMEvalKit checkout.  The output JSON follows the audited
Vision-OPD reference evaluator's input schema, while the source snapshot is
pinned for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

from export_eval_denylist import (
    SCHEMA_VERSION,
    hash_image,
    normalize_text,
    rebuild_aggregate,
    sha256_bytes,
    sha256_file,
    write_dataset_outputs,
)


DATASET_ID = "inclusionAI/ZoomBench"
DATASET_REVISION = "b788097e57d30510c6877824833234a73bf80d25"
EXPECTED_ROWS = 845
REFERENCE_COMMIT = "c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471"
OFFICIAL_EVAL_COMMIT = "fdc0ba1a3dee916d8c38304d543ad414879e0c99"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--denylist-root", required=True, type=Path)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def materialize(path: Path, payload: bytes) -> None:
    if path.is_file() and sha256_file(path) == sha256_bytes(payload):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def image_bytes(value: Any, row_number: int, field: str) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("bytes")
    if value is None:
        return None
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"row {row_number}: {field} does not contain bytes")
    return bytes(value)


def image_suffix(payload: bytes) -> str:
    with Image.open(io.BytesIO(payload)) as image:
        fmt = (image.format or "").upper()
    return {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "TIFF": ".tiff",
        "BMP": ".bmp",
    }.get(fmt, ".img")


def complete(dataset_root: Path, denylist_root: Path, revision: str) -> bool:
    manifest_path = dataset_root / "manifest.json"
    denylist_meta = denylist_root / "datasets" / "ZoomBench" / "metadata.json"
    benchmark_json = dataset_root / "zoombench.json"
    if not (manifest_path.is_file() and denylist_meta.is_file() and benchmark_json.is_file()):
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("dataset_revision") == revision
        and manifest.get("rows") == EXPECTED_ROWS
        and manifest.get("materialization_complete") is True
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    denylist_root = args.denylist_root.resolve()
    if complete(dataset_root, denylist_root, args.revision) and not args.force:
        print(f"Pinned ZoomBench is already complete: {dataset_root}")
        return

    denylist_dir = denylist_root / "datasets" / "ZoomBench"
    if denylist_dir.exists() and not args.force:
        raise FileExistsError(
            f"Incomplete output has an existing denylist at {denylist_dir}; "
            "inspect it, then rerun with --force"
        )
    if denylist_dir.exists():
        expected = {"records.jsonl", "images.jsonl", "metadata.json"}
        unexpected = {path.name for path in denylist_dir.iterdir()} - expected
        if unexpected:
            raise RuntimeError(f"Refusing to replace denylist with unexpected files: {unexpected}")
        for name in expected:
            path = denylist_dir / name
            if path.exists():
                path.unlink()
        denylist_dir.rmdir()

    dataset_root.mkdir(parents=True, exist_ok=True)
    snapshot_root = dataset_root / "snapshot"
    snapshot_download(
        DATASET_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(snapshot_root),
        allow_patterns=["README.md", "data/test.parquet", ".gitattributes"],
    )
    parquet_path = snapshot_root / "data" / "test.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    parquet_file = pq.ParquetFile(parquet_path)
    schema_names = set(parquet_file.schema_arrow.names)
    required = {"id", "query", "response", "image", "crop_image"}
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"ZoomBench schema is missing required columns: {missing}")
    columns = ["id", "query", "response", "image", "crop_image"]
    if "question_type" in schema_names:
        columns.append("question_type")

    full_root = dataset_root / "images" / "full"
    crop_root = dataset_root / "images" / "crop"
    benchmark_records: list[dict[str, Any]] = []
    denylist_records: list[dict[str, Any]] = []
    images_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    question_types: dict[str, int] = {}
    row_number = 0

    for batch in parquet_file.iter_batches(batch_size=8, columns=columns):
        for row in batch.to_pylist():
            full_payload = image_bytes(row.get("image"), row_number, "image")
            if full_payload is None:
                raise ValueError(f"row {row_number}: missing full image")
            crop_payload = image_bytes(row.get("crop_image"), row_number, "crop_image")

            full_path = full_root / f"{row_number:06d}{image_suffix(full_payload)}"
            materialize(full_path, full_payload)
            crop_path: Path | None = None
            if crop_payload is not None:
                crop_path = crop_root / f"{row_number:06d}{image_suffix(crop_payload)}"
                materialize(crop_path, crop_payload)

            question = str(row.get("query") or "").strip()
            answer = str(row.get("response") or "").strip()
            question_type = str(row.get("question_type") or "unknown")
            question_types[question_type] = question_types.get(question_type, 0) + 1
            sample_id = str(row.get("id") or row_number)
            benchmark_records.append(
                {
                    "index": row_number,
                    "id": sample_id,
                    "images": [str(full_path)],
                    "crop_images": [str(crop_path)] if crop_path is not None else [],
                    "query": question,
                    "response": answer,
                    "question_type": question_type,
                }
            )

            row_images: list[dict[str, Any]] = []
            for role, payload, path in (
                ("full", full_payload, full_path),
                ("crop", crop_payload, crop_path),
            ):
                if payload is None or path is None:
                    continue
                logical_path = str(path.relative_to(dataset_root.parent))
                image_info = hash_image(payload, logical_path)
                image_info["role"] = role
                images_by_key[(image_info["file_sha256"], logical_path)] = image_info
                row_images.append(image_info)

            normalized_question = normalize_text(question)
            denylist_records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset": "ZoomBench",
                    "index": str(row_number),
                    "sample_id_sha256": sha256_bytes(sample_id.encode("utf-8")),
                    "question_sha256": sha256_bytes(normalized_question.encode("utf-8")),
                    "prompt_sha256": sha256_bytes(normalized_question.encode("utf-8")),
                    "image_file_sha256": [item["file_sha256"] for item in row_images],
                    "image_rgb_sha256": [item["rgb_sha256"] for item in row_images],
                    "image_phash64_dct_v1": [item["phash64_dct_v1"] for item in row_images],
                }
            )
            row_number += 1

    if row_number != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, materialized {row_number}")

    benchmark_json = dataset_root / "zoombench.json"
    atomic_write_text(
        benchmark_json,
        json.dumps(benchmark_records, ensure_ascii=False, indent=2) + "\n",
    )
    images = sorted(images_by_key.values(), key=lambda item: (item["file_sha256"], item["logical_path"]))
    denylist_metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "ZoomBench",
        "rows": row_number,
        "unique_image_files": len({item["file_sha256"] for item in images}),
        "image_manifest_rows": len(images),
        "source_parquet": str(parquet_path),
        "source_parquet_sha256": sha256_file(parquet_path),
        "source_parquet_size_bytes": parquet_path.stat().st_size,
        "materialized_images": True,
        "includes_crop_oracle_images": True,
    }
    denylist_dir.mkdir(parents=True, exist_ok=False)
    write_dataset_outputs(denylist_dir, denylist_records, images, denylist_metadata)
    rebuild_aggregate(denylist_root)

    manifest = {
        "benchmark": "ZoomBench",
        "native_vlmevalkit": False,
        "dataset_id": DATASET_ID,
        "dataset_revision": args.revision,
        "rows": row_number,
        "question_type_counts": dict(sorted(question_types.items())),
        "source_parquet": str(parquet_path),
        "source_parquet_sha256": denylist_metadata["source_parquet_sha256"],
        "source_parquet_size_bytes": denylist_metadata["source_parquet_size_bytes"],
        "benchmark_json": str(benchmark_json),
        "benchmark_json_sha256": sha256_file(benchmark_json),
        "full_images": sum(1 for record in benchmark_records if record["images"]),
        "crop_images": sum(1 for record in benchmark_records if record["crop_images"]),
        "vision_opd_reference_commit": REFERENCE_COMMIT,
        "official_eval_commit_audited": OFFICIAL_EVAL_COMMIT,
        "primary_protocol": "full image only; crop image is oracle diagnostic and excluded",
        "scoring": "Vision-OPD/official hybrid MathRuler then Qwen LLM semantic judge",
        "materialization_complete": True,
    }
    atomic_write_text(dataset_root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
