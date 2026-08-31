#!/usr/bin/env python3
"""Download and materialize the exact 191-row VStar evaluation snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

REPO_ID = "lmms-lab/vstar-bench"
REVISION = "b44023b4dca749ed8a76b85eb576627d05a1c174"
SOURCE_PARQUET = "data/test-00000-of-00001.parquet"
SOURCE_SHA256 = "6f9a089e93e75931350157544f8e74713d9d108c5f73585609ca262c33528a27"
EXPECTED_JSON_SHA256 = "96312af877063b80b2a73936e3a2d223d8f3cd6966672759333bdf71e5dd6710"
EXPECTED_LOGICAL_ROWS_SHA256 = "1025f0c463281944034cd79f806aa80d44a82b35453463ab4288429c46f64a2f"
EXPECTED_IMAGE_MANIFEST_SHA256 = "0874addfe852aad054879a7020bf3dae880b6ea49ffb7f94eab5b23362444a6c"
EXPECTED_ROWS = 191
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class PreparationError(RuntimeError):
    """Raised when the pinned input or materialized output differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verify_prepared(output: Path) -> dict[str, object]:
    output = output.expanduser()
    if output.is_symlink():
        raise PreparationError("output cannot be a symlink")
    output = output.resolve()
    manifest_path = output / "manifest.json"
    data_path = output / "vstar.json"
    if not manifest_path.is_file() or not data_path.is_file():
        raise PreparationError(f"prepared VStar output is incomplete: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "hw_bjtu_opd_vstar_snapshot_v1",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "source_parquet_sha256": SOURCE_SHA256,
        "rows": EXPECTED_ROWS,
        "json_sha256": EXPECTED_JSON_SHA256,
        "logical_rows_sha256": EXPECTED_LOGICAL_ROWS_SHA256,
        "image_manifest_sha256": EXPECTED_IMAGE_MANIFEST_SHA256,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise PreparationError(f"prepared VStar manifest differs for {key}")
    if manifest.get("json_sha256") != sha256_file(data_path):
        raise PreparationError("prepared VStar JSON hash differs")
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise PreparationError("prepared VStar row count differs")
    image_records: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        images = row.get("images") if isinstance(row, dict) else None
        if not isinstance(images, list) or len(images) != 1:
            raise PreparationError(f"prepared VStar row {index} has invalid images")
        question_id = str(row.get("question_id", ""))
        expected_relative = f"images/{question_id}.jpg"
        if not _SAFE_ID.fullmatch(question_id) or images[0] != expected_relative:
            raise PreparationError(f"prepared VStar row {index} has an unsafe image path")
        image = output / expected_relative
        if image.is_symlink() or not image.is_file():
            raise PreparationError(f"prepared VStar image is missing: {image}")
        image_records.append(
            {
                "path": expected_relative,
                "bytes": image.stat().st_size,
                "sha256": sha256_file(image),
            }
        )
    actual_images = sorted(
        path.relative_to(output).as_posix()
        for path in (output / "images").iterdir()
        if path.is_file() and not path.is_symlink()
    )
    expected_images = sorted(str(row["images"][0]) for row in rows)
    if actual_images != expected_images:
        raise PreparationError("prepared VStar image file set differs")
    if hashlib.sha256(canonical_json(rows)).hexdigest() != EXPECTED_LOGICAL_ROWS_SHA256:
        raise PreparationError("prepared VStar logical row hash differs")
    if hashlib.sha256(canonical_json(image_records)).hexdigest() != EXPECTED_IMAGE_MANIFEST_SHA256:
        raise PreparationError("prepared VStar image content hash differs")
    return {"status": "verified", "output": str(output), **expected, "json_sha256": manifest["json_sha256"]}


def prepare(output: Path) -> dict[str, object]:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise PreparationError("install requirements-eval.txt before preparing VStar") from exc

    output = output.expanduser()
    if output.is_symlink():
        raise PreparationError("output cannot be a symlink")
    output = output.resolve()
    if output.exists():
        return verify_prepared(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        source_root = staging / "source"
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            local_dir=source_root,
            allow_patterns=[SOURCE_PARQUET],
        )
        parquet = source_root / SOURCE_PARQUET
        if parquet.is_symlink() or not parquet.is_file() or sha256_file(parquet) != SOURCE_SHA256:
            raise PreparationError("pinned VStar source parquet hash differs")
        table = pq.read_table(parquet, columns=["image", "text", "label", "question_id", "category"])
        source_rows = table.to_pylist()
        if len(source_rows) != EXPECTED_ROWS:
            raise PreparationError(f"expected {EXPECTED_ROWS} VStar rows, got {len(source_rows)}")

        image_dir = staging / "images"
        image_dir.mkdir()
        rows: list[dict[str, object]] = []
        image_records: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        for index, row in enumerate(source_rows):
            question_id = str(row.get("question_id") if row.get("question_id") is not None else index)
            if not _SAFE_ID.fullmatch(question_id) or question_id in seen_ids:
                raise PreparationError(f"unsafe or duplicate VStar question id: {question_id!r}")
            seen_ids.add(question_id)
            image_obj = row.get("image") or {}
            image_bytes = image_obj.get("bytes") if isinstance(image_obj, dict) else None
            if not isinstance(image_bytes, (bytes, bytearray)):
                raise PreparationError(f"VStar row {index} has no image bytes")
            relative = Path("images") / f"{question_id}.jpg"
            image_path = staging / relative
            image_path.write_bytes(bytes(image_bytes))
            query = str(row.get("text") or "").strip()
            if not query.endswith(POST_PROMPT.strip()):
                query += POST_PROMPT
            answer = str(row.get("label") or "").strip().upper()
            if answer not in set("ABCDE"):
                raise PreparationError(f"VStar row {index} has invalid answer {answer!r}")
            rows.append(
                {
                    "question_id": question_id,
                    "images": [relative.as_posix()],
                    "query": query,
                    "response": answer,
                    "category": str(row.get("category") or "unknown"),
                }
            )
            image_records.append(
                {
                    "path": relative.as_posix(),
                    "bytes": len(image_bytes),
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                }
            )

        data_path = staging / "vstar.json"
        data_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shutil.rmtree(source_root)
        manifest = {
            "schema_version": "hw_bjtu_opd_vstar_snapshot_v1",
            "repo_id": REPO_ID,
            "revision": REVISION,
            "source_parquet_sha256": SOURCE_SHA256,
            "rows": EXPECTED_ROWS,
            "json_sha256": sha256_file(data_path),
            "logical_rows_sha256": hashlib.sha256(canonical_json(rows)).hexdigest(),
            "image_manifest_sha256": hashlib.sha256(canonical_json(image_records)).hexdigest(),
            "images": len(image_records),
        }
        expected_hashes = {
            "json_sha256": EXPECTED_JSON_SHA256,
            "logical_rows_sha256": EXPECTED_LOGICAL_ROWS_SHA256,
            "image_manifest_sha256": EXPECTED_IMAGE_MANIFEST_SHA256,
        }
        for key, expected in expected_hashes.items():
            if manifest[key] != expected:
                raise PreparationError(f"materialized VStar {key} differs: {manifest[key]}")
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_prepared(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/vstar"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_prepared(args.output.expanduser().resolve()) if args.verify_only else prepare(args.output)
    except (PreparationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=os.sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
