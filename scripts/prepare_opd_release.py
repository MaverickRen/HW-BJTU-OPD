#!/usr/bin/env python3
"""Verify or stage the released Qwen3.5-9B Vision-OPD artifact.

The command deliberately has no network or credential handling.  It validates
the seven files in the local merged Hugging Face directory, builds a release in
a private sibling directory, and atomically renames that directory into place.
Hard links are used when possible; ``--copy`` forces independent copies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseError(RuntimeError):
    """Raised when an input or output violates the release contract."""


@dataclass(frozen=True)
class ArtifactFile:
    name: str
    bytes: int
    sha256: str


# This is intentionally a closed seven-file artifact.  In particular, no
# private training receipts, source paths, caches, or credentials are copied.
EXPECTED_FILES: tuple[ArtifactFile, ...] = (
    ArtifactFile(
        "chat_template.jinja",
        5415,
        "d9604b52b4e1f4b9ec68e065238c757a3d7efdebe1c3692d13a97df6f84c54db",
    ),
    ArtifactFile(
        "config.json",
        2900,
        "995196f6106dfbb228e3f198b3eaf14985ef437de88ca5fa5f8910eaf83b2353",
    ),
    ArtifactFile(
        "generation_config.json",
        115,
        "512889808b620c17f51634d88d035cd5efc23e0eee3843432a5ed07be821d87b",
    ),
    ArtifactFile(
        "model.safetensors",
        18819722392,
        "c86054edddaf186b5a0754fed55e4d8e80108ba2081ff7e6ba7c2d3e589ccdc7",
    ),
    ArtifactFile(
        "processor_config.json",
        1191,
        "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    ),
    ArtifactFile(
        "tokenizer.json",
        19989343,
        "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
    ),
    ArtifactFile(
        "tokenizer_config.json",
        1139,
        "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182",
    ),
)

ARTIFACT_ID = "qwen35-9b-vision-opd-sft9-teacher-full8-s65-v1"
MODEL_SHA256 = "c86054edddaf186b5a0754fed55e4d8e80108ba2081ff7e6ba7c2d3e589ccdc7"


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ReleaseError(f"{label} is a symlink: {path.name}")
    if not path.is_file():
        raise ReleaseError(f"{label} is missing or not a regular file: {path.name}")


def validate_source(
    source: Path,
    file_specs: Iterable[ArtifactFile] = EXPECTED_FILES,
) -> list[dict[str, int | str]]:
    """Validate and return public file metadata without recording ``source``."""

    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise ReleaseError(f"source must be a real directory: {source}")

    records: list[dict[str, int | str]] = []
    for spec in file_specs:
        path = source / spec.name
        _regular_file(path, f"source file {spec.name}")
        actual_sha, actual_size = _sha256_and_size(path)
        if actual_size != spec.bytes:
            raise ReleaseError(f"size mismatch for {spec.name}: {actual_size} != {spec.bytes}")
        if actual_sha != spec.sha256:
            raise ReleaseError(f"SHA256 mismatch for {spec.name}: {actual_sha} != {spec.sha256}")
        records.append({"name": spec.name, "bytes": actual_size, "sha256": actual_sha})
    return records


def _within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _validate_output(output: Path, source: Path) -> None:
    if output.is_symlink():
        raise ReleaseError(f"output cannot be a symlink: {output}")
    if output.exists() and not output.is_dir():
        raise ReleaseError(f"output exists but is not a directory: {output}")
    if output.is_dir():
        try:
            next(output.iterdir())
        except StopIteration:
            pass
        except OSError as exc:
            raise ReleaseError(f"cannot inspect output directory: {output}: {exc}") from exc
        else:
            raise ReleaseError(f"refusing to overwrite non-empty output directory: {output}")

    source_real = source.resolve()
    output_real = output.resolve(strict=False)
    if output_real == source_real or _within(output_real, source_real):
        raise ReleaseError("output must not be the source directory or inside it")


def _copy_file(source: Path, target: Path, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode in {"auto", "link"}:
        try:
            os.link(source, target, follow_symlinks=False)
            return "hardlink"
        except OSError:
            if mode == "link":
                raise
            target.unlink(missing_ok=True)
    shutil.copy2(source, target)
    return "copy"


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _readme() -> str:
    """Return the single source-of-truth Hugging Face model card."""

    return (ROOT / "hf_card/OPD9_TEACHER_README.md").read_text(encoding="utf-8")


def build_manifest(records: Sequence[dict[str, int | str]]) -> dict[str, object]:
    files = [dict(record) for record in records]
    model = next(record for record in files if record["name"] == "model.safetensors")
    return {
        "schema_version": "hw_bjtu_opd_opd9_teacher_artifact_v1",
        "artifact_id": ARTIFACT_ID,
        "artifact_type": "vision_opd_final_model",
        "repo_id": "HWBJTUOPD/Qwen3.5-9B-SFT10K-VisionOPD6K-SFT9BTeacher",
        "model": {
            "family": "Qwen3.5-9B",
            "role": "Vision-OPD student after fixed 9B SFT teacher training",
            "weight_file": model["name"],
            "sha256": model["sha256"],
            "bytes": model["bytes"],
        },
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(record["bytes"]) for record in files),
        "runtime": {
            "implementation": "Vision-OPD-reference",
            "commit": "c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471",
            "trainer_config": "verl/trainer/config/vopd.yaml",
            "teacher_source": "fixed_local_model",
        },
        "training": {
            "dataset": {
                "repo_id": "yuanqianhao/Vision-OPD-6K",
                "revision": "eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4",
                "processed_rows": 6241,
                "processed_parquet_sha256": "b8ac1cb2f17d5478af60feab2640d9526e7e816cb1506b5bd521e1598dfeb722",
            },
            "implementation_commit": "c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471",
            "student_initialization": "Qwen3.5-9B SFT_V1 10K",
            "teacher_initialization": "Qwen3.5-9B SFT_V1 10K",
            "teacher_mode": "fixed",
            "teacher_update_rate": 0.0,
            "student_image_key": "images",
            "teacher_image_key": "bbox_images",
            "world_size": 8,
            "seed": 42,
            "rollout_n": 8,
            "train_batch_size": 96,
            "total_training_steps": 65,
            "learning_rate": 2e-6,
            "warmup_steps": 10,
            "max_prompt_length": 8192,
            "max_response_length": 1024,
            "max_model_length": 9216,
            "distillation_topk": 100,
            "distillation_alpha": 0.5,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
        },
        "evaluation": {
            "protocol": "hw-bjtu-opd-fourbench-v1",
            "aggregate_only": True,
            "benchmarks": {
                "V*": {"correct": 176, "total": 191},
                "MMStar": {"correct": 1159, "total": 1500},
                "BLINK-v5": {"correct": 1263, "total": 1901},
                "ZoomBench": {"correct": 525, "total": 845},
            },
            "macro_percent": 74.49553937627918,
        },
        "source_paths_recorded": False,
        "credentials_recorded": False,
    }


def stage_release(
    source: Path,
    output: Path,
    *,
    mode: str = "auto",
    dry_run: bool = False,
    file_specs: Iterable[ArtifactFile] = EXPECTED_FILES,
) -> dict[str, object]:
    """Validate and atomically stage an artifact, or return a dry-run plan."""

    if mode not in {"auto", "link", "copy"}:
        raise ReleaseError(f"unknown staging mode: {mode}")
    source = Path(source)
    output = Path(output)
    specs = tuple(file_specs)
    records = validate_source(source, specs)
    _validate_output(output, source)
    manifest = build_manifest(records)
    plan: dict[str, object] = {
        "status": "dry_run" if dry_run else "staging",
        "artifact_id": ARTIFACT_ID,
        "output": str(output),
        "mode": mode,
        "file_count": len(records),
        "total_bytes": manifest["total_bytes"],
        "model_sha256": manifest["model"]["sha256"],
    }
    if dry_run:
        return plan

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        actual_modes: set[str] = set()
        for record in records:
            source_file = source / str(record["name"])
            target_file = staging / str(record["name"])
            actual_modes.add(_copy_file(source_file, target_file, mode))
        (staging / "README.md").write_text(_readme(), encoding="utf-8")
        _write_json(staging / "artifact_manifest.json", manifest)
        # Replacing an absent or empty output directory is atomic.  If another
        # process populated output while staging, os.replace fails and the
        # existing non-empty directory is left untouched.
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    plan.update({"status": "complete", "copy_modes": sorted(actual_modes)})
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="merged HF model directory")
    parser.add_argument("--output", type=Path, help="new staging directory")
    parser.add_argument(
        "--mode",
        choices=("auto", "link", "copy"),
        default="auto",
        help="hard-link when possible (default), require links, or force copies",
    )
    parser.add_argument("--copy", action="store_true", help="alias for --mode copy")
    parser.add_argument("--dry-run", action="store_true", help="validate and print a plan without writing")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the exact seven-file model artifact without staging it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verify_only:
            if args.output is not None or args.dry_run or args.copy:
                raise ReleaseError("--verify-only cannot be combined with output or staging options")
            records = validate_source(args.source)
            print(
                json.dumps(
                    {
                        "status": "verified",
                        "artifact_id": ARTIFACT_ID,
                        "file_count": len(records),
                        "model_sha256": MODEL_SHA256,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.output is None:
            raise ReleaseError("--output is required unless --verify-only is used")
        mode = "copy" if args.copy else args.mode
        result = stage_release(args.source, args.output, mode=mode, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ReleaseError, OSError, StopIteration, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
