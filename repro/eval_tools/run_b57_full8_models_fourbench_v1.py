#!/usr/bin/env python3
"""Formal four-benchmark queue for the serial B57 full-8-GPU OPD runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import run_b57_dual4_models_fourbench_v1 as formal


OUTPUT_ROOT = formal.OUTPUT_ROOT
DEFAULT_RUN_ROOT = OUTPUT_ROOT / "b57_full8_models_fourbench_v1"

MODEL_SPECS = (
    formal.ModelSpec(
        slug="b57_10k_init_vision6k_crop_raw9_teacher_full8_s65_v1",
        name="B57 10K Vision6K crop, Raw9 teacher, full8 s65",
        model_id="Qwen3.5-9B-B57-10K-Vision6K-Crop-Raw9Teacher-Full8-S65-v1",
        model_path=OUTPUT_ROOT
        / "b57_10k_init_vision6k_crop_raw9_teacher_full8_s65_v1/merged/final_hf_official_chat_v1",
        training_root=OUTPUT_ROOT
        / "b57_10k_init_vision6k_crop_raw9_teacher_full8_s65_v1",
    ),
    formal.ModelSpec(
        slug="b57_10k_init_vision6k_crop_b57_27b_teacher_full8_s65_v1",
        name="B57 10K Vision6K crop, B57-27B teacher, full8 s65",
        model_id="Qwen3.5-9B-B57-10K-Vision6K-Crop-B57-27BTeacher-Full8-S65-v1",
        model_path=OUTPUT_ROOT
        / "b57_10k_init_vision6k_crop_b57_27b_teacher_full8_s65_v1/merged/final_hf_official_chat_v1",
        training_root=OUTPUT_ROOT
        / "b57_10k_init_vision6k_crop_b57_27b_teacher_full8_s65_v1",
    ),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    result.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    result.add_argument(
        "--target",
        action="append",
        choices=("raw9_teacher", "b57_27b_teacher", "all"),
        help="repeat to select targets; default is all",
    )
    result.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=formal.DEFAULT_WAIT_TIMEOUT_SECONDS,
    )
    result.add_argument("--poll-seconds", type=float, default=formal.DEFAULT_POLL_SECONDS)
    result.add_argument("--no-wait", action="store_true")
    return result


def select(values: Sequence[str] | None) -> tuple[formal.ModelSpec, ...]:
    if not values or "all" in values:
        return MODEL_SPECS
    by_key = {
        "raw9_teacher": MODEL_SPECS[0],
        "b57_27b_teacher": MODEL_SPECS[1],
    }
    selected: list[formal.ModelSpec] = []
    for value in values:
        target = by_key[value]
        if target not in selected:
            selected.append(target)
    return tuple(selected)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    targets = select(args.target)
    try:
        if args.dry_run:
            plan = formal.render_plan(queue_root=args.run_root, targets=targets)
            print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
            return 0
        if args.wait_timeout_seconds < 0 or args.poll_seconds <= 0:
            raise formal.QueueError("invalid wait or poll interval")
        result = formal.execute(
            queue_root=args.run_root,
            targets=targets,
            timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.poll_seconds,
            no_wait=args.no_wait,
        )
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "summary": str(args.run_root / "summary.json"),
                    "completed_targets": result.get("completed_targets", []),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (formal.QueueError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "failed_closed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
