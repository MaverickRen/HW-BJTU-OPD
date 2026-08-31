#!/usr/bin/env python3
"""Simple public evaluation entry point.

The default is an eight-example VStar quick-check plan. Add ``--full`` for all
191 VStar examples. ``--benchmark all`` renders the complete four-cell plan
but is not executable here because BLINK-v5 and ZoomBench require the retained
high-resource frozen workflow. Requests are accepted only through a loopback
OpenAI-compatible server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw_bjtu_opd.eval.cli import EvaluationError, evaluate, preflight, render_plan
from hw_bjtu_opd.eval.protocol import BENCHMARKS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default="vstar",
        choices=("vstar", "mmstar", "blink", "zoombench", "all"),
        help="VStar/MMStar can execute; blink/zoombench/all are plan-only",
    )
    parser.add_argument("--model", type=Path, help="local checkpoint directory (metadata/preflight only)")
    parser.add_argument("--model-id", default="Qwen3.5-9B-SFT10K-Vision6K-Crop-SFT9Teacher")
    parser.add_argument(
        "--data", type=Path, help="prepared benchmark JSON/JSONL, or a directory containing named files"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument(
        "--api-base", default=os.environ.get("OPENAI_BASE_URL"), help="local OpenAI-compatible endpoint"
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"), help="API key; never written to output"
    )
    parser.add_argument(
        "--limit", type=int, help="number of rows; default is 8 for single VStar and full for --benchmark all"
    )
    parser.add_argument("--full", action="store_true", help="run all rows for a single benchmark")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true", help="send requests and write aggregate output")
    parser.add_argument(
        "--preflight", action="store_true", help="print CPU-only preflight instead of an evaluation plan"
    )
    args = parser.parse_args(argv)

    try:
        if args.preflight:
            report = preflight(
                root=ROOT,
                benchmark=args.benchmark,
                model=args.model,
                data=args.data,
                strict=False,
                execute=args.execute,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        limit = args.limit
        if args.full and limit is not None:
            raise EvaluationError("use either --full or --limit, not both")
        if args.full:
            if args.benchmark == "all":
                raise EvaluationError("--full is implicit with --benchmark all")
            limit = BENCHMARKS[args.benchmark]["rows"]
        plan = render_plan(
            benchmark=args.benchmark,
            model=args.model,
            data=args.data,
            output=args.output,
            model_id=args.model_id,
            api_base=args.api_base,
            limit=limit,
            execute=args.execute,
        )
        if not args.execute:
            print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if not args.api_base:
            raise EvaluationError("--api-base is required with --execute")
        if args.data is None:
            raise EvaluationError("--data is required with --execute")
        if args.limit is not None and args.limit < 1:
            raise EvaluationError("--limit must be positive")
        result = evaluate(
            benchmark=args.benchmark,
            model=args.model,
            data=args.data,
            output=args.output,
            model_id=args.model_id,
            api_base=args.api_base,
            api_key=args.api_key,
            limit=limit,
            timeout=args.timeout,
            workers=args.workers,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except EvaluationError as exc:
        print(json.dumps({"status": "failed_closed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
