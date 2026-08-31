#!/usr/bin/env python3
"""CPU-only preflight for the public evaluation command.

Examples::

    python scripts/preflight.py
    python scripts/preflight.py --model artifacts/checkpoint --data data/vstar.json
    python scripts/preflight.py --benchmark all --strict

The default command is informational and succeeds before large models or
benchmark payloads have been downloaded.  ``--strict`` turns missing runtime
inputs into a non-zero result for CI or a release check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw_bjtu_opd.eval.cli import preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="vstar", choices=("vstar", "mmstar", "blink", "zoombench", "all"))
    parser.add_argument("--model", type=Path, help="optional local checkpoint directory")
    parser.add_argument("--data", type=Path, help="optional prepared benchmark JSON/JSONL or directory")
    parser.add_argument("--strict", action="store_true", help="treat missing optional inputs as errors")
    parser.add_argument("--execute", action="store_true", help="also validate execute-time requirements")
    args = parser.parse_args(argv)
    report = preflight(
        root=ROOT, benchmark=args.benchmark, model=args.model, data=args.data, strict=args.strict, execute=args.execute
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
