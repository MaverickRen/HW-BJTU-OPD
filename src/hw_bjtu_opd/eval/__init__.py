"""Small, dependency-light evaluation entry points for HW-BJTU-OPD.

The public command line tools intentionally keep benchmark execution behind a
local OpenAI-compatible endpoint.  This makes a fresh checkout useful on a
CPU-only machine: preflight and dry-run never import CUDA, vLLM or Ray.
"""

from .protocol import BENCHMARKS, EvaluationError, build_plan, load_records, parse_vstar_choice, score_records

__all__ = ["BENCHMARKS", "EvaluationError", "build_plan", "load_records", "parse_vstar_choice", "score_records"]
