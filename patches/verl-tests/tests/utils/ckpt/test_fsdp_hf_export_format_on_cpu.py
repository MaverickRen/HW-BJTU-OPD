"""Regression guards for Transformers 5.x HF state-dict export semantics."""

from __future__ import annotations

import ast
from pathlib import Path


VERL_PACKAGE = Path(__file__).resolve().parents[3] / "verl"


def _save_pretrained_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "save_pretrained" and any(
            keyword.arg == "state_dict" for keyword in node.keywords
        ):
            calls.append(node)
    return calls


def _assert_current_format_is_explicit(path: Path) -> None:
    calls = _save_pretrained_calls(path)
    assert calls, f"no state_dict save_pretrained call found in {path}"
    for call in calls:
        values = {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }
        assert "save_original_format" in values, ast.dump(call)
        value = values["save_original_format"]
        assert isinstance(value, ast.Constant) and value.value is False, ast.dump(call)


def test_offline_fsdp_merger_disables_legacy_reverse_key_mapping() -> None:
    _assert_current_format_is_explicit(
        VERL_PACKAGE / "model_merger" / "base_model_merger.py"
    )


def test_inline_fsdp_hf_export_disables_legacy_reverse_key_mapping() -> None:
    _assert_current_format_is_explicit(
        VERL_PACKAGE / "utils" / "checkpoint" / "fsdp_checkpoint_manager.py"
    )
