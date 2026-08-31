from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hw_bjtu_opd.eval import cli as eval_cli
from hw_bjtu_opd.eval.cli import preflight, render_plan
from hw_bjtu_opd.eval.protocol import (
    EvaluationError,
    Record,
    load_records,
    parse_choice,
    parse_vstar_choice,
    score_records,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_preflight_is_ready_without_optional_runtime() -> None:
    report = preflight(root=ROOT)
    assert report["status"] == "ready"
    assert report["cpu_only"] is True
    assert report["errors"] == []
    assert report["benchmarks"]["vstar"]["evaluator"]["complete"] is True


def test_all_plan_is_json_serializable_and_does_not_probe_resources() -> None:
    plan = render_plan(
        benchmark="all", model=None, data=None, output=None, model_id="demo", api_base=None, limit=None, execute=False
    )
    assert plan["status"] == "dry_run"
    assert plan["raw_predictions_persisted"] is False
    assert json.loads(json.dumps(plan))["benchmark"] == "all"


def test_portable_json_reader_and_first_option_score(tmp_path: Path) -> None:
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"not used in CPU test")
    source = tmp_path / "vstar.json"
    source.write_text(
        json.dumps([{"images": [image.name], "query": "Pick one", "response": "A", "category": "direct_attributes"}])
        + "\n",
        encoding="utf-8",
    )
    records, digest = load_records(source, "vstar")
    assert len(digest) == 64
    assert records[0].images == (str(image),)
    assert parse_choice('{"answer":"A"}') == "A"
    assert parse_vstar_choice("Answer: (A)") == "A"
    assert score_records(records, ["A"])["correct"] == 1


def test_script_entrypoints_return_structured_output() -> None:
    command = [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--benchmark", "vstar"]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "hw_bjtu_opd_eval_plan_v1"
    assert payload["requested_rows"] == 8


def test_invalid_gold_fails_closed() -> None:
    records = [Record(index=0, images=("unused",), query="q", gold="")]
    with pytest.raises(EvaluationError, match="invalid gold"):
        score_records(records, [None])


def test_image_path_cannot_escape_dataset(tmp_path: Path) -> None:
    source = tmp_path / "vstar.json"
    source.write_text(
        json.dumps([{"images": ["../secret.jpg"], "query": "q", "response": "A"}]),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="escapes"):
        load_records(source, "vstar")


def test_evaluation_runtime_compatibility_boundary_is_pinned() -> None:
    requirements = (ROOT / "requirements-eval.txt").read_text(encoding="utf-8").splitlines()
    assert "vllm==0.18.0" in requirements
    assert "flashinfer-python==0.6.6" in requirements
    assert "quack-kernels==0.5.0" in requirements
    assert "nvidia-cutlass-dsl==4.5.3" in requirements

    installer = (ROOT / "scripts/install_eval.sh").read_text(encoding="utf-8")
    assert '"nvidia-cutlass-dsl-libs-base": "4.5.3"' in installer
    assert 'hasattr(cute_core, "ThrMma")' in installer

    wrapper = (ROOT / "scripts/evaluate_vstar.sh").read_text(encoding="utf-8")
    assert "--max-model-len 65536" in wrapper


def test_raw_http_request_flattens_vllm_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict[str, object]] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return b'{"choices":[{"message":{"content":"A"}}]}'

    def urlopen(request: object, *, timeout: float) -> Response:
        assert timeout == 5.0
        payloads.append(json.loads(request.data.decode("utf-8")))
        return Response()

    monkeypatch.setattr(eval_cli, "_mime_data_uri", lambda *_args, **_kwargs: "data:image/png;base64,AA==")
    monkeypatch.setattr(eval_cli.urllib.request, "urlopen", urlopen)
    row = Record(index=0, images=("image.jpg",), query="choose", gold="A")

    assert (
        eval_cli._api_request(
            api_base="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            model_id="model",
            row=row,
            benchmark="vstar",
            timeout=5.0,
        )
        == "A"
    )
    assert "extra_body" not in payloads[0]
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
