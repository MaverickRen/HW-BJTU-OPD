#!/usr/bin/env python3
"""Manifest and aggregate-only eligibility contract for the future OPD suite.

The manifest is intentionally independent of the historical required queue.
It describes the four historical/control artifacts plus C0, and the four
benchmark cells that every artifact must publish.  This module does not
import torch, CUDA, vLLM, or any benchmark implementation; reading it is
safe in a CPU-only test process.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "vision_opd_fourbench_manifest_v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("vision_opd_fourbench_manifest_v1.json")
BENCHMARK_ORDER = ("VStar", "MMStar", "BLINK-v5", "ZoomBench")
EXPECTED_TOTALS = {"VStar": 191, "MMStar": 1500, "BLINK-v5": 1901, "ZoomBench": 845}
PROTOCOLS = {
    "VStar": "vstar_frozen_first_option_v1",
    "MMStar": "mmstar_qwen35_modelcard_thinking_v2",
    "BLINK-v5": "blink_deterministic_checkpoint_comparison_v5",
    "ZoomBench": "zoombench_score_aggregate_v1",
}
BLINK_V5_PRESET = "blink_deterministic_checkpoint_comparison_v5"
BLINK_V5_SCHEMA = "mcq_blink_checkpoint_comparison_aggregate_v5"
BLINK_V5_PROTOCOL_HASH = "ab5754c61c01c3c761c9fd72ae37480163884e894ffdb38cb805fe96b54204dc"
BLINK_V5_RAW9_REFERENCE = {"correct": 1124, "total": 1901, "percent": 1124 / 1901 * 100.0}
BLINK_V5_B28_REFERENCE = {"correct": 1221, "total": 1901, "percent": 1221 / 1901 * 100.0}
BLINK_V5_RAW9_REPORTED_PERCENT = 59.13
BLINK_V5_B28_REPORTED_PERCENT = 64.23
GPU_IDS = tuple(range(8))
CUDA_VISIBLE_DEVICES = ",".join(str(item) for item in GPU_IDS)
LOCK_RELATIVE_PATH = "Locks/opd_gpu_0_7.lock"
REQUIRED_UID = REQUIRED_GID = 30_853


class ManifestError(ValueError):
    """Malformed, mixed, or unsafe future-artifact manifest."""


@dataclass(frozen=True)
class ArtifactSpec:
    slug: str
    name: str
    model_id: str
    model_tag: str
    model_path: str
    required_inputs: tuple[str, ...]
    artifact_role: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "name": self.name,
            "model_id": self.model_id,
            "model_tag": self.model_tag,
            "model_path": self.model_path,
            "required_inputs": list(self.required_inputs),
            "artifact_role": self.artifact_role,
        }


@dataclass(frozen=True)
class Manifest:
    path: str
    workspace: str
    benchmark_order: tuple[str, ...]
    benchmark_totals: Mapping[str, int]
    protocols: Mapping[str, str]
    blink_v5: Mapping[str, Any]
    resources: Mapping[str, Any]
    eligibility: Mapping[str, Any]
    targets: tuple[ArtifactSpec, ...]
    control_baselines: tuple[ArtifactSpec, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest": self.path,
            "workspace": self.workspace,
            "benchmark_order": list(self.benchmark_order),
            "benchmark_totals": dict(self.benchmark_totals),
            "protocols": dict(self.protocols),
            "blink_v5": dict(self.blink_v5),
            "resources": dict(self.resources),
            "eligibility": dict(self.eligibility),
            "targets": [target.as_dict() for target in self.targets],
            "control_baselines": [target.as_dict() for target in self.control_baselines],
        }


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ManifestError(f"manifest is unavailable: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ManifestError(f"manifest must be a single-link regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"manifest is not valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ManifestError("manifest must be one JSON object")
    return value


def _absolute(value: Any, *, workspace: Path, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a non-empty path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    # Do not resolve existing symlinks here.  Execute-time gates reject them;
    # this keeps a dry-run from silently changing the identity it advertises.
    if any(part == ".." for part in candidate.parts):
        raise ManifestError(f"{label} must not contain '..'")
    return os.path.normpath(str(candidate.absolute()))


def _str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a non-empty string")
    return value


def _validate_protocols(value: Mapping[str, Any]) -> None:
    order = value.get("benchmark_order")
    if tuple(order or ()) != BENCHMARK_ORDER:
        raise ManifestError("benchmark order must be VStar/MMStar/BLINK-v5/ZoomBench")
    totals = value.get("benchmark_totals")
    if not isinstance(totals, Mapping) or {key: totals.get(key) for key in BENCHMARK_ORDER} != EXPECTED_TOTALS:
        raise ManifestError("benchmark totals differ from the pinned four-benchmark suite")
    protocols = value.get("protocols")
    if not isinstance(protocols, Mapping) or {key: protocols.get(key) for key in BENCHMARK_ORDER} != PROTOCOLS:
        raise ManifestError("benchmark protocol labels differ from the pinned suite")
    blink = value.get("blink_v5")
    if not isinstance(blink, Mapping):
        raise ManifestError("BLINK-v5 protocol block is missing")
    if (
        blink.get("preset") != BLINK_V5_PRESET
        or blink.get("schema_version") != BLINK_V5_SCHEMA
        or blink.get("protocol_hash") != BLINK_V5_PROTOCOL_HASH
        or blink.get("thinking") is not False
        or blink.get("temperature") != 0.0
        or blink.get("seed") != 42
    ):
        raise ManifestError("BLINK-v5 is not the deterministic nonthinking protocol")
    raw = blink.get("raw9_reference")
    b28 = blink.get("b28_reference")
    for label, actual, expected in (("raw9", raw, BLINK_V5_RAW9_REFERENCE), ("B28", b28, BLINK_V5_B28_REFERENCE)):
        if not isinstance(actual, Mapping) or actual.get("correct") != expected["correct"] or actual.get("total") != expected["total"]:
            raise ManifestError(f"BLINK-v5 {label} reference differs")
        try:
            if abs(float(actual.get("percent")) - expected["percent"]) > 1e-9:
                raise ManifestError(f"BLINK-v5 {label} percentage differs")
        except (TypeError, ValueError):
            raise ManifestError(f"BLINK-v5 {label} percentage is malformed")
        reported = actual.get("reported_percent")
        expected_reported = BLINK_V5_RAW9_REPORTED_PERCENT if label == "raw9" else BLINK_V5_B28_REPORTED_PERCENT
        try:
            if abs(float(reported) - expected_reported) > 1e-9:
                raise ManifestError(f"BLINK-v5 {label} reported percentage differs")
        except (TypeError, ValueError):
            raise ManifestError(f"BLINK-v5 {label} reported percentage is malformed")


def load_manifest(path: Path | str = DEFAULT_MANIFEST_PATH, *, workspace: Path | str | None = None) -> Manifest:
    """Load and validate a future-artifact manifest without checking inputs.

    Existence checks are deliberately separate in :func:`inspect_target` so a
    caller can render a pure static plan even while all four future artifacts
    are still being built.
    """

    manifest_path = Path(path).absolute()
    value = _read_json(manifest_path)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("manifest schema_version differs")
    root = Path(workspace).absolute() if workspace is not None else Path(os.environ.get("OPD_QWEN35_WORKSPACE", "/minimax-3d-rw-backup/users/jiazhi/H_Workspace")).absolute()
    _validate_protocols(value)
    resources = value.get("resources")
    if not isinstance(resources, Mapping) or tuple(resources.get("gpu_ids", ())) != GPU_IDS or resources.get("cuda_visible_devices") != CUDA_VISIBLE_DEVICES or resources.get("world_size") != 8 or resources.get("serial") is not True or resources.get("additional_resources_requested") is not False:
        raise ManifestError("resource contract must be serial physical GPUs 0..7")
    eligibility = value.get("eligibility")
    if not isinstance(eligibility, Mapping) or tuple(eligibility.get("required_benchmarks", ())) != BENCHMARK_ORDER or eligibility.get("aggregate_only") is not True or eligibility.get("sample_level_output") is not False:
        raise ManifestError("eligibility contract must require all four aggregate cells")
    raw_targets = value.get("targets")
    if not isinstance(raw_targets, list) or len(raw_targets) != 5:
        raise ManifestError("manifest must contain the four historical targets plus C0")
    targets: list[ArtifactSpec] = []
    seen: set[str] = set()
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, Mapping):
            raise ManifestError(f"target {index} is malformed")
        slug = _str(raw_target.get("slug"), label=f"target {index} slug")
        if slug in seen:
            raise ManifestError(f"duplicate target slug: {slug}")
        seen.add(slug)
        model_path = _absolute(raw_target.get("model_path"), workspace=root, label=f"target {slug} model_path")
        required_raw = raw_target.get("required_inputs", [raw_target.get("model_path")])
        if not isinstance(required_raw, list) or not required_raw:
            raise ManifestError(f"target {slug} required_inputs is empty")
        required = tuple(_absolute(item, workspace=root, label=f"target {slug} required input") for item in required_raw)
        if model_path not in required:
            required = (model_path, *required)
        targets.append(
            ArtifactSpec(
                slug=slug,
                name=_str(raw_target.get("name"), label=f"target {slug} name"),
                model_id=_str(raw_target.get("model_id"), label=f"target {slug} model_id"),
                model_tag=_str(raw_target.get("model_tag"), label=f"target {slug} model_tag"),
                model_path=model_path,
                required_inputs=required,
                artifact_role=_str(raw_target.get("artifact_role"), label=f"target {slug} artifact_role"),
            )
        )
    raw_controls = value.get("control_baselines", [])
    if not isinstance(raw_controls, list):
        raise ManifestError("control_baselines must be a list when present")
    control_baselines: list[ArtifactSpec] = []
    for index, raw_control in enumerate(raw_controls):
        if not isinstance(raw_control, Mapping):
            raise ManifestError(f"control baseline {index} is malformed")
        slug = _str(raw_control.get("slug"), label=f"control baseline {index} slug")
        if slug in seen:
            raise ManifestError(f"duplicate target/control baseline slug: {slug}")
        seen.add(slug)
        model_path = _absolute(raw_control.get("model_path"), workspace=root, label=f"control baseline {slug} model_path")
        required_raw = raw_control.get("required_inputs", [raw_control.get("model_path")])
        if not isinstance(required_raw, list) or not required_raw:
            raise ManifestError(f"control baseline {slug} required_inputs is empty")
        required = tuple(_absolute(item, workspace=root, label=f"control baseline {slug} required input") for item in required_raw)
        if model_path not in required:
            required = (model_path, *required)
        control_baselines.append(
            ArtifactSpec(
                slug=slug,
                name=_str(raw_control.get("name"), label=f"control baseline {slug} name"),
                model_id=_str(raw_control.get("model_id"), label=f"control baseline {slug} model_id"),
                model_tag=_str(raw_control.get("model_tag"), label=f"control baseline {slug} model_tag"),
                model_path=model_path,
                required_inputs=required,
                artifact_role=_str(raw_control.get("artifact_role"), label=f"control baseline {slug} artifact_role"),
            )
        )
    return Manifest(
        path=str(manifest_path),
        workspace=str(root),
        benchmark_order=BENCHMARK_ORDER,
        benchmark_totals=dict(EXPECTED_TOTALS),
        protocols=dict(PROTOCOLS),
        blink_v5=dict(value["blink_v5"]),
        resources=dict(resources),
        eligibility=dict(eligibility),
        targets=tuple(targets),
        control_baselines=tuple(control_baselines),
    )


def _is_present(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) or (stat.S_ISREG(info.st_mode) and info.st_nlink == 1)


def inspect_target(target: ArtifactSpec) -> dict[str, Any]:
    """Return a metadata-only readiness record; missing inputs are not fatal."""

    missing = [path for path in target.required_inputs if not _is_present(Path(path))]
    return {
        **target.as_dict(),
        "input_status": "ready" if not missing else "hold_missing_input",
        "missing_inputs": missing,
        "missing_input_count": len(missing),
    }


def cells_are_complete(cells: Mapping[str, Any]) -> bool:
    """The sole promotion rule: all four pinned cells must be complete."""

    return all(
        isinstance(cells.get(name), Mapping)
        and cells[name].get("status") == "complete"
        and cells[name].get("complete", True) is True
        for name in BENCHMARK_ORDER
    )


def row_eligibility(cells: Mapping[str, Any]) -> tuple[bool, str]:
    missing = [name for name in BENCHMARK_ORDER if not (isinstance(cells.get(name), Mapping) and cells[name].get("status") == "complete" and cells[name].get("complete", True) is True)]
    if missing:
        return False, "required benchmark cells incomplete: " + ", ".join(missing)
    return True, "all four required benchmark cells complete"


__all__ = [
    "ArtifactSpec",
    "BENCHMARK_ORDER",
    "BLINK_V5_B28_REFERENCE",
    "BLINK_V5_PRESET",
    "BLINK_V5_PROTOCOL_HASH",
    "BLINK_V5_B28_REPORTED_PERCENT",
    "BLINK_V5_RAW9_REFERENCE",
    "BLINK_V5_RAW9_REPORTED_PERCENT",
    "BLINK_V5_SCHEMA",
    "CUDA_VISIBLE_DEVICES",
    "DEFAULT_MANIFEST_PATH",
    "EXPECTED_TOTALS",
    "GPU_IDS",
    "Manifest",
    "ManifestError",
    "PROTOCOLS",
    "cells_are_complete",
    "inspect_target",
    "load_manifest",
    "row_eligibility",
]
