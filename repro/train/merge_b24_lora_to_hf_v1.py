#!/usr/bin/env python3
"""Fail-closed CPU-only merger for a PEFT LoRA adapter.

The B24 experiment trains a small PEFT adapter against an already published
HF checkpoint.  ``verl.model_merger`` intentionally leaves the adapter next
to the HF model, which is useful for serving but is not a self-contained
checkpoint.  This utility performs the final, auditable conversion:

* the base model, PEFT configuration and an explicitly supplied checkpoint
  ``lora_train_meta.json`` (including its SHA256) are checked before loading;
* target modules are derived from the *adapter tensor names*, not from a
  suffix such as ``linear_fc1``.  The temporary PEFT config consequently
  contains exact full module names;
* PEFT's ``merge_and_unload(safe_merge=True)`` is used on CPU;
* publication is create-once and atomic, and an output containing adapter or
  LoRA tensors is rejected.

No evaluation artifact is read by this program.  ``--dry-run`` only validates
metadata and tensor headers and performs no writes.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "b24_lora_to_hf_v1"
WORKSPACE = Path("/minimax-3d-rw-backup/users/jiazhi/H_Workspace")
DEFAULT_BASE = WORKSPACE / "Output/opd_qwen35_9b/b12_strict_option_sft_768_v1_r2/merged/global_step_8_hf_v1"
DEFAULT_ADAPTER = WORKSPACE / "Output/opd_qwen35_9b/b24_lora_fine_multi_v1/adapter"
DEFAULT_OUTPUT = WORKSPACE / "Output/opd_qwen35_9b/b24_lora_fine_multi_v1/merged_hf"
DEFAULT_RECEIPT = DEFAULT_OUTPUT / "merge_receipt.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ADAPTER_WEIGHT_NAMES = frozenset({"adapter_model.safetensors", "adapter_model.bin"})
_LORA_MARKER_RE = re.compile(r"(?:^|[._])lora_(?:a|b|embedding_|magnitude_vector|_|$)", re.IGNORECASE)
_OLD_OUTPUT_MARKER_NAMES = frozenset({"merge_complete.json", "manifest.json", "model_manifest.json"})


class MergeError(RuntimeError):
    """A normal fail-closed validation or publication rejection."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    try:
        value = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise MergeError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise MergeError(f"{label} must be a single-link regular file: {path}")
    return path


def _directory(path: Path, label: str) -> Path:
    try:
        value = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise MergeError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise MergeError(f"{label} must be a real directory: {path}")
    return path


def _path(value: Path | str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.is_symlink():
        raise MergeError(f"{label} must not be a symlink: {path}")
    # ``resolve(strict=False)`` is used only for CLI identity.  Actual inputs
    # are still rejected if the final path component is a symlink.
    return path.resolve(strict=False)


def _output_path(value: Path | str) -> Path:
    """Normalize an output path without resolving a pre-existing symlink."""

    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    # This check deliberately precedes resolve(strict=False): resolve would
    # turn a symlink to a not-yet-created directory into an ordinary path and
    # defeat the create-once safety gate.
    probe = raw
    while probe != probe.parent:
        if probe.is_symlink():
            raise MergeError(f"output path must not contain a symlink component: {probe}")
        probe = probe.parent
    return raw.resolve(strict=False)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, MergeError) as exc:
        raise MergeError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise MergeError(f"{label} must be one JSON object: {path}")
    return value


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MergeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MergeError(f"{label} must be a positive integer")
    return value


def _target_from_tensor_name(name: str) -> tuple[str, str]:
    """Return ``(full_target_module, A-or-B)`` for a PEFT tensor key.

    veRL's adapter writer emits ``base_model.model.<module>.lora_A.weight``;
    vanilla PEFT emits the same prefix and may include ``.default`` between
    the factor and ``weight``.  The prefix is intentionally removed because
    the base model's named modules start at ``model``.
    """

    if not isinstance(name, str) or not name or "\x00" in name:
        raise MergeError(f"invalid adapter tensor key: {name!r}")
    match = re.fullmatch(r"(?P<prefix>(?:base_model\.model\.)?)(?P<target>[A-Za-z0-9_][A-Za-z0-9_.]*)\.lora_(?P<factor>A|B)(?:\.default)?\.weight", name)
    if match is None:
        raise MergeError(f"adapter contains a non-LoRA or malformed tensor key: {name}")
    target = match.group("target")
    # A target must be a full module path.  Requiring a dot prevents a broad
    # root-level match and catches the common accidental suffix-only config.
    if "." not in target or any(part in {"", ".", ".."} for part in target.split(".")):
        raise MergeError(f"adapter tensor does not identify a full target module: {name}")
    return target, match.group("factor")


def derive_full_target_modules(keys: Iterable[str]) -> list[str]:
    """Derive and validate exact full target module names from adapter keys."""

    pairs: dict[str, set[str]] = {}
    seen: set[str] = set()
    for name in keys:
        if name in seen:
            raise MergeError(f"duplicate adapter tensor key: {name}")
        seen.add(name)
        target, factor = _target_from_tensor_name(name)
        pairs.setdefault(target, set()).add(factor)
    if not pairs:
        raise MergeError("adapter has no LoRA tensors")
    incomplete = sorted(target for target, factors in pairs.items() if factors != {"A", "B"})
    if incomplete:
        raise MergeError(f"adapter is missing a LoRA A/B factor for: {incomplete}")
    return sorted(pairs)


def exact_target_regex(targets: Iterable[str]) -> str:
    """Build a full-match regex for exactly the supplied target paths."""

    values = sorted(set(targets))
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise MergeError("exact target regex requires non-empty targets")
    return r"^(?:" + "|".join(re.escape(value) for value in values) + r")$"


def _tensor_state(path: Path) -> Mapping[str, Any]:
    """Load adapter tensors lazily through safetensors (never pickle)."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise MergeError("safetensors is required for a LoRA adapter") from exc
    try:
        return load_file(str(path), device="cpu")
    except Exception as exc:
        raise MergeError(f"unable to read adapter safetensors: {path}") from exc


def _tensor_header_keys(path: Path) -> list[str]:
    """Read only safetensors metadata for dry-run; no tensor materialisation."""

    try:
        from safetensors import safe_open
    except ImportError:
        # A real execute path needs ``load_file`` later.  Keeping dry-run
        # fail-closed here is preferable to accepting an uninspected adapter.
        raise MergeError("safetensors is required for adapter header validation")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return sorted(handle.keys())
    except Exception as exc:
        raise MergeError(f"unable to inspect adapter safetensors: {path}") from exc


def _validate_tensor_pair_shapes(state: Mapping[str, Any], targets: Sequence[str], rank: int) -> dict[str, Any]:
    expected = {target: {"A": None, "B": None} for target in targets}
    for name, tensor in state.items():
        target, factor = _target_from_tensor_name(name)
        if target not in expected:
            raise MergeError(f"unexpected adapter target: {target}")
        shape = tuple(getattr(tensor, "shape", ()))
        if len(shape) != 2:
            raise MergeError(f"LoRA factor must be rank-2: {name}")
        if factor == "A" and int(shape[0]) != rank:
            raise MergeError(f"LoRA A rank differs from metadata: {name}")
        if factor == "B" and int(shape[1]) != rank:
            raise MergeError(f"LoRA B rank differs from metadata: {name}")
        expected[target][factor] = shape
    if any(value["A"] is None or value["B"] is None for value in expected.values()):
        raise MergeError("adapter A/B tensor pairs are incomplete")
    for target, value in expected.items():
        if int(value["A"][1]) != int(value["B"][0]):
            raise MergeError(f"LoRA input/output dimensions disagree for {target}")
    return {target: {key: list(shape) for key, shape in value.items()} for target, value in expected.items()}


def _canonical_adapter_key(name: str) -> str:
    """Return the PEFT canonical key, adding a missing base-model prefix."""

    target, factor = _target_from_tensor_name(name)
    return f"base_model.model.{target}.lora_{factor}.weight"


def _canonical_adapter_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize both veRL and prefix-free adapter state dictionaries.

    PEFT normally writes ``base_model.model.<target>`` keys.  A few veRL/PEFT
    combinations have emitted keys beginning directly at ``model``; copying
    those keys unchanged lets PEFT report a warning and continue with an
    uninitialized adapter.  Canonicalizing into a private temporary file
    makes that failure impossible and also detects duplicate aliases.
    """

    canonical: dict[str, Any] = {}
    for name, tensor in state.items():
        key = _canonical_adapter_key(name)
        if key in canonical:
            raise MergeError(f"adapter aliases collide after prefix normalization: {name}")
        canonical[key] = tensor
    return canonical


def _normalise_loaded_lora_key(name: str) -> tuple[str, str] | None:
    """Parse a model state key if it is an A/B LoRA parameter."""

    if "lora_" not in name.lower():
        return None
    try:
        target, factor = _target_from_tensor_name(name)
    except MergeError as exc:
        raise MergeError(f"unexpected or malformed loaded LoRA parameter: {name}") from exc
    return target, factor


def _validate_loaded_adapter(model: Any, state: Mapping[str, Any], targets: Sequence[str]) -> dict[str, Any]:
    """Prove PEFT injected and loaded every expected A/B parameter.

    ``PeftModel.from_pretrained`` only returns the model and older PEFT
    versions merely warn about missing keys.  We therefore inspect the actual
    model state and compare both names, shapes, parameter counts, and values.
    This catches missing and unexpected keys even when PEFT has no structured
    load result to expose.
    """

    expected_state = _canonical_adapter_state(state)
    expected_pairs: dict[tuple[str, str], Any] = {}
    for key, tensor in expected_state.items():
        target, factor = _normalise_loaded_lora_key(key) or (None, None)
        if target is None or factor is None:
            raise MergeError(f"canonical adapter key is not an A/B parameter: {key}")
        expected_pairs[(target, factor)] = tensor

    try:
        loaded_state = model.state_dict()
    except Exception as exc:
        raise MergeError("unable to inspect loaded PEFT state") from exc
    actual_pairs: dict[tuple[str, str], Any] = {}
    for name, tensor in loaded_state.items():
        parsed = _normalise_loaded_lora_key(name)
        if parsed is not None:
            if parsed in actual_pairs:
                raise MergeError(f"duplicate loaded LoRA parameter: {name}")
            actual_pairs[parsed] = tensor

    expected_keys = set(expected_pairs)
    actual_keys = set(actual_pairs)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise MergeError(f"PEFT adapter load mismatch: missing={missing}, unexpected={unexpected}")

    expected_parameters = 0
    loaded_parameters = 0
    for pair in sorted(expected_keys):
        source, loaded = expected_pairs[pair], actual_pairs[pair]
        source_shape = tuple(getattr(source, "shape", ()))
        loaded_shape = tuple(getattr(loaded, "shape", ()))
        if source_shape != loaded_shape:
            raise MergeError(f"loaded LoRA shape differs for {pair}: expected={source_shape}, got={loaded_shape}")
        source_numel = int(getattr(source, "numel", lambda: 0)())
        loaded_numel = int(getattr(loaded, "numel", lambda: 0)())
        expected_parameters += source_numel
        loaded_parameters += loaded_numel
        # ``autocast_adapter_dtype=False`` below preserves source dtype.  Use
        # torch.equal when available so a zero/uninitialized adapter cannot
        # pass merely because its shape is correct.
        try:
            source_cpu = source.detach().cpu()
            loaded_cpu = loaded.detach().cpu()
            equal = bool((source_cpu == loaded_cpu).all().item())
        except Exception:
            equal = source is loaded
        if not equal:
            raise MergeError(f"loaded LoRA values differ for {pair}")
    if expected_parameters != loaded_parameters:
        raise MergeError(f"loaded LoRA parameter count differs: expected={expected_parameters}, got={loaded_parameters}")

    injected: set[str] = set()
    try:
        modules = model.named_modules()
    except Exception as exc:
        raise MergeError("unable to inspect PEFT injected modules") from exc
    for name, module in modules:
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        try:
            target, _factor = _target_from_tensor_name(f"{name}.lora_A.weight")
        except MergeError as exc:
            raise MergeError(f"unable to identify injected PEFT target module: {name}") from exc
        injected.add(target)
    expected_targets = set(targets)
    if injected != expected_targets:
        raise MergeError(f"PEFT injected target mismatch: expected={sorted(expected_targets)}, got={sorted(injected)}")
    return {
        "expected_tensor_count": len(expected_pairs),
        "loaded_tensor_count": len(actual_pairs),
        "expected_parameter_count": expected_parameters,
        "loaded_parameter_count": loaded_parameters,
        "injected_target_modules": sorted(injected),
    }


def lora_delta(base: Any, lora_a: Any, lora_b: Any, *, lora_alpha: float, rank: int) -> Any:
    """Return ``base + (alpha / rank) * (B @ A)`` for a numerical smoke test."""

    rank = _positive_int(rank, "rank")
    if not isinstance(lora_alpha, (int, float)) or isinstance(lora_alpha, bool) or not math.isfinite(float(lora_alpha)):
        raise MergeError("lora_alpha must be finite")
    if float(lora_alpha) <= 0:
        raise MergeError("lora_alpha must be positive")
    return base + (float(lora_alpha) / rank) * (lora_b @ lora_a)


# Stable private aliases make the audit helpers convenient to call from the
# launcher/tests while keeping the public names readable.
_derive_full_target_modules = derive_full_target_modules
_exact_target_regex = exact_target_regex
_merge_lora_delta = lora_delta


def _validate_meta(base: Path, adapter: Path, checkpoint_meta: Path | str | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    base = _directory(base, "base model")
    adapter = _directory(adapter, "LoRA adapter")
    if base == adapter:
        raise MergeError("base model and adapter must be different directories")
    if checkpoint_meta is None:
        raise MergeError("an explicit checkpoint meta path is required; refusing to infer lora_train_meta")
    meta_path = _path(checkpoint_meta, "checkpoint lora_train_meta")
    meta_file = _regular_file(meta_path, "checkpoint lora_train_meta")
    base_config = _read_json(base / "config.json", "base model config")
    adapter_config = _read_json(adapter / "adapter_config.json", "adapter config")
    meta = _read_json(meta_file, "checkpoint lora_train_meta")
    _regular_file(adapter / "adapter_model.safetensors", "adapter weights")
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise MergeError("adapter_config.peft_type must be LORA")
    rank = _positive_int(adapter_config.get("r"), "adapter_config.r")
    alpha = adapter_config.get("lora_alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise MergeError("adapter_config.lora_alpha must be positive and finite")
    meta_rank = _positive_int(meta.get("r"), "lora_train_meta.r")
    if rank != meta_rank:
        raise MergeError("adapter_config.r and lora_train_meta.r differ")
    meta_alpha = meta.get("lora_alpha")
    if isinstance(meta_alpha, bool) or not isinstance(meta_alpha, (int, float)) or not math.isfinite(float(meta_alpha)) or float(meta_alpha) <= 0:
        raise MergeError("lora_train_meta.lora_alpha must be positive and finite")
    if float(alpha) != float(meta_alpha):
        raise MergeError("adapter_config.lora_alpha and lora_train_meta.lora_alpha differ")
    task = adapter_config.get("task_type", "CAUSAL_LM")
    meta_task = meta.get("task_type", task)
    if not isinstance(task, str) or not isinstance(meta_task, str) or task != meta_task:
        raise MergeError("adapter task_type differs from lora_train_meta")
    if task != "CAUSAL_LM":
        raise MergeError(f"unsupported LoRA task_type: {task}")
    reference = adapter_config.get("base_model_name_or_path")
    if isinstance(reference, str) and reference.startswith("/"):
        if _path(reference, "adapter base model") != base:
            raise MergeError("adapter base_model_name_or_path does not identify the supplied base model")
    if not isinstance(base_config, dict):
        raise MergeError("base model config is invalid")
    meta_receipt = {"path": str(meta_file), "sha256": _sha256(meta_file)}
    return base_config, adapter_config, meta, meta_receipt


def validate_inputs(base: Path | str, adapter: Path | str, checkpoint_meta: Path | str | None = None) -> dict[str, Any]:
    base_path, adapter_path = _path(base, "base model"), _path(adapter, "LoRA adapter")
    base_config, adapter_config, meta, meta_receipt = _validate_meta(base_path, adapter_path, checkpoint_meta)
    weight_path = adapter_path / "adapter_model.safetensors"
    keys = _tensor_header_keys(weight_path)
    targets = derive_full_target_modules(keys)
    rank = _positive_int(meta["r"], "lora_train_meta.r")
    # Header validation has no shapes; execute performs the full check after
    # loading tensors.  Keeping keys and exact targets in the receipt makes
    # accidental suffix-only target changes auditable.
    configured = adapter_config.get("target_modules")
    if isinstance(configured, list) and configured and set(configured) != set(targets):
        # Old veRL adapters often carry suffix-only targets.  They are not
        # trusted; the temporary config is rewritten from tensor keys below.
        pass
    return {
        "base": {"path": str(base_path), "config_sha256": _sha256(base_path / "config.json")},
        "adapter": {"path": str(adapter_path), "config_sha256": _sha256(adapter_path / "adapter_config.json"), "weights_sha256": _sha256(weight_path)},
        "checkpoint_meta": meta_receipt,
        "base_config": base_config,
        "adapter_config": adapter_config,
        "lora_train_meta": meta,
        "tensor_keys": keys,
        "target_modules": targets,
        "target_modules_regex": exact_target_regex(targets),
        "rank": rank,
        "lora_alpha": float(meta["lora_alpha"]),
    }


def _copy_non_weight_files(base: Path, target: Path) -> None:
    for source in sorted(base.iterdir(), key=lambda item: item.name):
        if source.name in {"model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"} or source.name.startswith("model-") or source.name.startswith("pytorch_model-"):
            continue
        if source.name in _OLD_OUTPUT_MARKER_NAMES or source.name == "merge_receipt.json" or source.name.startswith("manifest"):
            # These belong to a prior publication, not to the base HF model.
            continue
        if source.is_symlink() or not source.is_file():
            continue
        shutil.copy2(source, target / source.name)


def _assert_no_lora_residue(target: Path) -> dict[str, Any]:
    forbidden_files: list[str] = []
    for path in target.rglob("*"):
        if path.is_symlink():
            raise MergeError(f"published output contains a symlink: {path}")
        if path.is_file() and (path.name in _OLD_OUTPUT_MARKER_NAMES or path.name.startswith("manifest")):
            forbidden_files.append(str(path.relative_to(target)))
        if path.is_file() and (path.name.startswith("adapter_") or path.name == "lora_train_meta.json"):
            forbidden_files.append(str(path.relative_to(target)))
    if forbidden_files:
        raise MergeError(f"published output contains adapter files: {sorted(forbidden_files)}")
    weight_keys: list[str] = []
    for path in sorted(target.glob("*.safetensors")):
        try:
            keys = _tensor_header_keys(path)
        except MergeError:
            raise
        weight_keys.extend(key for key in keys if _LORA_MARKER_RE.search(key) or "lora_" in key.lower())
    if weight_keys:
        raise MergeError(f"published output contains LoRA tensor keys: {sorted(weight_keys)[:8]}")
    weight_paths = sorted(target.glob("*.safetensors"))
    if not weight_paths:
        raise MergeError("published output contains no safetensors model weights")
    weights = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in weight_paths
    }
    return {
        "weight_files": sorted(weights),
        "weights_sha256": weights,
        "weights_inventory_sha256": hashlib.sha256(_canonical(weights)).hexdigest(),
        "lora_tensor_keys": 0,
    }


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without ever replacing an existing one."""

    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise MergeError("renameat2 is unavailable; refusing non-atomic publication")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    source_parent = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    target_parent = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        result = function(source_parent, os.fsencode(source.name), target_parent, os.fsencode(target.name), 1)
        if result:
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise MergeError(f"refusing to overwrite existing HF target: {target}")
            raise MergeError(f"atomic publication failed: {os.strerror(error)}")
    finally:
        os.close(source_parent)
        os.close(target_parent)


def _fsync_tree(path: Path) -> None:
    directories: list[Path] = []
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise MergeError(f"staging tree contains a symlink: {entry}")
        if entry.is_dir():
            directories.append(entry)
            continue
        descriptor = os.open(entry, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _load_peft_model(base: Path, adapter_config: Mapping[str, Any], state: Mapping[str, Any], targets: Sequence[str]) -> tuple[Any, dict[str, Any]]:
    try:
        import torch
        import transformers
        from peft import PeftModel
    except ImportError as exc:
        raise MergeError("transformers, torch and peft are required for execution") from exc
    # Keep the actual adapter untouched: PeftModel reads from this temporary
    # copy after its config has been narrowed to exact full paths and its keys
    # have been normalized to the canonical base_model.model prefix.
    with tempfile.TemporaryDirectory(prefix="b24-peft-config-") as temporary:
        temp_adapter = Path(temporary) / "adapter"
        temp_adapter.mkdir()
        canonical_state = _canonical_adapter_state(state)
        try:
            from safetensors.torch import save_file
            save_file(canonical_state, str(temp_adapter / "adapter_model.safetensors"))
        except Exception as exc:
            raise MergeError(f"unable to write canonical temporary adapter: {exc}") from exc
        config = dict(adapter_config)
        config["target_modules"] = list(targets)
        (temp_adapter / "adapter_config.json").write_bytes(_canonical(config) + b"\n")
        classes = [getattr(transformers, "AutoModelForImageTextToText", None), getattr(transformers, "AutoModelForVision2Seq", None), getattr(transformers, "AutoModelForCausalLM", None)]
        model_class = next((item for item in classes if item is not None), None)
        if model_class is None:
            raise MergeError("transformers has no compatible AutoModel class")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["NVIDIA_VISIBLE_DEVICES"] = ""
        try:
            # ``auto`` retains the base checkpoint's dtype (notably bf16) and
            # avoids silently materialising an fp32 publication.
            base_model = model_class.from_pretrained(str(base), local_files_only=True, device_map="cpu", torch_dtype="auto", trust_remote_code=True)
            load_result_box: dict[str, Any] = {}
            original_load_adapter = getattr(PeftModel, "load_adapter", None)

            def _capture_load_result(instance: Any, *load_args: Any, **load_kwargs: Any) -> Any:
                result = original_load_adapter(instance, *load_args, **load_kwargs)
                load_result_box["result"] = result
                return result

            if callable(original_load_adapter):
                PeftModel.load_adapter = _capture_load_result
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = PeftModel.from_pretrained(base_model, str(temp_adapter), is_trainable=False, autocast_adapter_dtype=False)
            finally:
                if callable(original_load_adapter):
                    PeftModel.load_adapter = original_load_adapter
            load_result = load_result_box.get("result")
            if load_result is not None:
                missing_keys = list(getattr(load_result, "missing_keys", ()) or ())
                unexpected_keys = list(getattr(load_result, "unexpected_keys", ()) or ())
                mismatched_keys = list(getattr(load_result, "mismatched_keys", ()) or ())
                if missing_keys or unexpected_keys or mismatched_keys:
                    raise MergeError(f"PEFT adapter load mismatch: missing={missing_keys}, unexpected={unexpected_keys}, mismatched={mismatched_keys}")
            load_warnings = [str(item.message) for item in caught if re.search(r"missing|unexpected|mismatch|ignored", str(item.message), flags=re.IGNORECASE)]
            if load_warnings:
                raise MergeError(f"PEFT adapter load reported missing/unexpected keys: {load_warnings}")
            load_inventory = _validate_loaded_adapter(model, canonical_state, targets)
            merged = model.merge_and_unload(safe_merge=True)
        except Exception as exc:
            if isinstance(exc, MergeError):
                raise
            raise MergeError(f"PEFT safe merge failed: {exc}") from exc
    if hasattr(merged, "peft_config") and getattr(merged, "peft_config", None):
        raise MergeError("safe merge left a PEFT configuration on the model")
    return merged, load_inventory


def _atomic_publish(merged: Any, base: Path, output: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise MergeError(f"create-once output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    if staging.exists():
        raise MergeError("staging path unexpectedly exists")
    try:
        staging.mkdir(mode=0o755)
        merged.save_pretrained(str(staging), safe_serialization=True)
        _copy_non_weight_files(base, staging)
        inventory = _assert_no_lora_residue(staging)
        receipt_body = dict(receipt, status="passed", output={"path": str(output), **inventory})
        receipt_body["seal_sha256"] = hashlib.sha256(_canonical(receipt_body)).hexdigest()
        (staging / "merge_receipt.json").write_bytes(_canonical(receipt_body) + b"\n")
        _fsync_tree(staging)
        # A pre-check plus atomic rename gives a crash-safe publication.  If
        # another writer won the create-once race, never overwrite it.
        if output.exists() or output.is_symlink():
            raise MergeError(f"create-once output appeared during merge: {output}")
        _rename_noreplace(staging, output)
        return inventory
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def execute(base: Path | str, adapter: Path | str, output: Path | str, checkpoint_meta: Path | str | None = None) -> dict[str, Any]:
    base_path, adapter_path, output_path = _path(base, "base model"), _path(adapter, "LoRA adapter"), _output_path(output)
    if output_path.exists() or output_path.is_symlink():
        raise MergeError(f"create-once output already exists: {output_path}")
    evidence = validate_inputs(base_path, adapter_path, checkpoint_meta)
    state = _tensor_state(adapter_path / "adapter_model.safetensors")
    shapes = _validate_tensor_pair_shapes(state, evidence["target_modules"], evidence["rank"])
    merged, load_inventory = _load_peft_model(base_path, evidence["adapter_config"], state, evidence["target_modules"])
    receipt = {"schema_version": SCHEMA_VERSION, "base": evidence["base"], "adapter": evidence["adapter"], "checkpoint_meta": evidence["checkpoint_meta"], "target_modules": evidence["target_modules"], "target_modules_regex": evidence["target_modules_regex"], "rank": evidence["rank"], "lora_alpha": evidence["lora_alpha"], "tensor_shapes": shapes, "adapter_load": load_inventory, "gpu_used": False, "eval_gold_prediction_read": False}
    inventory = _atomic_publish(merged, base_path, output_path, receipt)
    receipt = dict(receipt, status="passed", output={"path": str(output_path), **inventory})
    receipt["seal_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--base-model", "--base-dir", dest="base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", "--adapter-dir", dest="adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", "--target-dir", dest="output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-meta", "--meta", dest="checkpoint_meta", type=Path, required=True, help="explicit checkpoint lora_train_meta.json path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        output_path = _output_path(args.output)
        evidence = validate_inputs(args.base, args.adapter, args.checkpoint_meta)
        if args.dry_run:
            if output_path.exists():
                raise MergeError(f"create-once output already exists: {output_path}")
            print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "dry_run_passed", "writes_performed": 0, "gpu_used": False, "base": evidence["base"], "adapter": evidence["adapter"], "checkpoint_meta": evidence["checkpoint_meta"], "target_modules": evidence["target_modules"], "target_modules_regex": evidence["target_modules_regex"], "tensor_count": len(evidence["tensor_keys"])}, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        result = execute(args.base, args.adapter, output_path, args.checkpoint_meta)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (MergeError, OSError, ValueError, ImportError) as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "blocked", "writes_performed": 0, "gpu_used": False, "eval_gold_prediction_read": False, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
