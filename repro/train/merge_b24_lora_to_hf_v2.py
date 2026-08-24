#!/usr/bin/env python3
"""B24 PEFT-to-HF merger with the rectangular-LoRA shape contract fixed.

The sealed B24 launch and adapter-conversion evidence pins the exact v1
merger source, so that file must remain immutable.  This narrowly scoped v2
loads that pinned implementation, replaces only its incorrect rectangular
linear-layer shape assertion, and then delegates the full fail-closed CPU
merge/publication lifecycle to it.

For ``Linear(in_features, out_features)``, LoRA A has shape
``[rank, in_features]`` and B has shape ``[out_features, rank]``.  Therefore
``B @ A`` requires the two rank axes to agree; it does not require the input
and output feature counts to be equal.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence


V1_PATH = Path(__file__).resolve().with_name("merge_b24_lora_to_hf_v1.py")
V1_SHA256 = "b3f4b4bab9544b6b87dc2bc5ea41a07dbad77e8b50dc11de6b97d02322b81da9"
SCHEMA_VERSION = "b24_lora_to_hf_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if _sha256(V1_PATH) != V1_SHA256:
    raise RuntimeError(f"pinned B24 v1 merger source changed: {V1_PATH}")

_SPEC = importlib.util.spec_from_file_location("b24_lora_merger_v1_pinned", V1_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load pinned B24 v1 merger: {V1_PATH}")
_V1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V1)

# Re-export the audited v1 helpers, including private validation helpers used
# by CPU contract tests.  The one overridden helper is assigned below.
for _name in dir(_V1):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_V1, _name))


def _validate_tensor_pair_shapes(
    state: Mapping[str, Any], targets: Sequence[str], rank: int
) -> dict[str, Any]:
    expected = {target: {"A": None, "B": None} for target in targets}
    for name, tensor in state.items():
        target, factor = _V1._target_from_tensor_name(name)
        if target not in expected:
            raise _V1.MergeError(f"unexpected adapter target: {target}")
        shape = tuple(getattr(tensor, "shape", ()))
        if len(shape) != 2:
            raise _V1.MergeError(f"LoRA factor must be rank-2: {name}")
        if factor == "A" and int(shape[0]) != rank:
            raise _V1.MergeError(f"LoRA A rank differs from metadata: {name}")
        if factor == "B" and int(shape[1]) != rank:
            raise _V1.MergeError(f"LoRA B rank differs from metadata: {name}")
        expected[target][factor] = shape
    if any(value["A"] is None or value["B"] is None for value in expected.values()):
        raise _V1.MergeError("adapter A/B tensor pairs are incomplete")
    for target, value in expected.items():
        if int(value["A"][0]) != int(value["B"][1]):
            raise _V1.MergeError(f"LoRA inner rank dimensions disagree for {target}")
    return {
        target: {factor: list(shape) for factor, shape in value.items()}
        for target, value in expected.items()
    }


def _assert_merged_model_has_no_lora(model: Any) -> dict[str, Any]:
    """Reject actual tuner modules/parameters while tolerating PEFT metadata.

    PEFT 0.18 returns the original Transformers model from
    ``merge_and_unload``.  That model can retain a non-empty ``peft_config``
    Python attribute even after every adapter layer was replaced by its base
    layer.  The attribute is not evidence of an active adapter.  Inspecting
    the concrete state and module tree is the fail-closed invariant; the
    subsequent staged safetensors/header audit remains a second boundary.
    """

    try:
        state_names = list(model.state_dict().keys())
        modules = list(model.named_modules())
    except Exception as exc:
        raise _V1.MergeError("unable to audit merged model for LoRA residue") from exc
    residue_keys = sorted(
        name for name in state_names
        if "lora_" in name.lower() or _V1._LORA_MARKER_RE.search(name)
    )
    if residue_keys:
        raise _V1.MergeError(f"safe merge left LoRA parameter keys: {residue_keys[:8]}")
    residue_modules: list[str] = []
    for name, module in modules:
        module_type = f"{type(module).__module__}.{type(module).__qualname__}"
        if (
            hasattr(module, "lora_A")
            or hasattr(module, "lora_B")
            or module_type.startswith("peft.tuners.")
        ):
            residue_modules.append(name or "<root>")
    if residue_modules:
        raise _V1.MergeError(f"safe merge left PEFT tuner modules: {sorted(residue_modules)[:8]}")
    return {
        "state_tensor_count_after_unload": len(state_names),
        "module_count_after_unload": len(modules),
        "lora_parameter_keys_after_unload": 0,
        "peft_tuner_modules_after_unload": 0,
        "peft_config_metadata_present": bool(getattr(model, "peft_config", None)),
    }


def _load_peft_model(
    base: Path,
    adapter_config: Mapping[str, Any],
    state: Mapping[str, Any],
    targets: Sequence[str],
) -> tuple[Any, dict[str, Any]]:
    """Pinned v1 loader with a concrete post-unload residue audit."""

    try:
        import transformers
        from peft import PeftModel
    except ImportError as exc:
        raise _V1.MergeError("transformers, torch and peft are required for execution") from exc
    with tempfile.TemporaryDirectory(prefix="b24-peft-config-") as temporary:
        temp_adapter = Path(temporary) / "adapter"
        temp_adapter.mkdir()
        canonical_state = _V1._canonical_adapter_state(state)
        try:
            from safetensors.torch import save_file

            save_file(canonical_state, str(temp_adapter / "adapter_model.safetensors"))
        except Exception as exc:
            raise _V1.MergeError(f"unable to write canonical temporary adapter: {exc}") from exc
        config = dict(adapter_config)
        config["target_modules"] = list(targets)
        (temp_adapter / "adapter_config.json").write_bytes(_V1._canonical(config) + b"\n")
        classes = [
            getattr(transformers, "AutoModelForImageTextToText", None),
            getattr(transformers, "AutoModelForVision2Seq", None),
            getattr(transformers, "AutoModelForCausalLM", None),
        ]
        model_class = next((item for item in classes if item is not None), None)
        if model_class is None:
            raise _V1.MergeError("transformers has no compatible AutoModel class")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["NVIDIA_VISIBLE_DEVICES"] = ""
        try:
            base_model = model_class.from_pretrained(
                str(base),
                local_files_only=True,
                device_map="cpu",
                torch_dtype="auto",
                trust_remote_code=True,
            )
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
                    model = PeftModel.from_pretrained(
                        base_model,
                        str(temp_adapter),
                        is_trainable=False,
                        autocast_adapter_dtype=False,
                    )
            finally:
                if callable(original_load_adapter):
                    PeftModel.load_adapter = original_load_adapter
            load_result = load_result_box.get("result")
            if load_result is not None:
                missing_keys = list(getattr(load_result, "missing_keys", ()) or ())
                unexpected_keys = list(getattr(load_result, "unexpected_keys", ()) or ())
                mismatched_keys = list(getattr(load_result, "mismatched_keys", ()) or ())
                if missing_keys or unexpected_keys or mismatched_keys:
                    raise _V1.MergeError(
                        "PEFT adapter load mismatch: "
                        f"missing={missing_keys}, unexpected={unexpected_keys}, mismatched={mismatched_keys}"
                    )
            load_warnings = [
                str(item.message)
                for item in caught
                if re.search(r"missing|unexpected|mismatch|ignored", str(item.message), flags=re.IGNORECASE)
            ]
            if load_warnings:
                raise _V1.MergeError(
                    f"PEFT adapter load reported missing/unexpected keys: {load_warnings}"
                )
            load_inventory = _V1._validate_loaded_adapter(model, canonical_state, targets)
            merged = model.merge_and_unload(safe_merge=True)
            unload_inventory = _assert_merged_model_has_no_lora(merged)
        except Exception as exc:
            if isinstance(exc, _V1.MergeError):
                raise
            raise _V1.MergeError(f"PEFT safe merge failed: {exc}") from exc
    return merged, {**load_inventory, "post_unload": unload_inventory}


# v1's execute() resolves these helpers and schema from its own module globals.
_V1._validate_tensor_pair_shapes = _validate_tensor_pair_shapes
_V1._load_peft_model = _load_peft_model
_V1.SCHEMA_VERSION = SCHEMA_VERSION


def main(argv: Sequence[str] | None = None) -> int:
    return _V1.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
