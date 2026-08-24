#!/usr/bin/env python3
"""B24 LoRA merger using a generic PEFT wrapper for the multimodal model.

The v2 recovery proved that every one of the 716 LoRA tensors loaded and
merged numerically, but PEFT selected ``PeftModelForCausalLM`` from the
training metadata.  Qwen3.5-VL is a conditional-generation wrapper around a
language model, so that task-specific PEFT wrapper returned the wrong nested
submodel and duplicated ``language_model`` prefixes in 759/760 saved keys.

This create-once v3 keeps the immutable v1/v2 implementations pinned and
changes only the temporary mechanical PEFT wrapper to the generic
``PeftModel`` (``task_type=None``).  The authoritative adapter metadata still
must say ``CAUSAL_LM`` before this point.  Before any publication, the merged
model's complete state-key universe must exactly equal the 760-key base HF
checkpoint.  The inherited staged safetensors audit then checks it again.
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


V2_PATH = Path(__file__).resolve().with_name("merge_b24_lora_to_hf_v2.py")
V2_SHA256 = "be7242e2a065006993e683b2481c809259e6ce53d815d29787d25fde4711c34f"
SCHEMA_VERSION = "b24_lora_to_hf_v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if _sha256(V2_PATH) != V2_SHA256:
    raise RuntimeError(f"pinned B24 v2 merger source changed: {V2_PATH}")
_SPEC = importlib.util.spec_from_file_location("b24_lora_merger_v2_pinned", V2_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load pinned B24 v2 merger: {V2_PATH}")
_V2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V2)

for _name in dir(_V2):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_V2, _name))


def _base_tensor_keys(base: Path) -> list[str]:
    weight = base / "model.safetensors"
    _V2._V1._regular_file(weight, "base model safetensors")
    index = base / "model.safetensors.index.json"
    if index.exists() or index.is_symlink():
        raise _V2.MergeError("B24 v3 requires the pinned single-file base checkpoint")
    keys = _V2._V1._tensor_header_keys(weight)
    if len(keys) != 760 or len(keys) != len(set(keys)):
        raise _V2.MergeError(f"base tensor-key universe differs: {len(keys)}")
    return sorted(keys)


def _validate_merged_key_universe(model: Any, base_keys: Sequence[str]) -> dict[str, Any]:
    try:
        merged_keys = sorted(model.state_dict().keys())
    except Exception as exc:
        raise _V2.MergeError("unable to inspect merged model state-key universe") from exc
    expected = list(base_keys)
    if merged_keys != expected:
        expected_set, observed_set = set(expected), set(merged_keys)
        missing = sorted(expected_set - observed_set)
        unexpected = sorted(observed_set - expected_set)
        raise _V2.MergeError(
            "generic PEFT merge changed the base state-key universe: "
            f"expected={len(expected)}, observed={len(merged_keys)}, "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    return {
        "base_tensor_key_count": len(expected),
        "merged_tensor_key_count": len(merged_keys),
        "exact_base_key_universe": True,
        "tensor_keys_sha256": hashlib.sha256(_V2._V1._canonical(expected)).hexdigest(),
    }


def _load_peft_model(
    base: Path,
    adapter_config: Mapping[str, Any],
    state: Mapping[str, Any],
    targets: Sequence[str],
) -> tuple[Any, dict[str, Any]]:
    try:
        import transformers
        from peft import PeftModel
    except ImportError as exc:
        raise _V2.MergeError("transformers, torch and peft are required for execution") from exc

    base_keys = _base_tensor_keys(base)
    with tempfile.TemporaryDirectory(prefix="b24-peft-generic-") as temporary:
        temp_adapter = Path(temporary) / "adapter"
        temp_adapter.mkdir()
        canonical_state = _V2._V1._canonical_adapter_state(state)
        try:
            from safetensors.torch import save_file

            save_file(canonical_state, str(temp_adapter / "adapter_model.safetensors"))
        except Exception as exc:
            raise _V2.MergeError(f"unable to write canonical temporary adapter: {exc}") from exc
        config = dict(adapter_config)
        config["target_modules"] = list(targets)
        # This changes only PEFT's wrapper selection.  The original adapter
        # and checkpoint metadata were already required to be CAUSAL_LM.
        config["task_type"] = None
        (temp_adapter / "adapter_config.json").write_bytes(_V2._V1._canonical(config) + b"\n")

        classes = [
            getattr(transformers, "AutoModelForImageTextToText", None),
            getattr(transformers, "AutoModelForVision2Seq", None),
            getattr(transformers, "AutoModelForCausalLM", None),
        ]
        model_class = next((item for item in classes if item is not None), None)
        if model_class is None:
            raise _V2.MergeError("transformers has no compatible AutoModel class")
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
                    raise _V2.MergeError(
                        "PEFT adapter load mismatch: "
                        f"missing={missing_keys}, unexpected={unexpected_keys}, mismatched={mismatched_keys}"
                    )
            load_warnings = [
                str(item.message)
                for item in caught
                if re.search(r"missing|unexpected|mismatch|ignored", str(item.message), flags=re.IGNORECASE)
            ]
            if load_warnings:
                raise _V2.MergeError(
                    f"PEFT adapter load reported missing/unexpected keys: {load_warnings}"
                )
            load_inventory = _V2._V1._validate_loaded_adapter(model, canonical_state, targets)
            merged = model.merge_and_unload(safe_merge=True)
            unload_inventory = _V2._assert_merged_model_has_no_lora(merged)
            key_inventory = _validate_merged_key_universe(merged, base_keys)
        except Exception as exc:
            if isinstance(exc, _V2.MergeError):
                raise
            raise _V2.MergeError(f"PEFT safe merge failed: {exc}") from exc
    return merged, {
        **load_inventory,
        "temporary_peft_task_type": None,
        "post_unload": {**unload_inventory, **key_inventory},
    }


_V2._V1._load_peft_model = _load_peft_model
_V2._V1.SCHEMA_VERSION = SCHEMA_VERSION


def main(argv: Sequence[str] | None = None) -> int:
    return _V2._V1.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
