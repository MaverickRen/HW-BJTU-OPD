"""Heterogeneous full-answer SFT adapter for the B54 10k publication.

All rows use the student images only.  The loss mask covers the complete
gold answer (strict MCQ letters and normalized/free-form answers alike).
Receipts and audits report only aggregate counts.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.dataset.vision_utils import process_image
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils import hf_tokenizer
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer.chat_template import apply_chat_template


_IMAGE_MARKER = re.compile(r"(<image>)")
_EXPECTED_SOURCES = frozenset(
    {
        "TIGER-Lab/Mantis-Instruct",
        "UCSC-VLAA",
        "UCSC-VLAA/VLM-CapCurriculum-VisualReasoning-Data",
        "aRefCOCO",
        "arefcoco/referring_expression_grounding",
        "datajuicer/VeriSciQA",
        "mantis_instruct/multi_vqa",
        "mantis_instruct/nlvr2",
        "verisciqa/scientific_figure_mcq",
        "visual_reasoning/synthesis",
    }
)


def _as_list(value: Any) -> Any:
    return convert_nested_value_to_list_recursive(value)


def _tokenizer_from_processor(processor: Any) -> Any:
    return getattr(processor, "tokenizer", None) or processor


def _encode_answer(tokenizer: Any, answer: str) -> list[int]:
    ids = list(tokenizer.encode(answer, add_special_tokens=False))
    if not ids:
        raise ValueError("gold answer tokenizes to an empty sequence")
    return ids


def build_length_bucket_order(
    dataframe: pd.DataFrame,
    *,
    global_batch_size: int,
    seed: int,
    image_token_proxy: int,
) -> list[int]:
    """Return deterministic, shuffled-batch, length-grouped row positions.

    Rows are globally sorted by a text-plus-image length proxy, divided into
    global batches, shuffled inside each batch, and then shuffled by batch.
    With ``DistributedSampler(shuffle=False)`` each DP rank receives its slice
    of the same approximately length-homogeneous global batch.
    """
    if global_batch_size < 1 or image_token_proxy < 0:
        raise ValueError("invalid B54 length-bucket controls")
    ranked: list[tuple[int, str, int]] = []
    prompts = dataframe["prompt"].tolist()
    images_column = dataframe["images"].tolist()
    stable_keys = dataframe["source_record_sha256"].astype(str).tolist()
    for position, (prompt_value, image_value, stable_key) in enumerate(zip(prompts, images_column, stable_keys, strict=True)):
        prompt = _as_list(prompt_value) or []
        images = _as_list(image_value) or []
        if not isinstance(prompt, list) or not isinstance(images, list):
            raise ValueError("B54 length bucketing received malformed prompt/images")
        text_chars = sum(len(str(message.get("content", ""))) for message in prompt if isinstance(message, Mapping))
        ranked.append((text_chars + image_token_proxy * len(images), stable_key, position))
    ranked.sort()
    batches = [[position for _, _, position in ranked[start : start + global_batch_size]] for start in range(0, len(ranked), global_batch_size)]
    generator = random.Random(seed)
    for batch in batches:
        generator.shuffle(batch)
    generator.shuffle(batches)
    order = [position for batch in batches for position in batch]
    if len(order) != len(dataframe) or len(set(order)) != len(dataframe):
        raise ValueError("B54 length-bucket order is not a permutation")
    return order


def locate_answer_span(input_ids: torch.Tensor, tokenizer: Any, answer: str) -> tuple[int, int]:
    answer_ids = _encode_answer(tokenizer, answer)
    ids = input_ids.detach().cpu().tolist()
    width = len(answer_ids)
    matches = [start for start in range(0, len(ids) - width + 1) if ids[start : start + width] == answer_ids]
    if not matches:
        raise ValueError("complete gold answer is absent from rendered conversation")
    start = matches[-1]
    stop = start + width
    if tokenizer.decode(ids[start:stop], skip_special_tokens=False) != answer:
        raise ValueError("gold answer decode mismatch")
    return start, stop


class B54TenKSFTDataset(Dataset):
    """Render B54 rows under veRL's no-padding multimodal SFT contract."""

    def __init__(self, parquet_files: str | list[str], tokenizer: Any, config: Any, processor: Optional[Any] = None, max_samples: int = -1) -> None:
        config = config or {}
        self.processor = processor
        if self.processor is None:
            raise ValueError("B54 SFT requires a multimodal processor")
        self.tokenizer = hf_tokenizer(tokenizer) if isinstance(tokenizer, str) else _tokenizer_from_processor(tokenizer)
        self.messages_key = str(config.get("messages_key", "prompt"))
        self.image_key = str(config.get("image_key", "images"))
        self.max_length = int(config.get("max_length", 16_384))
        self.truncation = str(config.get("truncation", "error"))
        self.pad_mode = str(config.get("pad_mode", "no_padding"))
        self.length_bucket_batch = bool(config.get("length_bucket_batch", False))
        self.global_batch_size = int(config.get("train_batch_size", 0))
        self.length_bucket_seed = int(config.get("length_bucket_seed", 42))
        self.image_token_proxy = int(config.get("length_bucket_image_token_proxy", 1024))
        self.image_patch_size = config.get("image_patch_size", getattr(getattr(self.processor, "image_processor", None), "patch_size", 16))
        if self.pad_mode != "no_padding" or self.truncation not in {"error", "right"}:
            raise ValueError("B54 requires no-padding SFT and error/right truncation")
        paths = list(parquet_files) if isinstance(parquet_files, (list, tuple)) else [parquet_files]
        self.parquet_files = [copy_local_path_from_hdfs(path, verbose=True) for path in paths]
        frames = [pd.read_parquet(path, dtype_backend="pyarrow") for path in self.parquet_files]
        if not frames:
            raise ValueError("B54 received no parquet files")
        self.dataframe = pd.concat(frames, ignore_index=True)
        if max_samples > 0:
            self.dataframe = self.dataframe.iloc[:max_samples].reset_index(drop=True)
        required = {self.messages_key, self.image_key, "reward_model", "extra_info", "data_source"}
        missing = required - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"B54 parquet is missing required columns: {sorted(missing)}")
        if self.length_bucket_batch:
            order = build_length_bucket_order(
                self.dataframe,
                global_batch_size=self.global_batch_size,
                seed=self.length_bucket_seed,
                image_token_proxy=self.image_token_proxy,
            )
            self.dataframe = self.dataframe.iloc[order].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    @staticmethod
    def _answer(row: Mapping[str, Any]) -> str:
        extra = _as_list(row["extra_info"]) or {}
        reward = _as_list(row["reward_model"]) or {}
        answer = extra.get("answer")
        ground_truth = reward.get("ground_truth")
        if not isinstance(answer, str) or not answer or not isinstance(ground_truth, str) or answer != ground_truth:
            raise ValueError("B54 requires identical nonempty answer and ground_truth")
        return answer

    def _messages(self, row: Mapping[str, Any], answer: str) -> list[dict[str, Any]]:
        prompt = _as_list(row[self.messages_key])
        images = _as_list(row[self.image_key]) or []
        if str(row.get("data_source")) not in _EXPECTED_SOURCES or not isinstance(prompt, list) or not prompt:
            raise ValueError("B54 row has unsupported source or empty prompt")
        if any(not isinstance(message, Mapping) or message.get("role") not in {"user", "assistant"} for message in prompt) or prompt[-1].get("role") != "user":
            raise ValueError("B54 prompt must be user/assistant messages ending with user")
        if not 1 <= len(images) <= 6:
            raise ValueError("B54 rows must contain one through six images")
        messages: list[dict[str, Any]] = []
        image_index = 0
        for message in prompt:
            content: list[dict[str, Any]] = []
            for segment in _IMAGE_MARKER.split(str(message.get("content", ""))):
                if not segment:
                    continue
                if segment == "<image>":
                    if image_index >= len(images):
                        raise ValueError("B54 prompt contains too many image markers")
                    content.append({"type": "image", "image": process_image(dict(_as_list(images[image_index])), image_patch_size=self.image_patch_size)})
                    image_index += 1
                else:
                    content.append({"type": "text", "text": segment})
            messages.append({"role": message["role"], "content": content})
        if image_index != len(images):
            raise ValueError("B54 prompt/image placeholder mismatch")
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return messages

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index].to_dict()
        answer = self._answer(row)
        rendered = dict(apply_chat_template(self.processor, messages=self._messages(row, answer), tools=None, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False))
        input_ids = rendered.pop("input_ids").squeeze(0)
        attention_mask = rendered.pop("attention_mask").squeeze(0)
        if input_ids.ndim != 1 or attention_mask.shape != input_ids.shape:
            raise ValueError("processor returned invalid token shapes")
        start, stop = locate_answer_span(input_ids, self.tokenizer, answer)
        loss_mask = torch.zeros_like(attention_mask)
        loss_mask[start:stop] = 1
        if input_ids.numel() > self.max_length:
            if self.truncation != "right" or stop > self.max_length:
                raise ValueError("B54 row exceeds max_length without retaining answer")
            input_ids, attention_mask, loss_mask = input_ids[: self.max_length], attention_mask[: self.max_length], loss_mask[: self.max_length]
        image_grid_thw = rendered.get("image_grid_thw")
        video_grid_thw = rendered.get("video_grid_thw")
        second_per_grid_ts = rendered.get("second_per_grid_ts")
        if "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            vision_position_ids = get_rope_index(self.processor, input_ids=input_ids, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, second_per_grid_ts=second_per_grid_ts, attention_mask=attention_mask)
            position_ids = torch.cat((torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0), vision_position_ids), dim=0)
        else:
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)
        multi_modal_inputs: dict[str, Any] = {}
        for key, value in rendered.items():
            if key == "mm_token_type_ids" or value is None:
                continue
            if key in {"image_grid_thw", "video_grid_thw"} and isinstance(value, torch.Tensor) and value.ndim == 1:
                value = value.unsqueeze(0)
            multi_modal_inputs[key] = value
        result: dict[str, Any] = {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": loss_mask}
        if multi_modal_inputs:
            result["multi_modal_inputs"] = multi_modal_inputs
        return result


def audit_dataset_rows(dataset: B54TenKSFTDataset) -> dict[str, Any]:
    """Exhaustively audit rendered rows while retaining aggregate counters only."""
    rows = 0
    image_counts: dict[str, int] = {}
    style_counts: dict[str, int] = {}
    answer_token_min: int | None = None
    answer_token_max = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        row = dataset.dataframe.iloc[index].to_dict()
        answer = dataset._answer(row)
        reward = _as_list(row["reward_model"]) or {}
        mask = sample["loss_mask"].to(dtype=torch.bool)
        token_count = int(mask.sum().item())
        if token_count != len(_encode_answer(dataset.tokenizer, answer)):
            raise ValueError("B54 loss mask does not cover complete answer")
        if dataset.tokenizer.decode(sample["input_ids"][mask].detach().cpu().tolist(), skip_special_tokens=False) != answer:
            raise ValueError("B54 masked tokens do not decode to answer")
        images = _as_list(row[dataset.image_key]) or []
        image_counts[str(len(images))] = image_counts.get(str(len(images)), 0) + 1
        style = str(reward.get("style"))
        style_counts[style] = style_counts.get(style, 0) + 1
        answer_token_min = token_count if answer_token_min is None else min(answer_token_min, token_count)
        answer_token_max = max(answer_token_max, token_count)
        rows += 1
    return {"rows": rows, "image_counts": image_counts, "styles": style_counts, "loss_mask": {"kind": "complete_gold_answer", "token_min": answer_token_min, "token_max": answer_token_max}, "student_images_only": True, "teacher_crop_input": False, "aggregate_only": True, "sample_payload_written": False}
