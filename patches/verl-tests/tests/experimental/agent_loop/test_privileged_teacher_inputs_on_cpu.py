# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.teacher_loop.teacher_manager import align_teacher_response_outputs
from verl.trainer.ppo.v1.agent_loop_tq import _validate_teacher_output_for_tq
from verl.trainer.ppo.v1.trainer_base import _normalize_tq_non_tensor_sequence
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.workers.config.distillation import DistillationConfig


class _FakeDataset:
    calls: list[list[dict[str, Any]]] = []

    @classmethod
    async def process_multi_modal_info(cls, messages, image_patch_size, config):
        assert image_patch_size == 14
        assert config.prompt_key == "prompt"
        cls.calls.append(messages)
        images = []
        for message in messages:
            for item in message.get("content", []):
                if isinstance(item, dict) and item.get("type") == "image":
                    images.append(item["image"])
        return images or None, None, None


class _FakeTokenizer:
    unk_token_id = -1

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        del token
        return -1


class _FakeProcessor:
    image_token_id = 900
    video_token_id = 901
    vision_start_token_id = 902
    vision_end_token_id = 903

    def __init__(self):
        self.image_processor = SimpleNamespace(patch_size=14)
        self.tokenizer = _FakeTokenizer()
        self.rendered_messages: list[list[dict[str, Any]]] = []
        self.processor_images: list[list[Image.Image]] = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True
        self.rendered_messages.append(messages)
        return "<teacher-prompt>"

    def __call__(self, *, text, images, videos, return_tensors, **kwargs):
        assert text == ["<teacher-prompt>"]
        assert videos is None
        assert return_tensors == "pt"
        assert kwargs == {"min_pixels": 16}
        self.processor_images.append(images)
        # Deliberately differs from the two-token student prompt.
        return {"input_ids": torch.tensor([[10, 11, 12, 13]], dtype=torch.long)}


class _FakeTeacherManager:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def compute_teacher_logprobs_single(self, **kwargs):
        self.calls.append(kwargs)
        sequence_length = len(kwargs["sequence_ids"])
        rows = torch.arange(sequence_length, dtype=torch.int32)
        teacher_ids = torch.stack((rows * 10 + 1, rows * 10 + 2), dim=1)
        teacher_logprobs = teacher_ids.to(torch.float32) / 100
        return teacher_ids, teacher_logprobs


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (8, 8), color=color)


def _worker(*, fallback: bool) -> tuple[AgentLoopWorker, _FakeProcessor, _FakeTeacherManager]:
    worker = object.__new__(AgentLoopWorker)
    processor = _FakeProcessor()
    teacher = _FakeTeacherManager()
    worker.processor = processor
    worker.tokenizer = processor.tokenizer
    worker.dataset_cls = _FakeDataset
    worker.data_config = OmegaConf.create({"prompt_key": "prompt"})
    worker.apply_chat_template_kwargs = {}
    worker.mm_processor_kwargs = {"min_pixels": 16}
    worker.rollout_config = SimpleNamespace(prompt_length=32)
    worker.loop = None
    worker.distillation_enabled = True
    worker.distillation_config = SimpleNamespace(
        teacher_image_key="bbox_images",
        fallback_to_student_images=fallback,
    )
    worker.teacher_key = "data_source"
    worker.teacher_server_manager = teacher
    return worker, processor, teacher


def _output(student_image: Image.Image, response_ids: list[int] | None = None) -> AgentLoopOutput:
    response_ids = response_ids or [20, 21, 22]
    return AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=response_ids,
        response_mask=[1] * len(response_ids),
        multi_modal_data={"images": [student_image]},
        mm_processor_kwargs={"student": True},
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )


def _compute(worker: AgentLoopWorker, output: AgentLoopOutput, sample_kwargs: dict[str, Any]) -> None:
    async def run() -> None:
        worker.loop = asyncio.get_running_loop()
        await worker._compute_teacher_logprobs(
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=False,
            sample_kwargs=sample_kwargs,
        )

    asyncio.run(run())


def test_privileged_response_alignment_uses_causal_prediction_rows_on_cpu():
    teacher_ids = torch.tensor(
        [[101], [102], [103], [200], [201], [202], [0]], dtype=torch.int32
    )
    teacher_logprobs = teacher_ids.to(torch.float32) / 10

    aligned_ids, aligned_logprobs = align_teacher_response_outputs(
        teacher_ids,
        teacher_logprobs,
        teacher_prompt_length=4,
        student_prompt_length=2,
        response_length=3,
    )

    assert aligned_ids[:, 0].tolist() == [0, 200, 201, 202, 0]
    assert aligned_logprobs[:, 0].tolist() == pytest.approx([0, 20, 20.1, 20.2, 0])


def test_privileged_teacher_rebuilds_prompt_and_aligns_response_on_cpu():
    student_image, teacher_crop = _image((0, 0, 255)), _image((255, 0, 0))
    worker, processor, teacher = _worker(fallback=False)
    output = _output(student_image)
    raw_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": student_image},
                {"type": "text", "text": "Where is the object?"},
            ],
        }
    ]

    _compute(
        worker,
        output,
        {
            "raw_prompt": raw_prompt,
            "bbox_images": [{"image": teacher_crop}],
            "data_source": "vision_opd",
            "index": 7,
        },
    )

    assert len(teacher.calls) == 1
    call = teacher.calls[0]
    assert call["sequence_ids"] == [10, 11, 12, 13, 20, 21, 22]
    assert call["routing_key"] == "vision_opd"
    assert call["mm_processor_kwargs"] == {"min_pixels": 16}
    assert call["multi_modal_data"]["images"][0].getpixel((0, 0)) == (255, 0, 0)
    assert processor.processor_images[0][0].getpixel((0, 0)) == (255, 0, 0)
    assert processor.rendered_messages[0][0]["content"][0]["image"].getpixel((0, 0)) == (255, 0, 0)
    # The raw student prompt remains untouched.
    assert raw_prompt[0]["content"][0]["image"].getpixel((0, 0)) == (0, 0, 255)

    teacher_ids = output.extra_fields["teacher_ids"]
    teacher_logprobs = output.extra_fields["teacher_logprobs"]
    assert teacher_ids.shape == teacher_logprobs.shape == (5, 2)
    assert teacher_ids.tolist() == [[0, 0], [31, 32], [41, 42], [51, 52], [0, 0]]
    evidence = output.extra_fields["teacher_input_evidence"]
    assert evidence == {
        "teacher_image_key": "bbox_images",
        "privileged_images_used": True,
        "fallback_to_student_images_used": False,
        "student_prompt_tokens": 2,
        "teacher_prompt_tokens": 4,
        "response_tokens": 3,
        "teacher_image_count": 1,
        "alignment": "response_logprob_positions",
    }


def test_empty_teacher_images_use_explicit_student_fallback_on_cpu():
    student_image = _image((0, 255, 0))
    worker, processor, teacher = _worker(fallback=True)
    output = _output(student_image)

    _compute(
        worker,
        output,
        {"raw_prompt": [], "bbox_images": [], "data_source": "general"},
    )

    assert processor.processor_images == []
    assert teacher.calls[0]["sequence_ids"] == [1, 2, 20, 21, 22]
    assert teacher.calls[0]["multi_modal_data"]["images"][0].getpixel((0, 0)) == (0, 255, 0)
    assert output.extra_fields["teacher_ids"].tolist() == [
        [1, 2],
        [11, 12],
        [21, 22],
        [31, 32],
        [41, 42],
    ]
    assert output.extra_fields["teacher_input_evidence"] == {
        "teacher_image_key": "bbox_images",
        "privileged_images_used": False,
        "fallback_to_student_images_used": True,
        "student_prompt_tokens": 2,
        "teacher_prompt_tokens": 2,
        "response_tokens": 3,
        "teacher_image_count": 1,
        "alignment": "identity_student_prompt",
    }


def test_empty_teacher_images_fail_closed_without_fallback_on_cpu():
    worker, _, teacher = _worker(fallback=False)
    output = _output(_image((0, 0, 255)))

    with pytest.raises(ValueError, match="fallback is disabled"):
        _compute(
            worker,
            output,
            {"raw_prompt": [], "bbox_images": [], "data_source": "vision_opd"},
        )
    assert teacher.calls == []


def test_visual_special_tokens_in_response_fail_before_teacher_on_cpu():
    worker, _, teacher = _worker(fallback=True)
    output = _output(_image((0, 0, 255)), response_ids=[20, 900, 22])

    with pytest.raises(ValueError, match="visual special tokens"):
        _compute(
            worker,
            output,
            {"raw_prompt": [], "bbox_images": [], "data_source": "general"},
        )
    assert teacher.calls == []


def test_rlhf_dataset_retains_non_student_bbox_image_column_on_cpu():
    teacher_ref = {"path": "/frozen/teacher-crop.png"}
    row = {
        "prompt": [{"role": "user", "content": "<image>question"}],
        "images": [{"path": "/frozen/student.png"}],
        "bbox_images": [teacher_ref],
        "data_source": "vision_opd",
        "extra_info": {"index": 17},
    }

    class _Frame:
        def __getitem__(self, item):
            assert item == 0
            return deepcopy(row)

    dataset = object.__new__(RLHFDataset)
    dataset.dataframe = _Frame()
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.audio_key = "audios"
    dataset.prompt_key = "prompt"
    dataset.need_tools_kwargs = False
    dataset._build_messages = lambda example, key: deepcopy(example[key])

    item = dataset[0]
    assert "images" not in item
    assert item["bbox_images"] == [teacher_ref]
    assert item["index"] == 17


def test_teacher_evidence_survives_agent_output_to_transfer_queue_on_cpu():
    output = _output(_image((0, 0, 255)))
    sequence_length = len(output.prompt_ids) + len(output.response_ids)
    output.extra_fields.update(
        {
            "teacher_ids": torch.ones((sequence_length, 2), dtype=torch.int32),
            "teacher_logprobs": torch.zeros((sequence_length, 2), dtype=torch.float32),
            "teacher_input_evidence": {
                "teacher_image_key": "bbox_images",
                "privileged_images_used": True,
            },
        }
    )

    _validate_teacher_output_for_tq(output, require_teacher_input_evidence=True)
    field = output.as_dict()
    field.pop("multi_modal_data", None)
    field["uid"] = "sample-0"
    field["reward_model"] = {"ground_truth": "A"}
    transfer_batch = list_of_dict_to_tensordict([field])
    extra_fields = _normalize_tq_non_tensor_sequence(transfer_batch["extra_fields"])

    assert extra_fields == [
        {
            "teacher_input_evidence": {
                "teacher_image_key": "bbox_images",
                "privileged_images_used": True,
            }
        }
    ]
    assert _normalize_tq_non_tensor_sequence(transfer_batch["uid"]) == ["sample-0"]
    assert _normalize_tq_non_tensor_sequence(transfer_batch["reward_model"]) == [
        {"ground_truth": "A"}
    ]
    assert transfer_batch["teacher_ids"][0].shape == (sequence_length, 2)


def test_transfer_queue_rejects_unaligned_teacher_rows_on_cpu():
    output = _output(_image((0, 0, 255)))
    output.extra_fields.update(
        {
            "teacher_ids": torch.ones((4, 2), dtype=torch.int32),
            "teacher_logprobs": torch.zeros((4, 2), dtype=torch.float32),
            "teacher_input_evidence": {},
        }
    )

    with pytest.raises(ValueError, match="Teacher rows must align"):
        _validate_teacher_output_for_tq(output, require_teacher_input_evidence=True)


@pytest.mark.parametrize("teacher_image_key", ["", " bbox_images", "bbox_images ", 3])
def test_distillation_config_rejects_invalid_teacher_image_key(teacher_image_key):
    with pytest.raises((TypeError, ValueError)):
        DistillationConfig(enabled=False, teacher_image_key=teacher_image_key)


def test_distillation_config_rejects_orphan_student_image_fallback():
    with pytest.raises(ValueError, match="requires teacher_image_key"):
        DistillationConfig(enabled=False, fallback_to_student_images=True)
