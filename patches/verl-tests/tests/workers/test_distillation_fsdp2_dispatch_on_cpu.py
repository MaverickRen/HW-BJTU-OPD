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
"""Regression guard for the FSDP2 top-k distillation backend alias."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from verl.trainer.distillation.losses import compute_topk_loss


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "veomni"])
def test_fsdp_family_uses_fsdp_topk_backend(strategy):
    student_logits = torch.randn(2, 3, 5)
    teacher_logprobs = object()
    teacher_ids = object()
    expected = {
        "distillation_losses": torch.zeros(2, 3),
        "student_mass": torch.ones(2, 3),
        "teacher_mass": torch.ones(2, 3),
    }

    with patch(
        "verl.trainer.distillation.fsdp.losses.compute_forward_kl_topk",
        return_value=expected,
    ) as backend:
        output = compute_topk_loss(
            config=SimpleNamespace(strategy=strategy),
            distillation_config=SimpleNamespace(),
            data={"teacher_logprobs": teacher_logprobs, "teacher_ids": teacher_ids},
            student_logits=student_logits,
            data_format="thd",
        )

    assert output is expected
    backend.assert_called_once()
    assert backend.call_args.kwargs["teacher_topk_log_probs"] is teacher_logprobs
    assert backend.call_args.kwargs["teacher_topk_ids"] is teacher_ids
