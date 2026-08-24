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
"""CPU regressions for distillation loss normalization across micro-batches."""

from types import SimpleNamespace

import pytest
import torch

from verl.trainer.distillation import losses as distillation_losses
from verl.utils.metric import AggregationType, reduce_metrics
from verl.utils.py_functional import append_to_dict

_LOSS_AGG_MODES = [
    "token-mean",
    "seq-mean-token-sum",
    "seq-mean-token-sum-norm",
    "seq-mean-token-mean",
]


def _synthetic_distillation_loss(config, distillation_config, model_output, data):
    """Return differentiable token losses while checking normalization is set first."""
    assert config.global_batch_info == {
        "dp_size": data["dp_size"],
        "batch_num_tokens": data["batch_num_tokens"],
        "global_batch_size": data["global_batch_size"],
        "loss_scale_factor": config.loss_scale_factor,
    }
    return model_output["synthetic_losses"], {}


@pytest.fixture(autouse=True)
def _register_synthetic_loss(monkeypatch):
    monkeypatch.setattr(
        distillation_losses,
        "get_distillation_loss_fn",
        lambda _loss_mode: _synthetic_distillation_loss,
    )


def _make_mask() -> torch.Tensor:
    lengths = [6, 2, 5, 1, 4, 3, 6, 2]
    mask = torch.zeros(len(lengths), max(lengths), dtype=torch.bool)
    for row, length in enumerate(lengths):
        mask[row, :length] = True
    return mask


def _make_token_losses() -> torch.Tensor:
    values = torch.arange(48, dtype=torch.float64).reshape(8, 6)
    return values.remainder(11).add(1).div(10)


def _call_distillation_loss(
    token_losses: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    loss_agg_mode: str,
    dp_size: int,
    batch_num_tokens: int,
    global_batch_size: int,
):
    loss_scale_factor = response_mask.shape[-1] if loss_agg_mode == "seq-mean-token-sum-norm" else None
    actor_config = SimpleNamespace(
        loss_agg_mode=loss_agg_mode,
        loss_scale_factor=loss_scale_factor,
        global_batch_info={},
    )
    distillation_loss_config = SimpleNamespace(
        loss_mode="synthetic",
        use_task_rewards=False,
        use_policy_gradient=False,
        distillation_loss_coef=1.0,
        loss_max_clamp=None,
    )
    distillation_config = SimpleNamespace(distillation_loss=distillation_loss_config)
    data = {
        "response_mask": response_mask,
        "dp_size": dp_size,
        "batch_num_tokens": batch_num_tokens,
        "global_batch_size": global_batch_size,
    }
    return distillation_losses.distillation_ppo_loss(
        config=actor_config,
        distillation_config=distillation_config,
        model_output={"synthetic_losses": token_losses},
        data=data,
    )


@pytest.mark.parametrize("loss_agg_mode", _LOSS_AGG_MODES)
@pytest.mark.parametrize("num_micro_batches", [2, 4, 8])
def test_distillation_loss_is_invariant_to_micro_batch_split(loss_agg_mode, num_micro_batches):
    """Summed micro-batch losses must equal the whole global mini-batch loss."""
    token_losses = _make_token_losses()
    response_mask = _make_mask()
    batch_size = response_mask.shape[0]
    global_tokens = int(response_mask.sum())

    whole_loss, _ = _call_distillation_loss(
        token_losses,
        response_mask,
        loss_agg_mode=loss_agg_mode,
        dp_size=1,
        batch_num_tokens=global_tokens,
        global_batch_size=batch_size,
    )

    micro_batch_size = batch_size // num_micro_batches
    accumulated_loss = sum(
        _call_distillation_loss(
            token_losses[start : start + micro_batch_size],
            response_mask[start : start + micro_batch_size],
            loss_agg_mode=loss_agg_mode,
            dp_size=1,
            batch_num_tokens=global_tokens,
            global_batch_size=batch_size,
        )[0]
        for start in range(0, batch_size, micro_batch_size)
    )

    torch.testing.assert_close(accumulated_loss, whole_loss)


def test_distillation_gradient_is_invariant_to_micro_batch_split():
    """Repeated backward calls must produce the same gradient as one whole-batch loss."""
    response_mask = _make_mask()
    batch_size = response_mask.shape[0]
    global_tokens = int(response_mask.sum())

    whole_token_losses = _make_token_losses().requires_grad_(True)
    whole_loss, _ = _call_distillation_loss(
        whole_token_losses,
        response_mask,
        loss_agg_mode="token-mean",
        dp_size=1,
        batch_num_tokens=global_tokens,
        global_batch_size=batch_size,
    )
    whole_loss.backward()

    accumulated_token_losses = _make_token_losses().requires_grad_(True)
    micro_batch_size = 2
    for start in range(0, batch_size, micro_batch_size):
        micro_loss, _ = _call_distillation_loss(
            accumulated_token_losses[start : start + micro_batch_size],
            response_mask[start : start + micro_batch_size],
            loss_agg_mode="token-mean",
            dp_size=1,
            batch_num_tokens=global_tokens,
            global_batch_size=batch_size,
        )
        micro_loss.backward()

    torch.testing.assert_close(accumulated_token_losses.grad, whole_token_losses.grad)


@pytest.mark.parametrize("dp_size", [2, 4])
def test_distillation_dp_multiplier_matches_fsdp_mean_reduction(dp_size):
    """The dp multiplier must cancel FSDP's gradient mean across data-parallel ranks."""
    token_losses = _make_token_losses()
    response_mask = _make_mask()
    batch_size = response_mask.shape[0]
    global_tokens = int(response_mask.sum())

    whole_loss, _ = _call_distillation_loss(
        token_losses,
        response_mask,
        loss_agg_mode="token-mean",
        dp_size=1,
        batch_num_tokens=global_tokens,
        global_batch_size=batch_size,
    )

    local_batch_size = batch_size // dp_size
    rank_losses = []
    for rank_start in range(0, batch_size, local_batch_size):
        rank_loss = sum(
            _call_distillation_loss(
                token_losses[micro_start : micro_start + 1],
                response_mask[micro_start : micro_start + 1],
                loss_agg_mode="token-mean",
                dp_size=dp_size,
                batch_num_tokens=global_tokens,
                global_batch_size=batch_size,
            )[0]
            for micro_start in range(rank_start, rank_start + local_batch_size)
        )
        rank_losses.append(rank_loss)

    fsdp_reduced_loss = torch.stack(rank_losses).mean()
    torch.testing.assert_close(fsdp_reduced_loss, whole_loss)


def test_distillation_loss_metric_uses_sum_for_global_mean():
    """Globally normalized micro-loss metrics are partial sums, not independent means."""
    token_losses = _make_token_losses()
    response_mask = _make_mask()
    batch_size = response_mask.shape[0]
    global_tokens = int(response_mask.sum())

    whole_loss, _ = _call_distillation_loss(
        token_losses,
        response_mask,
        loss_agg_mode="token-mean",
        dp_size=1,
        batch_num_tokens=global_tokens,
        global_batch_size=batch_size,
    )

    aggregated_metrics = {}
    micro_batch_size = 2
    for start in range(0, batch_size, micro_batch_size):
        _, metrics = _call_distillation_loss(
            token_losses[start : start + micro_batch_size],
            response_mask[start : start + micro_batch_size],
            loss_agg_mode="token-mean",
            dp_size=1,
            batch_num_tokens=global_tokens,
            global_batch_size=batch_size,
        )
        metric = metrics["distillation/loss"]
        assert metric.aggregation is AggregationType.SUM
        append_to_dict(aggregated_metrics, {"distillation/loss": metric})

    reduced_metrics = reduce_metrics(aggregated_metrics)
    assert reduced_metrics["distillation/loss"] == pytest.approx(whole_loss.item())
