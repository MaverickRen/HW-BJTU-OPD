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
"""CPU control-flow regressions for pure top-k distillation (Recovery 9)."""

import ast
import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def _distillation_config(*, use_task_rewards=False, use_policy_gradient=False):
    return SimpleNamespace(
        enabled=True,
        teacher_image_key=None,
        distillation_loss=SimpleNamespace(
            loss_mode="forward_kl_topk",
            loss_settings=SimpleNamespace(use_topk=True),
            use_task_rewards=use_task_rewards,
            use_policy_gradient=use_policy_gradient,
        ),
    )


class _Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _require_runtime_dependencies(*names):
    for name in names:
        try:
            importlib.import_module(name)
        except (ImportError, OSError) as exc:
            pytest.skip(f"runtime dependency {name!r} unavailable: {exc}")


def test_recovery9_config_is_distillation_only():
    _require_runtime_dependencies("torch")
    from verl.trainer.distillation import is_distillation_only

    assert is_distillation_only(_distillation_config())
    assert is_distillation_only(
        {
            "enabled": True,
            "distillation_loss": {
                "loss_mode": "forward_kl_topk",
                "use_task_rewards": False,
                "use_policy_gradient": False,
            },
        }
    )
    assert not is_distillation_only(_distillation_config(use_task_rewards=True))
    assert not is_distillation_only(_distillation_config(use_policy_gradient=True))


def test_agent_score_stage_is_not_entered_for_pure_distillation():
    """The no-reward path must not call a rule reward worker (the R9 failure)."""
    _require_runtime_dependencies("torch", "ray")
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    class FailingRewardWorker:
        def __getattr__(self, name):
            raise AssertionError(f"pure distillation called reward worker: {name}")

    worker = AgentLoopWorker.__new__(AgentLoopWorker)
    worker.distillation_config = _distillation_config()
    worker.reward_loop_worker_handles = [FailingRewardWorker()]
    output = SimpleNamespace(reward_score=None, extra_fields={})

    asyncio.run(worker._compute_score([output], kwargs={}))
    assert output.reward_score is None


def test_reward_manager_does_not_initialize_any_reward_component():
    """Pure mode must not reserve RM resources or create any Ray actors."""
    _require_runtime_dependencies("torch", "ray")
    reward_loop = importlib.import_module("verl.experimental.reward_loop.reward_loop")

    reward_model = SimpleNamespace(enable=True, enable_resource_pool=False)
    config = _Config(
        distillation=_distillation_config(),
        reward=SimpleNamespace(reward_model=reward_model, num_workers=4),
    )

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("pure distillation initialized a reward component")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(reward_loop, "RewardModelManager", _unexpected)
    monkeypatch.setattr(reward_loop, "resolve_reward_manager_cls", _unexpected)
    monkeypatch.setattr(reward_loop.ray, "remote", _unexpected)
    monkeypatch.setattr(reward_loop.ray, "nodes", _unexpected)
    try:
        manager = reward_loop.RewardLoopManager(config)
    finally:
        monkeypatch.undo()

    assert manager.reward_model_manager is None
    assert manager.reward_loop_workers == []
    assert manager.reward_loop_worker_handles is None


def test_tq_postprocess_skips_score_but_runs_teacher():
    """A real TQ postprocess invocation must reach teacher inference."""
    _require_runtime_dependencies("torch", "ray")
    import torch

    from verl.trainer.ppo.v1 import agent_loop_tq

    remote_class = agent_loop_tq.AgentLoopWorkerTQ
    metadata = getattr(remote_class, "__ray_metadata__", None)
    worker_class = getattr(metadata, "modified_class", None) or getattr(remote_class, "_modified_class", None)
    if worker_class is None:
        pytest.skip("Ray version does not expose the modified actor class")

    worker = worker_class.__new__(worker_class)
    worker.distillation_only = True
    worker.distillation_enabled = True
    worker.distillation_config = _distillation_config()
    calls = []

    async def _score(*_args, **_kwargs):
        calls.append("score")
        raise AssertionError("pure TQ postprocess called reward score")

    async def _teacher(output, **_kwargs):
        calls.append("teacher")
        output.extra_fields["teacher_ids"] = torch.zeros((3, 2), dtype=torch.long)
        output.extra_fields["teacher_logprobs"] = torch.zeros((3, 2))

    worker._compute_score = _score
    worker._compute_teacher_logprobs = _teacher
    worker._compute_multi_modal_inputs = lambda _output, _input_ids: {}
    worker._compute_position_ids = lambda input_ids, attention_mask, multi_modal_inputs: input_ids

    output = SimpleNamespace(
        prompt_ids=[1, 2],
        response_ids=[3],
        response_mask=[1],
        reward_score=None,
        extra_fields={},
        multi_modal_data=None,
        mm_processor_kwargs=None,
        as_dict=lambda: {
            "prompts": torch.tensor([1, 2]),
            "responses": torch.tensor([3]),
            "response_mask": torch.tensor([1]),
            "extra_fields": output.extra_fields,
        },
    )

    async def _put(**_kwargs):
        calls.append("put")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(agent_loop_tq.tq, "async_kv_batch_put", _put)
    monkeypatch.setattr(agent_loop_tq, "list_of_dict_to_tensordict", lambda fields: fields)
    try:
        asyncio.run(
            worker._agent_loop_postprocess(
                output,
                validate=False,
                uid="sample",
                session_id=0,
                global_steps=0,
            )
        )
    finally:
        monkeypatch.undo()
    assert calls == ["teacher", "put"]


def test_v1_step_only_balances_and_updates_actor_in_pure_mode():
    """Driver-side step behavior: no reward/PPO-only method may be touched."""
    _require_runtime_dependencies("torch", "ray")
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    class Batch:
        def __init__(self):
            self.extra_info = {}

        def __len__(self):
            return 1

    batch = Batch()
    config = _Config(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(temperature=0.7, n=1),
            actor=SimpleNamespace(ppo_mini_batch_size=1),
        ),
        trainer=SimpleNamespace(critic_warmup=0),
    )
    class ConcretePPOTrainer(PPOTrainer):
        def on_step_end(self):
            return None

        def on_sample_end(self):
            return None

    trainer = ConcretePPOTrainer.__new__(ConcretePPOTrainer)
    trainer.config = config
    trainer.distillation_only = True
    trainer.use_critic = False
    trainer.use_reference_policy = False
    trainer.global_steps = 0
    trainer.replay_buffer = SimpleNamespace(sample=lambda **_kwargs: (batch, {}))
    trainer.reward_loop_manager = SimpleNamespace(
        reward_loop_worker_handles=property(lambda _self: (_ for _ in ()).throw(AssertionError("RM touched")))
    )
    calls = []
    trainer.on_sample_begin = lambda: None
    trainer.on_sample_end = lambda: None
    trainer._balance_batch = lambda value, metrics: calls.append("balance") or value
    trainer._update_actor = lambda value, metrics: calls.append("actor") or value
    for method in ("_compute_reward_colocate", "_compute_old_log_prob", "_compute_ref_log_prob", "_compute_values", "_compute_advantage", "_update_critic"):
        setattr(trainer, method, lambda *_args, _method=method, **_kwargs: (_ for _ in ()).throw(AssertionError(_method)))

    trainer._step_once({}, {"step": 0.0}, 1)
    assert calls == ["balance", "actor"]


def test_v1_step_source_skips_ppo_only_stages():
    """Guard the exact V1 stage contract without initializing Ray or a model."""
    source = Path("verl/trainer/ppo/v1/trainer_base.py").read_text()
    assert "if not self.distillation_only and self.reward_loop_manager.reward_loop_worker_handles is None" in source
    assert "self.use_critic = False if self.distillation_only else need_critic(self.config)" in source
    assert "self.use_reference_policy = False if self.distillation_only else need_reference_policy(self.config)" in source
    assert "if not self.distillation_only:\n            with marked_timer(\"old_log_prob\"" in source
    assert "if self.use_reference_policy and not self.distillation_only" in source
    assert "if self.use_critic and not self.distillation_only" in source
    assert "if not self.distillation_only:\n            with marked_timer(\"adv\"" in source
    assert "if self.distillation_only or self.config.trainer.critic_warmup <= self.global_steps" in source
    assert 'raise RuntimeError("Pure distillation received no completed rollout outputs")' in source

    tq_source = Path("verl/trainer/ppo/v1/agent_loop_tq.py").read_text()
    assert "if not distillation_only:\n            await self._compute_score" in tq_source
    assert "await self._compute_teacher_logprobs" in tq_source
    agent_source = Path("verl/experimental/agent_loop/agent_loop.py").read_text()
    assert "if not self.distillation_only:\n            await self._compute_score([output], kwargs=kwargs)" in agent_source

    reward_source = Path("verl/experimental/reward_loop/reward_loop.py").read_text()
    assert "if self.distillation_only:\n            return" in reward_source
    assert "self.reward_loop_workers_class = None if self.distillation_only else ray.remote(RewardLoopWorker)" in reward_source
    assert "self.reward_manager_cls = None if self.distillation_only else resolve_reward_manager_cls(config)" in reward_source
    assert "if self.distillation_only:\n            raise RuntimeError(\"Pure distillation does not define task rewards\")" in reward_source


def test_legacy_prepare_stage_is_a_noop_without_reward_or_ppo_fields():
    """Legacy RayPPOTrainer must preserve teacher fields and skip reward/PPO stages."""
    _require_runtime_dependencies("torch", "ray")
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    class FailingTrainer:
        distillation_only = True

        def _compute_reward_colocate(self, *_args, **_kwargs):
            raise AssertionError("pure legacy path called reward")

        def _compute_old_log_prob(self, *_args, **_kwargs):
            raise AssertionError("pure legacy path called old log prob")

    batch = SimpleNamespace(batch={"teacher_topk_ids": object()}, non_tensor_batch={})
    prepared, reward_info = RayPPOTrainer._prepare_batch_for_actor(FailingTrainer(), batch, {}, {})
    assert prepared is batch
    assert reward_info == {}
    assert "teacher_topk_ids" in prepared.batch


def test_legacy_pure_mode_removes_prebuilt_reward_pool():
    """A legacy launcher may build the manager first; pure mode must remove RM demand."""
    _require_runtime_dependencies("torch", "ray")
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role

    manager = SimpleNamespace(
        resource_pool_spec={"global_pool": [8], "reward_pool": [2]},
        mapping={Role.RewardModel: "reward_pool"},
        resource_pool_dict={},
    )
    trainer = SimpleNamespace(resource_pool_manager=manager)
    RayPPOTrainer._disable_pure_distillation_reward_resources(trainer)
    assert manager.resource_pool_spec == {"global_pool": [8]}
    assert Role.RewardModel not in manager.mapping


def test_legacy_source_marks_pure_reward_unavailable_and_skips_remax_baseline():
    """Static regression guards the legacy KeyError/reward-pool failure surfaces."""
    source = Path("verl/trainer/ppo/ray_trainer.py").read_text()
    assert "self.use_rm = need_reward_model(self.config) and not self.distillation_only" in source
    assert "self._disable_pure_distillation_reward_resources()" in source
    assert "if not self.distillation_only and self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX" in source
    assert '"training/reward_available": False' in source
    assert '"val/reward/status": "unavailable"' in source
    assert "batch, reward_extra_infos_dict = self._prepare_batch_for_actor" in source


def _method_node(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"method {name!r} not found")


def test_v1_pure_metrics_are_unavailable_without_ppo_metric_fabrication():
    """Pure metrics must not require or synthesize reward/advantage tensors."""
    source = Path("verl/trainer/ppo/v1/trainer_base.py").read_text()
    method = _method_node(source, "_compute_metrics")
    pure_branch = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "self"
        and node.test.attr == "distillation_only"
        and any(
            isinstance(child, ast.Constant) and child.value == "training/reward_available"
            for child in ast.walk(node)
        )
    )
    pure_source = ast.unparse(ast.Module(body=pure_branch.body, type_ignores=[]))
    assert "compute_data_metrics(" not in pure_source
    method_source = ast.unparse(method)
    assert "torch.zeros(" not in method_source
    for forbidden in ("rm_scores", "token_level_rewards", "advantages", "returns", "critic/"):
        assert forbidden not in pure_source

    nonpure_calls = [
        node
        for node in ast.walk(ast.Module(body=pure_branch.orelse, type_ignores=[]))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compute_data_metrics"
    ]
    assert len(nonpure_calls) == 1
    pure_constants = {node.value for node in ast.walk(pure_branch) if isinstance(node, ast.Constant)}
    assert "training/reward_available" in pure_constants
    assert "training/reward_status" in pure_constants


def test_legacy_pure_kl_does_not_select_ref_role_but_nonpure_does():
    """The legacy launcher must apply pure-mode role selection before Ray wiring."""
    source = Path("verl/trainer/main_ppo_v0.py").read_text()
    assert "from verl.trainer.distillation import is_distillation_enabled, is_distillation_only" in source
    assert "distillation_only = is_distillation_only(config.get(\"distillation\"))" in source
    assert "if need_reference_policy(config) and not ref_in_actor and not distillation_only:" in source

    def select_role(*, need_ref: bool, ref_in_actor: bool, distillation_only: bool) -> str:
        return "ref" if need_ref and not ref_in_actor and not distillation_only else "actor"

    assert select_role(need_ref=True, ref_in_actor=False, distillation_only=True) == "actor"
    assert select_role(need_ref=True, ref_in_actor=False, distillation_only=False) == "ref"
