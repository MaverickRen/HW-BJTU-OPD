# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""CPU-only tests for visual processor metadata checkpoint preservation."""

import contextlib
import json
from pathlib import Path

import pytest

from verl.utils.checkpoint import fsdp_checkpoint_manager as fsdp_checkpoint_module
from verl.utils.checkpoint.fsdp_checkpoint_manager import (
    FSDPCheckpointManager,
    _copy_optional_visual_processor_metadata,
)


_METADATA = {
    "preprocessor_config.json": b'{"image": "source"}\n',
    "video_preprocessor_config.json": b'{"video": "source"}\n',
}


def test_copy_visual_processor_metadata_copies_both_and_safely_replaces_regular_file(tmp_path):
    source = tmp_path / "model"
    target = tmp_path / "checkpoint" / "huggingface"
    source.mkdir()
    target.mkdir(parents=True)
    for filename, payload in _METADATA.items():
        (source / filename).write_bytes(payload)
    (target / "preprocessor_config.json").write_bytes(b"stale processor output\n")

    copied = _copy_optional_visual_processor_metadata(source, target)

    assert copied == tuple(_METADATA)
    for filename, payload in _METADATA.items():
        assert (target / filename).read_bytes() == payload
        assert not (target / filename).is_symlink()
    assert not list(target.glob(".*.tmp"))


def test_copy_visual_processor_metadata_rejects_source_symlink(tmp_path):
    source = tmp_path / "model"
    target = tmp_path / "huggingface"
    source.mkdir()
    target.mkdir()
    payload = tmp_path / "outside-source.json"
    payload.write_bytes(b'{"outside": true}\n')
    (source / "preprocessor_config.json").symlink_to(payload)

    with pytest.raises(ValueError, match="source must be a regular non-symlink file"):
        _copy_optional_visual_processor_metadata(source, target)

    assert not (target / "preprocessor_config.json").exists()


def test_copy_visual_processor_metadata_rejects_existing_target_symlink(tmp_path):
    source = tmp_path / "model"
    target = tmp_path / "huggingface"
    source.mkdir()
    target.mkdir()
    (source / "preprocessor_config.json").write_bytes(_METADATA["preprocessor_config.json"])
    victim = tmp_path / "victim.json"
    victim.write_bytes(b"must not change\n")
    target_entry = target / "preprocessor_config.json"
    target_entry.symlink_to(victim)

    with pytest.raises(ValueError, match="target must be absent or a regular non-symlink file"):
        _copy_optional_visual_processor_metadata(source, target)

    assert target_entry.is_symlink()
    assert victim.read_bytes() == b"must not change\n"
    assert not list(target.glob(".*.tmp"))


def test_copy_visual_processor_metadata_missing_sources_create_nothing(tmp_path):
    source = tmp_path / "text-only-model"
    target = tmp_path / "huggingface"
    source.mkdir()
    target.mkdir()

    copied = _copy_optional_visual_processor_metadata(source, target)

    assert copied == ()
    assert list(target.iterdir()) == []


class _FakeConfig:
    def __init__(self, source_model_path: Path):
        self.name_or_path = str(source_model_path)

    def save_pretrained(self, target_path: str):
        Path(target_path, "config.json").write_text(json.dumps({"model_type": "test"}), encoding="utf-8")


class _FakeModel:
    def __init__(self, source_model_path: Path):
        self.config = _FakeConfig(source_model_path)

    @staticmethod
    def can_generate():
        return False


class _FakeProcessor:
    @staticmethod
    def save_pretrained(target_path: str):
        # Reproduce the incomplete checkpoint that prompted this fix: the
        # processor writes other assets but not the two visual metadata files.
        Path(target_path, "processor_config.json").write_text("{}\n", encoding="utf-8")


def test_checkpoint_installs_metadata_before_registering_completion(tmp_path, monkeypatch):
    source = tmp_path / "model"
    source.mkdir()
    for filename, payload in _METADATA.items():
        (source / filename).write_bytes(payload)

    monkeypatch.setattr(fsdp_checkpoint_module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(fsdp_checkpoint_module.torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(fsdp_checkpoint_module.torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(fsdp_checkpoint_module, "fsdp_version", lambda model: 2)
    monkeypatch.setattr(
        fsdp_checkpoint_module,
        "get_fsdp_state_ctx",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    manager = FSDPCheckpointManager(
        model=_FakeModel(source),
        optimizer=None,
        processing_class=_FakeProcessor(),
        checkpoint_config={"save_contents": []},
    )
    checkpoint = tmp_path / "checkpoint"
    completion_checks = []

    def assert_metadata_then_register(path, max_ckpt_to_keep):
        assert path == str(checkpoint)
        assert max_ckpt_to_keep is None
        for filename, payload in _METADATA.items():
            assert Path(path, "huggingface", filename).read_bytes() == payload
        completion_checks.append(True)

    monkeypatch.setattr(manager, "register_checkpoint", assert_metadata_then_register)

    manager.save_checkpoint(str(checkpoint), global_step=1)

    assert completion_checks == [True]
