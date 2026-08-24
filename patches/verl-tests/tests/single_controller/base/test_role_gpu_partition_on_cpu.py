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
"""CPU-only tests for the optional pre-model role GPU partition gate."""

import pytest

from verl.single_controller.base.worker import validate_role_gpu_partition


def _observations(*, student_ids=(0, 1, 2, 3), teacher_ids=(4, 5, 6, 7), node_ids=None, visible_ids=None, uuids=None):
    node_ids = node_ids or ["node-0"] * 8
    visible_ids = visible_ids or list(range(8))
    uuids = uuids or [f"GPU-{index}" for index in range(8)]
    records = []
    for role, ids in (("student", student_ids), ("teacher", teacher_ids)):
        for rank, physical in enumerate(ids):
            records.append(
                {
                    "role": role,
                    "rank": rank,
                    "pid": 1000 + physical,
                    "node_id": node_ids[physical],
                    "ray_accelerator_ids": [str(physical)],
                    "cuda_visible_devices": str(visible_ids[physical]),
                    "gpu_uuid": uuids[physical],
                }
            )
    return records


def test_role_gpu_partition_accepts_exact_disjoint_mapping():
    result = validate_role_gpu_partition(
        _observations(),
        student_physical_gpus=[0, 1, 2, 3],
        teacher_physical_gpus=[4, 5, 6, 7],
        require_gpu_uuid=True,
    )
    assert result["node_id"] == "node-0"
    assert [record["role"] for record in result["observations"]] == ["student"] * 4 + ["teacher"] * 4
    assert {record["physical_gpu"] for record in result["observations"]} == set(range(8))
    assert set(result["observations"][0]) == {
        "role",
        "rank",
        "pid",
        "node_id",
        "ray_accelerator_ids",
        "cuda_visible_devices",
        "physical_gpu",
        "gpu_uuid",
    }


def test_role_gpu_partition_rejects_swapped_roles():
    observations = _observations(student_ids=(4, 5, 6, 7), teacher_ids=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="student physical GPU partition"):
        validate_role_gpu_partition(
            observations,
            student_physical_gpus=[0, 1, 2, 3],
            teacher_physical_gpus=[4, 5, 6, 7],
            require_gpu_uuid=True,
        )


def test_role_gpu_partition_rejects_swapped_rank_mapping_within_role():
    observations = _observations(student_ids=(1, 0, 2, 3))
    with pytest.raises(ValueError, match="rank-to-physical-GPU mapping"):
        validate_role_gpu_partition(
            observations,
            student_physical_gpus=[0, 1, 2, 3],
            teacher_physical_gpus=[4, 5, 6, 7],
            require_gpu_uuid=True,
        )


def test_role_gpu_partition_rejects_overlapping_expected_config():
    with pytest.raises(ValueError, match="overlap"):
        validate_role_gpu_partition(
            _observations(),
            student_physical_gpus=[0, 1, 2, 3],
            teacher_physical_gpus=[3, 4, 5, 6],
        )


def test_role_gpu_partition_rejects_missing_worker():
    with pytest.raises(ValueError, match="expected 8"):
        validate_role_gpu_partition(
            _observations()[:-1],
            student_physical_gpus=[0, 1, 2, 3],
            teacher_physical_gpus=[4, 5, 6, 7],
            require_gpu_uuid=True,
        )


def test_role_gpu_partition_rejects_multiple_nodes():
    node_ids = ["node-0"] * 7 + ["node-1"]
    with pytest.raises(ValueError, match="one node"):
        validate_role_gpu_partition(
            _observations(node_ids=node_ids),
            student_physical_gpus=[0, 1, 2, 3],
            teacher_physical_gpus=[4, 5, 6, 7],
            require_gpu_uuid=True,
        )


def test_role_gpu_partition_rejects_inconsistent_visibility_ids():
    with pytest.raises(ValueError, match="Ray ID and CUDA visibility"):
        validate_role_gpu_partition(
            _observations(visible_ids=list(range(1, 9))),
            student_physical_gpus=[0, 1, 2, 3],
            teacher_physical_gpus=[4, 5, 6, 7],
            require_gpu_uuid=True,
        )


def test_role_gpu_partition_rejects_non_numeric_ray_id():
    observations = _observations()
    observations[0]["ray_accelerator_ids"] = ["GPU-0"]
    observations[0]["cuda_visible_devices"] = "GPU-0"
    with pytest.raises(ValueError, match="physical index"):
        validate_role_gpu_partition(
            observations,
            student_physical_gpus=[0, 1, 2, 3],
            teacher_physical_gpus=[4, 5, 6, 7],
            require_gpu_uuid=True,
        )


def test_role_gpu_partition_rejects_duplicate_uuid():
    uuids = ["GPU-same"] + [f"GPU-{index}" for index in range(1, 8)]
    uuids[4] = "GPU-same"
    with pytest.raises(ValueError, match="UUIDs are not unique"):
        validate_role_gpu_partition(
            _observations(uuids=uuids),
            student_physical_gpus=[0, 1, 2, 3],
            teacher_physical_gpus=[4, 5, 6, 7],
            require_gpu_uuid=True,
        )
