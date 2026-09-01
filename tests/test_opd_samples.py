from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from hw_bjtu_opd.data.opd_samples import (
    OPDDataError,
    OPDSample,
    load_opd_samples,
    load_rgb_image,
    overlay_bbox,
)

HINT = "Only focus on the objects inside the red bounding box in the image to answer this question."


def _write_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (12, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _raw_row() -> dict[str, object]:
    problem = f"<image>\nWhat color is the sign?\n\n{HINT}\n\nA. red\nB. blue\nC. green\nD. white"
    return {
        "images": ["images/student.png"],
        "teacher_images": ["teacher_images/teacher.png"],
        "original_images": ["original_images/original.png"],
        "bbox": [2, 1, 8, 7],
        "problem": problem,
        "answer": "B",
        "extra_info": {"answer": "B", "question": "What color is the sign?"},
    }


def test_load_jsonl_resolves_all_views_and_cleans_exact_hint(tmp_path: Path) -> None:
    media = tmp_path / "media"
    _write_image(media / "images/student.png", (1, 2, 3))
    _write_image(media / "teacher_images/teacher.png", (4, 5, 6), (6, 6))
    _write_image(media / "original_images/original.png", (7, 8, 9))
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(_raw_row()) + "\n", encoding="utf-8")

    samples = load_opd_samples(source, media)

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, OPDSample)
    assert sample.index == 0
    assert sample.source_id.startswith("vision_opd_6k_0000_")
    assert sample.answer == "B"
    assert sample.bbox == (2, 1, 8, 7)
    assert sample.question == "What color is the sign?\n\nA. red\nB. blue\nC. green\nD. white"
    assert sample.prompt.startswith("<image>")
    assert sample.student_images == (media / "images/student.png",)
    assert sample.teacher_images == (media / "teacher_images/teacher.png",)
    assert sample.original_images == (media / "original_images/original.png",)
    with pytest.raises((AttributeError, TypeError)):
        sample.answer = "A"  # type: ignore[misc]


def test_load_parquet_prefers_relative_media_fields(tmp_path: Path) -> None:
    media = tmp_path / "new-media"
    _write_image(media / "images/student.png", (1, 2, 3))
    _write_image(media / "teacher_images/teacher.png", (4, 5, 6), (6, 6))
    _write_image(media / "original_images/original.png", (7, 8, 9))
    old_root = tmp_path / "old-machine"
    problem = f"<image>\nRead the sign.\n\n{HINT}\n\nA. red\nB. blue\nC. green\nD. white"
    row = {
        "source_id": "vision_opd_6k_0007_test",
        "prompt": [{"role": "user", "content": problem}],
        "images": [{"path": str(old_root / "student.png"), "image": str(old_root / "student.png")}],
        "bbox_images": [{"path": str(old_root / "teacher.png"), "image": str(old_root / "teacher.png")}],
        "reward_model": {"style": "none", "ground_truth": "B"},
        "extra_info": {
            "answer": "B",
            "question": "Read the sign.",
            "row_index": 7,
            "bbox": [1, 2, 7, 8],
            "original_images": [{"path": str(old_root / "original.png")}],
            "relative_images": ["images/student.png"],
            "relative_teacher_images": ["teacher_images/teacher.png"],
            "relative_original_images": ["original_images/original.png"],
        },
    }
    source = tmp_path / "train.parquet"
    pd.DataFrame([row]).to_parquet(source, index=False)

    sample = load_opd_samples(source, media)[0]

    assert sample.index == 7
    assert sample.student_images == (media / "images/student.png",)
    assert sample.teacher_images == (media / "teacher_images/teacher.png",)
    assert sample.original_images == (media / "original_images/original.png",)
    assert sample.question == "Read the sign.\n\nA. red\nB. blue\nC. green\nD. white"


def test_parquet_uses_existing_absolute_refs_when_relative_fields_absent(tmp_path: Path) -> None:
    image = tmp_path / "absolute.png"
    _write_image(image, (20, 30, 40))
    row = {
        "source_id": "absolute-fallback",
        "prompt": [{"role": "user", "content": "<image>\nWhat is shown?"}],
        "images": [{"path": str(image)}],
        "bbox_images": [{"path": str(image)}],
        "reward_model": {"ground_truth": "A"},
        "extra_info": {"answer": "A", "bbox": [0, 0, 5, 5], "original_images": [{"path": str(image)}]},
    }
    source = tmp_path / "fallback.parquet"
    pd.DataFrame([row]).to_parquet(source, index=False)

    sample = load_opd_samples(source)[0]

    assert sample.student_images == (image,)
    assert sample.teacher_images == (image,)
    assert sample.original_images == (image,)


def test_relative_traversal_is_rejected(tmp_path: Path) -> None:
    row = _raw_row()
    row["images"] = ["images/../../outside.png"]
    source = tmp_path / "unsafe.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(OPDDataError, match="unsafe relative image path"):
        load_opd_samples(source, tmp_path / "media")


def test_overlay_bbox_returns_rgb_copy_without_mutating_input() -> None:
    image = Image.new("L", (12, 10), 80)

    output = overlay_bbox(image, (2, 1, 10, 9))

    assert output.mode == "RGB"
    assert image.mode == "L"
    assert image.getpixel((2, 1)) == 80
    assert output.getpixel((2, 1)) == (255, 0, 0)
    assert output.getpixel((9, 8)) == (255, 0, 0)
    assert output.getpixel((6, 5)) == (80, 80, 80)


def test_load_rgb_image_detaches_file_and_normalizes_mode(tmp_path: Path) -> None:
    source = tmp_path / "gray.png"
    Image.new("L", (3, 2), 123).save(source)

    image = load_rgb_image(source)

    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (123, 123, 123)
