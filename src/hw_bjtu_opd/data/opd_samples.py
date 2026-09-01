"""Load and inspect the pinned Vision-OPD sample format.

The viewer deliberately keeps this module independent from the training
runtime.  It parses the raw ``train.jsonl`` snapshot and the processed
``train.parquet`` produced by :mod:`repro.data_tools.prepare_vision_opd_6k`,
but opens image bytes only when :func:`load_rgb_image` is called.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageDraw, ImageOps

REMOVE_HINT = "Only focus on the objects inside the red bounding box in the image to answer this question."


class OPDDataError(ValueError):
    """Raised when an OPD row is malformed or unsafe to resolve."""


@dataclass(frozen=True, slots=True)
class OPDSample:
    """One target-conditioned Vision-OPD training example.

    ``student_images`` contain the full image with a red target box;
    ``teacher_images`` contain the privileged crop; ``original_images`` are
    retained for audit/visualization.  Tuples and a frozen dataclass prevent a
    cached Streamlit sample from being mutated accidentally.
    """

    index: int
    source_id: str
    prompt: str
    question: str
    answer: str
    bbox: tuple[int, int, int, int]
    student_images: tuple[Path, ...]
    teacher_images: tuple[Path, ...]
    original_images: tuple[Path, ...]

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise OPDDataError(f"sample index must be a non-negative integer, got {self.index!r}")
        for field in ("source_id", "prompt", "question", "answer"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise OPDDataError(f"sample {field} must be non-empty text")

        object.__setattr__(self, "bbox", _parse_bbox(self.bbox))
        for field in ("student_images", "teacher_images", "original_images"):
            values = tuple(Path(value) for value in getattr(self, field))
            if not values:
                raise OPDDataError(f"sample {field} must contain at least one image")
            object.__setattr__(self, field, values)


def _plain(value: Any) -> Any:
    """Convert Arrow/numpy scalar containers into ordinary Python values."""

    if hasattr(value, "as_py"):
        return _plain(value.as_py())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if type(value).__name__ == "ndarray" and hasattr(value, "tolist"):
        return _plain(value.tolist())
    return value


def _parse_bbox(value: Any) -> tuple[int, int, int, int]:
    value = _plain(value)
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise OPDDataError(f"bbox must be four integers, got {value!r}")
    x0, y0, x1, y1 = (int(item) for item in value)
    if x0 < 0 or y0 < 0 or x0 >= x1 or y0 >= y1:
        raise OPDDataError(f"bbox must satisfy 0 <= x0 < x1 and 0 <= y0 < y1, got {value!r}")
    return x0, y0, x1, y1


def _clean_question(problem: str) -> str:
    """Apply the exact cleanup used by the pinned Vision-OPD preparation."""

    text = (problem or "").replace("<image>", "").strip()
    text = text.replace(f"\n\n{REMOVE_HINT}", "")
    text = text.replace(REMOVE_HINT, "")
    return text.strip()


def _safe_relative_path(value: Any) -> Path:
    """Return a safe local path, rejecting POSIX and Windows traversal."""

    if not isinstance(value, str) or not value:
        raise OPDDataError(f"image path must be non-empty text, got {value!r}")
    # PurePosixPath catches the format used in the HF snapshot.  Also reject
    # backslash traversal so a Windows-style value cannot bypass the check on
    # a POSIX host (and vice versa when a snapshot is inspected elsewhere).
    if "\\" in value:
        windows_parts = value.replace("\\", "/").split("/")
        if value.startswith(("\\", "/")) or ":" in windows_parts[0] or ".." in windows_parts:
            raise OPDDataError(f"unsafe relative image path: {value!r}")
        value = value.replace("\\", "/")
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts:
        raise OPDDataError(f"unsafe relative image path: {value!r}")
    parts = tuple(part for part in posix.parts if part not in ("", "."))
    if not parts:
        raise OPDDataError(f"empty relative image path: {value!r}")
    return Path(*parts)


@lru_cache(maxsize=4_096)
def _resolve_media_directory(media_root: Path, relative_parent: Path) -> Path:
    """Resolve each media directory once and reject directory-symlink escapes."""

    resolved = (media_root / relative_parent).resolve()
    try:
        resolved.relative_to(media_root)
    except ValueError as exc:
        raise OPDDataError(f"media directory escapes media_root: {relative_parent}") from exc
    return resolved


def _resolve_reference(value: Any, media_root: Path | None, *, label: str) -> Path:
    """Resolve a row reference without allowing a media root escape.

    Absolute references are retained as a compatibility fallback for old
    processed Parquets. Relative references require ``media_root`` so that a
    downloaded data file cannot unexpectedly resolve against the process cwd.
    """

    if not isinstance(value, str) or not value:
        raise OPDDataError(f"{label} image reference must be non-empty text, got {value!r}")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    if media_root is None:
        raise OPDDataError(f"{label} image path is relative but media_root was not supplied: {value!r}")
    relative = _safe_relative_path(value)
    resolved_parent = _resolve_media_directory(media_root, relative.parent)
    return resolved_parent / relative.name


def _image_values(value: Any, *, label: str) -> list[Any]:
    value = _plain(value)
    if not isinstance(value, list) or not value:
        raise OPDDataError(f"{label} must be a non-empty list")
    return value


def _resolve_image_list(
    values: Any,
    media_root: Path | None,
    *,
    label: str,
    relative_values: Any = None,
) -> tuple[Path, ...]:
    """Resolve string or ``{path,image}`` image refs.

    Processed rows carry path-independent ``relative_*`` fields.  They take
    precedence whenever present; the old absolute ``path``/``image`` refs are
    retained as a compatibility fallback for historical Parquets.
    """

    refs = _plain(relative_values) if relative_values is not None else None
    if refs is not None:
        if not isinstance(refs, list):
            raise OPDDataError(f"{label} relative references must be a list")
        if refs:
            values = refs
    values = _image_values(values, label=label)
    paths: list[Path] = []
    for position, value in enumerate(values):
        value = _plain(value)
        if isinstance(value, Mapping):
            candidate = value.get("path") or value.get("image")
            if not isinstance(candidate, str) or not candidate:
                raise OPDDataError(f"{label}[{position}] has no path/image reference")
        else:
            candidate = value
        paths.append(_resolve_reference(candidate, media_root, label=f"{label}[{position}]"))
    return tuple(paths)


def _prompt_text(value: Any) -> str:
    value = _plain(value)
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise OPDDataError(f"prompt must be text or a message list, got {type(value).__name__}")
    pieces: list[str] = []
    for position, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise OPDDataError(f"prompt message {position} is not an object")
        content = message.get("content", "")
        if isinstance(content, str):
            pieces.append(content)
            continue
        if isinstance(content, list):
            text_parts = [
                str(segment["text"])
                for segment in content
                if isinstance(segment, Mapping) and isinstance(segment.get("text"), str)
            ]
            pieces.append("".join(text_parts))
            continue
        raise OPDDataError(f"prompt message {position} content is not text")
    text = "\n".join(piece for piece in pieces if piece)
    if not text:
        raise OPDDataError("prompt contains no text")
    return text


def _answer(row: Mapping[str, Any], *, extra: Mapping[str, Any] | None = None) -> str:
    candidates: list[tuple[str, Any]] = []
    if "answer" in row:
        candidates.append(("answer", row.get("answer")))
    if extra is not None and "answer" in extra:
        candidates.append(("extra_info.answer", extra.get("answer")))
    reward = _plain(row.get("reward_model"))
    if isinstance(reward, Mapping) and "ground_truth" in reward:
        candidates.append(("reward_model.ground_truth", reward.get("ground_truth")))
    valid = [(name, value) for name, value in candidates if isinstance(value, str) and value.strip()]
    if not valid:
        raise OPDDataError("row has no non-empty answer")
    answer = valid[0][1].strip()
    if any(value.strip() != answer for _, value in valid[1:]):
        raise OPDDataError("row answer fields disagree")
    return answer


def _source_id(row: Mapping[str, Any], index: int) -> str:
    value = row.get("source_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    record_hash = row.get("source_record_sha256")
    if isinstance(record_hash, str) and record_hash:
        return f"vision_opd_6k_{index:04d}_{record_hash[:12]}"
    try:
        source_record = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise OPDDataError(f"unable to derive source_id for row {index}") from exc
    digest = hashlib.sha256(source_record.encode("utf-8")).hexdigest()
    return f"vision_opd_6k_{index:04d}_{digest[:12]}"


def _sample_from_raw(index: int, row: Mapping[str, Any], media_root: Path | None) -> OPDSample:
    required = ("images", "teacher_images", "original_images", "bbox", "problem")
    missing = [key for key in required if key not in row]
    if missing:
        raise OPDDataError(f"JSONL row {index} is missing required fields: {missing}")
    prompt = row["problem"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise OPDDataError(f"JSONL row {index} problem must be non-empty text")
    extra = _plain(row.get("extra_info"))
    if extra is not None and not isinstance(extra, Mapping):
        raise OPDDataError(f"JSONL row {index} extra_info must be an object")
    extra_mapping = extra if isinstance(extra, Mapping) else None
    return OPDSample(
        index=index,
        source_id=_source_id(row, index),
        prompt=prompt,
        question=_clean_question(prompt),
        answer=_answer(row, extra=extra_mapping),
        bbox=_parse_bbox(row["bbox"]),
        student_images=_resolve_image_list(row["images"], media_root, label=f"row {index} student_images"),
        teacher_images=_resolve_image_list(row["teacher_images"], media_root, label=f"row {index} teacher_images"),
        original_images=_resolve_image_list(row["original_images"], media_root, label=f"row {index} original_images"),
    )


def _nested_relative(row: Mapping[str, Any], extra: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    if key in extra:
        return extra[key]
    return None


def _sample_from_parquet(index: int, row: Mapping[str, Any], media_root: Path | None) -> OPDSample:
    extra_value = _plain(row.get("extra_info", {}))
    if not isinstance(extra_value, Mapping):
        raise OPDDataError(f"Parquet row {index} extra_info must be an object")
    extra = extra_value
    prompt_value = row.get("prompt")
    if prompt_value is None:
        prompt_value = extra.get("question")
    prompt = _prompt_text(prompt_value)
    original_refs = extra.get("original_images", row.get("original_images"))
    bbox_value = extra.get("bbox", row.get("bbox"))
    if bbox_value is None:
        raise OPDDataError(f"Parquet row {index} has no bbox")
    student_relative = _nested_relative(row, extra, "relative_images")
    teacher_relative = _nested_relative(row, extra, "relative_teacher_images")
    original_relative = _nested_relative(row, extra, "relative_original_images")
    return OPDSample(
        index=int(extra.get("row_index", index)),
        source_id=_source_id(row, index),
        prompt=prompt,
        question=_clean_question(prompt),
        answer=_answer(row, extra=extra),
        bbox=_parse_bbox(bbox_value),
        student_images=_resolve_image_list(
            row.get("images"), media_root, label=f"row {index} student_images", relative_values=student_relative
        ),
        teacher_images=_resolve_image_list(
            row.get("bbox_images"), media_root, label=f"row {index} teacher_images", relative_values=teacher_relative
        ),
        original_images=_resolve_image_list(
            original_refs, media_root, label=f"row {index} original_images", relative_values=original_relative
        ),
    )


def _read_jsonl(data_path: Path, media_root: Path | None) -> tuple[OPDSample, ...]:
    samples: list[OPDSample] = []
    try:
        with data_path.open("r", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    raise OPDDataError(f"blank JSONL line at row {index}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise OPDDataError(f"invalid JSON at row {index}: {exc.msg}") from exc
                if not isinstance(row, Mapping):
                    raise OPDDataError(f"JSONL row {index} must be an object")
                samples.append(_sample_from_raw(index, row, media_root))
    except OSError as exc:
        raise OPDDataError(f"unable to read JSONL {data_path}: {exc}") from exc
    if not samples:
        raise OPDDataError(f"JSONL contains no samples: {data_path}")
    return tuple(samples)


def _read_parquet(data_path: Path, media_root: Path | None) -> tuple[OPDSample, ...]:
    try:
        import pyarrow.parquet as pq

        rows = pq.read_table(data_path).to_pylist()
    except Exception as exc:  # Arrow errors vary by installed backend and input corruption.
        raise OPDDataError(f"unable to read Parquet {data_path}: {exc}") from exc
    if not rows:
        raise OPDDataError(f"Parquet contains no samples: {data_path}")
    try:
        return tuple(_sample_from_parquet(index, _plain(row), media_root) for index, row in enumerate(rows))
    except OPDDataError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise OPDDataError(f"malformed Parquet row: {exc}") from exc


def load_opd_samples(data_path: str | Path, media_root: str | Path | None = None) -> tuple[OPDSample, ...]:
    """Load raw Vision-OPD JSONL or processed Parquet metadata.

    Relative media references are resolved beneath ``media_root`` and checked
    after symlink resolution.  No image bytes are read here; this keeps the
    Streamlit UI responsive and lets it report one missing image per panel.
    """

    path = Path(data_path).expanduser()
    if not path.is_file():
        raise OPDDataError(f"dataset file does not exist: {path}")
    root = Path(media_root).expanduser().resolve() if media_root is not None else None
    if root is not None and (root.exists() and not root.is_dir()):
        raise OPDDataError(f"media_root is not a directory: {root}")
    suffix = path.suffix.casefold()
    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl(path, root)
    if suffix == ".parquet":
        return _read_parquet(path, root)
    raise OPDDataError(f"unsupported OPD data format {path.suffix!r}; expected .jsonl or .parquet")


def load_rgb_image(path: str | Path) -> Image.Image:
    """Open an image, apply EXIF orientation, and return a detached RGB image."""

    image_path = Path(path)
    if image_path.is_symlink():
        raise OPDDataError(f"refusing symlink image: {image_path}")
    try:
        with Image.open(image_path) as image:
            oriented = ImageOps.exif_transpose(image)
            oriented.load()
            return oriented.convert("RGB").copy()
    except OSError as exc:
        raise OPDDataError(f"unable to open image {image_path}: {exc}") from exc


def overlay_bbox(image: Image.Image, bbox: Sequence[int]) -> Image.Image:
    """Return an RGB copy with a red outline for a half-open bbox."""

    if not isinstance(image, Image.Image):
        raise OPDDataError(f"overlay_bbox expects a PIL image, got {type(image).__name__}")
    parsed = _parse_bbox(bbox)
    x0, y0, x1, y1 = parsed
    width, height = image.size
    if x1 > width or y1 > height:
        raise OPDDataError(f"bbox {list(parsed)} is outside image dimensions {(width, height)}")
    output = image.convert("RGB").copy()
    line_width = max(1, min(3, width, height))
    ImageDraw.Draw(output).rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 0, 0), width=line_width)
    return output


__all__ = [
    "OPDDataError",
    "OPDSample",
    "load_opd_samples",
    "load_rgb_image",
    "overlay_bbox",
]
