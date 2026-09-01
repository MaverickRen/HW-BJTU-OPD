"""Interactive viewer for real Vision-OPD JSONL or prepared Parquet samples."""

from __future__ import annotations

import argparse
import os
import secrets
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import streamlit as st

from hw_bjtu_opd.data.opd_samples import (
    OPDDataError,
    OPDSample,
    load_opd_samples,
    load_rgb_image,
    overlay_bbox,
)

DEFAULT_DATA = Path("data/vision-opd-6k/raw/train.jsonl")
DEFAULT_MEDIA_ROOT = Path("data/vision-opd-6k/media")


def parse_viewer_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments placed after Streamlit's ``--`` separator."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--media-root", type=Path)
    args, _ = parser.parse_known_args(argv)
    return args


@st.cache_data(show_spinner="Loading OPD sample metadata...")
def cached_samples(data_path: str, media_root: str | None, modified_ns: int) -> tuple[OPDSample, ...]:
    """Cache metadata only; image payloads remain lazy and are opened per row."""

    del modified_ns
    return load_opd_samples(Path(data_path), Path(media_root) if media_root else None)


def render_images(
    container: st.delta_generator.DeltaGenerator,
    *,
    title: str,
    description: str,
    paths: tuple[Path, ...],
    bbox: tuple[int, int, int, int] | None = None,
) -> None:
    with container:
        st.subheader(title)
        st.caption(description)
        for position, path in enumerate(paths):
            try:
                image = load_rgb_image(path)
                if bbox is not None:
                    image = overlay_bbox(image, bbox)
            except (OSError, OPDDataError) as exc:
                st.error(f"Unable to open image {position + 1}: {exc}")
                continue
            suffix = f" · image {position + 1}/{len(paths)}" if len(paths) > 1 else ""
            st.image(
                image,
                caption=f"{image.width}×{image.height} · {path.name}{suffix}",
                width="stretch",
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_viewer_args(argv)
    default_data = args.data or Path(os.environ.get("HW_BJTU_OPD_DATA", DEFAULT_DATA))
    default_media = args.media_root or Path(os.environ.get("HW_BJTU_OPD_MEDIA_ROOT", DEFAULT_MEDIA_ROOT))

    st.set_page_config(page_title="Vision-OPD sample viewer", page_icon="🔎", layout="wide")
    st.title("Vision-OPD training sample viewer")
    st.caption("Real student, teacher and audit views from the 6,241-row Vision-OPD training set.")

    with st.sidebar:
        st.header("Dataset")
        data_value = st.text_input("JSONL or Parquet", value=str(default_data))
        media_value = st.text_input(
            "Extracted media root",
            value=str(default_media),
            help="Required for raw JSONL. For Parquet, it remaps the portable relative_* fields.",
        )

    data_path = Path(data_value).expanduser()
    media_root = Path(media_value).expanduser() if media_value.strip() else None
    if not data_path.is_file():
        st.error(f"Dataset file does not exist: {data_path}")
        st.code(
            "python repro/data_tools/prepare_vision_opd_6k.py --download "
            "--raw-root data/vision-opd-6k/raw --media-root data/vision-opd-6k/media "
            "--output-dir data/vision-opd-6k/processed",
            language="bash",
        )
        st.stop()

    try:
        samples = cached_samples(
            str(data_path.resolve()),
            str(media_root.resolve()) if media_root else None,
            data_path.stat().st_mtime_ns,
        )
    except (OSError, OPDDataError, ValueError) as exc:
        st.error(f"Dataset validation failed: {exc}")
        st.stop()

    with st.sidebar:
        st.success(f"Loaded {len(samples):,} samples")
        answer_counts = Counter(sample.answer for sample in samples)
        st.caption(
            "Answer distribution: " + " · ".join(f"{key} {answer_counts[key]:,}" for key in sorted(answer_counts))
        )
        search = st.text_input("Search questions", placeholder="color, logo, person...").strip().casefold()

        if search:
            matching = [position for position, sample in enumerate(samples) if search in sample.question.casefold()]
            st.caption(f"{len(matching):,} matching rows")
            if not matching:
                st.warning("No matching question")
                st.stop()
            visible_matches = matching[:500]
            row_position = st.selectbox(
                "Matching sample",
                visible_matches,
                format_func=lambda position: (
                    f"#{samples[position].index} · {samples[position].question.splitlines()[0][:72]}"
                ),
            )
            if len(matching) > len(visible_matches):
                st.caption("Showing the first 500 matches; refine the search to narrow them.")
        else:
            if "opd_row_index" not in st.session_state:
                st.session_state.opd_row_index = 0
            st.session_state.opd_row_index = min(max(int(st.session_state.opd_row_index), 0), len(samples) - 1)
            if st.button("Random sample", width="stretch"):
                st.session_state.opd_row_index = secrets.randbelow(len(samples))
            row_position = st.number_input(
                "Zero-based dataset position",
                min_value=0,
                max_value=len(samples) - 1,
                step=1,
                key="opd_row_index",
            )

        overlay_original = st.checkbox("Overlay bbox on original", value=True)
        show_paths = st.checkbox("Show resolved file paths", value=False)

    sample = samples[int(row_position)]
    bbox_width = sample.bbox[2] - sample.bbox[0]
    bbox_height = sample.bbox[3] - sample.bbox[1]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Row", f"{sample.index:,} / {len(samples) - 1:,}")
    metric_columns[1].metric("Gold answer", sample.answer)
    metric_columns[2].metric("BBox", str(list(sample.bbox)))
    metric_columns[3].metric("BBox size", f"{bbox_width} × {bbox_height}")

    st.subheader("Question")
    st.text(sample.question)
    st.success(
        f"Gold answer: {sample.answer}. This label is displayed for audit only; "
        "the released OPD objective does not train against it."
    )
    st.info(
        "The student is given the red target box, while the teacher is given the crop. "
        "This supervises target-conditioned perception, not autonomous no-box localization or bbox prediction."
    )

    original_column, student_column, teacher_column = st.columns(3)
    render_images(
        original_column,
        title="Original (audit only)",
        description="Not fed to either model. The red overlay is reconstructed from bbox coordinates.",
        paths=sample.original_images,
        bbox=sample.bbox if overlay_original else None,
    )
    render_images(
        student_column,
        title="Student input",
        description="Full image with the target region already marked by a red box (`images`).",
        paths=sample.student_images,
    )
    render_images(
        teacher_column,
        title="Teacher input",
        description="Privileged target crop used only by the fixed teacher (`bbox_images`).",
        paths=sample.teacher_images,
    )

    with st.expander("Raw prompt and sample metadata"):
        st.text(sample.prompt)
        metadata: dict[str, object] = {
            "index": sample.index,
            "source_id": sample.source_id,
            "bbox": list(sample.bbox),
            "answer": sample.answer,
            "student_image_count": len(sample.student_images),
            "teacher_image_count": len(sample.teacher_images),
            "original_image_count": len(sample.original_images),
        }
        if show_paths:
            metadata["student_images"] = [str(path) for path in sample.student_images]
            metadata["teacher_images"] = [str(path) for path in sample.teacher_images]
            metadata["original_images"] = [str(path) for path in sample.original_images]
        st.json(metadata)


if __name__ == "__main__":
    main()
