from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from prepare_opd_release import ArtifactFile, ReleaseError, _readme, stage_release

ROOT = Path(__file__).resolve().parents[1]


def _source_and_specs(tmp_path: Path) -> tuple[Path, tuple[ArtifactFile, ...]]:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    specs: list[ArtifactFile] = []
    for index, name in enumerate(
        (
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    ):
        content = (f"test artifact {index}\n").encode()
        if name == "model.safetensors":
            content = b"small test tensor\n"
        (source / name).write_bytes(content)
        specs.append(
            ArtifactFile(
                name=name,
                bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    return source, tuple(specs)


def test_dry_run_validates_without_creating_output(tmp_path: Path) -> None:
    source, specs = _source_and_specs(tmp_path)
    output = tmp_path / "release"

    plan = stage_release(source, output, dry_run=True, file_specs=specs)

    assert plan["status"] == "dry_run"
    assert plan["file_count"] == 7
    assert not output.exists()


def test_stage_is_closed_and_atomic_with_hardlinks(tmp_path: Path) -> None:
    source, specs = _source_and_specs(tmp_path)
    output = tmp_path / "release"

    result = stage_release(source, output, file_specs=specs)

    assert result["status"] == "complete"
    assert {item.name for item in output.iterdir()} == {
        "README.md",
        "artifact_manifest.json",
        *(spec.name for spec in specs),
    }
    assert (source / "model.safetensors").stat().st_ino == (output / "model.safetensors").stat().st_ino
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["file_count"] == 7
    assert manifest["model"]["weight_file"] == "model.safetensors"
    public_text = (output / "README.md").read_text(encoding="utf-8") + json.dumps(manifest)
    assert "/minimax-3d-rw-backup" not in public_text
    assert "hf_" not in public_text
    assert "ghp_" not in public_text


def test_copy_mode_and_non_empty_output_guard(tmp_path: Path) -> None:
    source, specs = _source_and_specs(tmp_path)
    output = tmp_path / "release"

    stage_release(source, output, mode="copy", file_specs=specs)
    assert (source / "model.safetensors").stat().st_ino != (output / "model.safetensors").stat().st_ino
    with pytest.raises(ReleaseError, match="non-empty"):
        stage_release(source, output, file_specs=specs)


def test_empty_output_directory_can_be_replaced_atomically(tmp_path: Path) -> None:
    source, specs = _source_and_specs(tmp_path)
    output = tmp_path / "empty-release"
    output.mkdir()

    result = stage_release(source, output, file_specs=specs)

    assert result["status"] == "complete"
    assert (output / "artifact_manifest.json").is_file()


def test_source_hash_mismatch_and_symlink_are_rejected(tmp_path: Path) -> None:
    source, specs = _source_and_specs(tmp_path)
    (source / "config.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ReleaseError, match="mismatch"):
        stage_release(source, tmp_path / "bad-hash", file_specs=specs)

    source, specs = _source_and_specs(tmp_path / "symlink-case")
    target = source / "tokenizer.json"
    target.unlink()
    target.symlink_to(source / "tokenizer_config.json")
    with pytest.raises(ReleaseError, match="symlink"):
        stage_release(source, tmp_path / "bad-link", file_specs=specs)


def test_published_config_binds_seven_files_and_result() -> None:
    config = json.loads((ROOT / "configs/opd9_teacher_artifact.json").read_text(encoding="utf-8"))
    required = config["model"]["required_files"]
    assert len(required) == 7
    assert config["model"]["sha256"] == "c86054edddaf186b5a0754fed55e4d8e80108ba2081ff7e6ba7c2d3e589ccdc7"
    assert config["evaluation"]["benchmarks"]["V*"] == {"correct": 176, "total": 191, "percent": 92.14659685863874}
    assert config["evaluation"]["macro_percent"] == 74.49553937627918
    assert config["training"]["dataset"]["processed_rows"] == 6241
    assert config["runtime"]["commit"] == "c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471"
    assert (ROOT / "hf_card/OPD9_TEACHER_README.md").read_text(encoding="utf-8") == _readme()
    public_text = json.dumps(config)
    assert "/minimax-3d-rw-backup" not in public_text
    assert "hf_" not in public_text
    assert "ghp_" not in public_text
