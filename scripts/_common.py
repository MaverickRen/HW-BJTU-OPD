"""Small dependency-free helpers shared by release launchers."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


class ReleaseError(RuntimeError):
    """A fail-closed release or launcher error."""


def load_config(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ReleaseError(f"unexpected config schema in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ReleaseError(f"missing regular {label}: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ReleaseError(f"missing real {label}: {path}")
    return path


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def printable(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def run(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("xb") as stream:
        result = subprocess.run(
            [str(item) for item in command],
            cwd=str(cwd),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise ReleaseError(f"command failed with status {result.returncode}; see {log}")


def training_env(*roots: Path) -> dict[str, str]:
    env = os.environ.copy()
    additions = [str(root) for root in roots if root]
    existing = env.get("PYTHONPATH")
    if existing:
        additions.append(existing)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(additions),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
    )
    return env
