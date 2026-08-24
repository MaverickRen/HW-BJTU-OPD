#!/usr/bin/env python3
"""Create a private GitHub repository and push this checkout without storing a PAT."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


API_ROOT = "https://api.github.com"


def request_json(
    method: str,
    path: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{API_ROOT}{path}",
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "hw-bjtu-opd-release",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}
    except HTTPError as error:
        raw = error.read()
        detail = json.loads(raw) if raw else {}
        return error.code, detail


def git(repo_root: Path, *args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo_root,
        env=env,
        check=True,
    )


def configure_remote(repo_root: Path, remote: str, url: str) -> None:
    result = subprocess.run(
        ["git", "remote", "get-url", remote],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        existing = result.stdout.strip()
        if existing != url:
            raise RuntimeError(
                f"remote {remote!r} already points to {existing!r}; refusing to replace it"
            )
    else:
        git(repo_root, "remote", "add", remote, url)


def push_without_storing_token(
    repo_root: Path,
    remote: str,
    branch: str,
    token: str,
) -> None:
    askpass_source = """#!/bin/sh
case "$1" in
  *Username*) printf '%s\\n' 'x-access-token' ;;
  *) printf '%s\\n' "$HW_BJTU_GITHUB_TOKEN" ;;
esac
"""
    askpass_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="hw-bjtu-opd-askpass-", suffix=".sh", delete=False
        ) as handle:
            handle.write(askpass_source)
            askpass_path = Path(handle.name)
        askpass_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        environment = os.environ.copy()
        environment.update(
            {
                "GIT_ASKPASS": str(askpass_path),
                "GIT_TERMINAL_PROMPT": "0",
                "HW_BJTU_GITHUB_TOKEN": token,
            }
        )
        git(repo_root, "push", "--set-upstream", remote, branch, env=environment)
    finally:
        if askpass_path is not None:
            askpass_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--name", default="HW-BJTU-OPD")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()

    token = getpass.getpass("GitHub token: ").strip()
    if not token:
        raise SystemExit("empty token")

    status, identity = request_json("GET", "/user", token)
    if status != 200 or not identity.get("login"):
        raise SystemExit(f"GitHub authentication failed (HTTP {status})")
    owner = identity["login"]
    repo_path = f"/repos/{owner}/{args.name}"
    status, repository = request_json("GET", repo_path, token)
    if status == 404:
        status, repository = request_json(
            "POST",
            "/user/repos",
            token,
            {
                "name": args.name,
                "description": "Reproducible multimodal SFT and crop-based OPD for Qwen3.5",
                "private": True,
                "has_issues": True,
                "has_projects": False,
                "has_wiki": False,
            },
        )
        if status != 201:
            raise SystemExit(f"GitHub repository creation failed (HTTP {status})")
    elif status != 200:
        raise SystemExit(f"GitHub repository lookup failed (HTTP {status})")

    if not repository.get("private", False):
        status, repository = request_json(
            "PATCH", repo_path, token, {"private": True}
        )
        if status != 200 or not repository.get("private", False):
            raise SystemExit(f"failed to enforce private visibility (HTTP {status})")

    repo_root = args.repo_root.resolve()
    remote_url = f"https://github.com/{owner}/{args.name}.git"
    configure_remote(repo_root, args.remote, remote_url)
    push_without_storing_token(repo_root, args.remote, args.branch, token)
    print(f"pushed {args.branch} to private repository {repository['html_url']}")


if __name__ == "__main__":
    main()
