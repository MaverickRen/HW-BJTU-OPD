#!/usr/bin/env python3
"""Create a public Hugging Face repository and upload a staged large folder."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--repo-id", default="HW_BJTU_OPD/HW-BJTU-OPD")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"release folder does not exist: {folder}")
    if not (folder / "README.md").is_file():
        raise SystemExit(f"release folder has no README.md: {folder}")

    token = getpass.getpass("Hugging Face token: ").strip()
    if not token:
        raise SystemExit("empty token")

    api = HfApi(token=token)
    identity = api.whoami()
    identity_name = identity.get("name") or identity.get("fullname") or "authenticated user"
    print(f"authenticated as {identity_name}; ensuring public repository {args.repo_id}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    info = api.repo_info(repo_id=args.repo_id, repo_type="model")
    if getattr(info, "private", False):
        api.update_repo_settings(
            repo_id=args.repo_id,
            repo_type="model",
            private=False,
        )

    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=folder,
        private=False,
        ignore_patterns=[".cache/**"],
        num_workers=args.workers,
        print_report=True,
        print_report_every=60,
    )
    final_info = api.repo_info(repo_id=args.repo_id, repo_type="model")
    if getattr(final_info, "private", True):
        raise SystemExit("upload completed but repository is not public")
    print(f"public release uploaded: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
