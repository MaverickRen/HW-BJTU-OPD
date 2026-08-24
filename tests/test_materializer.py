from __future__ import annotations

import json

from materialize_sft_dataset import scrub_metadata


def test_scrub_metadata_removes_local_source_paths() -> None:
    source = {
        "records": [
            {
                "source_path": "/minimax-3d-rw-backup/users/example/private.jsonl",
                "source_revision": "abc123",
            }
        ]
    }
    clean = scrub_metadata(source)
    assert clean == {"records": [{"source_revision": "abc123"}]}
    assert "/minimax" not in json.dumps(clean)
