from __future__ import annotations

import json
from pathlib import Path

import pytest

from omg.data.hindi_caption_sidecar import (
    CAPTION_MANIFEST_SCHEMA,
    canonical_bones_motion_id,
    load_bones_hindi_caption_manifest,
)


def _write_manifest(path: Path, records: list[dict]) -> None:
    rows = [{"schema": CAPTION_MANIFEST_SCHEMA, "motion_count": len(records)}, *records]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _record(source_motion_id: str, *, alias: str | None = None) -> dict:
    return {
        "source_motion_id": source_motion_id,
        "aliases": [] if alias is None else [alias],
        "group_id": source_motion_id.removesuffix("_M"),
        "is_mirror": source_motion_id.endswith("_M"),
        "captions": [
            {
                "caption_id": f"{source_motion_id}:1",
                "english": "walk forward slowly",
                "hindi": "धीरे आगे चलो",
            }
        ],
    }


def test_canonical_motion_id_resolves_omg_segment_and_part_suffix() -> None:
    assert (
        canonical_bones_motion_id("path/body_check_001__A512__seg0000__part0002.csv")
        == "body_check_001__A512"
    )
    assert canonical_bones_motion_id("body_check_001__A512__seg0003") == "body_check_001__A512"


def test_manifest_loads_aligned_pairs_and_aliases(tmp_path: Path) -> None:
    path = tmp_path / "captions.jsonl"
    _write_manifest(path, [_record("body_check_001__A512", alias="nested/body_check_001__A512.csv")])
    metadata, index, digest = load_bones_hindi_caption_manifest(path)
    assert metadata["schema"] == CAPTION_MANIFEST_SCHEMA
    assert len(digest) == 64
    entry = index[canonical_bones_motion_id("body_check_001__A512__seg0000__part0002")]
    assert entry["captions"][0]["hindi"] == "धीरे आगे चलो"
    assert entry["captions"][0]["english"] == "walk forward slowly"


def test_manifest_rejects_cross_motion_alias_collision(tmp_path: Path) -> None:
    path = tmp_path / "captions.jsonl"
    _write_manifest(
        path,
        [
            _record("motion_a", alias="shared.csv"),
            _record("motion_b", alias="shared.npy"),
        ],
    )
    with pytest.raises(ValueError, match="belongs to both"):
        load_bones_hindi_caption_manifest(path)


def test_manifest_rejects_missing_hindi_pair(tmp_path: Path) -> None:
    path = tmp_path / "captions.jsonl"
    record = _record("motion_a")
    record["captions"][0]["hindi"] = ""
    _write_manifest(path, [record])
    with pytest.raises(ValueError, match="must contain English and Hindi"):
        load_bones_hindi_caption_manifest(path)
