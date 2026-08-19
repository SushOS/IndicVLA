from __future__ import annotations

import json
from pathlib import Path

import pytest


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from omg.cli.data.audit_bones_hindi_alignment import (
    FORMAT,
    audit_bones_hindi_alignment,
)
from omg.data.hindi_caption_sidecar import CAPTION_MANIFEST_SCHEMA


def _manifest_entry(
    source_motion_id: str,
    *,
    group_id: str,
    aliases: list[str] | None = None,
) -> dict:
    return {
        "record_type": "motion",
        "source_motion_id": source_motion_id,
        "aliases": list(aliases or []),
        "group_id": group_id,
        "captions": [
            {
                "caption_id": f"{source_motion_id}:caption-{index}",
                "english": f"English caption {index} for {source_motion_id}",
                "hindi": f"मोशन {source_motion_id} का कैप्शन {index}",
            }
            for index in range(1, 5)
        ],
    }


def _write_manifest(path: Path, entries: list[dict]) -> None:
    records = [
        {
            "record_type": "metadata",
            "schema": CAPTION_MANIFEST_SCHEMA,
            "omg_data_revision": "6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7",
            "provenance": {"dataset_revision": "a" * 40},
        },
        *entries,
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_episode_metadata(
    root: Path,
    rows: list[dict],
    *,
    include_split_column: bool = True,
) -> None:
    episode_root = root / "meta" / "episodes" / "chunk-000"
    episode_root.mkdir(parents=True)
    table_rows = []
    for row in rows:
        value = dict(row)
        if not include_split_column:
            value.pop("omg/split", None)
        table_rows.append(value)
    pq.write_table(pa.Table.from_pylist(table_rows), episode_root / "file-000.parquet")
    split_names = [str(row["omg/split"]) for row in rows]
    split_ranges = {}
    for split in dict.fromkeys(split_names):
        indices = [
            int(row["episode_index"])
            for row in rows
            if str(row["omg/split"]) == split
        ]
        split_ranges[split] = f"{min(indices)}:{max(indices) + 1}"
    (root / "meta" / "info.json").write_text(
        json.dumps({"splits": split_ranges}),
        encoding="utf-8",
    )


def _episode(
    episode_index: int,
    split: str,
    source_id: str,
    *,
    source_dataset: str | None = None,
) -> dict:
    return {
        "episode_index": episode_index,
        "omg/split": split,
        "omg/dataset": source_dataset or f"bones_seed_{split}",
        "omg/source_id": source_id,
    }


def test_audit_resolves_aliases_reports_splits_and_writes_atomically(tmp_path: Path) -> None:
    manifest_path = tmp_path / "captions.jsonl"
    _write_manifest(
        manifest_path,
        [
            _manifest_entry("motion-a", group_id="group-a", aliases=["legacy/alias-a.csv"]),
            _manifest_entry("motion-b", group_id="group-b", aliases=["legacy/alias-b.npy"]),
            _manifest_entry("motion-c", group_id="group-c"),
        ],
    )
    dataset_root = tmp_path / "omg-data"
    _write_episode_metadata(
        dataset_root,
        [
            _episode(0, "train", "legacy/alias-a.csv"),
            _episode(1, "train", "motion-a.npz", source_dataset="bones_seed"),
            _episode(2, "val", "legacy/alias-b.npy"),
            _episode(3, "test", "motion-c.csv"),
            _episode(4, "test", "not-in-manifest", source_dataset="other_test"),
        ],
    )
    output_path = tmp_path / "reports" / "alignment.json"

    report = audit_bones_hindi_alignment(
        manifest_path,
        dataset_root=dataset_root,
        output_json=output_path,
    )

    assert report["format"] == FORMAT
    assert report["status"] == "pass"
    assert report["dataset"]["scanned_episode_rows"] == 5
    assert report["splits"] == {
        "train": {"episodes": 2, "unique_motions": 1, "caption_pairs": 4, "group_ids": 1},
        "val": {"episodes": 1, "unique_motions": 1, "caption_pairs": 4, "group_ids": 1},
        "test": {"episodes": 1, "unique_motions": 1, "caption_pairs": 4, "group_ids": 1},
    }
    assert report["totals"] == {
        "episodes": 4,
        "unique_motions": 3,
        "caption_pairs": 12,
        "group_ids": 3,
        "unresolved_episodes": 0,
        "leaking_group_ids": 0,
    }
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert not list(output_path.parent.glob(".alignment.json.*.tmp"))

    output_path.write_text("preserve me\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Output already exists"):
        audit_bones_hindi_alignment(
            manifest_path,
            dataset_root=dataset_root,
            output_json=output_path,
        )
    assert output_path.read_text(encoding="utf-8") == "preserve me\n"
    assert not list(output_path.parent.glob(".alignment.json.*.tmp"))


def test_audit_infers_splits_from_info_without_loading_motion_data(tmp_path: Path) -> None:
    manifest_path = tmp_path / "captions.jsonl"
    _write_manifest(
        manifest_path,
        [
            _manifest_entry("motion-a", group_id="group-a"),
            _manifest_entry("motion-b", group_id="group-b"),
            _manifest_entry("motion-c", group_id="group-c"),
        ],
    )
    dataset_root = tmp_path / "metadata-only"
    _write_episode_metadata(
        dataset_root,
        [
            _episode(0, "train", "motion-a"),
            _episode(1, "val", "motion-b"),
            _episode(2, "test", "motion-c"),
        ],
        include_split_column=False,
    )

    report = audit_bones_hindi_alignment(manifest_path, dataset_root=dataset_root)

    assert report["totals"]["episodes"] == 3
    assert not (dataset_root / "data").exists()


def test_audit_fails_when_bones_episode_has_no_manifest_resolution(tmp_path: Path) -> None:
    manifest_path = tmp_path / "captions.jsonl"
    _write_manifest(manifest_path, [_manifest_entry("motion-a", group_id="group-a")])
    dataset_root = tmp_path / "omg-data"
    _write_episode_metadata(dataset_root, [_episode(0, "train", "missing.csv")])

    with pytest.raises(ValueError, match="missing resolution.*missing.csv"):
        audit_bones_hindi_alignment(manifest_path, dataset_root=dataset_root)


def test_audit_fails_on_duplicate_alias_resolution(tmp_path: Path) -> None:
    manifest_path = tmp_path / "captions.jsonl"
    _write_manifest(
        manifest_path,
        [
            _manifest_entry("motion-a", group_id="group-a", aliases=["shared.csv"]),
            _manifest_entry("motion-b", group_id="group-b", aliases=["path/shared.npy"]),
        ],
    )

    with pytest.raises(ValueError, match="belongs to both"):
        audit_bones_hindi_alignment(manifest_path, dataset_root=tmp_path)


def test_audit_fails_on_group_id_overlap_across_splits(tmp_path: Path) -> None:
    manifest_path = tmp_path / "captions.jsonl"
    _write_manifest(
        manifest_path,
        [
            _manifest_entry("motion-a", group_id="shared-group"),
            _manifest_entry("motion-b", group_id="shared-group"),
        ],
    )
    dataset_root = tmp_path / "omg-data"
    _write_episode_metadata(
        dataset_root,
        [
            _episode(0, "train", "motion-a"),
            _episode(1, "val", "motion-b"),
        ],
    )

    with pytest.raises(ValueError, match="group_id overlap.*shared-group"):
        audit_bones_hindi_alignment(manifest_path, dataset_root=dataset_root)
