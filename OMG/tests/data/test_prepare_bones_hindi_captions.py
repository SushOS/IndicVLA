from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from omg.cli.data.prepare_bones_hindi_captions import (
    CAPTION_FIELD_PAIRS,
    FORMAT,
    OUTPUT_SCHEMA_SHA256,
    prepare_caption_manifest,
    sha256_file,
    sha256_json,
)


FIELDNAMES = [
    "filename",
    "move_name",
    "move_g1_path",
    "content_name",
    "is_mirror",
    *(field for pair in CAPTION_FIELD_PAIRS for field in pair),
]


def _row(name: str, *, is_mirror: str = "False") -> dict[str, str]:
    row = {
        "filename": name,
        "move_name": f"move-{name}",
        "move_g1_path": f"g1/csv/240101/{name}.csv",
        "content_name": "shared-action",
        "is_mirror": is_mirror,
    }
    for index, (english_field, hindi_field) in enumerate(CAPTION_FIELD_PAIRS, start=1):
        row[english_field] = f"  Cafe\u0301   action {index}  "
        row[hindi_field] = f"  क\u093cदम\t{index}  "
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_manifest_writes_metadata_then_grouped_motion_records(tmp_path: Path) -> None:
    input_csv = tmp_path / "translated.csv"
    output_jsonl = tmp_path / "captions.jsonl"
    _write_csv(input_csv, [_row("motion-a"), _row("motion-b_M", is_mirror="TRUE")])

    summary = prepare_caption_manifest(
        input_csv,
        output_jsonl,
        dataset_id="bones-studio/seed",
        dataset_revision="a" * 40,
    )

    records = _read_jsonl(output_jsonl)
    assert len(records) == 3
    metadata = records[0]
    assert metadata["record_type"] == "metadata"
    assert metadata["schema"] == FORMAT
    assert metadata["schema_sha256"] == OUTPUT_SCHEMA_SHA256
    assert metadata["provenance_sha256"] == sha256_json(metadata["provenance"])
    assert metadata["omg_data_revision"] == "6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7"
    assert metadata["provenance"]["source_csv_sha256"] == sha256_file(input_csv)
    assert metadata["caption_slots_per_motion"] == 4
    assert metadata["skipped_missing_pairs"] == 0

    first, second = records[1:]
    assert first["record_type"] == "motion"
    assert first["source_motion_id"] == "motion-a"
    assert first["group_id"] == "motion-a"
    assert first["is_mirror"] is False
    assert first["aliases"] == [
        "motion-a",
        "g1/csv/240101/motion-a.csv",
        "motion-a.csv",
    ]
    assert first["move_name"] == "move-motion-a"
    assert len(first["captions"]) == 4
    assert first["captions"][0]["english"] == "Café action 1"
    assert first["captions"][0]["hindi"] == "क़दम 1"
    assert first["captions"][0]["caption_id"] == "bones-seed:motion-a:natural-desc-1"
    assert second["is_mirror"] is True
    assert second["group_id"] == "motion-b"
    assert summary["source_rows"] == 2
    assert summary["caption_records"] == 8
    assert summary["output_jsonl_sha256"] == sha256_file(output_jsonl)


def test_prepare_manifest_rejects_duplicate_motion_and_preserves_existing_output(tmp_path: Path) -> None:
    input_csv = tmp_path / "translated.csv"
    output_jsonl = tmp_path / "captions.jsonl"
    _write_csv(input_csv, [_row("motion-a"), _row("motion-a")])
    output_jsonl.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate source_motion_id"):
        prepare_caption_manifest(
            input_csv,
            output_jsonl,
            dataset_id="bones-studio/seed",
            dataset_revision="a" * 40,
            overwrite=True,
        )

    assert output_jsonl.read_text(encoding="utf-8") == "existing\n"
    assert not list(tmp_path.glob(".captions.jsonl.*.tmp"))


def test_prepare_manifest_skips_missing_translation_pair_and_records_count(tmp_path: Path) -> None:
    input_csv = tmp_path / "translated.csv"
    output_jsonl = tmp_path / "captions.jsonl"
    row = _row("motion-a")
    row["content_natural_desc_3_hi"] = "   "
    _write_csv(input_csv, [row])

    summary = prepare_caption_manifest(
        input_csv,
        output_jsonl,
        dataset_id="bones-studio/seed",
        dataset_revision="a" * 40,
    )

    metadata, motion = _read_jsonl(output_jsonl)
    assert metadata["skipped_missing_pairs"] == 1
    assert metadata["caption_count"] == 3
    assert summary["skipped_missing_pairs"] == 1
    assert summary["caption_records"] == 3
    assert [caption["index"] for caption in motion["captions"]] == [1, 2, 4]


def test_prepare_manifest_rejects_motion_with_no_aligned_pair_atomically(tmp_path: Path) -> None:
    input_csv = tmp_path / "translated.csv"
    output_jsonl = tmp_path / "captions.jsonl"
    row = _row("motion-a")
    for _, hindi_field in CAPTION_FIELD_PAIRS:
        row[hindi_field] = ""
    _write_csv(input_csv, [row])

    with pytest.raises(ValueError, match="zero valid aligned"):
        prepare_caption_manifest(
            input_csv,
            output_jsonl,
            dataset_id="bones-studio/seed",
            dataset_revision="a" * 40,
        )

    assert not output_jsonl.exists()
