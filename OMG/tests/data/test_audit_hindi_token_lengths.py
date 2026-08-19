from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from omg.cli.data.audit_hindi_token_lengths import audit_token_lengths
from omg.cli.data.prepare_bones_hindi_captions import (
    CAPTION_FIELD_PAIRS,
    prepare_caption_manifest,
)


FIELDNAMES = [
    "filename",
    "move_name",
    "move_g1_path",
    "content_name",
    "is_mirror",
    *(field for pair in CAPTION_FIELD_PAIRS for field in pair),
]


class FakeMuRILTokenizer:
    vocab_size = 1000

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool, truncation: bool) -> list[int]:
        assert truncation is False
        content = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 100
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            content.append(self._token_to_id[token])
        return [101, *content, 102] if add_special_tokens else content

    @staticmethod
    def num_special_tokens_to_add(*, pair: bool) -> int:
        assert pair is False
        return 2

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)


def _words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def _write_manifest_fixture(tmp_path: Path) -> Path:
    input_csv = tmp_path / "translated.csv"
    output_jsonl = tmp_path / "captions.jsonl"
    lengths = ((1, 2, 3, 4), (5, 10, 48, 52))
    rows = []
    for row_index, caption_lengths in enumerate(lengths, start=1):
        name = f"motion-{row_index}"
        row = {
            "filename": name,
            "move_name": f"move-{name}",
            "move_g1_path": f"g1/csv/{name}.csv",
            "content_name": f"group-{row_index}",
            "is_mirror": "False",
        }
        for caption_index, ((english_field, hindi_field), length) in enumerate(
            zip(CAPTION_FIELD_PAIRS, caption_lengths, strict=True),
            start=1,
        ):
            row[english_field] = f"English caption {caption_index}"
            row[hindi_field] = _words(f"h{row_index}_{caption_index}_", length)
        rows.append(row)
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    prepare_caption_manifest(
        input_csv,
        output_jsonl,
        dataset_id="bones-studio/seed",
        dataset_revision="b" * 40,
    )
    return output_jsonl


def test_audit_reports_exact_nearest_rank_statistics_and_removed_suffix(tmp_path: Path) -> None:
    manifest = _write_manifest_fixture(tmp_path)
    output = tmp_path / "audit.json"
    tokenizer = FakeMuRILTokenizer()

    report = audit_token_lengths(
        manifest,
        output,
        tokenizer=tokenizer,
        tokenizer_model_name="fake-muril",
        tokenizer_revision="c" * 40,
    )

    assert report["caption_count"] == 8
    assert report["motion_count"] == 2
    assert report["statistics"] == {
        "min": 3,
        "p50": 6,
        "p90": 54,
        "p95": 54,
        "p99": 54,
        "max": 54,
    }
    assert report["over_50_count"] == 1
    assert report["over_50_rate"] == pytest.approx(1 / 8)
    assert report["longest"][0]["token_length"] == 54
    assert report["longest"][0]["removed_token_count"] == 4
    assert report["longest"][0]["decoded_removed_suffix"] == " ".join(
        f"h2_4_{index}" for index in range(48, 52)
    )
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_audit_accepts_manifest_with_recorded_missing_pair(tmp_path: Path) -> None:
    manifest = _write_manifest_fixture(tmp_path)
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    records[2]["captions"].pop(0)
    records[0]["caption_count"] -= 1
    records[0]["skipped_missing_pairs"] += 1
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    report = audit_token_lengths(
        manifest,
        tmp_path / "audit.json",
        tokenizer=FakeMuRILTokenizer(),
        tokenizer_model_name="fake-muril",
        tokenizer_revision="c" * 40,
    )

    assert report["caption_count"] == 7
    assert report["motion_count"] == 2


@pytest.mark.parametrize("mutation", ["duplicate", "missing"])
def test_audit_rejects_duplicate_or_missing_captions_atomically(tmp_path: Path, mutation: str) -> None:
    manifest = _write_manifest_fixture(tmp_path)
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if mutation == "duplicate":
        records[2]["captions"][0]["caption_id"] = records[1]["captions"][0]["caption_id"]
        expected = "Duplicate caption_id"
    else:
        del records[2]["captions"][0]["hindi"]
        expected = "Missing 'hindi'"
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"

    with pytest.raises(ValueError, match=expected):
        audit_token_lengths(
            manifest,
            output,
            tokenizer=FakeMuRILTokenizer(),
            tokenizer_model_name="fake-muril",
            tokenizer_revision="c" * 40,
        )

    assert not output.exists()
