from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Sequence, TextIO
import unicodedata


FORMAT = "omg.bones_hindi_captions.v1"
DEFAULT_OMG_DATA_REVISION = "6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7"
CAPTION_FIELD_PAIRS = tuple(
    (f"content_natural_desc_{index}", f"content_natural_desc_{index}_hi")
    for index in range(1, 5)
)
IDENTITY_FIELDS = (
    "filename",
    "move_name",
    "move_g1_path",
    "is_mirror",
)

OUTPUT_SCHEMA: dict[str, Any] = {
    "metadata_record": {
        "record_type": "metadata",
        "schema": FORMAT,
        "schema_sha256": "sha256",
        "provenance_sha256": "sha256",
        "omg_data_revision": "sha1",
        "provenance": {
            "dataset_id": "string",
            "dataset_revision": "string",
            "source_file": "string",
            "source_csv_sha256": "sha256",
            "source_schema_sha256": "sha256",
        },
        "caption_field_pairs": "list[{english, hindi}]",
        "caption_slots_per_motion": 4,
        "motion_count": "integer",
        "caption_count": "integer",
        "skipped_missing_pairs": "integer",
    },
    "motion_record": {
        "record_type": "motion",
        "source_row": "integer",
        "source_motion_id": "string",
        "aliases": "list[string]",
        "group_id": "string",
        "is_mirror": "boolean",
        "move_g1_path": "string",
        "move_name": "string",
        "content_name": "string|null",
        "captions": (
            "list[{caption_id, index, english_field, hindi_field, english, hindi, "
            "language, script, annotation_origin}]"
        ),
    },
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


OUTPUT_SCHEMA_SHA256 = sha256_json(OUTPUT_SCHEMA)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_caption(value: str) -> str:
    """Apply the only text normalization allowed by the Hindi caption manifest."""

    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def _required_columns() -> tuple[str, ...]:
    caption_columns = tuple(field for pair in CAPTION_FIELD_PAIRS for field in pair)
    return IDENTITY_FIELDS + caption_columns


def _required_value(row: dict[str, str | None], field: str, *, line_number: int) -> str:
    value = normalize_caption(row.get(field) or "")
    if not value:
        raise ValueError(f"Missing {field!r} at CSV line {line_number}")
    return value


def _parse_bool(value: str, *, field: str, line_number: int) -> bool:
    normalized = normalize_caption(value).casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean {field!r}={value!r} at CSV line {line_number}")


def _aliases(*values: str) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _resolution_alias(value: str) -> str:
    alias = unicodedata.normalize("NFC", value).strip().replace("\\", "/")
    alias = alias.rstrip("/").rsplit("/", 1)[-1]
    if alias.lower().endswith(".csv"):
        alias = alias[:-4]
    return alias


def _validated_fieldnames(input_csv: Path) -> list[str]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")
        fieldnames = [str(field) for field in reader.fieldnames]
    duplicate_columns = sorted({field for field in fieldnames if fieldnames.count(field) > 1})
    if duplicate_columns:
        raise ValueError(f"Duplicate CSV columns: {duplicate_columns}")
    missing_columns = sorted(set(_required_columns()) - set(fieldnames))
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {missing_columns}")
    return fieldnames


def _iter_motion_records(input_csv: Path) -> Iterator[tuple[dict[str, Any], int]]:
    motion_ids: set[str] = set()
    alias_owners: dict[str, str] = {}
    caption_ids: set[str] = set()
    with input_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            line_number = int(reader.line_num)
            filename = _required_value(row, "filename", line_number=line_number)
            move_name = _required_value(row, "move_name", line_number=line_number)
            g1_path = _required_value(row, "move_g1_path", line_number=line_number)
            content_name = normalize_caption(row.get("content_name") or "") or None
            is_mirror = _parse_bool(
                _required_value(row, "is_mirror", line_number=line_number),
                field="is_mirror",
                line_number=line_number,
            )
            source_motion_id = filename
            group_id = filename[:-2] if is_mirror and filename.endswith("_M") else filename
            if source_motion_id in motion_ids:
                raise ValueError(
                    f"Duplicate source_motion_id {source_motion_id!r} at CSV line {line_number}"
                )
            motion_ids.add(source_motion_id)
            aliases = _aliases(
                filename,
                g1_path,
                Path(g1_path).name,
                Path(g1_path).stem,
            )
            for alias in aliases:
                resolution_alias = _resolution_alias(alias)
                owner = alias_owners.get(resolution_alias)
                if owner is not None and owner != source_motion_id:
                    raise ValueError(
                        f"Duplicate resolution alias {resolution_alias!r} for motions {owner!r} and "
                        f"{source_motion_id!r} at CSV line {line_number}"
                    )
                alias_owners[resolution_alias] = source_motion_id

            captions = []
            skipped_missing_pairs = 0
            for caption_index, (english_field, hindi_field) in enumerate(CAPTION_FIELD_PAIRS, start=1):
                caption_en = normalize_caption(row.get(english_field) or "")
                caption_hi = normalize_caption(row.get(hindi_field) or "")
                if not caption_en or not caption_hi:
                    skipped_missing_pairs += 1
                    continue
                caption_id = f"bones-seed:{source_motion_id}:natural-desc-{caption_index}"
                if caption_id in caption_ids:
                    raise ValueError(f"Duplicate caption_id {caption_id!r} at CSV line {line_number}")
                caption_ids.add(caption_id)
                captions.append(
                    {
                        "caption_id": caption_id,
                        "index": caption_index,
                        "english_field": english_field,
                        "hindi_field": hindi_field,
                        "english": caption_en,
                        "hindi": caption_hi,
                        "language": "hi",
                        "script": "Devanagari",
                        "annotation_origin": "machine_translation",
                    }
                )
            if not captions:
                raise ValueError(
                    f"Motion {source_motion_id!r} has zero valid aligned English/Hindi pairs "
                    f"at CSV line {line_number}"
                )
            yield (
                {
                    "record_type": "motion",
                    "source_row": line_number,
                    "source_motion_id": source_motion_id,
                    "aliases": aliases,
                    "group_id": group_id,
                    "is_mirror": is_mirror,
                    "move_g1_path": g1_path,
                    "move_name": move_name,
                    "content_name": content_name,
                    "captions": captions,
                },
                skipped_missing_pairs,
            )


@contextmanager
def atomic_text_output(path: Path, *, overwrite: bool) -> Iterator[TextIO]:
    """Yield a same-directory temporary file and atomically publish it on success."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output appeared while preparing manifest: {path}")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def prepare_caption_manifest(
    input_csv: Path,
    output_jsonl: Path,
    *,
    dataset_id: str,
    dataset_revision: str,
    omg_data_revision: str = DEFAULT_OMG_DATA_REVISION,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_csv = Path(input_csv)
    output_jsonl = Path(output_jsonl)
    dataset_id = str(dataset_id).strip()
    dataset_revision = str(dataset_revision).strip()
    omg_data_revision = str(omg_data_revision).strip().lower()
    if not input_csv.is_file():
        raise FileNotFoundError(f"Translated BONES metadata CSV does not exist: {input_csv}")
    if not dataset_id:
        raise ValueError("dataset_id must be non-empty")
    if not dataset_revision:
        raise ValueError("dataset_revision must be pinned and non-empty")
    if len(omg_data_revision) != 40 or any(
        character not in "0123456789abcdef" for character in omg_data_revision
    ):
        raise ValueError(
            "omg_data_revision must be a full 40-character commit SHA, "
            f"got {omg_data_revision!r}"
        )

    source_csv_sha256 = sha256_file(input_csv)
    fieldnames = _validated_fieldnames(input_csv)
    source_schema_sha256 = sha256_json(fieldnames)
    provenance = {
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "source_file": input_csv.name,
        "source_csv_sha256": source_csv_sha256,
        "source_schema_sha256": source_schema_sha256,
    }
    provenance_sha256 = sha256_json(provenance)

    source_rows = 0
    caption_records = 0
    skipped_missing_pairs = 0
    for record, row_skipped_pairs in _iter_motion_records(input_csv):
        source_rows += 1
        caption_records += len(record["captions"])
        skipped_missing_pairs += row_skipped_pairs
    if source_rows == 0:
        raise ValueError(f"CSV contains no data rows: {input_csv}")

    with atomic_text_output(output_jsonl, overwrite=overwrite) as output:
        metadata_record = {
            "record_type": "metadata",
            "schema": FORMAT,
            "schema_sha256": OUTPUT_SCHEMA_SHA256,
            "provenance_sha256": provenance_sha256,
            "omg_data_revision": omg_data_revision,
            "provenance": provenance,
            "caption_field_pairs": [
                {"english": english_field, "hindi": hindi_field}
                for english_field, hindi_field in CAPTION_FIELD_PAIRS
            ],
            "caption_slots_per_motion": len(CAPTION_FIELD_PAIRS),
            "motion_count": source_rows,
            "caption_count": caption_records,
            "skipped_missing_pairs": skipped_missing_pairs,
        }
        output.write(json.dumps(metadata_record, ensure_ascii=False, sort_keys=True) + "\n")
        for record, _ in _iter_motion_records(input_csv):
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        if sha256_file(input_csv) != source_csv_sha256:
            raise RuntimeError(f"Input CSV changed while preparing the manifest: {input_csv}")

    return {
        "format": FORMAT,
        "schema_sha256": OUTPUT_SCHEMA_SHA256,
        "provenance_sha256": sha256_json(provenance),
        "omg_data_revision": omg_data_revision,
        "source_csv_sha256": source_csv_sha256,
        "source_schema_sha256": source_schema_sha256,
        "source_rows": source_rows,
        "caption_records": caption_records,
        "skipped_missing_pairs": skipped_missing_pairs,
        "output_jsonl": str(output_jsonl),
        "output_jsonl_sha256": sha256_file(output_jsonl),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a validated Hindi caption JSONL manifest from translated BONES-SEED metadata."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-id", default="bones-studio/seed")
    parser.add_argument(
        "--dataset-revision",
        required=True,
        help="Pinned BONES-SEED dataset revision or immutable artifact identifier.",
    )
    parser.add_argument(
        "--omg-data-revision",
        default=DEFAULT_OMG_DATA_REVISION,
        help="Pinned OMG-Data commit whose motion IDs this sidecar overlays.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    summary = prepare_caption_manifest(
        args.input_csv,
        args.output_jsonl,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        omg_data_revision=args.omg_data_revision,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
