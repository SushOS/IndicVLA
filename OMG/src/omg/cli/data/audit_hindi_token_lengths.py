from __future__ import annotations

import argparse
from collections import Counter
import heapq
import json
import math
from pathlib import Path
from typing import Any, Sequence

from omg.cli.data.prepare_bones_hindi_captions import (
    CAPTION_FIELD_PAIRS,
    FORMAT as CAPTION_MANIFEST_FORMAT,
    OUTPUT_SCHEMA_SHA256 as CAPTION_SCHEMA_SHA256,
    atomic_text_output,
    normalize_caption,
    sha256_file,
    sha256_json,
)


FORMAT = "omg.hindi_token_length_audit.v1"
REPORT_SCHEMA: dict[str, Any] = {
    "format": FORMAT,
    "schema_sha256": "sha256",
    "input": {
        "path": "string",
        "sha256": "sha256",
        "caption_schema": CAPTION_MANIFEST_FORMAT,
        "caption_schema_sha256": "sha256",
        "caption_provenance_sha256": "sha256",
    },
    "tokenizer": {
        "model_name": "string",
        "revision": "string",
        "class": "string",
        "vocab_size": "integer|null",
    },
    "max_length": "integer",
    "caption_count": "integer",
    "motion_count": "integer",
    "statistics": "{min,p50,p90,p95,p99,max}",
    "over_max_length_count": "integer",
    "over_max_length_rate": "number",
    "longest": "list[caption audit record]",
}
REPORT_SCHEMA_SHA256 = sha256_json(REPORT_SCHEMA)


def _required_string(value: Any, *, field: str, line_number: int) -> str:
    if value is None:
        raise ValueError(f"Missing {field!r} at JSONL line {line_number}")
    if not isinstance(value, str):
        raise ValueError(f"Expected string {field!r} at JSONL line {line_number}")
    normalized = normalize_caption(value)
    if not normalized:
        raise ValueError(f"Missing {field!r} at JSONL line {line_number}")
    return normalized


def _token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    values = tokenizer.encode(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
    )
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"Tokenizer.encode returned {type(values).__name__}, expected a token-id sequence")
    return [int(value) for value in values]


def _nearest_rank(histogram: Counter[int], count: int, percentile: float) -> int:
    rank = max(1, int(math.ceil(float(percentile) * count)))
    observed = 0
    for token_length in sorted(histogram):
        observed += histogram[token_length]
        if observed >= rank:
            return int(token_length)
    raise RuntimeError("Token-length histogram is inconsistent with caption count")


def _removed_suffix(tokenizer: Any, text: str, *, max_length: int) -> tuple[int, str]:
    content_ids = _token_ids(tokenizer, text, add_special_tokens=False)
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    content_capacity = max(0, int(max_length) - special_tokens)
    removed_ids = content_ids[content_capacity:]
    if not removed_ids:
        return 0, ""
    decoded = tokenizer.decode(
        removed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return len(removed_ids), normalize_caption(decoded)


def _tokenizer_vocab_size(tokenizer: Any) -> int | None:
    value = getattr(tokenizer, "vocab_size", None)
    if value is None:
        try:
            value = len(tokenizer)
        except (TypeError, AttributeError):
            return None
    return int(value)


def audit_token_lengths(
    input_jsonl: Path,
    output_json: Path,
    *,
    tokenizer: Any,
    tokenizer_model_name: str,
    tokenizer_revision: str,
    max_length: int = 50,
    longest_count: int = 25,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_jsonl = Path(input_jsonl)
    output_json = Path(output_json)
    tokenizer_model_name = str(tokenizer_model_name).strip()
    tokenizer_revision = str(tokenizer_revision).strip()
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"Hindi caption manifest does not exist: {input_jsonl}")
    if not tokenizer_model_name:
        raise ValueError("tokenizer_model_name must be non-empty")
    if not tokenizer_revision:
        raise ValueError("tokenizer_revision must be pinned and non-empty")
    if int(max_length) <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if int(longest_count) <= 0:
        raise ValueError(f"longest_count must be positive, got {longest_count}")
    max_length = int(max_length)
    longest_count = int(longest_count)

    input_sha256 = sha256_file(input_jsonl)
    histogram: Counter[int] = Counter()
    caption_ids: set[str] = set()
    motion_ids: set[str] = set()
    caption_count = 0
    over_max_count = 0
    metadata: dict[str, Any] | None = None
    longest_heap: list[tuple[int, int, int, str, str, str]] = []

    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"Blank JSONL record at line {line_number}")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at line {line_number}")

            if line_number == 1:
                if record.get("record_type") != "metadata":
                    raise ValueError("First JSONL record must be metadata")
                if record.get("schema") != CAPTION_MANIFEST_FORMAT:
                    raise ValueError(
                        f"Unexpected caption schema {record.get('schema')!r}; "
                        f"expected {CAPTION_MANIFEST_FORMAT!r}"
                    )
                if record.get("schema_sha256") != CAPTION_SCHEMA_SHA256:
                    raise ValueError("Caption manifest schema hash does not match this audit implementation")
                provenance = record.get("provenance")
                if not isinstance(provenance, dict):
                    raise ValueError("Metadata record is missing provenance")
                if record.get("provenance_sha256") != sha256_json(provenance):
                    raise ValueError("Caption manifest provenance hash is invalid")
                expected_pairs = [
                    {"english": english_field, "hindi": hindi_field}
                    for english_field, hindi_field in CAPTION_FIELD_PAIRS
                ]
                if record.get("caption_field_pairs") != expected_pairs:
                    raise ValueError("Metadata record does not declare the four expected English/Hindi pairs")
                if record.get("caption_slots_per_motion") != len(CAPTION_FIELD_PAIRS):
                    raise ValueError("Metadata caption_slots_per_motion is not four")
                for count_field in ("motion_count", "caption_count", "skipped_missing_pairs"):
                    value = record.get(count_field)
                    if not isinstance(value, int) or value < 0:
                        raise ValueError(f"Metadata {count_field} must be a non-negative integer")
                metadata = record
                continue

            if record.get("record_type") != "motion":
                raise ValueError(f"Expected motion record at JSONL line {line_number}")
            source_motion_id = _required_string(
                record.get("source_motion_id"),
                field="source_motion_id",
                line_number=line_number,
            )
            if source_motion_id in motion_ids:
                raise ValueError(f"Duplicate source_motion_id {source_motion_id!r} at JSONL line {line_number}")
            motion_ids.add(source_motion_id)
            captions = record.get("captions")
            if not isinstance(captions, list) or not 1 <= len(captions) <= len(CAPTION_FIELD_PAIRS):
                raise ValueError(
                    f"Motion {source_motion_id!r} must contain one to four captions at JSONL line {line_number}"
                )

            observed_indices: set[int] = set()
            for caption in captions:
                if not isinstance(caption, dict):
                    raise ValueError(
                        f"Expected caption object for motion {source_motion_id!r} at JSONL line {line_number}"
                    )
                caption_index = caption.get("index")
                if not isinstance(caption_index, int) or not 1 <= caption_index <= len(CAPTION_FIELD_PAIRS):
                    raise ValueError(
                        f"Invalid caption index for motion {source_motion_id!r} at JSONL line {line_number}"
                    )
                if caption_index in observed_indices:
                    raise ValueError(
                        f"Duplicate caption index {caption_index} for motion {source_motion_id!r} "
                        f"at JSONL line {line_number}"
                    )
                observed_indices.add(caption_index)
                expected_english_field, expected_hindi_field = CAPTION_FIELD_PAIRS[caption_index - 1]
                if (
                    caption.get("english_field") != expected_english_field
                    or caption.get("hindi_field") != expected_hindi_field
                ):
                    raise ValueError(
                        f"Caption field pair mismatch for motion {source_motion_id!r}, slot {caption_index} "
                        f"at JSONL line {line_number}"
                    )
                caption_id = _required_string(
                    caption.get("caption_id"),
                    field="caption_id",
                    line_number=line_number,
                )
                if caption_id in caption_ids:
                    raise ValueError(f"Duplicate caption_id {caption_id!r} at JSONL line {line_number}")
                caption_ids.add(caption_id)
                _required_string(caption.get("english"), field="english", line_number=line_number)
                hindi = _required_string(caption.get("hindi"), field="hindi", line_number=line_number)
                token_length = len(_token_ids(tokenizer, hindi, add_special_tokens=True))
                histogram[token_length] += 1
                caption_count += 1
                if token_length > max_length:
                    over_max_count += 1

                item = (
                    token_length,
                    -line_number,
                    caption_index,
                    source_motion_id,
                    caption_id,
                    hindi,
                )
                if len(longest_heap) < longest_count:
                    heapq.heappush(longest_heap, item)
                elif item[:3] > longest_heap[0][:3]:
                    heapq.heapreplace(longest_heap, item)

    if metadata is None:
        raise ValueError(f"Caption manifest is empty: {input_jsonl}")
    if not motion_ids:
        raise ValueError(f"Caption manifest contains no motion records: {input_jsonl}")
    if metadata["motion_count"] != len(motion_ids):
        raise ValueError(
            f"Metadata motion_count={metadata['motion_count']} does not match observed {len(motion_ids)}"
        )
    if metadata["caption_count"] != caption_count:
        raise ValueError(
            f"Metadata caption_count={metadata['caption_count']} does not match observed {caption_count}"
        )
    expected_skipped = len(motion_ids) * len(CAPTION_FIELD_PAIRS) - caption_count
    if metadata["skipped_missing_pairs"] != expected_skipped:
        raise ValueError(
            f"Metadata skipped_missing_pairs={metadata['skipped_missing_pairs']} does not match "
            f"observed {expected_skipped}"
        )
    if sha256_file(input_jsonl) != input_sha256:
        raise RuntimeError(f"Input JSONL changed while auditing token lengths: {input_jsonl}")

    statistics = {
        "min": min(histogram),
        "p50": _nearest_rank(histogram, caption_count, 0.50),
        "p90": _nearest_rank(histogram, caption_count, 0.90),
        "p95": _nearest_rank(histogram, caption_count, 0.95),
        "p99": _nearest_rank(histogram, caption_count, 0.99),
        "max": max(histogram),
    }
    longest = []
    for token_length, negative_line, caption_index, source_motion_id, caption_id, hindi in sorted(
        longest_heap,
        key=lambda item: (-item[0], -item[1], item[2]),
    ):
        removed_count, removed_suffix = _removed_suffix(tokenizer, hindi, max_length=max_length)
        longest.append(
            {
                "source_motion_id": source_motion_id,
                "caption_id": caption_id,
                "caption_index": caption_index,
                "jsonl_line": -negative_line,
                "hindi": hindi,
                "token_length": token_length,
                "removed_token_count": removed_count,
                "decoded_removed_suffix": removed_suffix,
            }
        )

    provenance = metadata["provenance"]
    report: dict[str, Any] = {
        "format": FORMAT,
        "schema_sha256": REPORT_SCHEMA_SHA256,
        "input": {
            "path": input_jsonl.name,
            "sha256": input_sha256,
            "caption_schema": CAPTION_MANIFEST_FORMAT,
            "caption_schema_sha256": metadata["schema_sha256"],
            "caption_provenance_sha256": metadata["provenance_sha256"],
            "source_csv_sha256": provenance["source_csv_sha256"],
            "source_schema_sha256": provenance["source_schema_sha256"],
        },
        "tokenizer": {
            "model_name": tokenizer_model_name,
            "revision": tokenizer_revision,
            "class": type(tokenizer).__name__,
            "vocab_size": _tokenizer_vocab_size(tokenizer),
        },
        "max_length": max_length,
        "quantile_method": "nearest_rank",
        "caption_count": caption_count,
        "motion_count": len(motion_ids),
        "statistics": statistics,
        "over_max_length_count": over_max_count,
        "over_max_length_rate": over_max_count / caption_count,
        "longest": longest,
    }
    if max_length == 50:
        report["over_50_count"] = over_max_count
        report["over_50_rate"] = over_max_count / caption_count

    with atomic_text_output(output_json, overwrite=overwrite) as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return report


def _load_tokenizer(model_name: str, revision: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "MuRIL token-length auditing requires transformers; install OMG with the training extras."
        ) from exc
    return AutoTokenizer.from_pretrained(model_name, revision=revision)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit MuRIL token lengths in a Hindi BONES caption manifest.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--tokenizer", default="google/muril-base-cased")
    parser.add_argument(
        "--tokenizer-revision",
        required=True,
        help="Pinned MuRIL tokenizer commit/revision.",
    )
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--longest-count", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    tokenizer = _load_tokenizer(args.tokenizer, args.tokenizer_revision)
    report = audit_token_lengths(
        args.input_jsonl,
        args.output_json,
        tokenizer=tokenizer,
        tokenizer_model_name=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        max_length=args.max_length,
        longest_count=args.longest_count,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
