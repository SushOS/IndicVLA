import argparse

from translation_pipeline_utils import (
    build_manifest_rows,
    load_jsonl,
    make_translation_ready_entry,
    summarize_translation_corpus,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a frozen, Kimodo-cleared English corpus for multilingual "
            "translation into Hindi, Bengali, and Telugu."
        )
    )
    parser.add_argument(
        "--input",
        default="corpus_build/corpus_raw.jsonl",
        help="Source English corpus JSONL.",
    )
    parser.add_argument(
        "--output",
        default="corpus_build/corpus_translation_ready.jsonl",
        help="Translation-ready corpus JSONL output path.",
    )
    parser.add_argument(
        "--manifest-output",
        default="corpus_build/translation_manifest.jsonl",
        help="Per-prompt per-language translation manifest JSONL output path.",
    )
    parser.add_argument(
        "--summary-output",
        default="corpus_build/translation_summary.json",
        help="Translation preparation summary JSON output path.",
    )
    parser.add_argument(
        "--validation-status-label",
        default="validated_assumed",
        help="Validation label to stamp onto the frozen English corpus.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of English prompts to prepare from the source corpus.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_records = load_jsonl(args.input)
    if args.limit is not None:
        source_records = source_records[:args.limit]
    ready_records = [
        make_translation_ready_entry(
            entry,
            validation_status_label=args.validation_status_label,
        )
        for entry in source_records
    ]

    manifest_rows = build_manifest_rows(ready_records)
    summary = summarize_translation_corpus(
        ready_records,
        assumptions={
            "english_source_frozen": True,
            "kimodo_validation_complete": True,
            "validation_status_label": args.validation_status_label,
        },
    )

    write_jsonl(args.output, ready_records)
    write_jsonl(args.manifest_output, manifest_rows)
    write_json(args.summary_output, summary)

    print(f"Prepared translation-ready corpus : {args.output}")
    print(f"Wrote translation manifest        : {args.manifest_output}")
    print(f"Wrote translation summary         : {args.summary_output}")
    print(f"Total prompts                     : {len(ready_records)}")
    print(f"Total translation jobs            : {len(manifest_rows)}")


if __name__ == "__main__":
    main()
