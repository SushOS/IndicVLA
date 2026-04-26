import argparse

from translation_pipeline_utils import (
    build_manifest_rows,
    finalize_entry_for_release,
    load_jsonl,
    summarize_translation_corpus,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize a multilingual corpus after LLM translation and optional "
            "bilingual review."
        )
    )
    parser.add_argument(
        "--input",
        default="corpus_build/corpus_translation_ready.jsonl",
        help="Input corpus JSONL containing translation blocks.",
    )
    parser.add_argument(
        "--output",
        default="corpus_build/corpus_multilingual_final.jsonl",
        help="Finalized multilingual corpus JSONL output path.",
    )
    parser.add_argument(
        "--manifest-output",
        default="corpus_build/translation_manifest_final.jsonl",
        help="Final manifest JSONL output path.",
    )
    parser.add_argument(
        "--summary-output",
        default="corpus_build/translation_summary_final.json",
        help="Final summary JSON output path.",
    )
    parser.add_argument(
        "--require-approved-review",
        action="store_true",
        help="Fail if any language block has not been marked review_status=approved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_jsonl(args.input)

    for entry in corpus:
        finalize_entry_for_release(entry)
        if args.require_approved_review:
            pending = [
                code
                for code in ("hi", "bn", "te")
                if entry[f"translation_{code}"]["review"]["review_status"] != "approved"
            ]
            if pending:
                raise ValueError(
                    f"{entry['prompt_id']} has untranslated or unapproved review blocks for {pending}."
                )

    write_jsonl(args.output, corpus)
    write_jsonl(args.manifest_output, build_manifest_rows(corpus))
    write_json(
        args.summary_output,
        summarize_translation_corpus(
            corpus,
            assumptions={
                "english_source_frozen": True,
                "kimodo_validation_complete": True,
                "release_ready": not args.require_approved_review,
                "review_gate": "approved_required" if args.require_approved_review else "not_enforced",
            },
        ),
    )

    print(f"Final multilingual corpus written : {args.output}")
    print(f"Final manifest written            : {args.manifest_output}")
    print(f"Final summary written             : {args.summary_output}")
    print(f"Total prompts                     : {len(corpus)}")


if __name__ == "__main__":
    main()
