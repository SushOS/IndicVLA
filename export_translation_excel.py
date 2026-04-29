import argparse
from pathlib import Path

import pandas as pd

from translation_pipeline_utils import LANGUAGES, VERIFICATION_FIELDS, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the multilingual translation corpus to an Excel workbook with "
            "English instructions, translated instructions, and key metadata."
        )
    )
    parser.add_argument(
        "--input",
        default="corpus_build/corpus_translation_ready.jsonl",
        help="Input translation corpus JSONL.",
    )
    parser.add_argument(
        "--output",
        default="corpus_build/translations.xlsx",
        help="Excel workbook output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of prompts to export.",
    )
    return parser.parse_args()


def flatten_entry(entry: dict) -> dict:
    row = {
        "prompt_id": entry.get("prompt_id"),
        "tier": entry.get("tier"),
        "subtier": entry.get("subtier"),
        "motion_family": entry.get("motion_family"),
        "constraint_type": entry.get("constraint_type"),
        "english_canonical": entry.get("instruction_en", {}).get("canonical"),
        "english_robot_natural": entry.get("instruction_en", {}).get("robot_natural"),
        "split": entry.get("metadata", {}).get("split"),
        "translation_status": entry.get("metadata", {}).get("translation_status"),
        "validation_status": entry.get("metadata", {}).get("validation_status"),
        "safe_for_direct_execution": entry.get("metadata", {}).get("safe_for_direct_execution"),
        "ambiguity_flag": entry.get("metadata", {}).get("ambiguity_flag"),
        "ambiguity_type": entry.get("metadata", {}).get("ambiguity_type"),
        "requires_scene_context": entry.get("metadata", {}).get("requires_scene_context"),
        "requires_body_reference": entry.get("metadata", {}).get("requires_body_reference"),
        "primitive_tags": ", ".join(entry.get("primitive_tags", [])),
        "landmarks": ", ".join(entry.get("landmarks", [])),
        "pipeline_stage": entry.get("pipeline", {}).get("stage"),
        "pipeline_next_required_step": entry.get("pipeline", {}).get("next_required_step"),
        "locomotion_type": entry.get("embodiment", {}).get("locomotion_type"),
        "robot_feasibility": entry.get("embodiment", {}).get("robot_feasibility"),
        "terrain_dependence": entry.get("embodiment", {}).get("terrain_dependence"),
        "contact_complexity": entry.get("embodiment", {}).get("contact_complexity"),
        "sequence_duration_s": entry.get("embodiment", {}).get("sequence_duration_s"),
        "target_body": entry.get("embodiment", {}).get("target_body"),
    }

    for code, language_name in LANGUAGES.items():
        block = entry.get(f"translation_{code}", {})
        verification = block.get("verification", {})
        review = block.get("review", {})
        provenance = block.get("provenance", {})

        prefix = f"{code}_{language_name.lower()}"
        language_key = language_name.lower()
        row[f"{language_key}_canonical"] = block.get("literal")
        row[f"{language_key}_robot"] = block.get("robot_natural")
        row[f"{prefix}_literal"] = block.get("literal")
        row[f"{prefix}_robot_natural"] = block.get("robot_natural")
        row[f"{prefix}_status"] = block.get("status")
        row[f"{prefix}_review_status"] = review.get("review_status")
        row[f"{prefix}_reviewer"] = review.get("bilingual_reviewer")
        row[f"{prefix}_adjudicator"] = review.get("adjudicator")
        row[f"{prefix}_translation_notes"] = " | ".join(block.get("translation_notes", []))
        row[f"{prefix}_translation_model"] = provenance.get("translation_model")
        row[f"{prefix}_translation_method"] = provenance.get("translation_method")
        for field in VERIFICATION_FIELDS:
            row[f"{prefix}_{field}"] = verification.get(field)

    return row


def autosize_columns(worksheet, frame: pd.DataFrame) -> None:
    for index, column in enumerate(frame.columns):
        values = frame[column].astype(str).tolist()
        max_len = max([len(column), *[len(value) for value in values]]) if values else len(column)
        worksheet.column_dimensions[worksheet.cell(row=1, column=index + 1).column_letter].width = min(max_len + 2, 50)


def main() -> None:
    args = parse_args()
    entries = load_jsonl(args.input)
    if args.limit is not None:
        entries = entries[:args.limit]

    frame = pd.DataFrame([flatten_entry(entry) for entry in entries])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="translations")
        worksheet = writer.sheets["translations"]
        autosize_columns(worksheet, frame)
        worksheet.freeze_panes = "A2"

    print(f"Exported {len(frame)} prompts to {output_path}")


if __name__ == "__main__":
    main()
