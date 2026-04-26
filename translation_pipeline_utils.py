import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LANGUAGES = {
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
}

VERIFICATION_FIELDS = (
    "action_preservation",
    "order_preservation",
    "direction_preservation",
    "safety_preservation",
)

SUBTIER_BY_TIER = {
    "T1": "atomic_primitive",
    "T2": "parameterized_atomic",
    "T3": "short_composite",
    "T4": "constraint_conditioned",
    "T5": "landmark_navigation",
    "T6": "safety_abstention_social",
    "T7": "long_horizon_multistep",
}

PRIMITIVE_PATTERNS = [
    ("follow_path", ("follow", "path", "route", "corridor")),
    ("turn_left", ("turn left", "rotate left", "pivot left", "face left")),
    ("turn_right", ("turn right", "rotate right", "pivot right", "face right")),
    ("turn_around", ("turn around", "rotate around", "pivot around")),
    ("move_forward", ("move forward", "walk forward", "step forward", "go forward", "ahead")),
    ("move_backward", ("move backward", "walk backward", "step back", "back up")),
    ("sidestep", ("sidestep", "step left", "step right")),
    ("stop", ("stop", "halt", "freeze", "pause", "hold position", "stand still")),
    ("crouch", ("crouch", "duck", "lower your body")),
    ("kneel", ("kneel",)),
    ("rise", ("rise", "stand up", "straighten up", "raise your body")),
    ("bow", ("bow", "lean forward")),
    ("wave", ("wave",)),
    ("clap", ("clap",)),
    ("nod", ("nod",)),
    ("salute", ("salute",)),
    ("yield", ("yield", "give way", "move to the side", "step aside")),
    ("inspect", ("inspect", "assess", "check", "look ahead")),
]

LANDMARK_PATTERNS = (
    "door",
    "gate",
    "marker",
    "wall",
    "corridor",
    "table",
    "chair",
    "ramp",
    "cone",
    "barrier",
    "pillar",
    "window",
    "shelf",
    "entrance",
    "exit",
    "desk",
    "cabinet",
    "sofa",
    "bench",
    "post",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=target.parent) as tmp:
        for record in records:
            tmp.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(target)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=target.parent) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(target)


def infer_subtier(tier: str) -> str:
    return SUBTIER_BY_TIER.get(tier, "unclassified")


def infer_primitive_tags(text: str) -> list[str]:
    lowered = text.casefold()
    tags = [tag for tag, needles in PRIMITIVE_PATTERNS if any(needle in lowered for needle in needles)]
    return tags or ["unclassified_motion"]


def extract_landmarks(entry: dict[str, Any]) -> list[str]:
    payload = entry.get("kimodo", {}).get("constraint_payload") or {}
    payload_landmarks = payload.get("landmarks") or []
    if payload_landmarks:
        return sorted(dict.fromkeys(payload_landmarks))

    canonical = entry.get("instruction_en", {}).get("canonical", "").casefold()
    found = []
    for pattern in LANDMARK_PATTERNS:
        phrase = f"the {pattern}"
        if phrase in canonical:
            found.append(phrase)
    return sorted(dict.fromkeys(found))


def empty_translation_entry(
    prompt_id: str,
    language_code: str,
    source_canonical: str,
    source_robot_natural: str,
    *,
    ambiguity_flag: bool,
    ambiguity_type: str | None,
    requires_scene_context: bool,
    requires_body_reference: bool,
    safe_for_direct_execution: bool,
) -> dict[str, Any]:
    return {
        "language": LANGUAGES[language_code],
        "literal": None,
        "robot_natural": None,
        "status": "pending",
        "translation_notes": [],
        "verification": {field: None for field in VERIFICATION_FIELDS},
        "metadata": {
            "translation_style_options": ["literal", "robot_natural"],
            "ambiguity_flag": ambiguity_flag,
            "ambiguity_type": ambiguity_type,
            "requires_scene_context": requires_scene_context,
            "requires_body_reference": requires_body_reference,
            "safe_for_direct_execution": safe_for_direct_execution,
        },
        "review": {
            "bilingual_reviewer": None,
            "adjudicator": None,
            "review_status": "pending",
            "review_notes": None,
        },
        "provenance": {
            "source_language": "en",
            "source_prompt_id": prompt_id,
            "source_canonical": source_canonical,
            "source_robot_natural": source_robot_natural,
        },
    }


def make_translation_ready_entry(
    entry: dict[str, Any],
    *,
    validation_status_label: str,
) -> dict[str, Any]:
    record = json.loads(json.dumps(entry))
    instruction_en = record.get("instruction_en", {})
    metadata = record.setdefault("metadata", {})
    kimodo = record.setdefault("kimodo", {})
    pipeline = record.setdefault("pipeline", {})

    canonical = instruction_en.get("canonical", "")
    robot_natural = instruction_en.get("robot_natural", canonical)

    kimodo["validated"] = True
    kimodo["validation_status"] = validation_status_label
    metadata["translation_status"] = "ready_for_translation"
    metadata["validation_status"] = validation_status_label

    record["subtier"] = record.get("subtier") or infer_subtier(record.get("tier", ""))
    record["primitive_tags"] = record.get("primitive_tags") or infer_primitive_tags(canonical)
    record["landmarks"] = record.get("landmarks") or extract_landmarks(record)
    record["translation_protocol"] = {
        "source_bank_frozen": True,
        "kimodo_validation_complete": True,
        "languages": list(LANGUAGES),
        "required_forms_per_language": ["literal", "robot_natural"],
        "required_verification_checks": list(VERIFICATION_FIELDS),
    }
    pipeline.update({
        "stage": "translation_ready",
        "next_required_step": "multilingual_translation",
        "translation_status": "ready_for_translation",
        "english_source_frozen": True,
        "kimodo_validation_complete": True,
    })

    instruction_multilingual = record.setdefault("instruction_multilingual", {})
    instruction_multilingual["en"] = {
        "canonical": canonical,
        "robot_natural": robot_natural,
    }

    for code in LANGUAGES:
        instruction_multilingual.setdefault(code, {"literal": None, "robot_natural": None})
        record[f"translation_{code}"] = empty_translation_entry(
            record["prompt_id"],
            code,
            canonical,
            robot_natural,
            ambiguity_flag=metadata.get("ambiguity_flag", False),
            ambiguity_type=metadata.get("ambiguity_type"),
            requires_scene_context=metadata.get("requires_scene_context", False),
            requires_body_reference=metadata.get("requires_body_reference", False),
            safe_for_direct_execution=metadata.get("safe_for_direct_execution", False),
        )

    return record


def build_manifest_rows(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in corpus:
        canonical = entry.get("instruction_en", {}).get("canonical")
        robot_natural = entry.get("instruction_en", {}).get("robot_natural")
        for code, language_name in LANGUAGES.items():
            block = entry.get(f"translation_{code}", {})
            rows.append({
                "prompt_id": entry["prompt_id"],
                "tier": entry.get("tier"),
                "subtier": entry.get("subtier"),
                "language": code,
                "language_name": language_name,
                "instruction_en": {
                    "canonical": canonical,
                    "robot_natural": robot_natural,
                },
                "translation_target": {
                    "literal": block.get("literal"),
                    "robot_natural": block.get("robot_natural"),
                },
                "context": {
                    "motion_family": entry.get("motion_family"),
                    "primitive_tags": entry.get("primitive_tags", []),
                    "constraint_type": entry.get("constraint_type"),
                    "landmarks": entry.get("landmarks", []),
                    "requires_scene_context": entry.get("metadata", {}).get("requires_scene_context"),
                    "requires_body_reference": entry.get("metadata", {}).get("requires_body_reference"),
                    "safe_for_direct_execution": entry.get("metadata", {}).get("safe_for_direct_execution"),
                },
                "verification": block.get("verification", {field: None for field in VERIFICATION_FIELDS}),
                "review_assignment": {
                    "bilingual_reviewer": block.get("review", {}).get("bilingual_reviewer"),
                    "adjudicator": block.get("review", {}).get("adjudicator"),
                    "status": block.get("review", {}).get("review_status", "pending"),
                },
                "notes": block.get("translation_notes", []),
            })
    return rows


def summarize_translation_corpus(
    corpus: list[dict[str, Any]],
    *,
    assumptions: dict[str, Any],
) -> dict[str, Any]:
    tier_counts = Counter(entry.get("tier") for entry in corpus)
    language_job_counts = {code: len(corpus) for code in LANGUAGES}
    tier_language_job_counts = {
        tier: {code: count for code in LANGUAGES}
        for tier, count in sorted(tier_counts.items())
    }

    status_counts = defaultdict(Counter)
    review_counts = defaultdict(Counter)
    verification_pass_counts = defaultdict(lambda: Counter({field: 0 for field in VERIFICATION_FIELDS}))

    for entry in corpus:
        for code in LANGUAGES:
            block = entry.get(f"translation_{code}", {})
            status_counts[code][block.get("status", "missing")] += 1
            review_counts[code][block.get("review", {}).get("review_status", "missing")] += 1
            verification = block.get("verification", {})
            for field in VERIFICATION_FIELDS:
                if verification.get(field) is True:
                    verification_pass_counts[code][field] += 1

    return {
        "assumptions": assumptions,
        "total_prompts": len(corpus),
        "languages": LANGUAGES,
        "total_translation_jobs": len(corpus) * len(LANGUAGES),
        "required_forms_per_language": 2,
        "required_verification_checks": len(VERIFICATION_FIELDS),
        "tier_counts": dict(sorted(tier_counts.items())),
        "language_job_counts": language_job_counts,
        "tier_language_job_counts": tier_language_job_counts,
        "translation_status_counts": {code: dict(counter) for code, counter in status_counts.items()},
        "review_status_counts": {code: dict(counter) for code, counter in review_counts.items()},
        "verification_pass_counts": {code: dict(counter) for code, counter in verification_pass_counts.items()},
        "next_steps": [
            "Fill literal and robot-natural translations for Hindi, Bengali, and Telugu.",
            "Run bilingual review for action, order, direction, and safety preservation.",
            "Adjudicate disagreements on overlapping subsets.",
            "Merge approved translations back into the release corpus.",
        ],
    }


def update_entry_with_model_translation(
    entry: dict[str, Any],
    *,
    language_code: str,
    literal: str,
    robot_natural: str,
    verification: dict[str, Any],
    translation_notes: list[str],
    model: str,
    translation_method: str,
) -> None:
    block = entry[f"translation_{language_code}"]
    block["literal"] = literal
    block["robot_natural"] = robot_natural
    block["status"] = "model_translated"
    block["translation_notes"] = translation_notes
    for field in VERIFICATION_FIELDS:
        block["verification"][field] = verification.get(field)
    block["review"]["review_status"] = "pending_bilingual_review"
    block["provenance"]["translation_model"] = model
    block["provenance"]["translation_method"] = translation_method

    entry["instruction_multilingual"][language_code]["literal"] = literal
    entry["instruction_multilingual"][language_code]["robot_natural"] = robot_natural

    statuses = [entry[f"translation_{code}"]["status"] for code in LANGUAGES]
    if all(status == "model_translated" for status in statuses):
        entry["metadata"]["translation_status"] = "model_translated_pending_review"
        entry["pipeline"]["stage"] = "translation_generated_pending_review"
        entry["pipeline"]["translation_status"] = "model_translated_pending_review"
        entry["pipeline"]["next_required_step"] = "bilingual_translation_qc"
    else:
        entry["metadata"]["translation_status"] = "translation_in_progress"
        entry["pipeline"]["stage"] = "translation_in_progress"
        entry["pipeline"]["translation_status"] = "translation_in_progress"
        entry["pipeline"]["next_required_step"] = "complete_remaining_language_translations"


def finalize_entry_for_release(entry: dict[str, Any]) -> None:
    review_statuses = []
    for code in LANGUAGES:
        block = entry[f"translation_{code}"]
        literal = block.get("literal")
        robot_natural = block.get("robot_natural")
        if not literal or not robot_natural:
            raise ValueError(f"{entry['prompt_id']} is missing {code} translations.")

        review_status = block.get("review", {}).get("review_status")
        review_statuses.append(review_status)
        entry["instruction_multilingual"][code]["literal"] = literal
        entry["instruction_multilingual"][code]["robot_natural"] = robot_natural

    entry["metadata"]["translation_status"] = "complete"
    entry["pipeline"]["translation_status"] = "complete"
    entry["pipeline"]["stage"] = "multilingual_complete"
    entry["pipeline"]["next_required_step"] = "downstream_multilingual_qc_or_release"
    if all(status == "approved" for status in review_statuses):
        entry["pipeline"]["review_gate"] = "approved"
    else:
        entry["pipeline"]["review_gate"] = "pending_or_partial"


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())
