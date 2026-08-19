from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import torch

from omg.data.lerobot_dataset import LeRobotG1MotionDataset
from omg.generation.conditions.muril import normalize_hindi_text


CAPTION_MANIFEST_SCHEMA = "omg.bones_hindi_captions.v1"
_KNOWN_MOTION_SUFFIXES = (".csv", ".npy", ".npz", ".pkl", ".pickle")


def canonical_bones_motion_id(value: str) -> str:
    normalized = unicodedata.normalize("NFC", str(value)).strip().replace("\\", "/")
    normalized = normalized.rstrip("/").rsplit("/", 1)[-1]
    lowered = normalized.lower()
    stripped = True
    while stripped:
        stripped = False
        for suffix in _KNOWN_MOTION_SUFFIXES:
            if lowered.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                lowered = lowered[: -len(suffix)]
                stripped = True
                break
    # OMG-Data may split one BONES motion into several immutable episodes,
    # e.g. ``body_check_001__A512__seg0000__part0002``. Caption supervision is
    # keyed to the CSV's source filename before this derived suffix.
    return re.sub(r"__seg\d+(?:__part\d+)?$", "", normalized)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bones_hindi_caption_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Hindi caption manifest does not exist: {manifest_path}")

    metadata: dict[str, Any] | None = None
    entries: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {manifest_path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {manifest_path}:{line_number}")
            if metadata is None and record.get("schema") == CAPTION_MANIFEST_SCHEMA:
                metadata = record
                continue
            entries.append(record)

    if metadata is None:
        raise ValueError(
            f"Hindi caption manifest must begin with schema={CAPTION_MANIFEST_SCHEMA!r}: {manifest_path}"
        )
    if not entries:
        raise ValueError(f"Hindi caption manifest has no motion entries: {manifest_path}")

    index: dict[str, dict[str, Any]] = {}
    owners: dict[str, str] = {}
    for row_number, entry in enumerate(entries, start=2):
        source_motion_id = canonical_bones_motion_id(entry.get("source_motion_id", ""))
        if not source_motion_id:
            raise ValueError(f"Missing source_motion_id in manifest record {row_number}")
        captions = entry.get("captions")
        if not isinstance(captions, list) or not captions:
            raise ValueError(f"Motion {source_motion_id!r} has no caption pairs")
        normalized_captions = []
        seen_caption_ids: set[str] = set()
        for pair_index, pair in enumerate(captions):
            if not isinstance(pair, dict):
                raise ValueError(f"Motion {source_motion_id!r} caption {pair_index} is not an object")
            english = " ".join(str(pair.get("english", "")).split())
            hindi = normalize_hindi_text(pair.get("hindi", ""))
            caption_id = str(pair.get("caption_id", f"{source_motion_id}:natural_desc_{pair_index + 1}"))
            if not english or not hindi:
                raise ValueError(
                    f"Motion {source_motion_id!r} caption {caption_id!r} must contain English and Hindi"
                )
            if caption_id in seen_caption_ids:
                raise ValueError(f"Duplicate caption_id {caption_id!r} for motion {source_motion_id!r}")
            seen_caption_ids.add(caption_id)
            normalized_captions.append({**pair, "caption_id": caption_id, "english": english, "hindi": hindi})
        normalized_entry = {**entry, "source_motion_id": source_motion_id, "captions": normalized_captions}
        aliases = [source_motion_id, *list(entry.get("aliases") or [])]
        for raw_alias in aliases:
            alias = canonical_bones_motion_id(raw_alias)
            if not alias:
                continue
            previous_owner = owners.get(alias)
            if previous_owner is not None and previous_owner != source_motion_id:
                raise ValueError(
                    f"Caption manifest alias {alias!r} belongs to both {previous_owner!r} and {source_motion_id!r}"
                )
            owners[alias] = source_motion_id
            index[alias] = normalized_entry
    expected_motion_count = metadata.get("motion_count")
    if expected_motion_count is not None and int(expected_motion_count) != len(entries):
        raise ValueError(
            f"Hindi caption manifest motion_count={expected_motion_count} does not match {len(entries)} records"
        )
    expected_caption_count = metadata.get("caption_count")
    observed_caption_count = sum(len(entry["captions"]) for entry in entries)
    if expected_caption_count is not None and int(expected_caption_count) != observed_caption_count:
        raise ValueError(
            "Hindi caption manifest caption_count="
            f"{expected_caption_count} does not match {observed_caption_count} pairs"
        )
    return metadata, index, _file_sha256(manifest_path)


class BonesSeedHindiLeRobotDataset(LeRobotG1MotionDataset):
    """BONES-SEED slice of OMG-Data with an aligned Hindi caption overlay."""

    def __init__(
        self,
        *args: Any,
        caption_manifest: str | Path,
        source_dataset_prefix: str = "bones_seed",
        strict_caption_coverage: bool = True,
        caption_selection_seed: int = 1234,
        **kwargs: Any,
    ) -> None:
        self.caption_manifest_path = str(Path(caption_manifest).expanduser())
        (
            self.caption_manifest_metadata,
            self._hindi_caption_index,
            self.caption_manifest_sha256,
        ) = load_bones_hindi_caption_manifest(self.caption_manifest_path)
        self.source_dataset_prefix = str(source_dataset_prefix).strip()
        self.strict_caption_coverage = bool(strict_caption_coverage)
        self.caption_selection_seed = int(caption_selection_seed)
        if not self.source_dataset_prefix:
            raise ValueError("source_dataset_prefix must be non-empty")
        super().__init__(*args, **kwargs)
        recorded_revision = self.caption_manifest_metadata.get("omg_data_revision")
        if recorded_revision is not None and str(recorded_revision) != self.revision:
            raise ValueError(
                "Hindi caption manifest OMG-Data revision mismatch: "
                f"manifest={recorded_revision!r} dataset={self.revision!r}"
            )

    def _load_episodes(self, info: dict[str, Any]) -> list[dict[str, Any]]:
        episodes = super()._load_episodes(info)
        accepted_source_names = {
            self.source_dataset_prefix,
            f"{self.source_dataset_prefix}_{self.split}",
        }
        selected: list[dict[str, Any]] = []
        missing: list[str] = []
        matched_motion_ids: set[str] = set()
        for episode in episodes:
            if str(episode.get("source_dataset", "")) not in accepted_source_names:
                continue
            source_id = canonical_bones_motion_id(episode.get("sequence_name", ""))
            entry = self._hindi_caption_index.get(source_id)
            if entry is None:
                missing.append(str(episode.get("sequence_name", "")))
                continue
            matched_motion_ids.add(str(entry["source_motion_id"]))
            selected.append(
                {
                    **episode,
                    "_hindi_caption_pairs": tuple(entry["captions"]),
                    "_hindi_source_motion_id": str(entry["source_motion_id"]),
                    "_hindi_group_id": str(entry.get("group_id", entry["source_motion_id"])),
                }
            )
        if missing and self.strict_caption_coverage:
            examples = sorted(set(missing))[:20]
            raise ValueError(
                f"Hindi caption manifest is missing {len(missing)} {self.source_dataset_prefix!r} "
                f"episodes in split {self.split!r}; examples={examples}"
            )
        if not selected:
            raise ValueError(
                f"No {self.source_dataset_prefix!r} episodes with Hindi captions found in split {self.split!r}"
            )
        print(
            "[INFO] BonesSeedHindiLeRobotDataset "
            f"split={self.split} episodes={len(selected)} unique_motions={len(matched_motion_ids)} "
            f"unresolved={len(missing)} manifest_sha256={self.caption_manifest_sha256}"
        )
        return selected

    def _caption_pair_index(self, sample: dict[str, Any], pairs: tuple[dict[str, Any], ...]) -> int:
        if self.training:
            return int(torch.randint(0, len(pairs), (1,)).item())
        window_start = int(sample.get("meta", {}).get("window_start", 0))
        episode_index = int(sample.get("meta", {}).get("lerobot_episode_index", 0))
        payload = f"{self.caption_selection_seed}:{self.split}:{episode_index}:{window_start}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_info = self.samples[int(idx)]
        pairs = tuple(sample_info.get("_hindi_caption_pairs") or ())
        if not pairs:
            raise RuntimeError(f"Dataset sample {idx} has no aligned Hindi caption pairs")
        sample = super().__getitem__(int(idx))
        pair = pairs[self._caption_pair_index(sample, pairs)]
        hindi = str(pair["hindi"])
        english = str(pair["english"])
        sample["caption"] = hindi
        sample["teacher_caption"] = english
        sample["has_text"] = torch.tensor(bool(hindi), dtype=torch.bool)
        sample["meta"] = {
            **sample["meta"],
            "caption_id": str(pair["caption_id"]),
            "caption_language": "hi",
            "caption_machine_translated": True,
            "teacher_caption_language": "en",
            "hindi_source_motion_id": str(sample_info["_hindi_source_motion_id"]),
            "hindi_group_id": str(sample_info["_hindi_group_id"]),
            "caption_manifest_sha256": self.caption_manifest_sha256,
            "source_csv_sha256": str(
                self.caption_manifest_metadata.get("provenance", {}).get("source_csv_sha256", "")
            ),
        }
        return sample
