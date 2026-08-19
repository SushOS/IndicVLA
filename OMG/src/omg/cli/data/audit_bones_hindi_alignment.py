from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence
import unicodedata

from omg.data.hindi_caption_sidecar import (
    canonical_bones_motion_id,
    load_bones_hindi_caption_manifest,
)


FORMAT = "omg.bones_hindi_alignment_audit.v1"
DEFAULT_REPO_ID = "THU-MARS/OMG-Data"
DEFAULT_REVISION = "6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7"
DEFAULT_SPLITS = ("train", "val", "test")
_REQUIRED_EPISODE_COLUMNS = ("episode_index", "omg/dataset", "omg/source_id")


def _load_parquet_runtime() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "BONES Hindi alignment auditing requires pyarrow; "
            "install OMG with `pip install -e '.[data]'`."
        ) from exc
    return pq


def _resolve_dataset_root(
    dataset_root: Path | None,
    *,
    repo_id: str,
    revision: str,
) -> Path:
    if dataset_root is not None:
        root = Path(dataset_root).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"OMG-Data root does not exist: {root}")
        return root

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Remote OMG-Data metadata resolution requires huggingface-hub; "
            "install OMG with `pip install -e '.[data]'`."
        ) from exc

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[
                "meta/info.json",
                "meta/episodes/*/*.parquet",
                "meta/episodes/**/*.parquet",
            ],
        )
    )


def _parse_split_range(value: Any, *, split: str) -> tuple[int, int]:
    if isinstance(value, str):
        parts = value.split(":")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        parts = value
    else:
        raise ValueError(f"Invalid episode range for split {split!r}: {value!r}")
    if len(parts) != 2:
        raise ValueError(f"Invalid episode range for split {split!r}: {value!r}")
    try:
        start, end = (int(parts[0]), int(parts[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid episode range for split {split!r}: {value!r}") from exc
    if start < 0 or end <= start:
        raise ValueError(f"Invalid episode range for split {split!r}: {start}:{end}")
    return start, end


def _load_split_ranges(dataset_root: Path) -> dict[str, tuple[int, int]]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return {}
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid OMG-Data metadata JSON: {info_path}") from exc
    raw_splits = info.get("splits", {})
    if not isinstance(raw_splits, dict):
        raise ValueError(f"OMG-Data info.json has invalid splits metadata: {info_path}")
    return {
        str(split): _parse_split_range(value, split=str(split))
        for split, value in raw_splits.items()
    }


def _infer_split(
    episode_index: int,
    split_ranges: dict[str, tuple[int, int]],
    *,
    parquet_path: Path,
) -> str:
    matches = [
        split
        for split, (start, end) in split_ranges.items()
        if start <= episode_index < end
    ]
    if len(matches) != 1:
        raise ValueError(
            "Cannot resolve a unique split for episode metadata without `omg/split`: "
            f"episode_index={episode_index} matches={matches} parquet={parquet_path}"
        )
    return matches[0]


def _normalize_splits(splits: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(split).strip() for split in splits)
    if not normalized or any(not split for split in normalized):
        raise ValueError("At least one non-empty split is required")
    duplicates = sorted({split for split in normalized if normalized.count(split) > 1})
    if duplicates:
        raise ValueError(f"Duplicate requested splits: {duplicates}")
    return normalized


def _manifest_inventory(index: dict[str, dict[str, Any]]) -> tuple[int, int]:
    entries_by_source: dict[str, set[int]] = defaultdict(set)
    entries: dict[str, dict[str, Any]] = {}
    for entry in index.values():
        source_motion_id = str(entry["source_motion_id"])
        entries_by_source[source_motion_id].add(id(entry))
        entries[source_motion_id] = entry
    duplicates = sorted(
        source_motion_id
        for source_motion_id, entry_ids in entries_by_source.items()
        if len(entry_ids) > 1
    )
    if duplicates:
        raise ValueError(
            "Duplicate manifest source_motion_id resolution through aliases; "
            f"examples={duplicates[:20]}"
        )
    caption_owners: dict[str, str] = {}
    caption_pair_count = 0
    for source_motion_id, entry in entries.items():
        for caption in entry["captions"]:
            caption_id = str(caption["caption_id"])
            previous_owner = caption_owners.get(caption_id)
            if previous_owner is not None and previous_owner != source_motion_id:
                raise ValueError(
                    f"Duplicate caption_id resolution {caption_id!r} for motions "
                    f"{previous_owner!r} and {source_motion_id!r}"
                )
            caption_owners[caption_id] = source_motion_id
            caption_pair_count += 1
    return len(entries_by_source), caption_pair_count


def _write_json_atomically(path: Path, value: dict[str, Any], *, overwrite: bool) -> None:
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
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output appeared while preparing audit summary: {path}")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def audit_bones_hindi_alignment(
    caption_manifest: Path,
    *,
    dataset_root: Path | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    source_dataset_prefix: str = "bones_seed",
    splits: Sequence[str] = DEFAULT_SPLITS,
    output_json: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Audit BONES-SEED episode/caption alignment using metadata parquet only."""

    caption_manifest = Path(caption_manifest).expanduser()
    repo_id = str(repo_id).strip()
    revision = str(revision).strip()
    source_dataset_prefix = str(source_dataset_prefix).strip()
    requested_splits = _normalize_splits(splits)
    if not repo_id:
        raise ValueError("repo_id must be non-empty")
    if not revision:
        raise ValueError("revision must be pinned and non-empty")
    if not source_dataset_prefix:
        raise ValueError("source_dataset_prefix must be non-empty")
    if (
        output_json is not None
        and Path(output_json).expanduser().resolve() == caption_manifest.resolve()
    ):
        raise ValueError("output_json must not overwrite the caption manifest")

    metadata, caption_index, manifest_sha256 = load_bones_hindi_caption_manifest(caption_manifest)
    recorded_revision = str(metadata.get("omg_data_revision", "")).strip()
    if recorded_revision != revision:
        raise ValueError(
            "Hindi caption manifest OMG-Data revision mismatch: "
            f"manifest={recorded_revision!r} audit={revision!r}"
        )
    manifest_motion_count, manifest_caption_pair_count = _manifest_inventory(caption_index)
    resolved_root = _resolve_dataset_root(
        dataset_root,
        repo_id=repo_id,
        revision=revision,
    )
    episode_root = resolved_root / "meta" / "episodes"
    parquet_paths = sorted(episode_root.rglob("*.parquet")) if episode_root.is_dir() else []
    if not parquet_paths:
        raise FileNotFoundError(f"No OMG-Data episode metadata parquet files under {episode_root}")

    pq = _load_parquet_runtime()
    split_ranges = _load_split_ranges(resolved_root)
    accumulators: dict[str, dict[str, Any]] = {
        split: {
            "episodes": 0,
            "motion_ids": set(),
            "caption_pairs": set(),
            "group_ids": set(),
        }
        for split in requested_splits
    }
    seen_episode_indices: dict[int, Path] = {}
    caption_owners: dict[str, str] = {}
    unresolved: list[dict[str, Any]] = []
    scanned_episode_rows = 0
    selected_episode_rows = 0

    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        available_columns = set(parquet_file.schema_arrow.names)
        missing_columns = sorted(set(_REQUIRED_EPISODE_COLUMNS) - available_columns)
        if missing_columns:
            raise ValueError(
                f"OMG episode parquet is missing required columns {missing_columns}: {parquet_path}"
            )
        columns = list(_REQUIRED_EPISODE_COLUMNS)
        if "omg/split" in available_columns:
            columns.append("omg/split")
        rows = pq.read_table(parquet_path, columns=columns).to_pylist()
        for row in rows:
            scanned_episode_rows += 1
            try:
                episode_index = int(row["episode_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid episode_index={row.get('episode_index')!r} in {parquet_path}"
                ) from exc
            previous_path = seen_episode_indices.get(episode_index)
            if previous_path is not None:
                raise ValueError(
                    "Duplicate episode_index in OMG episode metadata: "
                    f"episode_index={episode_index} files={[str(previous_path), str(parquet_path)]}"
                )
            seen_episode_indices[episode_index] = parquet_path

            source_dataset = str(row.get("omg/dataset") or "").strip()
            if source_dataset != source_dataset_prefix and not source_dataset.startswith(
                f"{source_dataset_prefix}_"
            ):
                continue
            raw_split = str(row.get("omg/split") or "").strip()
            split = raw_split or _infer_split(
                episode_index,
                split_ranges,
                parquet_path=parquet_path,
            )
            if split not in accumulators:
                continue

            if source_dataset not in {
                source_dataset_prefix,
                f"{source_dataset_prefix}_{split}",
            }:
                continue
            selected_episode_rows += 1

            raw_source_id = str(row.get("omg/source_id") or "").strip()
            source_alias = canonical_bones_motion_id(raw_source_id)
            entry = caption_index.get(source_alias)
            if not source_alias or entry is None:
                unresolved.append(
                    {
                        "episode_index": episode_index,
                        "split": split,
                        "omg/source_id": raw_source_id,
                    }
                )
                continue

            source_motion_id = str(entry["source_motion_id"])
            group_id = " ".join(
                unicodedata.normalize("NFC", str(entry.get("group_id", ""))).split()
            )
            if not group_id:
                raise ValueError(
                    f"Resolved manifest motion {source_motion_id!r} has no non-empty group_id"
                )
            accumulator = accumulators[split]
            accumulator["episodes"] += 1
            accumulator["motion_ids"].add(source_motion_id)
            accumulator["group_ids"].add(group_id)
            for caption in entry["captions"]:
                caption_id = str(caption["caption_id"])
                previous_owner = caption_owners.get(caption_id)
                if previous_owner is not None and previous_owner != source_motion_id:
                    raise ValueError(
                        f"Duplicate caption_id resolution {caption_id!r} for motions "
                        f"{previous_owner!r} and {source_motion_id!r}"
                    )
                caption_owners[caption_id] = source_motion_id
                accumulator["caption_pairs"].add((source_motion_id, caption_id))

    if unresolved:
        raise ValueError(
            f"Hindi caption manifest is missing resolution for {len(unresolved)} "
            f"{source_dataset_prefix!r} episodes; examples={unresolved[:20]}"
        )
    if selected_episode_rows == 0:
        raise ValueError(
            f"No {source_dataset_prefix!r} episodes found in requested splits {list(requested_splits)}"
        )

    group_splits: dict[str, list[str]] = defaultdict(list)
    for split, accumulator in accumulators.items():
        for group_id in accumulator["group_ids"]:
            group_splits[group_id].append(split)
    leaking_groups = {
        group_id: split_names
        for group_id, split_names in sorted(group_splits.items())
        if len(split_names) > 1
    }
    if leaking_groups:
        examples = dict(list(leaking_groups.items())[:20])
        raise ValueError(
            f"group_id overlap across train/val/test splits: "
            f"groups={len(leaking_groups)} examples={examples}"
        )

    split_summary = {
        split: {
            "episodes": int(accumulator["episodes"]),
            "unique_motions": len(accumulator["motion_ids"]),
            "caption_pairs": len(accumulator["caption_pairs"]),
            "group_ids": len(accumulator["group_ids"]),
        }
        for split, accumulator in accumulators.items()
    }
    all_motion_ids = set().union(*(value["motion_ids"] for value in accumulators.values()))
    all_caption_pairs = set().union(*(value["caption_pairs"] for value in accumulators.values()))
    all_group_ids = set().union(*(value["group_ids"] for value in accumulators.values()))
    report: dict[str, Any] = {
        "format": FORMAT,
        "status": "pass",
        "dataset": {
            "repo_id": repo_id,
            "revision": revision,
            "root": str(resolved_root),
            "episode_parquet_files": len(parquet_paths),
            "scanned_episode_rows": scanned_episode_rows,
            "source_dataset_prefix": source_dataset_prefix,
        },
        "caption_manifest": {
            "path": str(caption_manifest),
            "sha256": manifest_sha256,
            "schema": metadata.get("schema"),
            "motions": manifest_motion_count,
            "caption_pairs": manifest_caption_pair_count,
            "aliases": len(caption_index),
        },
        "requested_splits": list(requested_splits),
        "splits": split_summary,
        "totals": {
            "episodes": selected_episode_rows,
            "unique_motions": len(all_motion_ids),
            "caption_pairs": len(all_caption_pairs),
            "group_ids": len(all_group_ids),
            "unresolved_episodes": 0,
            "leaking_group_ids": 0,
        },
    }
    if output_json is not None:
        _write_json_atomically(Path(output_json).expanduser(), report, overwrite=overwrite)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit BONES-SEED Hindi caption alignment and split leakage from "
            "OMG-Data episode metadata only."
        )
    )
    parser.add_argument("--caption-manifest", "--input-jsonl", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Local OMG-Data root. If omitted, only meta/info.json and meta/episodes are downloaded.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--source-dataset-prefix", default="bones_seed")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    report = audit_bones_hindi_alignment(
        args.caption_manifest,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        revision=args.revision,
        source_dataset_prefix=args.source_dataset_prefix,
        splits=args.splits,
        output_json=args.output_json,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
