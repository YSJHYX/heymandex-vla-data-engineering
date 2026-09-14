"""Independent persisted-manifest validator that rereads every source artifact."""

import json
from dataclasses import dataclass
from pathlib import Path

from vla_data.batch.discovery import canonical_episode_id, discover_curated_episodes
from vla_data.manifest.eligibility import inspect_episode
from vla_data.manifest.schema import OUTPUT_FILES
from vla_data.manifest.split import SplitConfig, assign_splits
from vla_data.manifest.summary import summarize


@dataclass(frozen=True)
class ManifestValidationResult:
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


def validate_training_manifest(output_root: str | Path) -> ManifestValidationResult:
    errors = []
    try:
        _validate(Path(output_root), errors)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
    ) as exc:
        errors.append(f"MANIFEST_INVALID: {exc}")
    return ManifestValidationResult(tuple(errors))


def _validate(root: Path, errors: list[str]) -> None:
    groups = {}
    for name in OUTPUT_FILES[:-1]:
        rows = [
            json.loads(line)
            for line in (root / name).read_text().splitlines()
            if line.strip()
        ]
        ids = [canonical_episode_id(row["episode_id"]) for row in rows]
        if ids != sorted(set(ids)):
            errors.append(f"{name}: episode IDs must be unique and sorted")
        groups[name] = rows
    episodes, train, val, excluded = (groups[name] for name in OUTPUT_FILES[:-1])
    train_ids = {r["episode_id"] for r in train}
    val_ids = {r["episode_id"] for r in val}
    if train_ids & val_ids:
        errors.append("TRAIN_VAL_EPISODE_OVERLAP")
    if sorted(train + val, key=lambda r: r["episode_id"]) != episodes:
        errors.append("train/val must be exact partitions of episodes.jsonl")
    if any(r.get("split") != "train" for r in train) or any(
        r.get("split") != "val" for r in val
    ):
        errors.append("INVALID_SPLIT")
    records = sorted(episodes + excluded, key=lambda r: r["episode_id"])
    policies = {
        json.dumps(r.get("verification_policy"), sort_keys=True)
        for r in records
        if r.get("verification_policy") is not None
    }
    if len(policies) > 1:
        errors.append("MIXED_VERIFICATION_POLICIES")
    ids = [r["episode_id"] for r in records]
    if len(ids) != len(set(ids)):
        errors.append("duplicate eligible/excluded episode ID")
    summary = json.loads((root / OUTPUT_FILES[-1]).read_text())
    roots = tuple(
        Path(summary["input_roots"][name]).resolve()
        for name in ("curated", "quality", "annotation", "verification")
    )
    discoveries = [discover_curated_episodes(path) for path in roots]
    if any(d.issues for d in discoveries):
        errors.append("noncanonical source episode directory")
    discovered = sorted({item.episode_id for d in discoveries for item in d.episodes})
    selection = summary["episode_selection"]
    if selection is not None:
        canonical_episode_id(selection)
        discovered = [value for value in discovered if value == selection]
    if discovered != ids:
        errors.append("manifest must account for every selected source episode")
    config = SplitConfig(**summary["split_config"])
    splits = assign_splits([r["episode_id"] for r in episodes], config)
    for record in records:
        eid = record["episode_id"]
        actual = inspect_episode(
            eid,
            roots[0] / eid,
            roots[1] / eid,
            roots[2] / eid / "annotation.json",
            roots[3] / eid / "verification.json",
        )
        actual["split"] = splits.get(eid)
        if actual != record:
            errors.append(f"{eid}: record differs from current source resolution")
        if record in episodes:
            if (
                actual["reason"]
                or not actual["final_instruction"]
                or actual["transition_count_selected"] <= 0
            ):
                errors.append(f"{eid}: TRAINING_INELIGIBLE: {actual['reason']}")
            if record["split"] not in {"train", "val"}:
                errors.append(f"{eid}: INVALID_SPLIT")
        elif not actual["reason"] or record["split"] is not None:
            errors.append(f"{eid}: invalid exclusion")
    for key, value in summarize(records).items():
        if summary.get(key) != value:
            errors.append(f"summary mismatch: {key}")
    from vla_data.manifest.builder import input_fingerprint

    if input_fingerprint(ids, roots, config) != summary["input_fingerprint"]:
        errors.append("INPUT_FINGERPRINT_CHANGED")
