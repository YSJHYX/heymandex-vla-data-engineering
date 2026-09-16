"""Dataset-wide reference manifest generation with content-aware resume."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from vla_data.batch.discovery import discover_curated_episodes
from vla_data.manifest.eligibility import inspect_training_units
from vla_data.manifest.schema import OUTPUT_FILES, SCHEMA_VERSION
from vla_data.manifest.split import SplitConfig, assign_splits
from vla_data.manifest.summary import summarize


@dataclass(frozen=True)
class ManifestBuildResult:
    status: str
    summary: dict
    records: tuple[dict, ...]


def input_fingerprint(
    ids: list[str], roots: tuple[Path, Path, Path, Path], config: SplitConfig
) -> str:
    """Hash small artifacts; use stat identity for large trajectory data."""
    entries = []
    curated, quality, annotation, verification = roots
    for episode in ids:
        for path in (
            curated / episode / "metadata.json",
            curated / episode / "cleaning_report.json",
            quality / episode / "quality_report.json",
            quality / episode / "quality_mask.npy",
            annotation / episode / "annotation.json",
            verification / episode / "verification.json",
        ):
            try:
                identity = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                identity = type(exc).__name__
            entries.append([str(path), identity])
        path = curated / episode / "trajectory.npz"
        try:
            stat = path.stat()
            identity = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]
        except FileNotFoundError:
            identity = "MISSING"
        entries.append([str(path), identity])
    payload = [SCHEMA_VERSION, ids, [str(p) for p in roots], asdict(config), entries]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_training_manifest(
    curated_root: str | Path,
    quality_root: str | Path,
    annotation_root: str | Path,
    output_root: str | Path,
    *,
    verification_root: str | Path,
    episode: str | None = None,
    config: SplitConfig | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> ManifestBuildResult:
    config = config or SplitConfig()
    roots = tuple(
        Path(p).resolve()
        for p in (curated_root, quality_root, annotation_root, verification_root)
    )
    output = Path(output_root).resolve()
    if any(
        output == root or output in root.parents or root in output.parents
        for root in roots
    ):
        raise ValueError("manifest output must be separate from all input trees")
    # Union includes orphaned quality/annotation episodes, with missing Curated
    # recorded as an exclusion rather than silently dropping their existence.
    discoveries = [discover_curated_episodes(root) for root in roots]
    if any(d.issues for d in discoveries):
        raise ValueError("noncanonical episode directories in manifest input roots")
    ids = sorted({item.episode_id for d in discoveries for item in d.episodes})
    if episode is not None:
        from vla_data.batch.discovery import canonical_episode_id

        episode = canonical_episode_id(episode)
        if episode not in ids:
            raise ValueError(f"selected episode not found: {episode}")
        ids = [episode]
    fingerprint = input_fingerprint(ids, roots, config)
    records = [
        record
        for value in ids
        for record in inspect_training_units(
            value,
            roots[0] / value,
            roots[1] / value,
            roots[2] / value / "annotation.json",
            roots[3] / value / "verification.json",
        )
    ]
    splits = assign_splits(
        sorted(
            {
                r.get("source_episode_id", r["episode_id"])
                for r in records
                if not r["reason"]
            }
        ),
        config,
    )
    policies = {
        json.dumps(r["verification_policy"], sort_keys=True)
        for r in records
        if r["verification_policy"] is not None
    }
    if len(policies) > 1:
        raise ValueError("mixed verification policies; verify the whole dataset first")
    for record in records:
        assigned = splits.get(record.get("source_episode_id", record["episode_id"]))
        if "semantic_segment_id" in record:
            record["source_split"] = assigned
        record["split"] = assigned if not record["reason"] else None
    summary = summarize(records)
    summary.update(
        {
            "input_fingerprint": fingerprint,
            "split_config": asdict(config),
            "input_roots": dict(
                zip(
                    ("curated", "quality", "annotation", "verification"),
                    map(str, roots),
                    strict=True,
                )
            ),
            "episode_selection": episode,
        }
    )
    if dry_run:
        return ManifestBuildResult("DRY_RUN", summary, tuple(records))
    if not force and (output / "dataset_manifest_summary.json").is_file():
        from vla_data.manifest.validator import validate_training_manifest

        validation = validate_training_manifest(output)
        if validation.passed:
            previous = json.loads(
                (output / "dataset_manifest_summary.json").read_text()
            )
            if previous == summary:
                return ManifestBuildResult("SKIPPED", summary, tuple(records))
    output.mkdir(parents=True, exist_ok=True)

    def unit_id(record: dict) -> str:
        return record.get("training_unit_id", record["episode_id"])

    groups = {
        "episodes.jsonl": [r for r in records if not r["reason"]],
        "train.jsonl": [r for r in records if r["split"] == "train"],
        "val.jsonl": [r for r in records if r["split"] == "val"],
        "excluded.jsonl": [r for r in records if r["reason"]],
    }
    groups = {name: sorted(values, key=unit_id) for name, values in groups.items()}
    # Stage the complete set; publish summary last. Partial/interrupted sets fail
    # independent validation and cannot satisfy resume.
    with tempfile.TemporaryDirectory(prefix=".manifest-", dir=output) as temporary:
        staging = Path(temporary)
        for name, values in groups.items():
            (staging / name).write_text(
                "".join(
                    json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                    for r in values
                ),
                encoding="utf-8",
            )
        (staging / "dataset_manifest_summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if input_fingerprint(ids, roots, config) != fingerprint:
            raise ValueError("inputs changed during manifest build; retry")
        for name in OUTPUT_FILES:
            os.replace(staging / name, output / name)
    return ManifestBuildResult("BUILT", summary, tuple(records))
