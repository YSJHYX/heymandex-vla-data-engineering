"""Atomic local-only LeRobot export over direct D2+D3 or optional manifest runs."""

import json
import tempfile
import time
from pathlib import Path

from vla_data.batch.runner import _publish_directory
from vla_data.export.plan import make_mainline_plan, make_plan, plan_summary
from vla_data.export.runtime import worker_call
from vla_data.export.validator import validate_lerobot_export
from vla_data.manifest import SplitConfig


def export_lerobot(
    manifest_root: str | Path | None,
    output_root: str | Path,
    *,
    curated_root: str | Path | None = None,
    quality_root: str | Path | None = None,
    episode: str | None = None,
    validation_fraction: float = 0.0,
    split_seed: int = 0,
    dataset_name: str = "vla-local",
    lerobot_python: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    started = time.monotonic()
    direct = curated_root is not None or quality_root is not None
    if manifest_root is not None and direct:
        raise ValueError(
            "choose direct --curated-root/--quality-root or --manifest-root"
        )
    if manifest_root is None and not (
        curated_root is not None and quality_root is not None
    ):
        raise ValueError("direct export requires both curated_root and quality_root")
    split_config = SplitConfig(
        validation_fraction=validation_fraction,
        seed=split_seed,
    )
    plan = (
        make_mainline_plan(
            curated_root,
            quality_root,
            dataset_name,
            split_config=split_config,
            episode=episode,
        )
        if direct
        else make_plan(manifest_root, dataset_name)
    )
    output = Path(output_root).absolute()
    if output.is_symlink():
        raise ValueError("export output must not be a symlink")
    output = output.resolve()
    inputs = [Path(p).resolve() for p in plan["input_roots"].values()]
    if manifest_root is not None:
        inputs.append(Path(manifest_root).resolve())
    if any(output == p or output in p.parents or p in output.parents for p in inputs):
        raise ValueError("export output must be separate from every input tree")
    summary = {
        **plan_summary(plan),
        "schema_name": "vla_lerobot_export",
        "schema_version": 1,
        "source_mode": plan.get("source_mode", "OPTIONAL_SEMANTIC_MANIFEST"),
        "input_roots": plan["input_roots"],
        "episode": plan.get("episode"),
        "split_config": plan.get("split_config"),
        "manifest_root": plan.get("manifest_root"),
        "dataset_name": dataset_name,
        "camera_transforms": plan["camera_transforms"],
    }
    if dry_run:
        return {**summary, "status": "DRY_RUN"}
    runtime = worker_call("probe", {}, lerobot_python)
    if output.exists():
        if not (output / "export_summary.json").is_file():
            raise ValueError("refusing to replace unmanaged output directory")
        if not force:
            valid = validate_lerobot_export(
                output, lerobot_python=lerobot_python, manifest_root=manifest_root
            )
            if (
                valid.passed
                and json.loads((output / "export_summary.json").read_text())[
                    "fingerprint"
                ]
                == plan["fingerprint"]
                and json.loads((output / "export_summary.json").read_text())["runtime"]
                == runtime
            ):
                return {
                    **summary,
                    "status": "SKIPPED",
                    "wall_time_s": time.monotonic() - started,
                }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.staging-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        worker_call(
            "write", {"plan": plan, "output_root": str(staging)}, lerobot_python
        )
        summary["runtime"] = runtime
        (staging / "export_summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n"
        )
        (staging / "export_provenance.jsonl").write_text(
            "".join(
                json.dumps(
                    {k: v for k, v in run.items() if k != "images"},
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
                for run in plan["runs"]
            )
        )
        for split in ("train", "val"):
            split_runs = [run for run in plan["runs"] if run["split"] == split]
            if not split_runs:
                continue
            (staging / split / "meta" / "source_provenance.jsonl").write_text(
                "".join(
                    json.dumps(
                        {key: value for key, value in run.items() if key != "images"},
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    + "\n"
                    for run in split_runs
                )
            )
        valid = validate_lerobot_export(
            staging, lerobot_python=lerobot_python, manifest_root=manifest_root
        )
        if not valid.passed:
            raise ValueError(f"independent export validation failed: {valid.errors}")
        current = (
            make_mainline_plan(
                curated_root,
                quality_root,
                dataset_name,
                split_config=split_config,
                episode=episode,
            )
            if direct
            else make_plan(manifest_root, dataset_name)
        )
        if current["fingerprint"] != plan["fingerprint"]:
            raise ValueError("source changed during export")
        (staging / "export_validation.json").write_text(
            json.dumps(valid.evidence, sort_keys=True, indent=2) + "\n"
        )
        _publish_directory(staging, output)
    return {**summary, "status": "BUILT", "wall_time_s": time.monotonic() - started}
