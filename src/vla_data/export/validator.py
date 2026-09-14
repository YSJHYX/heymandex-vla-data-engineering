"""Independent source-plan reconstruction and official on-disk LeRobot reload."""

import json
from dataclasses import dataclass
from pathlib import Path

from vla_data.export.plan import make_plan
from vla_data.export.runtime import worker_call


@dataclass(frozen=True)
class ExportValidation:
    errors: tuple[str, ...]
    evidence: dict

    @property
    def passed(self) -> bool:
        return not self.errors


def validate_lerobot_export(
    output_root: str | Path,
    *,
    lerobot_python: str | Path | None = None,
    manifest_root: str | Path | None = None,
) -> ExportValidation:
    try:
        root = Path(output_root).resolve()
        summary = json.loads((root / "export_summary.json").read_text())
        if (
            summary.get("schema_name") != "vla_lerobot_export"
            or summary.get("schema_version") != 1
        ):
            raise ValueError("unsupported export summary schema")
        if summary.get("runtime", {}).get("codebase_version") != "v2.1":
            raise ValueError("missing or unsupported writer runtime identity")
        plan = make_plan(
            manifest_root or summary["manifest_root"], summary["dataset_name"]
        )
        if summary["fingerprint"] != plan["fingerprint"]:
            raise ValueError("export fingerprint is stale")
        provenance = [
            json.loads(line)
            for line in (root / "export_provenance.jsonl").read_text().splitlines()
        ]
        expected = [{k: v for k, v in r.items() if k != "images"} for r in plan["runs"]]
        if provenance != expected:
            raise ValueError("export provenance differs from contiguous source runs")
        train = {r["source_episode_id"] for r in provenance if r["split"] == "train"}
        val = {r["source_episode_id"] for r in provenance if r["split"] == "val"}
        if train & val:
            raise ValueError("source train/val overlap")
        evidence = worker_call(
            "validate", {"plan": plan, "output_root": str(root)}, lerobot_python
        )
        return ExportValidation((), evidence)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return ExportValidation((str(exc),), {})
