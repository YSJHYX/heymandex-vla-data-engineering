"""Three source episodes: continuous, D2 split, and D3 internal gap."""

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vla_data.annotation.schema import build_annotation, write_annotation
from vla_data.manifest import SplitConfig, build_training_manifest
from vla_data.quality.report import QUALITY_SCHEMA_NAME, QUALITY_SCHEMA_VERSION
from vla_data.verification import VerificationPolicy, verify_annotations


@pytest.fixture
def export_tree(curated_v1_episode, tmp_path):
    roots = {
        name: tmp_path / name
        for name in ("curated", "quality", "annotation", "verification", "manifest")
    }
    for number in range(3):
        eid = f"episode_{number:06d}"
        dest = roots["curated"] / eid
        shutil.copytree(curated_v1_episode, dest)
        metadata = json.loads((dest / "metadata.json").read_text())
        metadata.update(
            episode_id=eid,
            language_instruction="test",
            segment_offsets=[0, 2, 3] if number == 1 else [0, 3],
        )
        (dest / "metadata.json").write_text(json.dumps(metadata))
        with np.load(dest / "trajectory.npz") as archive:
            values = dict(archive)
        ticks = np.array([0, 1, 3] if number == 1 else [0, 1, 2], dtype=np.int64)
        values["source_tick_index"], values["next_state_tick_index"] = ticks, ticks + 1
        values["robot_qcmd_17d_rad"] += number
        values["robot_qcmd_17d_rad"][2] += 0.5
        np.savez(dest / "trajectory.npz", **values)
        for role, colors in (
            ("head", [(220, 20, 20), (180, 60, 20)]),
            ("right_wrist", [(20, 30, 220), (20, 80, 180)]),
        ):
            for index, color in enumerate(colors):
                Image.new("RGB", (128, 96), color).save(
                    dest / "media" / role / "rgb" / f"{index:06d}.jpg"
                )
        quality = roots["quality"] / eid
        quality.mkdir(parents=True)
        mask = np.array([True, number != 2, True])
        np.save(quality / "quality_mask.npy", mask)
        (quality / "quality_report.json").write_text(
            json.dumps(
                {
                    "schema_name": QUALITY_SCHEMA_NAME,
                    "schema_version": QUALITY_SCHEMA_VERSION,
                    "episode_id": eid,
                    "status": "ACCEPT",
                    "transition_count": 3,
                    "clean_transition_count": int(mask.sum()),
                }
            )
        )
        write_annotation(
            roots["annotation"] / eid / "annotation.json",
            build_annotation(
                episode_id=eid,
                quality_outcome="ACCEPT",
                model_annotation={
                    "provider": "synthetic",
                    "model": "fixture",
                    "prompt_version": "test",
                    "instruction": f"Move the SYNTHETIC block {number}.",
                    "confidence": 0.9,
                    "objects": [],
                    "task_type": None,
                    "uncertainty": None,
                },
            ),
        )
    verify_annotations(
        roots["annotation"],
        roots["quality"],
        roots["verification"],
        policy=VerificationPolicy(0.5),
    )

    def refresh():
        return build_training_manifest(
            roots["curated"],
            roots["quality"],
            roots["annotation"],
            roots["manifest"],
            verification_root=roots["verification"],
            config=SplitConfig(validation_fraction=0.34, seed=17),
        )

    refresh()
    return SimpleNamespace(**roots, output=tmp_path / "export", refresh=refresh)
