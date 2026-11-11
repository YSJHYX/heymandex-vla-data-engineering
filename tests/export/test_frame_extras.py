"""Depth/event alignment, cut boundaries, cumulative rebasing and integrity."""

import hashlib
import json
import os
import shutil

import numpy as np
import pytest
from PIL import Image

from tests.export.test_mainline import _tree
from vla_data.export import export_lerobot, validate_lerobot_export
from vla_data.export.frame_extras import enrich_plan, validate_package
from vla_data.export.plan import make_mainline_plan
from vla_data.export.runtime import worker_call
from vla_data.publish.local import (
    InvalidDatasetRootError,
    build_canonical_manifest,
    validate_dataset_root,
)
from vla_data.publish.merge import plan_logical_merge, rebuild_logical_dataset


def _manifest(curated, path):
    """Sparse annotations: filtered tick 1 must never shift onto exported tick 2."""
    calibration = path.parent / "calibration.json"
    calibration.write_text(json.dumps({"K": [[100, 0, 64], [0, 100, 48], [0, 0, 1]]}))
    pixels = np.arange(96 * 128, dtype=np.uint16).reshape(96, 128)
    depth = path.parent / "depth.png"
    Image.fromarray(pixels).save(depth)
    rows = []
    for episode in sorted(curated.iterdir()):
        with np.load(episode / "trajectory.npz") as trajectory:
            for index, tick in enumerate(trajectory["source_tick_index"]):
                action_ts = int(trajectory["action_timestamp_ns"][index])
                row = {
                    "source_episode_id": episode.name,
                    "source_tick_index": int(tick),
                    "success": 1 if tick == 2 else 0,
                    "success_source": "human:test-fixture",
                }
                for name, role in (
                    ("depth_top", "head"),
                    ("depth_hand", "right_wrist"),
                ):
                    row[name] = {
                        "path": depth.name,
                        "timestamp_ns": action_ts - 1,
                        "rgb_timestamp_ns": action_ts - 1,
                        "max_skew_ns": 0,
                        "rgb_frame_index": int(
                            trajectory[f"{role}_rgb_frame_index"][index]
                        ),
                        "scale_m_per_unit": 0.001,
                        "invalid_value": 0,
                        "calibration": calibration.name,
                        "aligned_to_rgb": True,
                    }
                rows.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return pixels


def test_source_ticks_cut_success_unknown_and_asset_fingerprints(
    curated_v1_episode, tmp_path
):
    curated, quality, _ = _tree(curated_v1_episode, tmp_path, count=1)
    manifest = tmp_path / "frames.jsonl"
    _manifest(curated, manifest)
    plan = make_mainline_plan(curated, quality)
    enriched = enrich_plan(plan, manifest)
    rows = list(enriched["frame_extras"]["runs"].values())
    assert [[r["source_tick_index"] for r in run] for run in rows] == [[0], [2]]
    assert [r[0]["success"] for r in rows] == [0, 1]
    assert enriched["frame_extras"]["ignored_source_ticks"] == [["episode_000000", 1]]
    labels = [json.loads(line) for line in manifest.read_text().splitlines()]
    labels[1]["success"], labels[2]["success"] = 1, 0
    manifest.write_text("".join(json.dumps(label) + "\n" for label in labels))
    cut_event = enrich_plan(plan, manifest)
    assert all(
        row["success"] == 0
        for run in cut_event["frame_extras"]["runs"].values()
        for row in run
    )
    manifest.write_text("")
    unknown = enrich_plan(plan, manifest)
    assert all(
        not row["success_valid"] and row["success"] == 0
        for run in unknown["frame_extras"]["runs"].values()
        for row in run
    )
    _manifest(curated, manifest)
    old = enrich_plan(plan, manifest)["fingerprint"]
    (tmp_path / "calibration.json").write_text(
        '{"K": [[110, 0, 64], [0, 100, 48], [0, 0, 1]]}'
    )
    assert enrich_plan(plan, manifest)["fingerprint"] != old


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda rows: rows.append(rows[0]), "duplicate"),
        (lambda rows: rows[0].update(success=True), "success must"),
        (lambda rows: rows[0].pop("success_source"), "success_source"),
        (lambda rows: rows[0]["depth_hand"].update(timestamp_ns=0), "causality"),
        (
            lambda rows: rows[0]["depth_hand"].update(rgb_frame_index=999),
            "different source RGB",
        ),
        (lambda rows: rows[0]["depth_hand"].update(aligned_to_rgb=False), "aligned"),
        (lambda rows: rows[0]["depth_hand"].update(scale_m_per_unit=0), "scale"),
    ],
)
def test_rejects_ambiguous_or_invalid_annotations(
    curated_v1_episode, tmp_path, change, message
):
    curated, quality, _ = _tree(curated_v1_episode, tmp_path, count=1)
    manifest = tmp_path / "frames.jsonl"
    _manifest(curated, manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    change(rows)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=message):
        enrich_plan(make_mainline_plan(curated, quality), manifest)


def test_rgbd_export_pi05_and_merge_roundtrip(curated_v1_episode, tmp_path):
    python = os.environ.get("VLA_LEROBOT_PYTHON")
    if not python:
        pytest.skip("set VLA_LEROBOT_PYTHON for official LeRobot/OpenPI integration")
    curated, quality, _ = _tree(curated_v1_episode, tmp_path, count=2)
    # Give each synthetic source immutable, distinct RAW bytes for merge identity.
    for episode in curated.iterdir():
        raw = tmp_path / f"{episode.name}.npz"
        np.savez(raw, fixture_id=episode.name)
        metadata = json.loads((episode / "metadata.json").read_text())
        metadata["source_episode_path"] = str(raw)
        (episode / "metadata.json").write_text(json.dumps(metadata))
    manifest = tmp_path / "frames.jsonl"
    pixels = _manifest(curated, manifest)
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    outputs = []
    for index in range(2):
        output = tmp_path / f"export-{index}"
        export_lerobot(
            None,
            output,
            curated_root=curated,
            quality_root=quality,
            episode=f"episode_{index:06d}",
            frame_extras=manifest,
            lerobot_python=python,
        )
        assert validate_lerobot_export(output, lerobot_python=python).passed
        outputs.append(output)
    baseline, incoming = [output / "train" for output in outputs]
    repeated = plan_logical_merge(baseline, baseline, source_export_fingerprint="same")
    assert repeated["already_present"] == 2 and repeated["to_append"] == 0
    changed = tmp_path / "changed-annotations"
    shutil.copytree(baseline, changed)
    changed_manifest = changed / "extras/manifest.json"
    changed_data = json.loads(changed_manifest.read_text())
    changed_data["episodes"][0]["source_fingerprint"] = "0" * 64
    changed_manifest.write_text(json.dumps(changed_data))
    with pytest.raises(InvalidDatasetRootError, match="different frame extras"):
        plan_logical_merge(changed, baseline, source_export_fingerprint="changed")
    first = baseline / "extras/episodes/episode_000000/depth_hand/000000.png"
    with Image.open(first) as image:
        np.testing.assert_array_equal(np.asarray(image), np.rot90(pixels, 2))
    view = worker_call(
        "frame_view",
        {"dataset_root": str(baseline), "episode_index": 0, "frame_index": 0},
        python,
    )
    assert len(view["joint_pos"]) == len(view["action"]) == 17
    assert view["success"] == 0 and view["success_valid"]
    assert all(
        p.relative_to(baseline).as_posix()
        in {f["relative_path"] for f in build_canonical_manifest(baseline)["files"]}
        for p in (baseline / "extras").rglob("*")
        if p.is_file()
    )
    plan = enrich_plan(
        make_mainline_plan(curated, quality, episode="episode_000000"), manifest
    )
    smoke = worker_call(
        "openpi", {"plan": plan, "output_root": str(outputs[0])}, python
    )
    assert smoke["openpi_loader"] == "PASS" and all(
        r["boundary_safe"] for r in smoke["runs"]
    )
    merge_plan = plan_logical_merge(
        incoming, baseline, source_export_fingerprint="test"
    )
    merged = tmp_path / "merged"
    result = rebuild_logical_dataset(
        incoming, baseline, merge_plan, merged, lerobot_python=python
    )
    assert result["frame_extras"] == {"episodes": 3, "frames": 5}
    assert set(validate_dataset_root(merged)["frame_extras"]) == {0, 1, 2}
    merged_view = worker_call(
        "frame_view",
        {"dataset_root": str(merged), "episode_index": 2, "frame_index": 2},
        python,
    )
    assert (
        merged_view["source_episode_id"] == "episode_000001"
        and merged_view["success"] == 1
    )
    legacy = tmp_path / "legacy"
    shutil.copytree(baseline, legacy)
    shutil.rmtree(legacy / "extras")
    partial_plan = plan_logical_merge(
        incoming, legacy, source_export_fingerprint="partial"
    )
    partial = tmp_path / "partial"
    partial_result = rebuild_logical_dataset(
        incoming, legacy, partial_plan, partial, lerobot_python=python
    )
    assert partial_result["frame_extras"] == {"episodes": 1, "frames": 3}
    assert set(validate_package(partial)) == {2}
    tokenizer = os.environ.get("VLA_PI05_TOKENIZER_PATH")
    if tokenizer:
        batch = worker_call(
            "pi05_batch",
            {"dataset_root": str(merged), "tokenizer_path": tokenizer},
            python,
        )
        assert batch["pi05_batch"] == "PASS"
        assert batch["state_shape"] == [2, 32]
        assert batch["action_shape"] == [2, 50, 32]
    assert all(
        hashlib.sha256(p.read_bytes()).hexdigest() == checksum
        for p, checksum in before.items()
    )
    # Fresh-copy integrity validation needs no RAW/manifest access.
    first.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_package(baseline)
    assert not validate_lerobot_export(outputs[0], lerobot_python=python).passed
