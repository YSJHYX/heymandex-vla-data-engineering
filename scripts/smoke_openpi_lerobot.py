"""Offline 003-006 LeRobot -> registered OpenPI pi0.5 one-batch smoke.

Run with the existing OpenPI interpreter. All generated files stay under the
explicit smoke root; this neither trains nor contacts Hugging Face.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

SOURCE_IDS = {f"episode_{number:06d}" for number in range(3, 7)}
EXPECTED_LENGTHS = [343, 116, 286, 310]


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def validate_smoke_scope(summary: dict, rows: list[dict]) -> set[str]:
    """Refuse unrelated, omitted, extra, or repartitioned source episodes."""

    sources = {row["source_episode_id"] for row in rows}
    if (
        sources != SOURCE_IDS
        or len(rows) != 4
        or [row["transition_count"] for row in rows] != EXPECTED_LENGTHS
        or any(row["split"] != "train" for row in rows)
        or summary["selected_transitions"] != 1055
        or summary["source_mode"] != "DIRECT_D2_D3"
    ):
        raise ValueError("unexpected source selection or clean-run decomposition")
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    args = parser.parse_args()
    export = args.export_root.resolve(strict=True)
    smoke = args.smoke_root.resolve()
    openpi = args.openpi_root.resolve(strict=True)
    if smoke == export or smoke in export.parents or export in smoke.parents:
        raise ValueError("smoke root must be separate from exported dataset")
    if not (openpi / "src/openpi/training/config.py").is_file():
        raise ValueError("OpenPI source root not found")
    summary = json.loads((export / "export_summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (export / "export_provenance.jsonl").read_text().splitlines()
    ]
    sources = validate_smoke_scope(summary, rows)
    for row in rows:
        metadata = json.loads((Path(row["curated_path"]) / "metadata.json").read_text())
        task = metadata["language_instruction"]
        with np.load(metadata["source_episode_path"], allow_pickle=False) as raw_source:
            raw_task = raw_source["language_instruction"].item()
        if (
            raw_task != task
            or not isinstance(raw_task, str)
            or task != row["task_instruction"]
            or hashlib.sha256(task.encode("utf-8")).hexdigest()
            != row["task_instruction_sha256"]
            or "SYNTHETIC" in str(row["source_dataset_status"]).upper()
        ):
            raise ValueError("collection-time task/provenance mismatch")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HF_LEROBOT_HOME"] = str(smoke / "lerobot_cache")
    os.environ["OPENPI_DATA_HOME"] = "/data/openpi_cache"
    sys.path.insert(0, str(openpi / "src"))
    sys.path.insert(0, str(openpi / "scripts"))

    import compute_norm_stats
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.policies import arm_hand_policy
    from openpi.shared import normalize
    from openpi.training import config, data_loader

    repo_id = f"local/{summary['dataset_name']}-train"
    target = export / "train"
    link = Path(os.environ["HF_LEROBOT_HOME"]) / repo_id
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve(strict=True) != target:
            raise ValueError("local LeRobot cache alias points to a different export")
    elif link.exists():
        raise ValueError("local LeRobot cache alias exists and is not a symlink")
    else:
        link.symlink_to(target, target_is_directory=True)

    dataset = LeRobotDataset(repo_id)
    lengths = [
        int(dataset.episode_data_index["to"][i] - dataset.episode_data_index["from"][i])
        for i in range(dataset.num_episodes)
    ]
    if lengths != EXPECTED_LENGTHS or dataset.num_frames != 1055:
        raise ValueError("LeRobot episode boundaries differ from clean runs")
    if dataset.meta.info["codebase_version"] != "v2.1":
        raise ValueError("local dataset is not LeRobot v2.1")
    for key in ("observation.state", "action"):
        feature = dataset.features[key]
        if feature["dtype"] != "float32" or list(feature["shape"]) != [17]:
            raise ValueError(f"{key} must be physical float32 [17]")
    for key in ("observation.images.head", "observation.images.wrist"):
        if dataset.features[key]["dtype"] != "video":
            raise ValueError(f"missing video modality: {key}")
    expected_task = rows[0]["task_instruction"]
    if any(row["task_instruction"] != expected_task for row in rows):
        raise ValueError("source instructions unexpectedly differ")
    if set(dataset.meta.tasks.values()) != {expected_task}:
        raise ValueError("LeRobot task differs from collection-time instruction")

    model = Pi0Config(pi05=True, action_dim=32, action_horizon=50)
    assets = config.AssetsConfig(assets_dir=str(smoke / "assets"), asset_id=repo_id)
    train = config.TrainConfig(
        name="openpi_mainline_smoke_003_006_TEST_ONLY",
        model=model,
        data=config.LeRobotArmHandDataConfig(
            repo_id=repo_id,
            assets=assets,
            physical_arm_action_dim=6,
            physical_hand_action_dim=11,
        ),
        assets_base_dir=str(smoke / "assets"),
        batch_size=2,
        num_workers=0,
        num_train_steps=1,  # construction only; no training is performed here
        wandb_enabled=False,
    )
    before = train.data.create(train.assets_dirs, model)
    if not before.prompt_from_task or before.action_sequence_keys != ("action",):
        raise ValueError("registered ArmHand config does not read task/action")

    raw = data_loader.create_torch_dataset(before, model.action_horizon, model)
    frame_start = 0
    for row, length in zip(rows, lengths, strict=True):
        with np.load(
            Path(row["curated_path"]) / "trajectory.npz", allow_pickle=False
        ) as archive:
            expected_actions = archive["robot_qcmd_17d_rad"][
                row["source_curated_start"] : row["source_curated_end"] + 1
            ].astype(np.float32)
        for local in sorted({0, length - 1}):
            sample = raw[frame_start + local]
            expected_chunk = expected_actions[
                np.minimum(np.arange(50) + local, length - 1)
            ]
            if not np.array_equal(_array(sample["action"]), expected_chunk):
                raise ValueError("50-step action horizon crossed a clean-run boundary")
            if sample["prompt"] != row["task_instruction"]:
                raise ValueError("OpenPI prompt differs from source task")
            if _array(sample["observation.state"]).shape != (17,):
                raise ValueError("OpenPI raw state is not 17D")
            if sample["task"] != row["task_instruction"]:
                raise ValueError("LeRobot sample task differs from source task")
            for camera in ("observation.images.head", "observation.images.wrist"):
                if _array(sample[camera]).ndim != 3:
                    raise ValueError(f"{camera} could not be decoded")
        frame_start += length

    # Verify exact string and camera routing immediately before model tokenization.
    mapped = raw[0]
    for transform in (*before.repack_transforms.inputs, *before.data_transforms.inputs):
        mapped = transform(mapped)
        if mapped["prompt"] != expected_task:
            raise ValueError("prompt was rewritten in an OpenPI data transform")
    if not (
        mapped["image_mask"]["base_0_rgb"]
        and mapped["image_mask"]["right_wrist_0_rgb"]
        and not mapped["image_mask"]["left_wrist_0_rgb"]
    ):
        raise ValueError("camera modality mapping/mask incorrect")
    if np.array_equal(
        mapped["image"]["base_0_rgb"], mapped["image"]["right_wrist_0_rgb"]
    ):
        raise ValueError("head and wrist camera views unexpectedly identical")
    for source, target_key in (
        ("observation.images.head", "base_0_rgb"),
        ("observation.images.wrist", "right_wrist_0_rgb"),
    ):
        if not np.array_equal(
            mapped["image"][target_key],
            arm_hand_policy._parse_image(_array(raw[0][source])),
        ):
            raise ValueError(f"{source} was not mapped to {target_key}")

    stats_dir = smoke / "assets" / repo_id
    stats_path = stats_dir / "norm_stats.json"
    marker_path = stats_dir / "TEST_ONLY_NORM_STATS_MARKER.json"
    if stats_path.exists() != marker_path.exists():
        raise ValueError("incomplete smoke-only normalization assets")
    if not stats_path.exists():
        stats_loader, num_batches = compute_norm_stats.create_torch_dataloader(
            before,
            model.action_horizon,
            batch_size=32,
            model_config=model,
            num_workers=0,
        )
        stats = compute_norm_stats.compute_statistics(stats_loader, num_batches)
        stats_dir.mkdir(parents=True, exist_ok=True)
        normalize.save(stats_dir, stats)
        marker_path.write_text(
            json.dumps(
                {
                    "status": "TEST_ONLY / NOT_PRODUCTION_NORM_STATS",
                    "source": str(target),
                    "training_split": "four 003-006 clean runs; smoke only",
                    "stats_frames": num_batches * 32,
                    "stats_selection": "first 1024 of 1055 frames (drop_last)",
                    "stats_batches": num_batches,
                },
                indent=2,
            )
            + "\n"
        )
    stats = normalize.load(stats_dir)
    stats_marker = json.loads(marker_path.read_text())
    if stats_marker["status"] != "TEST_ONLY / NOT_PRODUCTION_NORM_STATS":
        raise ValueError("normalization asset is not marked smoke-only")
    if set(stats) != {"state", "actions"} or any(
        np.asarray(stats[key].mean).shape != (17,)
        or np.asarray(stats[key].q01).shape != (17,)
        or np.asarray(stats[key].q99).shape != (17,)
        for key in ("state", "actions")
    ):
        raise ValueError("smoke norm stats must remain physical 17D with quantiles")
    configured = train.data.create(train.assets_dirs, model)
    if configured.norm_stats is None or not configured.use_quantile_norm:
        raise ValueError("registered config did not load smoke-only quantile stats")

    transformed = data_loader.transform_dataset(raw, configured)
    first = transformed[0]
    if first["state"].shape != (32,) or first["actions"].shape != (50, 32):
        raise ValueError("OpenPI transform did not map 17D to native 32D")
    if not (np.all(first["state"][17:] == 0) and np.all(first["actions"][:, 17:] == 0)):
        raise ValueError("32D pad region is not zero")
    expected_state = transforms.Normalize(configured.norm_stats, use_quantiles=True)(
        {"state": mapped["state"].copy()}
    )["state"]
    np.testing.assert_allclose(
        first["state"][:17], expected_state, rtol=1e-5, atol=1e-5
    )

    loader = data_loader.create_data_loader(
        train, shuffle=False, num_batches=1, framework="pytorch"
    )
    observation, actions = next(iter(loader))
    images = {key: _array(value) for key, value in observation.images.items()}
    masks = {key: _array(value) for key, value in observation.image_masks.items()}
    state, action = _array(observation.state), _array(actions)
    if state.shape != (2, 32) or action.shape != (2, 50, 32):
        raise ValueError("one-batch shape mismatch")
    if not (
        bool(masks["base_0_rgb"].all())
        and bool(masks["right_wrist_0_rgb"].all())
        and not bool(masks["left_wrist_0_rgb"].any())
    ):
        raise ValueError("one-batch camera masks incorrect")
    if any(value.shape != (2, 3, 224, 224) for value in images.values()):
        raise ValueError("one-batch image shape mismatch")
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        raise ValueError("one-batch state/action nonfinite")
    if not (np.all(state[:, 17:] == 0) and np.all(action[:, :, 17:] == 0)):
        raise ValueError("one-batch pad region nonzero")
    np.testing.assert_allclose(state[0], first["state"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(action[0], first["actions"], rtol=1e-5, atol=1e-5)

    report = {
        "status": "PASS",
        "scope": [
            "LOCAL_OFFLINE",
            "TEST_ONLY_NORM_STATS",
            "NO_HF_UPLOAD",
            "NO_TRAINING",
        ],
        "export_root": str(export),
        "repo_id": repo_id,
        "source_episode_ids": sorted(sources),
        "exported_episode_lengths": lengths,
        "frames": dataset.num_frames,
        "task_exact": expected_task,
        "task_sha256": hashlib.sha256(expected_task.encode("utf-8")).hexdigest(),
        "prompt_trace": {
            "source_task": expected_task,
            "lerobot_task": dataset.meta.tasks[int(raw[0]["task_index"])],
            "openpi_prompt": raw[0]["prompt"],
        },
        "prompt_equals_source_task_before_tokenization": True,
        "camera_map": {
            "observation.images.head": "base_0_rgb",
            "observation.images.wrist": "right_wrist_0_rgb",
        },
        "norm_stats_path": str(stats_path),
        "norm_stats_status": "TEST_ONLY / NOT_PRODUCTION_NORM_STATS",
        "norm_stats_frames": (len(raw) // 32) * 32,
        "batch_size": 2,
        "action_horizon": 50,
        "state_shape": list(state.shape),
        "state_dtype": str(state.dtype),
        "action_shape": list(action.shape),
        "action_dtype": str(action.dtype),
        "images": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in images.items()
        },
        "image_masks": {key: bool(value.all()) for key, value in masks.items()},
        "prompt_tokens_shape": list(_array(observation.tokenized_prompt).shape),
        "prompt_tokens_dtype": str(_array(observation.tokenized_prompt).dtype),
        "storage_width": 17,
        "model_width": 32,
        "padding_after_normalization": True,
        "boundary_safe": True,
        "optimizer_step": "NOT RUN: no accessible GPU driver; CPU 3B+ model step is not a short smoke",
    }
    reports = smoke.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "openpi_batch_smoke.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    (reports / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
