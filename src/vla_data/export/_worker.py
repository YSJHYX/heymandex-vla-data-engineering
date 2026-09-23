"""Standalone Python 3.11+ bridge to the existing LeRobot/OpenPI environment.

Intentionally imports no vla_data modules: the orchestration package uses 3.12.
Only the official LeRobot API writes canonical dataset artifacts.
"""

import hashlib
import importlib.metadata
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .camera_transform import (
        CAMERA_EXPORT_TRANSFORMS,
        CAMERA_FEATURE_TO_EXPORT_ROLE,
        apply_camera_export_transform,
    )
except ImportError:  # Standalone execution in the audited LeRobot environment.
    from camera_transform import (  # type: ignore[no-redef]
        CAMERA_EXPORT_TRANSFORMS,
        CAMERA_FEATURE_TO_EXPORT_ROLE,
        apply_camera_export_transform,
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def library():
    from lerobot.common.datasets import lerobot_dataset

    require(
        lerobot_dataset.CODEBASE_VERSION == "v2.1",
        "expected LeRobot CODEBASE_VERSION v2.1",
    )
    return lerobot_dataset


def probe():
    module = library()
    from lerobot.common.datasets import video_utils

    source = Path(module.__file__)
    return {
        "codebase_version": module.CODEBASE_VERSION,
        "distribution_version": importlib.metadata.version("lerobot"),
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "video_source_sha256": hashlib.sha256(
            Path(video_utils.__file__).read_bytes()
        ).hexdigest(),
        "video_backend": video_utils.get_safe_default_codec(),
        "video_dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ("av", "torch", "torchvision")
        },
        "direct_url": importlib.metadata.distribution("lerobot").read_text(
            "direct_url.json"
        ),
    }


def decode_rgb(path):
    with Image.open(path) as image:
        require(image.mode == "RGB", "source must be RGB")
        return np.asarray(image, dtype=np.uint8).copy()


def rgb(path, camera_role):
    return apply_camera_export_transform(decode_rgb(path), camera_role=camera_role)


def task_instruction(run):
    """Direct exports use task_instruction; legacy semantic manifests remain loadable."""

    value = run.get("task_instruction", run.get("final_instruction"))
    require(isinstance(value, str) and bool(value.strip()), "task instruction missing")
    return value


def write(request):
    module = library()
    plan, output = request["plan"], Path(request["output_root"])
    require(
        plan.get("camera_transforms") == CAMERA_EXPORT_TRANSFORMS,
        "camera export transform contract mismatch",
    )
    for split in ("train", "val"):
        runs = [r for r in plan["runs"] if r["split"] == split]
        if not runs:
            continue  # no invalid zero-episode LeRobot repository
        dataset = module.LeRobotDataset.create(
            repo_id=f"local/{plan['dataset_name']}-{split}",
            root=output / split,
            robot_type="rm65_sg100",
            fps=plan["fps"],
            features={
                key: {**feature, "shape": tuple(feature["shape"])}
                for key, feature in plan["features"].items()
            },
            use_videos=True,
            image_writer_threads=0,
            image_writer_processes=0,
        )
        for run in runs:
            require(
                run.get("camera_transforms") == CAMERA_EXPORT_TRANSFORMS,
                "run camera export transform contract mismatch",
            )
            with np.load(
                Path(run["curated_path"]) / "trajectory.npz", allow_pickle=False
            ) as archive:
                state = archive["robot_qpos_17d_rad"]
                action = archive["robot_qcmd_17d_rad"]
                for local, index in enumerate(
                    range(run["source_curated_start"], run["source_curated_end"] + 1)
                ):
                    frame = {
                        "observation.state": state[index].astype(np.float32),
                        "action": action[index].astype(np.float32),
                        "task": task_instruction(run),
                    }
                    for key in ("observation.state", "action"):
                        require(
                            frame[key].shape == (17,) and np.isfinite(frame[key]).all(),
                            "finite physical17 required",
                        )
                    frame.update(
                        {
                            key: rgb(paths[local], CAMERA_FEATURE_TO_EXPORT_ROLE[key])
                            for key, paths in run["images"].items()
                        }
                    )
                    dataset.add_frame(frame)
            dataset.save_episode()
        dataset.stop_image_writer()
    return {"written_runs": len(plan["runs"])}


def video_check(
    path,
    source_paths,
    other_paths,
    expected_shape,
    camera_role,
    other_camera_role,
):
    import av

    selected = sorted({0, len(source_paths) // 2, len(source_paths) - 1})
    samples, count = {}, 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        for frame in container.decode(stream):
            require(
                [frame.height, frame.width, 3] == expected_shape,
                "video resolution differs from source",
            )
            if count in selected:
                samples[count] = frame.to_ndarray(format="rgb24").astype(np.float32)
            count += 1
    require(count == len(source_paths), "video frame count != source run count")
    evidence = []
    for index in selected:
        decoded = samples[index]
        error = float(
            np.mean(
                np.abs(
                    decoded - rgb(source_paths[index], camera_role).astype(np.float32)
                )
            )
        )
        untransformed_error = float(
            np.mean(
                np.abs(decoded - decode_rgb(source_paths[index]).astype(np.float32))
            )
        )
        alternatives = [
            float(
                np.mean(
                    np.abs(
                        decoded - rgb(source_paths[j], camera_role).astype(np.float32)
                    )
                )
            )
            for j in selected
            if j != index
        ]
        other = rgb(other_paths[index], other_camera_role).astype(np.float32)
        swap_error = (
            float(np.mean(np.abs(decoded - other)))
            if other.shape == decoded.shape
            else None
        )
        # Relative correspondence, not an arbitrary visual-quality threshold.
        require(
            not alternatives or error <= min(alternatives) + 1e-6,
            "decoded frame matches another temporal anchor better",
        )
        require(
            swap_error is None or error <= swap_error + 1e-6,
            "decoded frame matches other camera better",
        )
        evidence.append(
            {
                "frame_index": index,
                "mean_abs_pixel_difference": error,
                "untransformed_source_difference": untransformed_error,
                "other_camera_difference": swap_error,
                "other_anchor_differences": alternatives,
            }
        )
    return {
        "frame_count": count,
        "fps": fps,
        "resolution": expected_shape,
        "camera_role": camera_role,
        "transform": CAMERA_EXPORT_TRANSFORMS[camera_role],
        "samples": evidence,
    }


def validate(request):
    import torch

    module = library()
    plan, output = request["plan"], Path(request["output_root"])
    require(
        plan.get("camera_transforms") == CAMERA_EXPORT_TRANSFORMS,
        "camera export transform contract mismatch",
    )
    evidence = {"official_reload": "PASS", "runs": [], "total_frames": 0}
    for split in ("train", "val"):
        runs = [r for r in plan["runs"] if r["split"] == split]
        if not runs:
            require(
                not (output / split).exists(), "empty split must not contain a dataset"
            )
            continue
        root = output / split
        for name in (
            "info.json",
            "tasks.jsonl",
            "episodes.jsonl",
            "episodes_stats.jsonl",
        ):
            require((root / "meta" / name).is_file(), f"missing metadata {name}")
        dataset = module.LeRobotDataset(
            f"local/{plan['dataset_name']}-{split}", root=root
        )
        require(
            dataset.meta.info["codebase_version"] == "v2.1", "stored format mismatch"
        )
        expected_keys = set(plan["features"]) | {
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        }
        require(
            set(dataset.features) == expected_keys,
            "unexpected model features or missing features",
        )
        for key, feature in plan["features"].items():
            actual = dataset.features[key]
            require(
                actual["dtype"] == feature["dtype"]
                and list(actual["shape"]) == feature["shape"],
                f"feature mismatch {key}",
            )
        require(dataset.num_episodes == len(runs), "episode count mismatch")
        require(len(dataset.meta.episodes_stats) == len(runs), "episode stats missing")
        require(dataset.fps == plan["fps"], "FPS mismatch")
        require(
            dataset.num_frames == sum(r["transition_count"] for r in runs),
            "total frame count mismatch",
        )
        require(len(list(root.rglob("*.mp4"))) == len(runs) * 2, "MP4 count mismatch")
        require(
            len(list(root.rglob("*.parquet"))) == len(runs), "Parquet count mismatch"
        )
        for key in ("observation.state", "action"):
            require(
                dataset.hf_dataset.features[key].feature.dtype == "float32",
                "Parquet physical vectors not float32",
            )
        for run in runs:
            episode_index = run["lerobot_episode_index"]
            start = int(dataset.episode_data_index["from"][episode_index])
            stop = int(dataset.episode_data_index["to"][episode_index])
            count = run["transition_count"]
            require(stop - start == count, "episode boundary length mismatch")
            require(
                dataset.meta.episodes[episode_index]["length"] == count,
                "metadata episode length mismatch",
            )
            table = dataset.hf_dataset.select(range(start, stop))
            require(
                [int(i) for i in table["frame_index"]] == list(range(count)),
                "frame_index order mismatch",
            )
            require(
                all(int(i) == episode_index for i in table["episode_index"]),
                "episode_index mismatch",
            )
            require(
                [int(i) for i in table["index"]] == list(range(start, stop)),
                "global index mismatch",
            )
            ts = torch.stack(table["timestamp"]).numpy().reshape(-1)
            require(
                np.allclose(ts, np.arange(count) / plan["fps"], rtol=0, atol=1e-5),
                "timestamp not export-local fixed FPS",
            )
            tasks = [dataset.meta.tasks[int(t)] for t in table["task_index"]]
            require(
                all(t == task_instruction(run) for t in tasks),
                "task is not exact source instruction",
            )
            with np.load(
                Path(run["curated_path"]) / "trajectory.npz", allow_pickle=False
            ) as archive:
                for key, source in (
                    ("observation.state", "robot_qpos_17d_rad"),
                    ("action", "robot_qcmd_17d_rad"),
                ):
                    expected = archive[source][
                        run["source_curated_start"] : run["source_curated_end"] + 1
                    ].astype(np.float32)
                    values = torch.stack(table[key]).numpy()
                    require(
                        values.shape == (count, 17) and values.dtype == np.float32,
                        "stored vectors must be float32 [N,17]",
                    )
                    require(
                        np.isfinite(values).all() and np.array_equal(values, expected),
                        "vectors differ from pure source float32 cast",
                    )
            video_evidence = {}
            keys = list(run["images"])
            for key in keys:
                other = next(k for k in keys if k != key)
                path = root / dataset.meta.get_video_file_path(episode_index, key)
                video_evidence[key] = video_check(
                    path,
                    run["images"][key],
                    run["images"][other],
                    plan["features"][key]["shape"],
                    CAMERA_FEATURE_TO_EXPORT_ROLE[key],
                    CAMERA_FEATURE_TO_EXPORT_ROLE[other],
                )
                require(video_evidence[key]["fps"] == plan["fps"], "video FPS mismatch")
            # Actual standard __getitem__, including both video decoders and task resolution.
            for local in sorted({0, count // 2, count - 1}):
                sample = dataset[start + local]
                require(
                    sample["task"] == task_instruction(run),
                    "official loader task mismatch",
                )
                for key in keys:
                    height, width, channels = plan["features"][key]["shape"]
                    require(
                        tuple(sample[key].shape) == (channels, height, width),
                        "official video tensor shape mismatch",
                    )
            evidence["runs"].append(
                {
                    "export_run_id": run["export_run_id"],
                    "split": split,
                    "parquet_rows": count,
                    "task": task_instruction(run),
                    "videos": video_evidence,
                    "max_abs_state_cast_error": run["max_abs_state_cast_error"],
                    "max_abs_action_cast_error": run["max_abs_action_cast_error"],
                }
            )
            evidence["total_frames"] += count
    require(
        evidence["total_frames"] == sum(r["transition_count"] for r in plan["runs"]),
        "frame accounting mismatch",
    )
    return evidence


def openpi_smoke(request):
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config, data_loader

    plan, output = request["plan"], Path(request["output_root"])
    evidence = []
    model = Pi0Config(pi05=True)
    for split in ("train", "val"):
        runs = [r for r in plan["runs"] if r["split"] == split]
        if not runs:
            continue
        repo_id = f"local/{plan['dataset_name']}-{split}"
        link = Path(os.environ["HF_LEROBOT_HOME"]) / repo_id
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(output / split, target_is_directory=True)
        cfg = config.DataConfig(
            repo_id=repo_id,
            action_sequence_keys=("action",),
            prompt_from_task=True,
            repack_transforms=transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                            "image": {
                                "base_0_rgb": "observation.images.head",
                                "right_wrist_0_rgb": "observation.images.wrist",
                            },
                        }
                    )
                ]
            ),
            model_transforms=transforms.Group(
                inputs=[transforms.PadStatesAndActions(32)]
            ),
        )
        dataset = data_loader.create_torch_dataset(cfg, model.action_horizon, model)
        transformed = data_loader.transform_dataset(dataset, cfg, skip_norm_stats=True)
        start = 0
        for run in runs:
            count = run["transition_count"]
            with np.load(
                Path(run["curated_path"]) / "trajectory.npz", allow_pickle=False
            ) as archive:
                actions = archive["robot_qcmd_17d_rad"][
                    run["source_curated_start"] : run["source_curated_end"] + 1
                ].astype(np.float32)
            for local in sorted({0, count - 1}):
                sample = dataset[start + local]
                expected = actions[
                    np.minimum(np.arange(model.action_horizon) + local, count - 1)
                ]
                require(
                    np.array_equal(sample["action"], expected),
                    "OpenPI action horizon crossed run boundary",
                )
                require(
                    sample["observation.state"].shape == (17,), "physical state not17"
                )
                require(
                    sample["prompt"] == task_instruction(run),
                    "OpenPI prompt authority mismatch",
                )
                padded = transformed[start + local]
                require(
                    padded["state"].shape == (32,)
                    and padded["actions"].shape == (model.action_horizon, 32),
                    "OpenPI padding shape mismatch",
                )
                require(
                    np.array_equal(padded["actions"][..., :17], expected),
                    "smoke changed physical action",
                )
                require(np.all(padded["actions"][..., 17:] == 0), "padding not zero")
            evidence.append(
                {
                    "export_run_id": run["export_run_id"],
                    "length": count,
                    "horizon": model.action_horizon,
                    "boundary_safe": True,
                    "storage_width": 17,
                    "model_width": 32,
                    "normalization": "SKIPPED; no stats computed or loaded",
                }
            )
            start += count
    return {
        "openpi_loader": "PASS",
        "runs": evidence,
        "full_tokenizer_model_pipeline": "not run",
    }


def main():
    # Fail closed on accidental network calls, in addition to HF offline envs.
    def no_network(*args, **kwargs):
        raise RuntimeError("network prohibited in local export worker")

    socket.socket.connect = no_network
    mode, request_path, response_path = sys.argv[1:]
    request = json.loads(Path(request_path).read_text())
    result = {
        "probe": lambda _: probe(),
        "write": write,
        "validate": validate,
        "openpi": openpi_smoke,
    }[mode](request)
    Path(response_path).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
