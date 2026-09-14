"""Human QC video from Curated references and the existing D3 mask.

Selection compacts time: this contact sheet is NOT a training video or a replay.
MISSING/decode-failed references are skipped with reasons, never fabricated.
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vla_data.batch.discovery import canonical_episode_id
from vla_data.io.curated_episode import CuratedEpisode


def review_plan(curated_root, quality_root, episode, selection="all") -> dict:
    if selection not in {"all", "valid", "invalid"}:
        raise ValueError("selection must be valid, invalid or all")
    eid = canonical_episode_id(episode)
    source = CuratedEpisode.load(Path(curated_root) / eid)
    mask_path = Path(quality_root).resolve() / eid / "quality_mask.npy"
    mask = np.load(mask_path, allow_pickle=False)
    if mask.dtype != np.dtype(bool) or mask.shape != (source.transition_count,):
        raise ValueError("quality mask must be bool [Curated transition count]")
    fps = source.metadata.get("dataset_hz")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("positive source dataset_hz required for review playback")
    selected = np.flatnonzero(
        mask
        if selection == "valid"
        else ~mask
        if selection == "invalid"
        else np.ones_like(mask)
    )
    rows, skipped = [], []
    dimensions = {"head": [0, 0], "right_wrist": [0, 0]}
    checked = {}
    for row in selected.tolist():
        images, reasons = {}, []
        for role, current_size in dimensions.items():
            try:
                index = int(source.trajectory[f"{role}_rgb_frame_index"][row])
                path = source.media.rgb_path(role, index).resolve()
                if path not in checked:
                    try:
                        with Image.open(path) as image:
                            image.load()
                            if image.width <= 0 or image.height <= 0:
                                raise ValueError("zero-sized image")
                            checked[path] = (image.size, None)
                    except (OSError, ValueError) as exc:
                        checked[path] = (None, f"{type(exc).__name__}: {exc}")
                size, error = checked[path]
                if error:
                    raise ValueError(error)
                images[role] = str(path)
                dimensions[role] = [
                    max(a, b) for a, b in zip(current_size, size, strict=True)
                ]
            except (OSError, ValueError, KeyError, IndexError) as exc:
                reasons.append(f"MISSING_OR_UNDECODABLE {role}: {exc}")
        if reasons:
            skipped.append({"curated_row": row, "reasons": reasons})
            continue
        rows.append(
            {
                "curated_row": row,
                "quality_valid": bool(mask[row]),
                "source_tick": int(source.trajectory["source_tick_index"][row]),
                "segment": int(
                    np.searchsorted(source.segment_offsets, row, side="right") - 1
                ),
                "images": images,
            }
        )
    width = sum(d[0] for d in dimensions.values())
    height = max(d[1] for d in dimensions.values()) + 48
    return {
        "schema_name": "vla_rgb_review",
        "schema_version": 1,
        "scope": "DIAGNOSTIC_ONLY; compacted selection, NOT training timeline",
        "episode_id": eid,
        "selection": selection,
        "fps": fps,
        "mask_path": str(mask_path),
        "source_path": str(source.episode_dir),
        "selected_count": len(selected),
        "frame_count": len(rows),
        "skipped": skipped,
        "rows": rows,
        "camera_dimensions": dimensions,
        "video_shape": [height + height % 2, width + width % 2, 3],
    }


def compose_frame(plan: dict, row: dict) -> np.ndarray:
    height, width, _ = plan["video_shape"]
    canvas = Image.new("RGB", (width, height))
    x = 0
    for role in ("head", "right_wrist"):
        with Image.open(row["images"][role]) as image:
            canvas.paste(image.convert("RGB"), (x, 24))
        x += plan["camera_dimensions"][role][0]
    draw = ImageDraw.Draw(canvas)
    draw.text((2, 2), "HEAD", fill="white")
    draw.text((plan["camera_dimensions"]["head"][0] + 2, 2), "WRIST", fill="white")
    draw.text(
        (2, height - 20),
        f"QC ONLY row={row['curated_row']} tick={row['source_tick']} seg={row['segment']} D3={row['quality_valid']}",
        fill="white",
    )
    return np.asarray(canvas)


def _encode(plan: dict, output: Path) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise ValueError(
            "RGB review requires existing ffmpeg; use --dry-run to inspect selection"
        )
    height, width, _ = plan["video_shape"]
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(plan["fps"]),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "2",
        str(output),
    ]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for row in plan["rows"]:
                process.stdin.write(compose_frame(plan, row).tobytes())
            process.stdin.close()
            if process.wait() != 0:
                errors.seek(0)
                raise ValueError(
                    f"ffmpeg failed: {errors.read().decode(errors='replace')[-2000:]}"
                )
        except BaseException:
            if process.poll() is None:
                process.terminate()
            process.wait()
            if not process.stdin.closed:
                process.stdin.close()
            raise


def render_rgb_review(
    curated_root, quality_root, output_root, *, episode, selection="all", dry_run=False
) -> dict:
    plan = review_plan(curated_root, quality_root, episode, selection)
    output = Path(output_root).resolve()
    inputs = (Path(curated_root).resolve(), Path(quality_root).resolve())
    if any(output == p or output in p.parents or p in output.parents for p in inputs):
        raise ValueError(
            "review output must be separate from Curated and Quality trees"
        )
    if dry_run:
        return {**plan, "status": "DRY_RUN"}
    target = output / plan["episode_id"]
    if target.is_symlink():
        raise ValueError("review output episode must not be a symlink")
    stem = f"{selection}_head_wrist"
    video, report = target / f"{stem}.mp4", target / f"{stem}.json"
    if video.exists() or report.exists():
        raise FileExistsError("review exists; choose a new diagnostic output root")
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rgb-review-", dir=target) as temp:
        staging = Path(temp)
        if plan["rows"]:
            _encode(plan, staging / video.name)
        result = {
            **plan,
            "status": "RENDERED" if plan["rows"] else "EMPTY",
            "video_path": str(video) if plan["rows"] else None,
        }
        (staging / report.name).write_text(json.dumps(result, indent=2) + "\n")
        if plan["rows"]:
            os.replace(staging / video.name, video)
        os.replace(staging / report.name, report)
    return result
