"""Dual-view inspection videos for human annotation auditing.

One MP4 per benchmark episode: left = global agentview, right = wrist view,
strictly same source frame per row, minimal overlay (episode id, frame i/N,
t=seconds) — NEVER the source task instruction. Video frame index equals
LIBERO source frame index one-to-one, so an annotation boundary [a,b) maps
directly to video frames a..b-1.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

OVERLAY_BAR_HEIGHT = 28
JPEG_QUALITY_RENDER = 95


def build_dual_view_inspection_video(
    episode_dir: str | Path,
    *,
    fps: int,
    output_path: str | Path,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict[str, object]:
    """Encode the synchronized dual-view MP4 and verify it frame-by-frame.

    ``episode_dir`` must contain ``benchmark_episode.json`` and the full
    ``media/{head,right_wrist}/rgb/NNNNNN.jpg`` cache. Source JPEGs are
    read-only; overlays are drawn on a fresh frame buffer.
    """

    episode_dir = Path(episode_dir)
    marker = json.loads((episode_dir / "benchmark_episode.json").read_text())
    episode_id = str(marker["episode_id"])
    frame_count = int(marker["source_frame_count"])
    timestamps_ns = marker["timestamps_ns"]

    first_head = episode_dir / "media" / "head" / "rgb" / f"{0:06d}.jpg"
    with Image.open(first_head) as probe:
        frame_w, frame_h = probe.size
    canvas_w, canvas_h = 2 * frame_w, frame_h + OVERLAY_BAR_HEIGHT
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{canvas_w}x{canvas_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=None
    )
    assert process.stdin is not None
    try:
        for frame_index in range(frame_count):
            composite = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
            draw = ImageDraw.Draw(composite)
            seconds = timestamps_ns[frame_index] / 1e9
            draw.text(
                (8, 7),
                f"{episode_id}  frame {frame_index:03d} / {frame_count - 1}  "
                f"t={seconds:.1f}s",
                fill=(235, 235, 235),
            )
            draw.text(
                (8 + frame_w // 2 - 30, canvas_h - 14), "GLOBAL", fill=(180, 180, 180)
            )
            draw.text(
                (canvas_w - frame_w // 2 - 20, canvas_h - 14),
                "WRIST",
                fill=(180, 180, 180),
            )
            for offset, role in ((0, "head"), (frame_w, "right_wrist")):
                source = episode_dir / "media" / role / "rgb" / f"{frame_index:06d}.jpg"
                with Image.open(source) as image:
                    composite.paste(image.convert("RGB"), (offset, OVERLAY_BAR_HEIGHT))
            process.stdin.write(composite.tobytes())
    finally:
        process.stdin.close()
        process.wait()

    verification = _verify_video(output_path, ffprobe)
    if int(verification["frame_count"]) != frame_count:
        raise RuntimeError(
            f"{output_path}: encoded {verification['frame_count']} frames, "
            f"expected {frame_count}"
        )
    frame_map = {
        "episode_id": episode_id,
        "fps": fps,
        "video_frame_index_equals_source_frame_index": True,
        "frames": [
            {
                "video_frame_index": index,
                "source_frame_index": index,
                "timestamp": timestamps_ns[index] / 1e9,
            }
            for index in range(frame_count)
        ],
    }
    map_path = output_path.parent / f"{episode_id}_inspection_frame_map.json"
    map_path.write_text(json.dumps(frame_map, indent=2) + "\n")
    return {
        "episode_id": episode_id,
        "video": str(output_path),
        "frame_map": str(map_path),
        "source_frame_count": frame_count,
        "encoded_frame_count": verification["frame_count"],
        "encoded_fps": verification["fps"],
        "resolution": [canvas_w, canvas_h],
    }


def _verify_video(path: Path, ffprobe: str) -> dict[str, object]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    rate = str(stream["avg_frame_rate"]).split("/")
    fps = float(rate[0]) / float(rate[1]) if len(rate) == 2 and float(rate[1]) else 0.0
    return {"frame_count": int(stream["nb_read_frames"]), "fps": round(fps, 3)}
