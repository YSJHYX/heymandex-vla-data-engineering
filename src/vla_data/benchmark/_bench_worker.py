"""Benchmark extraction worker: runs under the pyarrow-capable interpreter.

Reads LIBERO parquet and writes the FULL-frame dual-view JPEG cache plus a
sidecar JSON (timestamps). The main process orchestrates everything else.
No vla_data imports here: only stdlib + pyarrow + PIL.
"""

import json
import sys

import pyarrow.parquet as pq
from PIL import Image


def main() -> int:
    request = json.loads(sys.stdin.read())
    results = []
    for spec in request["episodes"]:
        table = pq.read_table(
            spec["parquet"],
            columns=["image", "wrist_image", "timestamp"],
        )
        if table.num_rows != spec["frame_count"]:
            raise SystemExit(
                f"episode {spec['episode_index']}: parquet has {table.num_rows} "
                f"rows, metadata says {spec['frame_count']}"
            )
        view_dirs = spec["view_dirs"]  # {role: absolute dir}
        for frame_index in range(table.num_rows):
            for role, column in (
                ("head", "image"),
                ("right_wrist", "wrist_image"),
            ):
                cell = table.column(column)[frame_index].as_py()
                blob = (cell or {}).get("bytes")
                if not blob:
                    raise SystemExit(
                        f"episode {spec['episode_index']} frame {frame_index}: "
                        f"{column} has no embedded bytes"
                    )
                destination = f"{view_dirs[role]}/{frame_index:06d}.jpg"
                with Image.open(__import__("io").BytesIO(blob)) as image:
                    image.convert("RGB").save(destination, format="JPEG", quality=92)
        timestamps_ns = [
            round(float(value) * 1e9) for value in table.column("timestamp").to_pylist()
        ]
        results.append(
            {"episode_index": spec["episode_index"], "timestamps_ns": timestamps_ns}
        )
        print(
            f"extracted {spec['episode_index']}: {table.num_rows} frames",
            file=sys.stderr,
        )
    sys.stdout.write(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
