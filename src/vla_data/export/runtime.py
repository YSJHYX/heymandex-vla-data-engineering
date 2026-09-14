"""Run the audited optional writer in an existing environment, without installing."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def worker_call(mode: str, request: dict, python: str | Path | None = None) -> dict:
    executable = str(python or sys.executable)
    with tempfile.TemporaryDirectory(prefix="vla-lerobot-runtime-") as temporary:
        root = Path(temporary)
        request_path, response_path = root / "request.json", root / "response.json"
        request_path.write_text(json.dumps(request))
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HOME": str(root / "hf"),
                "HF_DATASETS_CACHE": str(root / "datasets"),
                "HF_LEROBOT_HOME": str(root / "lerobot"),
                "JAX_PLATFORMS": "cpu",
                "OMP_NUM_THREADS": "2",
            }
        )
        result = subprocess.run(
            [
                executable,
                "-B",
                str(Path(__file__).with_name("_worker.py")),
                mode,
                str(request_path),
                str(response_path),
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise ValueError(
                f"LeRobot {mode} failed. Use --lerobot-python pointing to an existing environment with audited LeRobot v2.1; no packages were installed.\n{result.stderr[-5000:]}"
            )
        return json.loads(response_path.read_text())
