"""Parent-process bridge to the Hugging Face worker interpreter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORKER_SOURCE = Path(__file__).with_name("_hf_worker.py")


class HFWorkerError(RuntimeError):
    """The Hub worker failed; the message must stay credential-free."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message


def hf_call(mode: str, payload: dict, hf_python: str | None = None) -> dict:
    """Run one Hub operation in the interpreter that has huggingface_hub."""

    interpreter = hf_python or sys.executable
    request = json.dumps({"mode": mode, **payload})
    completed = subprocess.run(
        [interpreter, str(WORKER_SOURCE)],
        input=request,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise HFWorkerError(
            "HFWorkerFailed",
            (completed.stderr or completed.stdout or "no worker output").strip()[-400:],
        )
    response = json.loads(completed.stdout)
    if not response.get("ok"):
        raise HFWorkerError(
            str(response.get("error_type", "HFWorkerFailed")),
            str(response.get("error", "unknown worker error"))[:400],
        )
    return response
