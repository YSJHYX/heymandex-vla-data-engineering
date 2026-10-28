"""Parent-process bridge to the Hugging Face worker interpreter."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

WORKER_SOURCE = Path(__file__).with_name("_hf_worker.py")
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
LOOPBACK_PROXY_CONNECT_TIMEOUT_SECONDS = 0.25


class HFWorkerError(RuntimeError):
    """The Hub worker failed; the message must stay credential-free."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message


def _loopback_proxy_endpoint(value: str) -> tuple[str, int] | None:
    """Return a local proxy endpoint, leaving non-local proxies untouched."""

    try:
        parsed = urlsplit(value if "://" in value else f"//{value}")
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if host is None or port is None:
        return None
    if host.lower() == "localhost":
        return host, port
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        if addresses and all(
            info[4][0].startswith("127.") or info[4][0] == "::1"
            for info in addresses
        ):
            return host, port
    except socket.gaierror:
        return None
    return None


def _worker_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """Ignore a stale loopback proxy while preserving usable/remote proxies."""

    environment = dict(os.environ if source is None else source)
    reachable: dict[tuple[str, int], bool] = {}
    for name in PROXY_ENV_VARS:
        endpoint = _loopback_proxy_endpoint(environment.get(name, ""))
        if endpoint is None:
            continue
        if endpoint not in reachable:
            try:
                with socket.create_connection(
                    endpoint, timeout=LOOPBACK_PROXY_CONNECT_TIMEOUT_SECONDS
                ):
                    reachable[endpoint] = True
            except OSError:
                reachable[endpoint] = False
        if not reachable[endpoint]:
            environment.pop(name, None)
    return environment


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
        timeout=7200,
        env=_worker_environment(),
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
