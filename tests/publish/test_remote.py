"""Network environment isolation for Hugging Face worker processes."""

from __future__ import annotations

from contextlib import nullcontext

from vla_data.publish import remote


def test_download_timeout_is_configurable_and_does_not_leak_stderr(monkeypatch):
    import subprocess

    import pytest

    observed = []

    def timeout(*args, **kwargs):
        observed.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(
            args[0], kwargs["timeout"], stderr="signed-credential"
        )

    monkeypatch.setattr(remote.subprocess, "run", timeout)
    monkeypatch.setenv("VLA_HF_DOWNLOAD_TIMEOUT_SECONDS", "12345")
    with pytest.raises(remote.HFWorkerError, match="download exceeded 12345") as exc:
        remote.hf_call("download", {})
    assert "signed-credential" not in str(exc.value)
    monkeypatch.delenv("VLA_HF_DOWNLOAD_TIMEOUT_SECONDS")
    with pytest.raises(remote.HFWorkerError):
        remote.hf_call("download", {})
    assert observed == [12345, 43200]


def test_unavailable_loopback_proxy_is_removed(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise ConnectionRefusedError

    monkeypatch.setattr(remote.socket, "create_connection", unavailable)
    environment = remote._worker_environment(
        {
            "HTTP_PROXY": "http://127.0.0.1:17891",
            "https_proxy": "http://localhost:17891",
            "ALL_PROXY": "socks5://proxy.example:1080",
            "NO_PROXY": "localhost,127.0.0.1",
        }
    )

    assert "HTTP_PROXY" not in environment
    assert "https_proxy" not in environment
    assert environment["ALL_PROXY"] == "socks5://proxy.example:1080"
    assert environment["NO_PROXY"] == "localhost,127.0.0.1"


def test_listening_loopback_proxy_is_preserved(monkeypatch) -> None:
    monkeypatch.setattr(
        remote.socket,
        "create_connection",
        lambda *_args, **_kwargs: nullcontext(),
    )
    source = {
        "HTTP_PROXY": "http://127.0.0.1:17891",
        "HTTPS_PROXY": "http://127.0.0.1:17891",
    }

    assert remote._worker_environment(source) == source
