"""Network environment isolation for Hugging Face worker processes."""

from __future__ import annotations

from contextlib import nullcontext

from vla_data.publish import remote


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
