"""Launch the local operator UI without package sync or network binding."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from vla_data.ui.config import load_config
from vla_data.ui.server import UIServer


def main() -> None:
    parser = argparse.ArgumentParser(description="VLA Data Engineering 本地操作界面")
    parser.add_argument("--config", default="config/ui.toml", help="UI TOML 配置路径")
    parser.add_argument("--port", type=int, help="覆盖本地监听端口")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    if args.port is not None:
        if not 0 <= args.port <= 65535:
            parser.error("端口必须在 0 到 65535 之间")
        config = replace(config, port=args.port)
    server = UIServer(config)
    print(f"VLA Data UI: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
