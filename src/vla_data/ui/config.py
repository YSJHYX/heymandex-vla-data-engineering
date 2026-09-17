"""Strict local-path configuration; no user-supplied path escapes its allowlist."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from vla_data.publish.local import validate_repo_id

RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
EPISODE_ID = re.compile(r"^episode_[0-9]{6}$")


def inside(path: Path, roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in roots)


@dataclass(frozen=True)
class RunPaths:
    name: str
    run_root: Path
    raw_root: Path
    work_root: Path
    curated_root: Path
    quality_root: Path
    export_root: Path
    episode_ids: tuple[str, ...] = ()
    read_only: bool = False


@dataclass(frozen=True)
class UIConfig:
    allowed_data_roots: tuple[Path, ...]
    allowed_raw_roots: tuple[Path, ...]
    state_dir: Path
    default_hf_repo: str
    uv_cache_dir: Path
    lerobot_python: Path
    hf_python: Path
    host: str
    port: int
    run_overrides: dict[str, dict]

    def run(self, name: str) -> RunPaths:
        if not RUN_NAME.fullmatch(name) or name in {".", ".."}:
            raise ValueError("无效的 Run 名称")
        candidates = [root / name for root in self.allowed_data_roots]
        existing = [path for path in candidates if path.is_dir()]
        if len(existing) != 1:
            raise ValueError("Run 不存在或名称不唯一")
        run_root = existing[0].resolve()
        if not inside(run_root, self.allowed_data_roots):
            raise ValueError("Run 路径超出允许范围")
        override = self.run_overrides.get(name, {})
        work = run_root / "work"
        raw = Path(override.get("raw_root", run_root / "raw")).resolve()
        export = Path(override.get("export_root", work / "lerobot")).resolve()
        if not inside(raw, self.allowed_raw_roots):
            raise ValueError("RAW 路径超出允许范围")
        if not inside(export, self.allowed_data_roots):
            raise ValueError("导出路径超出允许范围")
        if not inside(work, self.allowed_data_roots):
            raise ValueError("工作路径超出允许范围")
        curated = (work / "curated").resolve()
        quality = (work / "quality").resolve()
        if not inside(curated, self.allowed_data_roots) or not inside(
            quality, self.allowed_data_roots
        ):
            raise ValueError("处理结果路径超出允许范围")
        ids = tuple(override.get("episode_ids", ()))
        if any(not EPISODE_ID.fullmatch(value) for value in ids):
            raise ValueError("配置中存在无效 episode ID")
        return RunPaths(
            name,
            run_root,
            raw,
            work,
            curated,
            quality,
            export,
            ids,
            bool(override.get("read_only", False)),
        )

    def list_runs(self) -> list[str]:
        names: set[str] = set()
        for root in self.allowed_data_roots:
            if not root.is_dir():
                continue
            for candidate in root.iterdir():
                if (
                    candidate.is_dir()
                    and RUN_NAME.fullmatch(candidate.name)
                    and not candidate.name.startswith(".")
                    and (
                        (candidate / "raw").is_dir()
                        or candidate.name in self.run_overrides
                    )
                ):
                    try:
                        self.run(candidate.name)
                    except ValueError:
                        continue
                    names.add(candidate.name)
        return sorted(names)


def load_config(path: str | Path) -> UIConfig:
    source = Path(path).resolve(strict=True)
    with source.open("rb") as stream:
        value = tomllib.load(stream)
    roots = tuple(Path(p).resolve() for p in value["allowed_data_roots"])
    raw_roots = tuple(Path(p).resolve() for p in value["allowed_raw_roots"])
    if not roots or not raw_roots:
        raise ValueError("至少配置一个允许的数据根目录和 RAW 根目录")
    state_dir = Path(value["state_dir"]).resolve()
    if not inside(state_dir, roots) or state_dir in roots:
        raise ValueError("UI 状态目录必须位于数据根目录内")
    repo = value["default_hf_repo"]
    validate_repo_id(repo)
    host = str(value.get("host", "127.0.0.1"))
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("本地 UI 仅允许绑定 127.0.0.1")
    port = int(value.get("port", 8765))
    if not 0 <= port <= 65535:
        raise ValueError("无效的 UI 端口")
    for name in value.get("run_overrides", {}):
        if not RUN_NAME.fullmatch(name):
            raise ValueError("无效的 Run 配置名称")
    return UIConfig(
        roots,
        raw_roots,
        state_dir,
        repo,
        Path(value["uv_cache_dir"]).resolve(),
        # Invoking the venv symlink path is essential: resolving it to the base
        # interpreter loses the virtual environment's site-packages.
        Path(value["lerobot_python"]).expanduser().absolute(),
        Path(value.get("hf_python", value["lerobot_python"])).expanduser().absolute(),
        host,
        port,
        value.get("run_overrides", {}),
    )
