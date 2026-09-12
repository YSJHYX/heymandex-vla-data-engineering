"""Deterministic discovery of isolated RAW and Curated episodes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

EPISODE_ID_PATTERN = re.compile(r"episode_(\d{6})$")
RAW_FILE_PATTERN = re.compile(r"episode_(\d{6})\.npz$")


class DiscoveryError(ValueError):
    """Global discovery or selection error."""


@dataclass(frozen=True)
class RawEpisodeInput:
    episode_id: str
    numeric_id: int
    raw_path: Path
    media_path: Path


@dataclass(frozen=True)
class CuratedEpisodeInput:
    episode_id: str
    numeric_id: int
    episode_path: Path


@dataclass(frozen=True)
class DiscoveryIssue:
    path: Path
    error_type: str
    message: str
    episode_id: str | None = None


@dataclass(frozen=True)
class DiscoveryResult:
    episodes: tuple[RawEpisodeInput | CuratedEpisodeInput, ...]
    issues: tuple[DiscoveryIssue, ...]
    ignored_paths: tuple[Path, ...]


def canonical_episode_id(value: str) -> str:
    match = EPISODE_ID_PATTERN.fullmatch(str(value))
    if match is None:
        raise DiscoveryError(
            f"episode must use canonical form episode_000001, got {value!r}"
        )
    return f"episode_{int(match.group(1)):06d}"


def discover_raw_episodes(
    input_root: str | Path,
    *,
    episode: str | None = None,
) -> DiscoveryResult:
    root = _input_directory(input_root)
    selected = canonical_episode_id(episode) if episode is not None else None
    episodes: list[RawEpisodeInput] = []
    issues: list[DiscoveryIssue] = []
    ignored: list[Path] = []

    for path in sorted(root.glob("*.npz"), key=lambda item: item.name):
        match = RAW_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            if path.name.startswith("episode_"):
                issues.append(
                    DiscoveryIssue(
                        path=path.resolve(),
                        error_type="INVALID_EPISODE_NAME",
                        message="expected episode_<six-digit-id>.npz",
                    )
                )
            else:
                ignored.append(path.resolve())
            continue
        episode_id = path.stem
        if selected is not None and episode_id != selected:
            continue
        media = path.with_name(f"{episode_id}_media")
        if not media.is_dir():
            issues.append(
                DiscoveryIssue(
                    episode_id=episode_id,
                    path=path.resolve(),
                    error_type="MISSING_MEDIA_DIRECTORY",
                    message=f"missing sibling directory {media.name}",
                )
            )
            continue
        episodes.append(
            RawEpisodeInput(
                episode_id=episode_id,
                numeric_id=int(match.group(1)),
                raw_path=path.resolve(),
                media_path=media.resolve(),
            )
        )

    ordered = tuple(
        sorted(episodes, key=lambda item: (item.numeric_id, item.episode_id))
    )
    selected_issues = tuple(
        issue for issue in issues if selected is None or issue.episode_id == selected
    )
    if selected is not None and not ordered and not selected_issues:
        raise DiscoveryError(f"selected RAW episode was not found: {selected}")
    return DiscoveryResult(ordered, selected_issues, tuple(ignored))


def discover_curated_episodes(
    input_root: str | Path,
    *,
    episode: str | None = None,
) -> DiscoveryResult:
    root = _input_directory(input_root)
    selected = canonical_episode_id(episode) if episode is not None else None
    episodes: list[CuratedEpisodeInput] = []
    issues: list[DiscoveryIssue] = []
    ignored: list[Path] = []

    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        match = EPISODE_ID_PATTERN.fullmatch(path.name)
        if match is None:
            if path.name.startswith("episode_"):
                issues.append(
                    DiscoveryIssue(
                        path=path.resolve(),
                        error_type="INVALID_EPISODE_NAME",
                        message="expected episode_<six-digit-id> directory",
                    )
                )
            else:
                ignored.append(path.resolve())
            continue
        if selected is not None and path.name != selected:
            continue
        episodes.append(
            CuratedEpisodeInput(
                episode_id=path.name,
                numeric_id=int(match.group(1)),
                episode_path=path.resolve(),
            )
        )

    ordered = tuple(
        sorted(episodes, key=lambda item: (item.numeric_id, item.episode_id))
    )
    selected_issues = tuple(
        issue for issue in issues if selected is None or issue.episode_id == selected
    )
    if selected is not None and not ordered and not selected_issues:
        raise DiscoveryError(f"selected Curated episode was not found: {selected}")
    return DiscoveryResult(ordered, selected_issues, tuple(ignored))


def _input_directory(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_dir():
        raise DiscoveryError(f"input root is not a directory: {root}")
    return root.resolve()
