"""Small, stable result contracts for dataset runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    status: str
    stage: str
    input_path: str
    output_path: str | None = None
    message: str | None = None
    error_type: str | None = None
    outcome: str | None = None
    transitions_total: int = 0
    transitions_valid: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class DatasetRunResult:
    stage: str
    results: tuple[EpisodeResult, ...]
    summary: dict[str, Any]
    summary_path: Path | None
    wall_time_s: float

    @property
    def failed_count(self) -> int:
        return sum(result.status == "FAILED" for result in self.results)
