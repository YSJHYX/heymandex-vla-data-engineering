"""D3 read-only quality evaluation package."""

from vla_data.quality.report import aggregate_quality_reports
from vla_data.quality.temporal import TemporalQualityConfig
from vla_data.quality.trajectory import ActivityConfig
from vla_data.quality.visual import VisualQualityConfig

__all__ = [
    "ActivityConfig",
    "TemporalQualityConfig",
    "VisualQualityConfig",
    "aggregate_quality_reports",
]
