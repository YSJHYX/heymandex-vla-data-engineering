"""Independent episode annotations, never data-quality or eligibility gates."""

from collections.abc import Mapping

EPISODE_TAG_LABELS = {
    "failure_recovery": "失败后恢复的数据",
    "dagger": "DAgger数据",
}


def episode_tags(metadata: Mapping[str, object]) -> list[str]:
    """Read known annotations; legacy episodes have no implied correction tag."""
    value = metadata.get("episode_tags", [])
    if not isinstance(value, list):
        return []
    return [tag for tag in EPISODE_TAG_LABELS if tag in value]
