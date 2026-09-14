"""Versioned reference-only training manifest contract."""

SCHEMA_NAME = "vla_training_manifest_episode"
SCHEMA_VERSION = 2
SUMMARY_SCHEMA_NAME = "vla_training_manifest_summary"
OUTPUT_FILES = (
    "episodes.jsonl",
    "train.jsonl",
    "val.jsonl",
    "excluded.jsonl",
    "dataset_manifest_summary.json",
)
SPLIT_LIMITATION = (
    "episode-level random split does not guarantee scene/task independence; "
    "session/group-aware split is not implemented"
)
