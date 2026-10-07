"""Fixed writer execution options for the audited LeRobot subprocesses."""

from __future__ import annotations

# Asynchronous PNG intermediate writer thread count for LeRobotDataset.create.
# 0 preserves the legacy synchronous main-thread writer. The PNG intermediates
# are temporary (deleted during save_episode) and their content is unaffected,
# so this only changes write timing, never the encoded MP4 bytes; regression
# requires byte-identical MP4/parquet output against the synchronous writer.
IMAGE_WRITER_THREADS = 4
