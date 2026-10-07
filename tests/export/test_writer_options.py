"""Fixed writer execution options shared by the audited LeRobot subprocesses."""

from __future__ import annotations

from vla_data.export.writer_options import IMAGE_WRITER_THREADS


def test_async_png_writer_thread_count_is_pinned() -> None:
    # Byte-identical output against the synchronous writer was verified on the
    # real batch4 export and the legacy cumulative migration rebuild; changing
    # this value invalidates that verification and requires re-running it.
    assert IMAGE_WRITER_THREADS == 4
