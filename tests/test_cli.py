from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(*arguments: object, cwd: Path | None = None):
    environment = os.environ.copy()
    return subprocess.run(
        [sys.executable, "-m", "vla_data.cli", *(str(value) for value in arguments)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_all_cli_help_commands() -> None:
    for arguments in (
        ("--help",),
        ("build-curated", "--help"),
        ("validate-curated", "--help"),
        ("quality", "--help"),
        ("run", "--help"),
        ("annotate", "--help"),
        ("export-lerobot", "--help"),
        ("validate-lerobot", "--help"),
        ("publish-hf", "--help"),
    ):
        result = _run(*arguments)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout

    root_help = _run("--help")
    normalized_help = " ".join(root_help.stdout.split())
    assert "0=no episode failures" in normalized_help
    assert "1=one or more episode failures" in normalized_help
    assert "2=global error" in normalized_help


def test_cli_global_error_exit_code_is_two(tmp_path) -> None:
    result = _run(
        "build-curated",
        "--input-root",
        tmp_path / "missing",
        "--output-root",
        tmp_path / "output",
    )
    assert result.returncode == 2


def test_synthetic_end_to_end_cli(raw_dataset_factory, tmp_path) -> None:
    raw = raw_dataset_factory(tmp_path / "raw", (0, 1))
    work = tmp_path / "processed"

    result = _run(
        "run",
        "--input-root",
        raw,
        "--work-root",
        work,
        "--workers",
        2,
        "--expert-exclude",
        "episode_000000",
    )

    assert result.returncode == 0, result.stderr
    assert "episode_000000" in result.stdout
    assert "episode_000001" in result.stdout
    assert (work / "quality" / "dataset_quality_summary.json").is_file()


def test_cli_run_dry_run_reports_plan_without_artifacts(
    raw_dataset_factory, tmp_path
) -> None:
    raw = raw_dataset_factory(tmp_path / "raw", (0, 1))
    work = tmp_path / "processed"

    result = _run(
        "run",
        "--input-root",
        raw,
        "--work-root",
        work,
        "--expert-exclude",
        "episode_000000",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("WOULD_PROCESS") >= 6
    assert not work.exists()


def test_cli_episode_failure_exit_code_is_one(synthetic_raw_episode, tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "episode_000001.npz").write_bytes(synthetic_raw_episode.read_bytes())

    result = _run(
        "build-curated",
        "--input-root",
        raw,
        "--output-root",
        tmp_path / "curated",
    )

    assert result.returncode == 1
    assert "FAILED" in result.stdout
