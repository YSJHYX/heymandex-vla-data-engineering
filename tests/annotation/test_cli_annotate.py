"""CLI `annotate` subprocess tests over the synthetic annotated tree."""

from __future__ import annotations

import os
import subprocess
import sys


def _run_annotate(annotated_tree, *extra: str) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment.pop("GLM_API_KEY", None)
    environment.pop("ZHIPU_API_KEY", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "vla_data.cli",
            "annotate",
            "--curated-root",
            str(annotated_tree.curated),
            "--quality-root",
            str(annotated_tree.quality),
            "--output-root",
            str(annotated_tree.output),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def test_annotate_without_api_key_is_a_global_error(annotated_tree) -> None:
    result = _run_annotate(annotated_tree)
    assert result.returncode == 2
    assert "GLM_API_KEY" in result.stderr or "ZHIPU_API_KEY" in result.stderr
    assert not annotated_tree.output.exists()


def test_annotate_dry_run_needs_no_api_key(annotated_tree) -> None:
    result = _run_annotate(annotated_tree, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("WOULD_PROCESS") == 2
    assert result.stdout.count("INELIGIBLE") == 1
    assert not annotated_tree.output.exists()


def test_annotate_global_input_error_exit_code_is_two(tmp_path) -> None:
    environment = os.environ.copy()
    environment.pop("GLM_API_KEY", None)
    environment.pop("ZHIPU_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vla_data.cli",
            "annotate",
            "--curated-root",
            str(tmp_path / "missing_curated"),
            "--quality-root",
            str(tmp_path / "missing_quality"),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 2
