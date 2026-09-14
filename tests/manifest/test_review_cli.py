import json
import subprocess
import sys

import pytest


def run_review(tree, status, *extra):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "vla_data.cli",
            "review-annotation",
            "--verification-root",
            str(tree.verification),
            "--episode",
            "episode_000000",
            "--status",
            status,
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("status", ["HUMAN_VERIFIED", "HUMAN_CORRECTED", "REJECTED"])
def test_review_cli_preserves_original_model(manifest_tree, status):
    tree = manifest_tree(status="AUTO_LABELED")
    path = tree.annotation / "episode_000000/annotation.json"
    original = json.loads(path.read_text())
    extra = (
        ["--instruction", "Move the synthetic green block."]
        if status == "HUMAN_CORRECTED"
        else []
    )
    result = run_review(tree, status, *extra)
    assert result.returncode == 0, result.stderr
    assert json.loads(path.read_text()) == original
    updated = json.loads(
        (tree.verification / "episode_000000/verification.json").read_text()
    )
    assert updated["verification_status"] == status
    if status == "HUMAN_CORRECTED":
        assert (
            updated["final_instruction"]
            == updated["human_review"]["corrected_instruction"]
            == extra[1]
        )
    elif status == "REJECTED":
        assert updated["final_instruction"] is None
    else:
        assert (
            updated["final_instruction"] == original["model_annotation"]["instruction"]
        )


def test_missing_correction_and_invalid_transition_leave_file_intact(manifest_tree):
    tree = manifest_tree(status="HUMAN_VERIFIED")
    path = tree.annotation / "episode_000000/annotation.json"
    before = path.read_bytes()
    assert run_review(tree, "HUMAN_CORRECTED").returncode == 2
    assert run_review(tree, "HUMAN_VERIFIED").returncode == 2
    assert path.read_bytes() == before


@pytest.mark.parametrize("command", ["build-manifest", "review-annotation"])
def test_new_cli_help(command):
    result = subprocess.run(
        [sys.executable, "-m", "vla_data.cli", command, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
