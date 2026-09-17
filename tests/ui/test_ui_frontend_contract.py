"""Static frontend behavior contract for environments without a browser runtime."""

from html.parser import HTMLParser
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "src" / "vla_data" / "ui" / "static"


class IdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.ids.update(value for name, value in attrs if name == "id" and value)


def test_operator_controls_and_confirmation_present() -> None:
    parser = IdParser()
    parser.feed((STATIC / "index.html").read_text())
    assert {
        "stepper",
        "run-select",
        "raw-summary",
        "task-preview",
        "d2-status",
        "d3-status",
        "episode-body",
        "clean-btn",
        "export-btn",
        "validate-btn",
        "dry-btn",
        "publish-btn",
        "modal-check",
        "modal-confirm",
        "job-logs",
        "history-body",
        "hf-auth",
        "hf-private",
        "publish-result",
    } <= parser.ids


def test_frontend_safety_state_wiring() -> None:
    source = (STATIC / "app.js").read_text()
    for contract in (
        'const validated = () => latestAny("validate")?.status === "PASS"',
        'result?.would_action==="NO_NEW_EPISODES"',
        "state.hf?.private!==true",
        'dry?.status!=="PASS"',
        'dry.context?.repo_id!==$("hf-repo").value',
        'confirmation:"我已检查 Dry Run 结果"',
        '$("modal-confirm").disabled=!event.target.checked',
        'job?.status==="FAILED"',
        "state.snapshot.read_only",
    ):
        assert contract in source
    assert "innerHTML" not in source
    assert "localStorage" not in source
