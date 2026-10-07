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
        "diagnostic-card",
        "diagnostic-summary",
        "root-cause",
        "rejection-groups",
        "signal-health",
        "camera-timeline",
        "rejection-reason-filter",
        "representative-rejections",
        "copy-diagnostic-report",
        "comparison-baseline",
        "compare-episode",
        "comparison-result",
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


def test_episode_diagnostics_are_lazy_and_dependency_free() -> None:
    html = (STATIC / "index.html").read_text()
    source = (STATIC / "app.js").read_text()

    assert "拒绝原因分组" in html and "不同分组之间可能存在重叠" in html
    assert "D2 拒绝证据与根据 RAW 信号推断的诊断分开显示" in html
    assert "/episodes/${encodeURIComponent(episodeId)}/diagnostics" in source
    assert "loadEpisodeDiagnostics(entry.episode_id)" in source
    assert "document.createElementNS" in source
    assert "timeline.point_count" in source
    assert "chart.js" not in html.lower()
    assert "plotly" not in html.lower()


def test_diagnostics_have_no_known_user_visible_english_copy() -> None:
    html = (STATIC / "index.html").read_text()
    source = (STATIC / "app.js").read_text()
    combined = html + "\n" + source
    for phrase in (
        "View D2 Diagnostics",
        "Loading diagnostics",
        "Failed to load diagnostics",
        "Primary Root Cause",
        "Suggested subsystem",
        "Suggested checks",
        "Why was this Episode rejected",
        "Diagnostic Report",
        "Camera Health",
        "Action Health",
        "Feedback Health",
        "Rejection Groups",
        "Signal Health",
        "Representative Rejected Rows",
        "Baseline Episode",
        '"D2 Status"',
        '"Candidate transitions"',
        '"Acceptance rate"',
    ):
        assert phrase not in combined


def test_chinese_operator_copy_and_offline_help() -> None:
    html = (STATIC / "index.html").read_text()
    script = (STATIC / "app.js").read_text()
    for phrase in (
        "选择数采批次（Run）",
        "验证集比例",
        "数据划分随机种子",
        "数据处理任务",
        'href="/help"',
    ):
        assert phrase in html
    for phrase in (
        "原始数采 Episode",
        "未知数据问题",
        "技术代码：",
        "原始详情：",
        "已完成（有警告）",
        "等待中",
        "处理中",
    ):
        assert (
            phrase in script
            or phrase
            in (
                Path(__file__).resolve().parents[2]
                / "src"
                / "vla_data"
                / "ui"
                / "reasons.py"
            ).read_text()
        )


def test_reason_expansion_state_survives_rerender() -> None:
    """查看原因 must stay open across the 2.5 s polling re-render cycle."""
    source = (STATIC / "app.js").read_text()
    # Expansion state lives in JS state, keyed by stable identity.
    assert "expandedReasons: new Set()" in source
    assert "wireReasonDetails" in source
    # State follows details.open (add when open, delete when closed) so a
    # programmatic restore cannot invert the state.
    assert "if(detail.open)state.expandedReasons.add(key)" in source
    assert "state.expandedReasons.delete(key)" in source
    # Restore happens from state on every rebuild.
    assert "detail.open=state.expandedReasons.has(key)" in source
    # Stable keys: episode id + D2/D3 category (never a row index), plus a
    # per-reason suffix for the nested 高级信息 details.
    assert "`${entry.episode_id}:${category}`" in source
    assert "`${key}:adv${index}`" in source
    assert "dataset.reasonKey" in source
    # Switching runs (select or history jump) clears stale expansion state.
    assert source.count("state.expandedReasons.clear();") == 2
    # Polling itself is untouched.
    assert (
        'setInterval(async()=>{try{await loadJobs();}catch(error){notice(error.message,"error");}},2500);'
        in source
    )
    # Advanced-info details use the same generic mechanism (shared root cause).
    assert 'wireReasonDetails(el("details")' in source
