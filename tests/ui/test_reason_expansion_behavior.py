"""Behavioral test of the shipped reason-expansion JS (node, no browser dep).

Extracts the real ``wireReasonDetails``/``reasonDetails`` from app.js and
drives them through a minimal details shim whose ``toggle`` event fires
asynchronously after ``.open`` changes (matching the HTML spec), then
asserts expansion state survives simulated polling re-renders.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "src" / "vla_data" / "ui" / "static"

HARNESS = r"""
const queued = [];
class FakeDetails {
  constructor() { this.children = []; this._open = false; this.dataset = {}; this._listeners = []; }
  set open(value) {
    const changed = this._open !== Boolean(value);
    this._open = Boolean(value);
    if (changed) queued.push(() => this._listeners.forEach((fn) => fn()));
  }
  get open() { return this._open; }
  append(child) { this.children.push(child); return child; }
  addEventListener(_type, fn) { this._listeners.push(fn); }
}
const flushToggles = async () => { while (queued.length) { const fn = queued.shift(); fn(); } };
const state = { expandedReasons: new Set() };
const el = (tag, text) => { const node = new FakeDetails(); if (text !== undefined && text !== null) node.append(String(text)); node.tag = tag; return node; };

/*__EXTRACTED_FUNCTIONS__*/

const assert = (condition, label) => { if (!condition) { console.error("FAIL: " + label); process.exit(1); } };
const reasons = [
  { summary: "缺少腕部相机数据", technical_code: "MISSING_WRIST_CAMERA", original: "MISSING_WRIST_CAMERA: ..." },
  { summary: "时间同步检查发现 3 个超出阈值的数据点", technical_code: null, original: "head: 3 transitions exceed ..." },
];

(async () => {
  // 1. User opens 003's D3 reasons -> survives a full re-render.
  let details = reasonDetails("episode_000003:d3", "查看原因 · 质量检查", reasons);
  details.open = true; await flushToggles();
  assert(state.expandedReasons.has("episode_000003:d3"), "open records key");
  details = reasonDetails("episode_000003:d3", "查看原因 · 质量检查", reasons);
  await flushToggles();
  assert(details.open === true, "re-render restores open");

  // 2. User closes it -> stays closed after another re-render.
  details.open = false; await flushToggles();
  assert(!state.expandedReasons.has("episode_000003:d3"), "close removes key");
  details = reasonDetails("episode_000003:d3", "查看原因 · 质量检查", reasons);
  await flushToggles();
  assert(details.open === false, "re-render keeps closed");

  // 3. Two episodes independently expanded.
  for (const id of ["episode_000003", "episode_000004"]) {
    const d = reasonDetails(`${id}:d3`, "查看原因 · 质量检查", reasons);
    d.open = true; await flushToggles();
  }
  const again3 = reasonDetails("episode_000003:d3", "查看原因 · 质量检查", reasons);
  const again4 = reasonDetails("episode_000004:d3", "查看原因 · 质量检查", reasons);
  await flushToggles();
  assert(again3.open && again4.open, "multi-episode both restored");
  again3.open = false; await flushToggles();
  const after3 = reasonDetails("episode_000003:d3", "查看原因 · 质量检查", reasons);
  const after4 = reasonDetails("episode_000004:d3", "查看原因 · 质量检查", reasons);
  await flushToggles();
  assert(after3.open === false && after4.open === true, "independent per-episode");

  // 4. D2 vs D3 independence for the same episode.
  const d2 = reasonDetails("episode_000005:d2", "查看原因 · 物理检查", reasons);
  d2.open = true; await flushToggles();
  const d3 = reasonDetails("episode_000005:d3", "查看原因 · 质量检查", reasons);
  await flushToggles();
  assert(d2.open === true && d3.open === false, "d2/d3 independent");
  const d3b = reasonDetails("episode_000005:d3", "查看原因 · 质量检查", reasons);
  d3b.open = true; await flushToggles();
  const d2b = reasonDetails("episode_000005:d2", "查看原因 · 物理检查", reasons);
  await flushToggles();
  assert(d2b.open === true, "d2 unaffected by d3 toggle");

  // 5. Nested 高级信息 details share the mechanism and their own keys.
  const outer = reasonDetails("episode_000006:d3", "查看原因 · 质量检查", reasons);
  const body = outer.children[outer.children.length - 1];
  const advanced = body.children.find((c) => c.dataset.reasonKey === "episode_000006:d3:adv0");
  assert(advanced !== undefined, "advanced details keyed separately");
  advanced.open = true; await flushToggles();
  const outer2 = reasonDetails("episode_000006:d3", "查看原因 · 质量检查", reasons);
  const body2 = outer2.children[outer2.children.length - 1];
  const advanced2 = body2.children.find((c) => c.dataset.reasonKey === "episode_000006:d3:adv0");
  await flushToggles();
  assert(advanced2.open === true, "advanced expansion restored");
  assert(outer2.open === false, "outer untouched by advanced toggle");

  // 6. Programmatic restore does not invert state (toggle follows open).
  state.expandedReasons.add("episode_000003:d3");
  const restore = reasonDetails("episode_000003:d3", "查看原因 · 质量检查", reasons);
  await flushToggles();
  assert(restore.open === true && state.expandedReasons.has("episode_000003:d3"), "no toggle inversion");

  // 7. Run switch clears everything.
  state.expandedReasons.clear();
  const cleared = reasonDetails("episode_000003:d3", "查看原因 · 质量检查", reasons);
  await flushToggles();
  assert(cleared.open === false, "run switch collapses all");

  console.log(JSON.stringify({ behavioral: "PASS" }));
})().catch((error) => { console.error(error); process.exit(1); });
"""


def test_reason_expansion_behavior_via_node() -> None:
    source = (STATIC / "app.js").read_text()
    match = re.search(
        r"function wireReasonDetails\(detail,key\)\{.*?\}\n"
        r"function reasonDetails\(key,label,reasons\)\{.*?\}\n",
        source,
        re.DOTALL,
    )
    assert match is not None, "reason expansion functions missing from app.js"
    script = HARNESS.replace("/*__EXTRACTED_FUNCTIONS__*/", match.group(0))
    completed = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["behavioral"] == "PASS"
