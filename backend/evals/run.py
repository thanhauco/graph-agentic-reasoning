"""Evaluation harness.

Runs each gold Q through the agent, captures the answer + citations, and
scores with lightweight heuristics. If `ragas` + Azure OpenAI are available,
also reports faithfulness / answer_relevancy / context_precision.

Usage:
    python -m evals.run
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from app.agent.graph import run_agent
from app.config import get_settings
from app.state import AppState
from evals.gold import GOLD_QA

log = logging.getLogger("icm.evals")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


async def _run_one(store: AppState, q: str) -> dict[str, Any]:
    answer_parts: list[str] = []
    citations: list[str] = []
    contexts: list[str] = []
    chosen_mode = ""
    async for ev in run_agent(store, q):
        if ev["type"] == "plan":
            chosen_mode = ev.get("mode", "")
        elif ev["type"] == "token":
            answer_parts.append(ev["text"])
        elif ev["type"] == "final":
            if ev.get("answer"):
                answer_parts = [ev["answer"]]
            citations = ev.get("citations", [])
            for piece in ev.get("evidence", []):
                r = piece.get("result") or {}
                for inc in (r.get("incidents") or []):
                    contexts.append(
                        f"[{inc.get('incidentId')}] {inc.get('title')} | "
                        f"rc={inc.get('rootCauseCategory')} mit={inc.get('mitigation')}"
                    )
                for c in (r.get("communities") or []):
                    contexts.append(f"[{c.get('communityId')}] {c.get('summary')}")
    return {
        "answer": "".join(answer_parts),
        "citations": citations,
        "contexts": contexts[:30],
        "mode": chosen_mode,
    }


def _heuristic_score(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    ans = (actual.get("answer") or "").lower()
    cites = set(actual.get("citations") or [])
    mode_ok = not expected.get("mode") or actual.get("mode") == expected["mode"]
    inc_ok = True
    for need in expected.get("must_include", []) or []:
        if need not in cites and need.lower() not in ans:
            inc_ok = False
    any_ok = True
    opts = expected.get("must_include_any") or []
    if opts:
        any_ok = any(o.lower() in ans for o in opts)
    cited = len(cites) > 0
    return {"modeOk": mode_ok, "citations": cited, "mustInclude": inc_ok, "mustAny": any_ok}


async def amain() -> None:
    settings = get_settings()
    store = AppState.load(settings)
    log.info("Loaded index: %d incidents, %d communities", len(store.incidents), len(store.communities))

    rows: list[dict[str, Any]] = []
    for gold in GOLD_QA:
        q = str(gold["q"])
        log.info("Q: %s", q)
        actual = await _run_one(store, q)
        scores = _heuristic_score(gold, actual)
        rows.append({"q": q, "expected": gold, "actual": actual, "scores": scores})

    totals = {
        "modeOk": sum(1 for r in rows if r["scores"]["modeOk"]),
        "citations": sum(1 for r in rows if r["scores"]["citations"]),
        "mustInclude": sum(1 for r in rows if r["scores"]["mustInclude"]),
        "mustAny": sum(1 for r in rows if r["scores"]["mustAny"]),
    }
    n = len(rows) or 1
    summary = {k: f"{v}/{n} ({v / n * 100:.0f}%)" for k, v in totals.items()}

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "evals_report.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    print("\n=== Heuristic eval summary ===")
    for k, v in summary.items():
        print(f"  {k:12s}: {v}")
    print(f"Full report: {out_path}")

    # Optional ragas block (only if configured).
    if settings.has_azure_openai:
        try:
            _run_ragas(rows)
        except Exception as e:  # noqa: BLE001
            log.warning("ragas skipped: %s", e)


def _run_ragas(rows: list[dict[str, Any]]) -> None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness, context_precision
    except ImportError:
        log.info("ragas not installed; install with `pip install .[evals]`")
        return

    ds = Dataset.from_list([
        {
            "question": r["q"],
            "answer": r["actual"]["answer"],
            "contexts": r["actual"]["contexts"] or ["(no context)"],
        }
        for r in rows
    ])
    log.info("Running ragas…")
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    print("\n=== Ragas metrics ===")
    print(result)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
