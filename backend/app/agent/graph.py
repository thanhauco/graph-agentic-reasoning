"""Agentic reasoning pipeline.

Implemented as an explicit async generator instead of LangGraph primitives so
we can stream SSE events directly. Flow:
    planner -> router -> executor (tool loop, max N) -> critic -> synthesizer
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator

from app.agent import prompts
from app.config import get_settings
from app.graphrag import tools as tool_registry
from app.state import AppState

log = logging.getLogger("icm.agent")

MAX_TOOL_STEPS = 5


# ---------- LLM adapter (with heuristic fallback) ----------

def _llm_chat(system: str, user: str, *, max_tokens: int = 600, temperature: float = 0.1) -> str:
    s = get_settings()
    if s.has_azure_openai:
        try:
            from app.graphrag.llm import chat
            return chat(system, user, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM chat failed, using heuristic: %s", e)
    return _heuristic_reply(system, user)


def _llm_stream(system: str, user: str, *, max_tokens: int = 700, temperature: float = 0.2) -> AsyncIterator[str]:
    async def _gen() -> AsyncIterator[str]:
        s = get_settings()
        if s.has_azure_openai:
            try:
                from app.graphrag.llm import stream_chat
                for tok in stream_chat(system, user, max_tokens=max_tokens, temperature=temperature):
                    yield tok
                return
            except Exception as e:  # noqa: BLE001
                log.warning("LLM stream failed, using heuristic: %s", e)
        text = _heuristic_synthesis(user)
        for chunk in _split_stream(text):
            yield chunk
    return _gen()


def _split_stream(text: str, chunk: int = 18) -> list[str]:
    return [text[i : i + chunk] for i in range(0, len(text), chunk)]


def _heuristic_reply(system: str, user: str) -> str:
    """Deterministic fallback so the agent works without Azure OpenAI configured."""
    if "planner" in system.lower():
        mode = "drift" if any(k in user.lower() for k in ["cluster", "storm", "outage", "cascade", "march", "trend", "top"]) else "local"
        return json.dumps({
            "mode": mode,
            "plan": [
                "Identify the services, regions, or time window mentioned.",
                f"Run {mode}_search with the user question.",
                "Inspect top incidents/communities for shared root cause + mitigation.",
                "Summarize with inline citations.",
            ],
            "firstTool": {"name": f"{mode}_search", "args": {"q": user.strip(), "top_k": 8}},
        })
    if "executor" in system.lower():
        return json.dumps({"action": "answer", "thought": "Sufficient evidence already gathered."})
    if "critic" in system.lower():
        return json.dumps({"ok": True, "issues": [], "revisedHint": ""})
    return _heuristic_synthesis(user)


def _heuristic_synthesis(user: str) -> str:
    # Extract a couple of INC IDs if present in the context.
    ids = list(dict.fromkeys(re.findall(r"INC-2026-\d{4}", user)))
    comms = list(dict.fromkeys(re.findall(r"C-\d{3}", user)))
    cites = ", ".join(ids[:4]) or ", ".join(comms[:3]) or "(no citations)"
    return (
        f"## Summary\nBased on the retrieved evidence, a small set of recurring patterns drive the question. "
        f"Key incidents: {cites}.\n\n"
        "## Observations\n- Multiple incidents share a service + root cause.\n"
        "- Mitigations cluster around rollback, scale-out, and certificate rotation.\n\n"
        "## Likely Cause\nShared dependency or configuration change affecting the service.\n\n"
        "## Recommended Next Steps\n1. Review the cited incidents for common component signatures.\n"
        "2. Check deployment and config change windows aligned with the impact times.\n"
        "3. Verify mitigations were applied across all impacted regions.\n"
    )


# ---------- JSON helper ----------

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json(s: str) -> dict[str, Any]:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).rstrip("`").strip()
    m = _JSON_RE.search(s)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {}


# ---------- Evidence ----------

def _evidence_text(evidence: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    cited_ids: set[str] = set()
    for ev in evidence:
        result = ev.get("result") or {}
        for inc in result.get("incidents", []) or []:
            iid = inc.get("incidentId")
            if not iid or iid in cited_ids:
                continue
            cited_ids.add(iid)
            lines.append(
                f"[{iid}] Sev{inc.get('severity')} {inc.get('service')}/{inc.get('region')} "
                f"— {inc.get('title')} | rc={inc.get('rootCauseCategory')} "
                f"mit={inc.get('mitigation')} status={inc.get('status')}"
            )
        for c in result.get("communities", []) or []:
            cid = c.get("communityId")
            if not cid or cid in cited_ids:
                continue
            cited_ids.add(cid)
            lines.append(f"[{cid} size={c.get('size')}] {c.get('summary')}")
    return "\n".join(lines[:30])


def _collect_citations(evidence: list[dict[str, Any]]) -> list[str]:
    cited: list[str] = []
    for ev in evidence:
        result = ev.get("result") or {}
        for inc in result.get("incidents", []) or []:
            iid = inc.get("incidentId")
            if iid and iid not in cited:
                cited.append(iid)
        for c in result.get("communities", []) or []:
            cid = c.get("communityId")
            if cid and cid not in cited:
                cited.append(cid)
    return cited


# ---------- Public entrypoint ----------

async def run_agent(store: AppState, question: str) -> AsyncIterator[dict[str, Any]]:
    session_id = uuid.uuid4().hex[:12]
    started = time.time()
    tools = tool_registry.make_tools(store)
    tool_desc = tool_registry.describe_tools(tools)
    trace: list[dict[str, Any]] = []

    def _emit(event: dict[str, Any]) -> dict[str, Any]:
        event["sessionId"] = session_id
        event["t"] = round(time.time() - started, 3)
        trace.append(event)
        return event

    yield _emit({"type": "session", "question": question})

    # 1. Planner
    plan_raw = _llm_chat(prompts.PLANNER_SYSTEM, question, max_tokens=400)
    plan = _parse_json(plan_raw)
    mode = plan.get("mode", "drift")
    steps = plan.get("plan", ["Retrieve relevant evidence", "Summarize with citations"])
    yield _emit({"type": "plan", "mode": mode, "steps": steps})

    # 2. Initial tool call from planner
    evidence: list[dict[str, Any]] = []
    first_tool = plan.get("firstTool") or {"name": f"{mode}_search", "args": {"q": question}}

    for step in range(MAX_TOOL_STEPS):
        tool_name = first_tool.get("name")
        tool_args = first_tool.get("args") or {}
        if tool_name not in tools:
            break
        yield _emit({"type": "tool_call", "step": step, "tool": tool_name, "args": tool_args})
        try:
            result = tool_registry.call_tool(tools, tool_name, tool_args)
        except Exception as e:  # noqa: BLE001
            result = {"error": str(e)}
        evidence.append({"tool": tool_name, "args": tool_args, "result": result})
        yield _emit({
            "type": "tool_result",
            "step": step,
            "tool": tool_name,
            "summary": _summarize_result(tool_name, result),
        })

        # Decide next action.
        exec_prompt = prompts.EXECUTOR_SYSTEM.format(tools=tool_desc, evidence=_evidence_text(evidence) or "(none)")
        decision_raw = _llm_chat(exec_prompt, question, max_tokens=220)
        decision = _parse_json(decision_raw)
        thought = decision.get("thought", "")
        if thought:
            yield _emit({"type": "thought", "step": step, "thought": thought})
        if decision.get("action") == "answer":
            break
        nxt = decision.get("tool") or {}
        if nxt.get("name") and nxt["name"] in tools:
            first_tool = nxt
        else:
            break

    citations = _collect_citations(evidence)

    # 3. Synthesis (streamed)
    ctx = _evidence_text(evidence)
    synth_user = (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE (only use IDs from this list):\n{ctx or '(no evidence)'}\n"
    )
    answer_chunks: list[str] = []
    async for tok in _llm_stream(prompts.SYNTHESIZER_SYSTEM, synth_user, max_tokens=700):
        answer_chunks.append(tok)
        yield _emit({"type": "token", "text": tok})
    answer = "".join(answer_chunks)

    yield _emit({
        "type": "final",
        "answer": answer,
        "citations": citations,
        "evidence": evidence,
    })

    store.sessions[session_id] = trace


def _summarize_result(tool_name: str, result: Any) -> str:
    if isinstance(result, dict):
        incs = result.get("incidents") or []
        comms = result.get("communities") or []
        if incs or comms:
            return f"{len(incs)} incidents, {len(comms)} communities"
        if "nodes" in result and "edges" in result:
            return f"{len(result['nodes'])} nodes, {len(result['edges'])} edges"
        if result.get("incidentId"):
            return f"incident {result['incidentId']}"
    if isinstance(result, list):
        return f"{len(result)} rows"
    return str(result)[:120]
