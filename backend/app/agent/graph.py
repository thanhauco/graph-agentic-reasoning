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
from app.agent.nl_query import GraphQuery, parse_query, to_filters
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
    """Deterministic fallback so the agent works without Azure OpenAI configured.

    NOTE: the planner path here is intentionally minimal — the real routing
    happens in `_plan_from_query()` using the parsed `GraphQuery`. This fallback
    is only used if the LLM planner is enabled and fails mid-stream.
    """
    if "executor" in system.lower():
        return json.dumps({"action": "answer", "thought": "Sufficient evidence already gathered."})
    if "critic" in system.lower():
        return json.dumps({"ok": True, "issues": [], "revisedHint": ""})
    return _heuristic_synthesis(user)


def _plan_from_query(gq: GraphQuery) -> dict[str, Any]:
    """Translate the structured NL query into a retrieval plan + first tool call."""
    # 1. Direct incident lookup.
    if gq.intent == "lookup" and gq.incidentId:
        return {
            "mode": "local",
            "plan": [
                f"Look up incident {gq.incidentId} in the graph.",
                "Expand to service, region, team, and root-cause neighbors.",
                "Return the full record with citations.",
            ],
            "firstTool": {"name": "incident_lookup", "args": {"incident_id": gq.incidentId}},
        }
    # 2. Cluster / storm / cascade.
    if gq.intent == "cluster":
        args: dict[str, Any] = {"q": gq.question, "top_communities": 3, "top_k": 8}
        plan = ["Detect the cluster via Louvain community summaries.",
                "Drift-search inside the top communities for citation incidents."]
        if gq.service or gq.region or gq.rootCause or gq.startIso:
            plan.append(
                "Filter candidates by "
                + ", ".join(f"{k}={v}" for k, v in to_filters(gq).items())
            )
        plan.append("Summarize shared root cause + mitigation.")
        return {"mode": "drift", "plan": plan, "firstTool": {"name": "drift_search", "args": args}}
    # 3. Summary / trend / executive overview.
    if gq.intent == "summarize":
        return {
            "mode": "global",
            "plan": [
                "Map over community summaries matching the question.",
                "Reduce into thematic observations.",
                "Cite communities and anchor incidents.",
            ],
            "firstTool": {"name": "global_search", "args": {"q": gq.question, "top_k": 5}},
        }
    # 4. Temporal-only filter (month/quarter with no service/region/cluster).
    if (gq.startIso or gq.endIso) and not (gq.service or gq.region or gq.rootCause or gq.team):
        return {
            "mode": "local",
            "plan": [
                f"Filter incidents by time window {gq.startIso or '*'} → {gq.endIso or '*'}.",
                "Rank by severity and impacted customers.",
                "Summarize with citations.",
            ],
            "firstTool": {
                "name": "temporal_filter",
                "args": {"start": gq.startIso, "end": gq.endIso, "service": gq.service},
            },
        }
    # 5. Default: filtered local search.
    args = {"q": gq.question, "top_k": 10}
    filters = to_filters(gq)
    args.update(filters)
    plan = []
    if filters:
        plan.append("Parsed filters: " + ", ".join(f"{k}={v}" for k, v in filters.items()))
    plan += [
        "Anchor on matched entities in the KG.",
        "Rank candidates with embedding similarity.",
        "Answer with inline incident citations.",
    ]
    return {"mode": "local", "plan": plan, "firstTool": {"name": "local_search", "args": args}}


def _heuristic_synthesis(user: str) -> str:
    """Data-driven fallback: parse the EVIDENCE block embedded in `user` and
    produce a query-specific answer (not a boilerplate template)."""
    from collections import Counter

    # Evidence lines look like:
    #   [INC-2026-0137] Sev1 Front Door/westus — <title> | rc=Certificate mit=<text> status=Resolved
    #   [C-003 size=12] <summary>
    incident_re = re.compile(
        r"\[(?P<id>INC-2026-\d{4})\]\s+Sev(?P<sev>\d)\s+(?P<svc>[^/]+)/(?P<reg>\S+)\s+—\s+"
        r"(?P<title>[^|]+)\|\s+rc=(?P<rc>[^ ]+(?:\s+[^ |]+)*?)\s+mit=(?P<mit>.+?)\s+status=(?P<st>\S+)"
    )
    comm_re = re.compile(r"\[(?P<id>C-\d{3})(?:\s+size=(?P<sz>\d+))?\]\s+(?P<sum>.+)")

    question_match = re.search(r"QUESTION:\s*(.+?)\n", user)
    question = question_match.group(1).strip() if question_match else ""

    incs: list[dict[str, str]] = []
    comms: list[dict[str, str]] = []
    for line in user.splitlines():
        mi = incident_re.search(line)
        if mi:
            incs.append(mi.groupdict())
            continue
        mc = comm_re.search(line)
        if mc:
            comms.append(mc.groupdict())

    if not incs and not comms:
        ids = list(dict.fromkeys(re.findall(r"INC-2026-\d{4}", user)))
        cites = ", ".join(ids[:4]) or "(no matching incidents)"
        return (
            f"## Summary\nNo evidence rows matched the query `{question}`. "
            f"Known IDs in context: {cites}.\n"
        )

    # Aggregate structured stats.
    svcs = Counter(i["svc"].strip() for i in incs)
    regs = Counter(i["reg"].strip() for i in incs)
    rcs = Counter(i["rc"].strip() for i in incs)
    sevs = Counter(i["sev"] for i in incs)
    statuses = Counter(i["st"] for i in incs)
    mits = Counter(i["mit"].strip() for i in incs)

    top_svc = ", ".join(f"{s} ({n})" for s, n in svcs.most_common(3))
    top_reg = ", ".join(f"{r} ({n})" for r, n in regs.most_common(3))
    top_rc = ", ".join(f"{r} ({n})" for r, n in rcs.most_common(3))
    sev_dist = ", ".join(f"Sev{k}:{v}" for k, v in sorted(sevs.items()))
    status_dist = ", ".join(f"{k}:{v}" for k, v in statuses.most_common())
    top_mit = ", ".join(f"{m} ({n})" for m, n in mits.most_common(2))

    cite_list = [i["id"] for i in incs[:6]]
    inc_cites = ", ".join(f"[{cid}]" for cid in cite_list)

    lines: list[str] = []
    lines.append(f"## Summary")
    if question:
        lines.append(f"For **{question}**, the graph returned **{len(incs)}** matching incident(s)"
                     + (f" across **{len(comms)}** community cluster(s)" if comms else "") + ".")
    else:
        lines.append(f"Retrieved {len(incs)} incidents" + (f" and {len(comms)} community cluster(s)." if comms else "."))

    lines.append("\n## Observations")
    if svcs:
        lines.append(f"- **Services**: {top_svc}")
    if regs:
        lines.append(f"- **Regions**: {top_reg}")
    if rcs:
        lines.append(f"- **Root causes**: {top_rc}")
    if sevs:
        lines.append(f"- **Severity mix**: {sev_dist}")
    if statuses:
        lines.append(f"- **Status**: {status_dist}")

    # Show up to 5 incidents with context, each one different.
    lines.append("\n## Evidence")
    for i in incs[:6]:
        title = i["title"].strip()
        lines.append(
            f"- [{i['id']}] Sev{i['sev']} {i['svc'].strip()} / {i['reg'].strip()} — {title} "
            f"(cause: {i['rc'].strip()}, mitigation: {i['mit'].strip()}, status: {i['st']})"
        )

    if comms:
        lines.append("\n## Community context")
        for c in comms[:3]:
            summary = c["sum"].strip()
            if len(summary) > 220:
                summary = summary[:220] + "…"
            size = f" (size={c['sz']})" if c.get("sz") else ""
            lines.append(f"- [{c['id']}]{size} {summary}")

    lines.append("\n## Likely Cause & Mitigations")
    rc_top = rcs.most_common(1)[0][0] if rcs else "mixed"
    lines.append(
        f"The dominant root-cause signal is **{rc_top}** "
        + (f"(out of {sum(rcs.values())} observations). " if rcs else ". ")
        + (f"Most common mitigations: {top_mit}." if mits else "")
    )

    lines.append("\n## Recommended Next Steps")
    lines.append(f"1. Drill into the top incidents ({inc_cites}) for shared component signatures.")
    if regs:
        lines.append(f"2. Confirm whether the issue is regional (top regions: {top_reg}) or global.")
    if rcs:
        lines.append(f"3. Check change-management windows for **{rc_top}** categories.")
    return "\n".join(lines)


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

    def _row(inc: dict[str, Any]) -> str | None:
        iid = inc.get("incidentId")
        if not iid or iid in cited_ids:
            return None
        cited_ids.add(iid)
        return (
            f"[{iid}] Sev{inc.get('severity')} {inc.get('service')}/{inc.get('region')} "
            f"— {inc.get('title')} | rc={inc.get('rootCauseCategory')} "
            f"mit={inc.get('mitigation')} status={inc.get('status')}"
        )

    for ev in evidence:
        result = ev.get("result")
        if result is None:
            continue
        # Shape A: list of incidents (temporal_filter).
        if isinstance(result, list):
            for inc in result:
                line = _row(inc) if isinstance(inc, dict) else None
                if line:
                    lines.append(line)
            continue
        # Shape B: single incident dict (incident_lookup).
        if isinstance(result, dict) and result.get("incidentId") and "incidents" not in result:
            line = _row(result)
            if line:
                lines.append(line)
            continue
        # Shape C: structured retrieval result.
        if isinstance(result, dict):
            for inc in result.get("incidents", []) or []:
                line = _row(inc)
                if line:
                    lines.append(line)
            for c in result.get("communities", []) or []:
                cid = c.get("communityId")
                if not cid or cid in cited_ids:
                    continue
                cited_ids.add(cid)
                lines.append(f"[{cid} size={c.get('size')}] {c.get('summary')}")
    return "\n".join(lines[:30])


def _collect_citations(evidence: list[dict[str, Any]]) -> list[str]:
    cited: list[str] = []

    def _add(val: str | None) -> None:
        if val and val not in cited:
            cited.append(val)

    for ev in evidence:
        result = ev.get("result")
        if result is None:
            continue
        if isinstance(result, list):
            for inc in result:
                if isinstance(inc, dict):
                    _add(inc.get("incidentId"))
            continue
        if isinstance(result, dict):
            if result.get("incidentId") and "incidents" not in result:
                _add(result.get("incidentId"))
                continue
            for inc in result.get("incidents", []) or []:
                _add(inc.get("incidentId"))
            for c in result.get("communities", []) or []:
                _add(c.get("communityId"))
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

    # 0. Parse NL → structured graph query (always runs; grounds the whole agent).
    gq: GraphQuery = parse_query(question, store)
    yield _emit({
        "type": "graph_query",
        "query": gq.to_dict(),
        "describe": gq.describe(),
    })

    # 1. Planner — prefer the LLM when configured, else use the structured query.
    heur_plan = _plan_from_query(gq)
    if get_settings().has_azure_openai:
        plan_raw = _llm_chat(prompts.PLANNER_SYSTEM, question, max_tokens=400)
        plan = _parse_json(plan_raw) or heur_plan
    else:
        plan = heur_plan
    mode = plan.get("mode", heur_plan["mode"])
    steps = plan.get("plan", heur_plan["plan"])
    yield _emit({"type": "plan", "mode": mode, "steps": steps})

    # 2. Initial tool call from planner
    evidence: list[dict[str, Any]] = []
    first_tool = plan.get("firstTool") or heur_plan["firstTool"]

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
            return f"incident {result['incidentId']} (Sev{result.get('severity')} {result.get('service')}/{result.get('region')})"
    if isinstance(result, list):
        return f"{len(result)} rows"
    if result is None:
        return "no result"
    return str(result)[:120]
