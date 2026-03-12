"""LangGraph `StateGraph` wrapper around the agent pipeline.

The original async generator in ``app.agent.graph`` is an explicit state
machine. Here we expose the *same* logic composed as a LangGraph graph:

    parse_nl → plan → execute_tool_chain → synthesize → END

Running via LangGraph gives us proper checkpointing, visualization, and the
standard ``app.astream()`` / ``app.get_graph().draw_mermaid()`` affordances.
The module degrades gracefully if ``langgraph`` isn't installed.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from app.agent import prompts  # noqa: F401  (used by graph.py; keep import path hot)
from app.agent.nl_query import GraphQuery, parse_query, to_filters
from app.graphrag import tools as tool_registry
from app.state import AppState

log = logging.getLogger("icm.agent.langgraph")


class AgentState(TypedDict, total=False):
    question: str
    store: AppState
    graph_query: GraphQuery
    plan: dict[str, Any]
    evidence: list[dict[str, Any]]
    citations: list[str]
    answer: str


def _plan_node(state: AgentState) -> AgentState:
    from app.agent.graph import _plan_from_query  # reuse existing planner
    gq = state["graph_query"]
    return {"plan": _plan_from_query(gq)}


def _parse_node(state: AgentState) -> AgentState:
    gq = parse_query(state["question"], state["store"])
    return {"graph_query": gq}


def _execute_node(state: AgentState) -> AgentState:
    plan = state["plan"]
    tools = tool_registry.make_tools(state["store"])
    chain = list(plan.get("toolChain") or [plan.get("firstTool")])
    evidence: list[dict[str, Any]] = []
    for call in chain[:5]:
        if not call:
            continue
        name = call.get("name")
        if name not in tools:
            continue
        try:
            result = tool_registry.call_tool(tools, name, call.get("args") or {})
        except Exception as e:  # noqa: BLE001
            result = {"error": str(e)}
        evidence.append({"tool": name, "args": call.get("args") or {}, "result": result})
    return {"evidence": evidence}


def _synthesize_node(state: AgentState) -> AgentState:
    from app.agent.graph import _collect_citations, _evidence_text, _heuristic_synthesis
    ev = state.get("evidence", [])
    ctx = _evidence_text(ev)
    synth_input = f"QUESTION:\n{state['question']}\n\nEVIDENCE (only use IDs from this list):\n{ctx or '(no evidence)'}\n"
    answer = _heuristic_synthesis(synth_input)
    return {"answer": answer, "citations": _collect_citations(ev)}


def build_app():
    """Compile the StateGraph. Returns None if LangGraph isn't importable."""
    try:
        from langgraph.graph import END, StateGraph
    except Exception as e:  # noqa: BLE001
        log.warning("langgraph unavailable (%s) — falling back to async generator only.", e)
        return None

    g = StateGraph(AgentState)
    g.add_node("parse_nl", _parse_node)
    g.add_node("plan", _plan_node)
    g.add_node("execute", _execute_node)
    g.add_node("synthesize", _synthesize_node)
    g.set_entry_point("parse_nl")
    g.add_edge("parse_nl", "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def run_sync(store: AppState, question: str) -> dict[str, Any]:
    """One-shot invocation for smoke testing / CLI use."""
    app = build_app()
    if app is None:
        # Inline-run the nodes manually.
        state: AgentState = {"store": store, "question": question}
        state.update(_parse_node(state))
        state.update(_plan_node(state))
        state.update(_execute_node(state))
        state.update(_synthesize_node(state))
        return state  # type: ignore[return-value]
    return app.invoke({"store": store, "question": question})
