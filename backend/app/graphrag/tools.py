"""Plain Python tools exposed to the agent loop.

Kept provider-neutral (no LangChain dependency here) so we can drive them
from our LangGraph-style state machine directly.
"""

from __future__ import annotations

from typing import Any, Callable

from app.graphrag import retriever
from app.state import AppState


def make_tools(store: AppState) -> dict[str, dict[str, Any]]:
    def _local(q: str, top_k: int = 10, **filters: Any) -> dict[str, Any]:
        allowed = {"service", "region", "team", "rootCause", "status", "severity", "start", "end"}
        clean = {k: v for k, v in filters.items() if k in allowed and v not in (None, "")}
        return retriever.local_search(store, q, top_k=top_k, **clean).to_dict()

    def _global(q: str, top_k: int = 5) -> dict[str, Any]:
        return retriever.global_search(store, q, top_k=top_k).to_dict()

    def _drift(q: str, top_communities: int = 3, top_k: int = 8) -> dict[str, Any]:
        return retriever.drift_search(store, q, top_communities=top_communities, top_k=top_k).to_dict()

    def _neighbors(node_id: str, limit: int = 25) -> dict[str, Any]:
        return retriever.neighbors(store, node_id, limit=limit)

    def _temporal(start: str | None = None, end: str | None = None, service: str | None = None) -> list[dict[str, Any]]:
        return retriever.temporal_filter(store, start, end, service=service)

    def _incident(incident_id: str) -> dict[str, Any] | None:
        inc = store.incident_by_id(incident_id)
        if inc:
            return inc
        if store.graph.has_node(incident_id):
            return retriever._incident_row(store.graph, incident_id)  # type: ignore[attr-defined]
        return None

    return {
        "local_search": {
            "fn": _local,
            "description": "Entity-anchored search with optional structured filters (service, region, team, rootCause, status, severity, start, end).",
            "args": {"q": "string", "top_k": "int (default 10)", "service": "string?", "region": "string?", "rootCause": "string?", "severity": "int?", "start": "ISO?", "end": "ISO?"},
        },
        "global_search": {
            "fn": _global,
            "description": "Map-reduce over community summaries. Best for broad/thematic questions like 'what were the top issues last quarter?'.",
            "args": {"q": "string", "top_k": "int (default 5)"},
        },
        "drift_search": {
            "fn": _drift,
            "description": "Hybrid: pick top communities via global search, then rank incidents inside them. Best for cluster/storm analysis.",
            "args": {"q": "string", "top_communities": "int (default 3)", "top_k": "int (default 8)"},
        },
        "neighbor_expand": {
            "fn": _neighbors,
            "description": "Expand neighbors of a graph node (Incident, Service, Region, Team, etc.).",
            "args": {"node_id": "string", "limit": "int (default 25)"},
        },
        "temporal_filter": {
            "fn": _temporal,
            "description": "Return incidents filtered by createdAt window and optional service.",
            "args": {"start": "ISO-8601 or null", "end": "ISO-8601 or null", "service": "string or null"},
        },
        "incident_lookup": {
            "fn": _incident,
            "description": "Fetch a single incident's full record by ID (e.g., INC-2026-0137).",
            "args": {"incident_id": "string"},
        },
    }


def describe_tools(tools: dict[str, dict[str, Any]]) -> str:
    lines = []
    for name, meta in tools.items():
        args = ", ".join(f"{k}: {v}" for k, v in meta["args"].items())
        lines.append(f"- {name}({args}): {meta['description']}")
    return "\n".join(lines)


def call_tool(tools: dict[str, dict[str, Any]], name: str, args: dict[str, Any]) -> Any:
    if name not in tools:
        raise KeyError(f"Unknown tool: {name}")
    fn: Callable[..., Any] = tools[name]["fn"]
    return fn(**(args or {}))
