"""Plain Python tools exposed to the agent loop.

Kept provider-neutral (no LangChain dependency here) so we can drive them
from our LangGraph-style state machine directly.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from app.graphrag import retriever
from app.state import AppState


_CYPHER_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD|FOREACH|CALL\s+dbms|CALL\s+db\.)\b",
    re.IGNORECASE,
)


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
        # Memgraph first when enabled.
        try:
            from app.graphdb import is_memgraph_enabled, get_client, queries as gq
            if is_memgraph_enabled():
                res = gq.incident_by_id(get_client(), incident_id)
                if res:
                    return res
        except Exception:  # noqa: BLE001
            pass
        inc = store.incident_by_id(incident_id)
        if inc:
            return inc
        if store.graph.has_node(incident_id):
            return retriever._incident_row(store.graph, incident_id)  # type: ignore[attr-defined]
        return None

    def _related(incident_id: str, hops: int = 2, min_shared: int = 2, limit: int = 12) -> dict[str, Any]:
        return retriever.related_incidents(store, incident_id, hops=hops, min_shared=min_shared, limit=limit)

    def _compare(left: str, right: str, dimension: str = "service") -> dict[str, Any]:
        return retriever.compare_entities(store, left, right, dimension=dimension)

    def _path(src: str, dst: str, max_len: int = 6) -> dict[str, Any]:
        return retriever.shortest_path_between(store, src, dst, max_len=max_len)

    def _cooccur(anchor: str, dimension: str = "service", top: int = 5) -> dict[str, Any]:
        return retriever.cooccurrence(store, anchor, dimension=dimension, top=top)

    def _cypher(cypher: str, params: dict[str, Any] | None = None, limit: int = 50) -> dict[str, Any]:
        """Execute a read-only Cypher query against Memgraph and return the rows."""
        try:
            from app.graphdb import get_client, is_memgraph_enabled
        except Exception as e:  # noqa: BLE001
            return {"error": f"memgraph unavailable: {e}", "rows": []}
        if not is_memgraph_enabled():
            return {"error": "memgraph not enabled", "rows": []}
        if not cypher or not isinstance(cypher, str):
            return {"error": "cypher must be a non-empty string", "rows": []}
        if _CYPHER_FORBIDDEN.search(cypher):
            return {"error": "write/admin statements are not allowed", "rows": []}
        try:
            rows = get_client().read(cypher, **(params or {}))
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "cypher": cypher, "params": params or {}, "rows": []}
        capped = rows[: max(1, int(limit))]
        return {
            "cypher": cypher,
            "params": params or {},
            "count": len(capped),
            "total": len(rows),
            "rows": capped,
        }

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
        "related_incidents": {
            "fn": _related,
            "description": "MULTI-HOP: from an anchor incident, traverse to its Service/Region/Team/RootCause nodes and return other incidents sharing >=min_shared of those dimensions, grouped by shared signature. Use for 'similar to', 'related to', 'like this', 'why does this keep happening'.",
            "args": {"incident_id": "string", "hops": "int (default 2)", "min_shared": "int (default 2)", "limit": "int (default 12)"},
        },
        "compare_entities": {
            "fn": _compare,
            "description": "Side-by-side incident breakdown for two services / regions / teams. Use for 'compare X vs Y', 'difference between ...'.",
            "args": {"left": "string", "right": "string", "dimension": "service|region|team|rootCauseCategory (default service)"},
        },
        "shortest_path": {
            "fn": _path,
            "description": "Shortest relational path between two KG nodes (Incident/Service/Region/Team). Use for 'how are X and Y connected', 'path between ...'.",
            "args": {"src": "node id", "dst": "node id", "max_len": "int (default 6)"},
        },
        "cooccurrence": {
            "fn": _cooccur,
            "description": "Inside Louvain communities containing an anchor (service/region/cause), which peer services/root-causes/regions co-occur most? Use for 'what fails alongside X', 'common dependencies of X'.",
            "args": {"anchor": "string", "dimension": "service|region|rootCause (default service)", "top": "int (default 5)"},
        },
        "cypher_query": {
            "fn": _cypher,
            "description": (
                "Execute a READ-ONLY Memgraph Cypher query. USE THIS for aggregate, "
                "structural, or counting questions that other tools cannot answer "
                "(e.g. 'incidents impacting >= N teams', 'services with the most "
                "incidents', 'count by month'). "
                "Schema: "
                "(:Incident {incidentId,title,description,severity,status,service,region,team,"
                "rootCauseCategory,mitigation,createdAt,impactedCustomers}); "
                "(:Service{name}); (:Region{name}); (:Team{name}); (:Owner{name}); "
                "(:RootCauseCategory{name}); (:Mitigation{name}); (:Component{name}); "
                "(:Community{communityId,size,summary}). Relationships: "
                "(i:Incident)-[:BELONGS_TO]->(:Service), -[:IN_REGION]->(:Region), "
                "-[:OWNED_BY]->(:Team), -[:ASSIGNED_TO]->(:Owner), "
                "-[:HAS_ROOT_CAUSE]->(:RootCauseCategory), -[:MITIGATED_BY]->(:Mitigation), "
                "-[:AFFECTS]->(:Component), -[:LINKED_TO]->(:Incident), -[:IN_COMMUNITY]->(:Community). "
                "Rules: (1) READ-ONLY — no CREATE/MERGE/SET/DELETE; "
                "(2) ALWAYS include LIMIT ≤ 50; "
                "(3) Prefer $params over inline literals; "
                "(4) Return concrete incidentId values so the synthesizer can cite them."
            ),
            "args": {
                "cypher": "string — read-only Cypher",
                "params": "object — optional parameters",
                "limit": "int (default 50)",
            },
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
