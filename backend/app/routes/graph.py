from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.agent.nl2cypher import run_pipeline
from app.graphrag import retriever

router = APIRouter(tags=["graph"])


def _store(req: Request):
    return req.app.state.store


@router.get("/stats")
def stats(req: Request) -> dict[str, Any]:
    return retriever.top_stats(_store(req))


@router.get("/incidents")
def incidents(
    req: Request,
    q: str | None = None,
    service: str | None = None,
    region: str | None = None,
    severity: int | None = None,
    status: str | None = None,
    limit: int = Query(100, le=500),
    offset: int = 0,
) -> dict[str, Any]:
    store = _store(req)
    rows = store.incidents
    ql = (q or "").lower()
    filtered = []
    for inc in rows:
        if service and inc["service"] != service:
            continue
        if region and inc["region"] != region:
            continue
        if severity is not None and inc["severity"] != severity:
            continue
        if status and inc["status"] != status:
            continue
        if ql and ql not in (
            inc["title"] + " " + inc["description"] + " " + inc["incidentId"]
        ).lower():
            continue
        filtered.append(inc)
    total = len(filtered)
    page = filtered[offset : offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "rows": page}


@router.get("/incidents/{incident_id}")
def incident(req: Request, incident_id: str) -> dict[str, Any]:
    store = _store(req)
    inc = store.incident_by_id(incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    return inc


@router.get("/graph")
def graph(
    req: Request,
    limit_incidents: int = Query(100, ge=10, le=500),
    service: str | None = None,
    region: str | None = None,
) -> dict[str, Any]:
    store = _store(req)
    g = store.graph
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    candidates = [
        n for n, d in g.nodes(data=True)
        if d.get("type") == "Incident"
        and (not service or d.get("service") == service)
        and (not region or d.get("region") == region)
    ]
    # Rank "main" incidents first: highest severity (lowest sev #) then most connected.
    candidates.sort(key=lambda n: (
        g.nodes[n].get("severity", 4),
        -g.degree(n),
    ))
    incident_nodes = candidates[:limit_incidents]
    total_incidents = len(candidates)
    inc_set = set(incident_nodes)
    kept_entities: set[str] = set()
    for iid in incident_nodes:
        for _, v, data in g.out_edges(iid, data=True):
            if g.nodes[v].get("type") == "Incident" and v not in inc_set:
                continue
            edges.append({"source": iid, "target": v, "relation": data.get("relation")})
            if g.nodes[v].get("type") != "Incident":
                kept_entities.add(v)
    for iid in incident_nodes:
        d = g.nodes[iid]
        nodes.append({
            "id": iid, "type": "Incident", "label": iid,
            "severity": d.get("severity"), "service": d.get("service"),
            "region": d.get("region"), "status": d.get("status"),
            "title": d.get("title"),
        })
    for ent in kept_entities:
        d = g.nodes[ent]
        nodes.append({"id": ent, "type": d.get("type"), "label": d.get("label")})
    return {
        "nodes": nodes,
        "edges": edges,
        "incidents": len(incident_nodes),
        "totalIncidents": total_incidents,
    }


@router.get("/graph/neighbors/{node_id:path}")
def graph_neighbors(req: Request, node_id: str, limit: int = 50) -> dict[str, Any]:
    return retriever.neighbors(_store(req), node_id, limit=limit)


@router.get("/communities")
def communities(req: Request, limit: int = 20) -> list[dict[str, Any]]:
    return _store(req).communities[:limit]


class NLQueryBody(BaseModel):
    question: str
    limit: int = 50


@router.post("/graph/query")
def graph_query(req: Request, body: NLQueryBody) -> dict[str, Any]:
    """LLM-powered natural-language → Cypher pipeline.

    Steps: (1) LLM translates the question to read-only Cypher; (2) we execute
    it against Memgraph with guardrails; (3) results are semantically
    re-ranked against the pre-computed incident embeddings; (4) an LLM
    generates a grounded answer. Falls back to a heuristic rule-based
    translator if Azure OpenAI is not configured or the LLM output is unsafe.
    """
    store = _store(req)
    result = run_pipeline(body.question, store, limit=body.limit)
    return {
        "question": result.question,
        "intent": result.plan.intent,
        "cypher": result.plan.cypher,
        "cypherParams": result.plan.params,
        "cypherSource": result.plan.source,  # "llm" | "heuristic"
        "explanation": result.plan.explanation,
        "answer": result.answer,
        "matches": result.matches,
        "matchIds": result.match_ids,
        "anchorIds": result.anchor_ids,
        "rows": result.rows[:25],  # raw rows for UI inspection
        "total": result.total,
    }
