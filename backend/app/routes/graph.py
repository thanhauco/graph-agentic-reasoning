from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.agent.nl_query import parse_query
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
    """Translate a natural-language question into structured filters and return
    matching incident ids (plus the entities they touch) so the Explorer can
    highlight / zoom to them on top of the currently-loaded graph.

    This reuses the same heuristic NL parser the agent uses, so it works
    without calling an LLM and understands vocabulary like severity, region,
    service, team, root cause, date windows, and free-text keywords.
    """
    store = _store(req)
    gq = parse_query(body.question, store)

    g = store.graph
    matches: list[dict[str, Any]] = []

    # 1) Specific incident lookup
    if gq.incidentId:
        iid = gq.incidentId.upper()
        if g.has_node(iid):
            d = g.nodes[iid]
            matches.append({
                "id": iid, "type": "Incident", "label": iid,
                "severity": d.get("severity"), "service": d.get("service"),
                "region": d.get("region"), "status": d.get("status"),
                "title": d.get("title"),
            })

    # 2) Structured filter over incidents
    keywords_lc = [k.lower() for k in (gq.keywords or [])]
    for iid, d in g.nodes(data=True):
        if d.get("type") != "Incident":
            continue
        if gq.service and d.get("service") != gq.service:
            continue
        if gq.region and d.get("region") != gq.region:
            continue
        if gq.team and d.get("team") != gq.team:
            continue
        if gq.rootCause and d.get("rootCauseCategory") != gq.rootCause:
            continue
        if gq.status and d.get("status") != gq.status:
            continue
        if gq.severity is not None and d.get("severity") != gq.severity:
            continue
        created = d.get("createdAt") or ""
        if gq.startIso and created and created < gq.startIso:
            continue
        if gq.endIso and created and created > gq.endIso:
            continue
        if keywords_lc:
            hay = " ".join([
                str(d.get("title") or ""),
                str(d.get("description") or ""),
                str(d.get("mitigation") or ""),
            ]).lower()
            if not any(k in hay for k in keywords_lc):
                continue
        if any(m["id"] == iid for m in matches):
            continue
        matches.append({
            "id": iid, "type": "Incident", "label": iid,
            "severity": d.get("severity"), "service": d.get("service"),
            "region": d.get("region"), "status": d.get("status"),
            "title": d.get("title"),
        })

    # Rank: severity asc (Sev0 first), then most connected, then recency desc.
    matches.sort(key=lambda m: (
        m.get("severity", 4),
        -g.degree(m["id"]) if g.has_node(m["id"]) else 0,
    ))
    matches = matches[: body.limit]

    # Collect highlight entity ids (service/region/team/rootCause anchors)
    anchors: list[str] = []
    for key, label in (
        (gq.service, "Service"),
        (gq.region, "Region"),
        (gq.team, "Team"),
        (gq.rootCause, "RootCauseCategory"),
    ):
        if key:
            nid = f"{label}:{key}"
            if g.has_node(nid):
                anchors.append(nid)

    return {
        "question": body.question,
        "intent": gq.intent,
        "filters": {
            "service": gq.service,
            "region": gq.region,
            "team": gq.team,
            "rootCause": gq.rootCause,
            "status": gq.status,
            "severity": gq.severity,
            "start": gq.startIso,
            "end": gq.endIso,
            "keywords": gq.keywords,
            "incidentId": gq.incidentId,
        },
        "matches": matches,
        "matchIds": [m["id"] for m in matches],
        "anchorIds": anchors,
        "total": len(matches),
    }
