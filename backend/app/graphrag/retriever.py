"""GraphRAG retrieval: local, global, drift search.

- local_search: entity-anchored — find entities matching the query, expand
  their incident neighbors, rank by similarity.
- global_search: map-reduce over community summaries to answer broad questions.
- drift_search: hybrid — global hit selects communities; local search within
  them for citation-quality evidence.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from app.state import AppState

log = logging.getLogger("icm.graphrag.retriever")


# ---------- helpers ----------

def _cosine(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (Bn @ a).astype(np.float32)


def _embed_query(store: AppState, q: str) -> np.ndarray:
    from app.config import get_settings
    s = get_settings()
    if s.has_azure_openai:
        try:
            from app.graphrag.llm import embed_texts
            return embed_texts([q])[0]
        except Exception as e:  # noqa: BLE001
            log.warning("Query embed via Azure failed: %s — using fallback", e)
    from app.graphrag.index import _fake_embedding  # type: ignore
    dim = store.incident_embeddings.shape[1] if store.incident_embeddings is not None else 256
    return _fake_embedding(q, dim=dim)


def _match_entities(store: AppState, q: str) -> list[str]:
    """Return node IDs whose label matches tokens in q."""
    ql = q.lower()
    hits: list[tuple[str, int]] = []
    for n, d in store.graph.nodes(data=True):
        if d.get("type") == "Incident":
            continue
        label = str(d.get("label", "")).lower()
        if label and label in ql:
            hits.append((n, len(label)))
    hits.sort(key=lambda x: -x[1])
    return [h[0] for h in hits[:6]]


def _incident_row(g: nx.MultiDiGraph, iid: str) -> dict[str, Any]:
    d = g.nodes[iid]
    return {
        "incidentId": iid,
        "title": d.get("title"),
        "severity": d.get("severity"),
        "status": d.get("status"),
        "service": d.get("service"),
        "region": d.get("region"),
        "team": d.get("team"),
        "rootCauseCategory": d.get("rootCauseCategory"),
        "mitigation": d.get("mitigation"),
        "createdAt": d.get("createdAt"),
        "impactedCustomers": d.get("impactedCustomers"),
    }


# ---------- results ----------

@dataclass
class RetrievalResult:
    mode: str
    query: str
    incidents: list[dict[str, Any]] = field(default_factory=list)
    communities: list[dict[str, Any]] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    context_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "query": self.query,
            "incidents": self.incidents,
            "communities": self.communities,
            "matchedEntities": self.matched_entities,
            "contextText": self.context_text,
        }


# ---------- retrievers ----------

def local_search(
    store: AppState,
    query: str,
    *,
    top_k: int = 10,
    service: str | None = None,
    region: str | None = None,
    team: str | None = None,
    rootCause: str | None = None,
    status: str | None = None,
    severity: int | None = None,
    start: str | None = None,
    end: str | None = None,
) -> RetrievalResult:
    entity_hits = _match_entities(store, query)
    candidates: set[str] = set()
    for node in entity_hits:
        for u, _ in store.graph.in_edges(node):
            if store.graph.nodes[u].get("type") == "Incident":
                candidates.add(u)
    if not candidates:
        candidates = {n for n, d in store.graph.nodes(data=True) if d.get("type") == "Incident"}

    # Apply structured filters derived from the NL query.
    def _keep(iid: str) -> bool:
        d = store.graph.nodes[iid]
        if service and d.get("service") != service:
            return False
        if region and d.get("region") != region:
            return False
        if team and d.get("team") != team:
            return False
        if rootCause and d.get("rootCauseCategory") != rootCause:
            return False
        if status and d.get("status") != status:
            return False
        if severity is not None and d.get("severity") != severity:
            return False
        created = d.get("createdAt") or ""
        if start and created < start:
            return False
        if end and created >= end:
            return False
        return True

    filtered = {iid for iid in candidates if _keep(iid)}
    # If filters eliminate everything, relax entity-anchoring but keep filters.
    if not filtered and any([service, region, team, rootCause, status, severity is not None, start, end]):
        all_inc = {n for n, d in store.graph.nodes(data=True) if d.get("type") == "Incident"}
        filtered = {iid for iid in all_inc if _keep(iid)}
    if filtered:
        candidates = filtered

    inc_ids = store.incident_ids
    inc_embeds = store.incident_embeddings
    if inc_embeds is None:
        ranked = list(candidates)[:top_k]
    else:
        q_vec = _embed_query(store, query)
        sims = _cosine(q_vec, inc_embeds)
        mask = np.array([iid in candidates for iid in inc_ids])
        sims = np.where(mask, sims, -1.0)
        top_idx = np.argsort(-sims)[:top_k]
        ranked = [inc_ids[i] for i in top_idx if sims[i] > -1.0]

    rows = [_incident_row(store.graph, iid) for iid in ranked if store.graph.has_node(iid)]
    ctx_lines = [
        f"[{r['incidentId']}] Sev{r['severity']} {r['service']}/{r['region']} — "
        f"{r['title']} (rc={r['rootCauseCategory']}, mit={r['mitigation']})"
        for r in rows
    ]
    return RetrievalResult(
        mode="local",
        query=query,
        incidents=rows,
        matched_entities=entity_hits,
        context_text="\n".join(ctx_lines),
    )


def global_search(store: AppState, query: str, *, top_k: int = 5) -> RetrievalResult:
    if store.community_embeddings is None or len(store.community_ids) == 0:
        return RetrievalResult(mode="global", query=query, context_text="(no communities indexed)")
    q_vec = _embed_query(store, query)
    sims = _cosine(q_vec, store.community_embeddings)
    top_idx = np.argsort(-sims)[:top_k]
    hits: list[dict[str, Any]] = []
    ctx: list[str] = []
    for i in top_idx:
        cid = store.community_ids[i]
        match = next((c for c in store.communities if c["communityId"] == cid), None)
        if not match:
            continue
        hits.append({**match, "score": float(sims[i])})
        ctx.append(
            f"[{cid} | size={match['size']}] services={match.get('topServices')} "
            f"causes={match.get('topRootCauses')} regions={match.get('topRegions')}\n"
            f"{match['summary']}"
        )
    return RetrievalResult(mode="global", query=query, communities=hits, context_text="\n\n".join(ctx))


def drift_search(store: AppState, query: str, *, top_communities: int = 3, top_k: int = 8) -> RetrievalResult:
    g_res = global_search(store, query, top_k=top_communities)
    allowed_ids: set[str] = set()
    for c in g_res.communities:
        allowed_ids.update(c.get("incidentIds", []))
    if not allowed_ids:
        return local_search(store, query, top_k=top_k)

    if store.incident_embeddings is None:
        ranked = list(allowed_ids)[:top_k]
    else:
        q_vec = _embed_query(store, query)
        sims = _cosine(q_vec, store.incident_embeddings)
        mask = np.array([iid in allowed_ids for iid in store.incident_ids])
        sims = np.where(mask, sims, -1.0)
        top_idx = np.argsort(-sims)[:top_k]
        ranked = [store.incident_ids[i] for i in top_idx if sims[i] > -1.0]

    rows = [_incident_row(store.graph, iid) for iid in ranked if store.graph.has_node(iid)]
    ctx = g_res.context_text + "\n\n-- Relevant incidents --\n"
    ctx += "\n".join(
        f"[{r['incidentId']}] Sev{r['severity']} {r['service']}/{r['region']} — {r['title']}"
        for r in rows
    )
    return RetrievalResult(
        mode="drift", query=query,
        incidents=rows, communities=g_res.communities, context_text=ctx,
    )


def neighbors(store: AppState, node_id: str, *, limit: int = 50) -> dict[str, Any]:
    g = store.graph
    if not g.has_node(node_id):
        return {"node": None, "edges": []}
    edges: list[dict[str, Any]] = []
    for _, v, data in g.out_edges(node_id, data=True):
        edges.append({"source": node_id, "target": v, "relation": data.get("relation")})
    for u, _, data in g.in_edges(node_id, data=True):
        edges.append({"source": u, "target": node_id, "relation": data.get("relation")})
    edges = edges[:limit]
    node_ids = {node_id} | {e["source"] for e in edges} | {e["target"] for e in edges}
    nodes = [{"id": n, **{k: v for k, v in g.nodes[n].items() if k != "description"}} for n in node_ids]
    return {"node": node_id, "nodes": nodes, "edges": edges}


def temporal_filter(store: AppState, start: str | None, end: str | None, *, service: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for iid, d in store.graph.nodes(data=True):
        if d.get("type") != "Incident":
            continue
        created = d.get("createdAt") or ""
        if start and created < start:
            continue
        if end and created > end:
            continue
        if service and d.get("service") != service:
            continue
        rows.append(_incident_row(store.graph, iid))
    rows.sort(key=lambda r: r.get("createdAt") or "")
    return rows


def top_stats(store: AppState) -> dict[str, Any]:
    incs = [d for _, d in store.graph.nodes(data=True) if d.get("type") == "Incident"]
    by_service = Counter(i["service"] for i in incs)
    by_severity = Counter(str(i["severity"]) for i in incs)
    by_region = Counter(i["region"] for i in incs)
    by_status = Counter(i["status"] for i in incs)
    by_cause = Counter(i["rootCauseCategory"] for i in incs)
    by_month: Counter[str] = Counter()
    for i in incs:
        created = i.get("createdAt") or ""
        by_month[created[:7]] += 1
    return {
        "total": len(incs),
        "byService": dict(by_service.most_common()),
        "bySeverity": dict(sorted(by_severity.items())),
        "byRegion": dict(by_region.most_common()),
        "byStatus": dict(by_status),
        "byRootCause": dict(by_cause.most_common()),
        "byMonth": dict(sorted(by_month.items())),
        "communities": len(store.communities),
    }
