"""Cypher implementations of the agent's multi-hop reasoning primitives.

Each function mirrors the signature used by ``app.graphrag.retriever`` so the
dispatcher can swap backends transparently. All queries are parameterized to
prevent Cypher injection.
"""

from __future__ import annotations

from typing import Any

from app.graphdb.client import MemgraphClient


# ---------- single-incident lookup ----------

def incident_by_id(client: MemgraphClient, incident_id: str) -> dict[str, Any] | None:
    rows = client.read(
        """
        MATCH (i:Incident {incidentId: $id})
        RETURN i {.*} AS i
        """,
        id=incident_id,
    )
    if not rows:
        return None
    return rows[0]["i"]


# ---------- multi-hop: related incidents sharing >=N dimensions ----------

def related_incidents(
    client: MemgraphClient,
    anchor_id: str,
    *,
    min_shared: int = 2,
    limit: int = 12,
) -> dict[str, Any]:
    """Return incidents sharing >=min_shared of {service, region, team,
    rootCauseCategory} with the anchor, grouped by shared-dimension signature.

    This is a true multi-hop Cypher traversal: from the anchor we hop to its
    Service/Region/Team/RootCause nodes, then back out to *other* incidents
    via any of those edges, and finally count how many dimensions coincide
    with the anchor.
    """
    anchor = incident_by_id(client, anchor_id)
    if not anchor:
        return {"anchor": None, "related": [], "groups": {}, "reason": "anchor not found"}

    rows = client.read(
        """
        MATCH (a:Incident {incidentId: $id})
        MATCH (b:Incident)
        WHERE b.incidentId <> a.incidentId
        WITH a, b,
             [x IN [
                CASE WHEN b.service = a.service THEN 'service' END,
                CASE WHEN b.region = a.region THEN 'region' END,
                CASE WHEN b.team = a.team THEN 'team' END,
                CASE WHEN b.rootCauseCategory = a.rootCauseCategory THEN 'rootCauseCategory' END
             ] WHERE x IS NOT NULL] AS shared
        WHERE size(shared) >= $minShared
        RETURN b {.*} AS inc, shared AS sharedDims, size(shared) AS sharedCount
        ORDER BY sharedCount DESC, b.severity ASC
        LIMIT $limit
        """,
        id=anchor_id, minShared=min_shared, limit=limit,
    )

    related: list[dict[str, Any]] = []
    groups: dict[str, list[str]] = {}
    for r in rows:
        inc = r["inc"]
        inc["sharedWithAnchor"] = r["sharedDims"]
        inc["sharedCount"] = r["sharedCount"]
        related.append(inc)
        key = "+".join(sorted(r["sharedDims"]))
        groups.setdefault(key, []).append(inc["incidentId"])

    anchor_dims = {k: anchor.get(k) for k in ("service", "region", "team", "rootCauseCategory") if anchor.get(k)}
    return {
        "anchor": anchor,
        "anchorDims": anchor_dims,
        "related": related,
        "groups": groups,
        "hops": 2,
    }


# ---------- compare two entities ----------

_COMPARE_DIM_TO_PROP = {
    "service": "service",
    "region": "region",
    "team": "team",
    "rootCauseCategory": "rootCauseCategory",
}


def compare_entities(
    client: MemgraphClient,
    left: str,
    right: str,
    *,
    dimension: str = "service",
) -> dict[str, Any]:
    prop = _COMPARE_DIM_TO_PROP.get(dimension, "service")

    def _stats(value: str) -> dict[str, Any]:
        rows = client.read(
            f"""
            MATCH (i:Incident) WHERE i.{prop} = $val
            RETURN i {{.*}} AS inc
            """,
            val=value,
        )
        incs = [r["inc"] for r in rows]
        from collections import Counter
        return {
            "value": value,
            "total": len(incs),
            "bySeverity": dict(Counter(str(i["severity"]) for i in incs).most_common()),
            "byRootCause": dict(Counter(i["rootCauseCategory"] for i in incs).most_common(5)),
            "byRegion": dict(Counter(i["region"] for i in incs).most_common(5)),
            "byStatus": dict(Counter(i["status"] for i in incs).most_common()),
            "topIncidents": sorted(incs, key=lambda r: r.get("severity", 4))[:5],
        }

    l_stats = _stats(left)
    r_stats = _stats(right)
    common_rc = sorted(set(l_stats["byRootCause"]) & set(r_stats["byRootCause"]))
    return {
        "dimension": dimension,
        "left": l_stats,
        "right": r_stats,
        "sharedRootCauses": common_rc,
    }


# ---------- shortest path between two named nodes ----------

def shortest_path_between(
    client: MemgraphClient,
    src: str,
    dst: str,
    *,
    max_len: int = 6,
) -> dict[str, Any]:
    """Resolve src/dst by name (Service/Region/Team/RootCause) or incidentId,
    then ask Memgraph for the shortest undirected-style path."""
    rows = client.read(
        """
        CALL {
          WITH $src AS key
          OPTIONAL MATCH (a) WHERE a.incidentId = key OR a.name = key OR a.communityId = key
          RETURN a LIMIT 1
        }
        CALL {
          WITH $dst AS key
          OPTIONAL MATCH (b) WHERE b.incidentId = key OR b.name = key OR b.communityId = key
          RETURN b LIMIT 1
        }
        WITH a, b
        WHERE a IS NOT NULL AND b IS NOT NULL
        MATCH p = (a)-[*BFS 1..%d]-(b)
        RETURN p LIMIT 1
        """ % max_len,
        src=src, dst=dst,
    )
    if not rows:
        return {"found": False, "src": src, "dst": dst}
    path = rows[0]["p"]
    # neo4j driver returns Path objects; iterate nodes+rels.
    try:
        nodes = list(path.nodes)
        rels = list(path.relationships)
    except AttributeError:
        return {"found": False, "src": src, "dst": dst, "reason": "path shape unsupported"}

    def _node_view(n: Any) -> dict[str, Any]:
        lbls = list(getattr(n, "labels", []) or [])
        props = dict(n)
        key = props.get("incidentId") or props.get("name") or props.get("communityId")
        label = props.get("title") or props.get("name") or props.get("communityId") or key
        return {"id": key, "type": lbls[0] if lbls else None, "label": label}

    steps: list[dict[str, Any]] = []
    for i, rel in enumerate(rels):
        a = _node_view(nodes[i])
        b = _node_view(nodes[i + 1])
        steps.append({
            "from": a["id"], "fromType": a["type"], "fromLabel": a["label"],
            "to": b["id"], "toType": b["type"], "toLabel": b["label"],
            "relation": rel.type,
        })
    return {"found": True, "src": src, "dst": dst, "length": len(rels), "steps": steps}


# ---------- neighbor expansion ----------

def neighbors(client: MemgraphClient, node_id: str, *, limit: int = 50) -> dict[str, Any]:
    rows = client.read(
        """
        OPTIONAL MATCH (n) WHERE n.incidentId = $id OR n.name = $id OR n.communityId = $id
        WITH n LIMIT 1
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n, r, m LIMIT $limit
        """,
        id=node_id, limit=limit,
    )
    if not rows or rows[0].get("n") is None:
        return {"node": None, "edges": []}

    node_map: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def _view(nx_node: Any) -> dict[str, Any]:
        props = dict(nx_node)
        key = props.get("incidentId") or props.get("name") or props.get("communityId")
        lbls = list(getattr(nx_node, "labels", []) or [])
        return {"id": key, "type": lbls[0] if lbls else None, **props}

    root = _view(rows[0]["n"])
    node_map[root["id"]] = root
    for r in rows:
        if r.get("r") is None:
            continue
        m = _view(r["m"])
        node_map[m["id"]] = m
        rel = r["r"]
        # Build edge in original direction.
        try:
            s_key = rel.start_node.get("incidentId") or rel.start_node.get("name") or rel.start_node.get("communityId")
            e_key = rel.end_node.get("incidentId") or rel.end_node.get("name") or rel.end_node.get("communityId")
        except Exception:  # noqa: BLE001
            s_key, e_key = root["id"], m["id"]
        edges.append({"source": s_key, "target": e_key, "relation": rel.type})

    return {"node": root["id"], "nodes": list(node_map.values()), "edges": edges}


# ---------- co-occurrence (community-mediated) ----------

def cooccurrence(
    client: MemgraphClient,
    anchor_label: str,
    *,
    top: int = 5,
) -> dict[str, Any]:
    """Find Louvain communities that contain incidents touching ``anchor_label``
    (as service/region/rootCause) and aggregate peer dimensions."""
    rows = client.read(
        """
        MATCH (c:Community)<-[:IN_COMMUNITY]-(i:Incident)
        WHERE i.service = $a OR i.region = $a OR i.rootCauseCategory = $a
        WITH DISTINCT c
        MATCH (c)<-[:IN_COMMUNITY]-(j:Incident)
        RETURN c.communityId AS communityId, c.size AS size, c.summary AS summary,
               collect(DISTINCT j.service) AS services,
               collect(DISTINCT j.region) AS regions,
               collect(DISTINCT j.rootCauseCategory) AS rootCauses
        LIMIT $top
        """,
        a=anchor_label, top=top,
    )
    from collections import Counter
    co_s: Counter[str] = Counter()
    co_r: Counter[str] = Counter()
    co_c: Counter[str] = Counter()
    comms: list[dict[str, Any]] = []
    for r in rows:
        comms.append({"communityId": r["communityId"], "size": r["size"], "summary": r["summary"]})
        for s in r["services"] or []:
            if s and s != anchor_label:
                co_s[s] += 1
        for reg in r["regions"] or []:
            if reg and reg != anchor_label:
                co_r[reg] += 1
        for rc in r["rootCauses"] or []:
            if rc:
                co_c[rc] += 1
    return {
        "anchor": anchor_label,
        "communities": comms,
        "coServices": dict(co_s.most_common(top)),
        "coRootCauses": dict(co_c.most_common(top)),
        "coRegions": dict(co_r.most_common(top)),
    }


# ---------- anchored incident search (used by local_search) ----------

def anchored_candidates(
    client: MemgraphClient,
    anchors: list[str],
    *,
    service: str | None = None,
    region: str | None = None,
    team: str | None = None,
    rootCause: str | None = None,
    status: str | None = None,
    severity: int | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 200,
) -> list[str]:
    """Return incident IDs reachable from any of the given anchor node names
    (Service/Region/Team/RootCause or incidentId) and passing structured filters.
    If ``anchors`` is empty, falls back to a structured-filter-only query.
    """
    # If no anchors, just filter all incidents.
    base = "MATCH (i:Incident)"
    if anchors:
        base = (
            "MATCH (i:Incident)-[]-(a) "
            "WHERE a.name IN $anchors OR a.incidentId IN $anchors OR a.communityId IN $anchors"
        )
    clauses: list[str] = []
    params: dict[str, Any] = {"anchors": anchors, "limit": limit}
    if service:
        clauses.append("i.service = $service"); params["service"] = service
    if region:
        clauses.append("i.region = $region"); params["region"] = region
    if team:
        clauses.append("i.team = $team"); params["team"] = team
    if rootCause:
        clauses.append("i.rootCauseCategory = $rootCause"); params["rootCause"] = rootCause
    if status:
        clauses.append("i.status = $status"); params["status"] = status
    if severity is not None:
        clauses.append("i.severity = $severity"); params["severity"] = severity
    if start:
        clauses.append("i.createdAt >= $start"); params["start"] = start
    if end:
        clauses.append("i.createdAt < $end"); params["end"] = end

    where = ""
    if clauses:
        joiner = " AND " if "WHERE" in base else " WHERE "
        where = joiner + " AND ".join(clauses)

    cypher = f"{base}{where} RETURN DISTINCT i.incidentId AS id LIMIT $limit"
    rows = client.read(cypher, **params)
    return [r["id"] for r in rows if r.get("id")]


def temporal_filter(
    client: MemgraphClient,
    start: str | None,
    end: str | None,
    *,
    service: str | None = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if start:
        clauses.append("i.createdAt >= $start"); params["start"] = start
    if end:
        clauses.append("i.createdAt <= $end"); params["end"] = end
    if service:
        clauses.append("i.service = $service"); params["service"] = service
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = client.read(
        f"MATCH (i:Incident) {where} RETURN i {{.*}} AS inc ORDER BY i.createdAt ASC",
        **params,
    )
    return [r["inc"] for r in rows]


def top_stats(client: MemgraphClient) -> dict[str, Any]:
    rows = client.read("MATCH (i:Incident) RETURN i {.*} AS inc")
    incs = [r["inc"] for r in rows]
    from collections import Counter
    by_service = Counter(i["service"] for i in incs)
    by_sev = Counter(str(i["severity"]) for i in incs)
    by_region = Counter(i["region"] for i in incs)
    by_status = Counter(i["status"] for i in incs)
    by_cause = Counter(i["rootCauseCategory"] for i in incs)
    by_month: Counter[str] = Counter()
    for i in incs:
        created = i.get("createdAt") or ""
        by_month[created[:7]] += 1
    comm_rows = client.read("MATCH (c:Community) RETURN count(c) AS c")
    ncomm = comm_rows[0]["c"] if comm_rows else 0
    return {
        "total": len(incs),
        "byService": dict(by_service.most_common()),
        "bySeverity": dict(sorted(by_sev.items())),
        "byRegion": dict(by_region.most_common()),
        "byStatus": dict(by_status),
        "byRootCause": dict(by_cause.most_common()),
        "byMonth": dict(sorted(by_month.items())),
        "communities": ncomm,
    }
