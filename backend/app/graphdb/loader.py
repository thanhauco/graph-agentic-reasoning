"""Load the NetworkX-built KG into Memgraph via Cypher UNWIND batches.

The in-process index is still authoritative for embeddings/communities; this
layer just mirrors the structural graph (incidents + entity nodes + typed
relationships) into Memgraph so the agent can run Cypher traversals.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from app.graphdb.client import MemgraphClient
from app.graphdb.schema import apply_schema, drop_all

log = logging.getLogger("icm.graphdb.loader")


_ENTITY_LABELS = {
    "Incident": "Incident",
    "Service": "Service",
    "Region": "Region",
    "Team": "Team",
    "RootCauseCategory": "RootCauseCategory",
    "Community": "Community",
}

# Relation -> (source label, target label, cypher relation type)
_RELATION_MAP = {
    "affects": ("Incident", "Service", "AFFECTS"),
    "in_region": ("Incident", "Region", "IN_REGION"),
    "owned_by": ("Incident", "Team", "OWNED_BY"),
    "caused_by": ("Incident", "RootCauseCategory", "CAUSED_BY"),
    "in_community": ("Incident", "Community", "IN_COMMUNITY"),
    "hosted_in": ("Service", "Region", "HOSTED_IN"),
    "owns": ("Team", "Service", "OWNS"),
}


def _node_rows_by_label(g: nx.MultiDiGraph) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {lbl: [] for lbl in _ENTITY_LABELS}
    for n, d in g.nodes(data=True):
        t = d.get("type")
        if t not in _ENTITY_LABELS:
            continue
        row: dict[str, Any] = {"id": n}
        if t == "Incident":
            row.update({
                "incidentId": n,
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
            })
        elif t == "Community":
            row.update({
                "communityId": n,
                "size": d.get("size"),
                "summary": d.get("summary"),
            })
        else:
            row.update({"name": d.get("label") or n})
        buckets[t].append(row)
    return buckets


def _edge_rows_by_rel(g: nx.MultiDiGraph) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {k: [] for k in _RELATION_MAP}
    seen: set[tuple[str, str, str]] = set()
    for u, v, data in g.edges(data=True):
        rel = data.get("relation")
        if rel not in _RELATION_MAP:
            continue
        key = (rel, u, v)
        if key in seen:
            continue
        seen.add(key)
        buckets[rel].append({"src": u, "dst": v})
    return buckets


_MERGE_TEMPLATES = {
    "Incident": """
        UNWIND $rows AS row
        MERGE (n:Incident {incidentId: row.incidentId})
        SET n.title = row.title,
            n.severity = row.severity,
            n.status = row.status,
            n.service = row.service,
            n.region = row.region,
            n.team = row.team,
            n.rootCauseCategory = row.rootCauseCategory,
            n.mitigation = row.mitigation,
            n.createdAt = row.createdAt,
            n.impactedCustomers = row.impactedCustomers
    """,
    "Service": "UNWIND $rows AS row MERGE (n:Service {name: row.name})",
    "Region": "UNWIND $rows AS row MERGE (n:Region {name: row.name})",
    "Team": "UNWIND $rows AS row MERGE (n:Team {name: row.name})",
    "RootCauseCategory": "UNWIND $rows AS row MERGE (n:RootCauseCategory {name: row.name})",
    "Community": """
        UNWIND $rows AS row
        MERGE (n:Community {communityId: row.communityId})
        SET n.size = row.size, n.summary = row.summary
    """,
}


def _edge_cypher(rel_key: str) -> str:
    src_lbl, dst_lbl, rel_type = _RELATION_MAP[rel_key]
    src_key = "incidentId" if src_lbl == "Incident" else ("communityId" if src_lbl == "Community" else "name")
    dst_key = "incidentId" if dst_lbl == "Incident" else ("communityId" if dst_lbl == "Community" else "name")
    return (
        f"UNWIND $rows AS row "
        f"MATCH (a:{src_lbl} {{{src_key}: row.src}}), (b:{dst_lbl} {{{dst_key}: row.dst}}) "
        f"MERGE (a)-[:{rel_type}]->(b)"
    )


def load_graph(client: MemgraphClient, g: nx.MultiDiGraph, *, wipe: bool = True) -> dict[str, int]:
    """Mirror the NetworkX graph into Memgraph. Returns counts per kind."""
    if wipe:
        drop_all(client)
    apply_schema(client)

    stats: dict[str, int] = {}

    node_buckets = _node_rows_by_label(g)
    for label, rows in node_buckets.items():
        if not rows:
            continue
        n = client.write_many(_MERGE_TEMPLATES[label], rows)
        stats[f"nodes:{label}"] = n

    edge_buckets = _edge_rows_by_rel(g)
    for rel_key, rows in edge_buckets.items():
        if not rows:
            continue
        n = client.write_many(_edge_cypher(rel_key), rows)
        stats[f"edges:{rel_key}"] = n

    return stats


def count_graph(client: MemgraphClient) -> dict[str, int]:
    rows = client.read("MATCH (n) RETURN count(n) AS c")
    nodes = rows[0]["c"] if rows else 0
    rows = client.read("MATCH ()-[r]->() RETURN count(r) AS c")
    edges = rows[0]["c"] if rows else 0
    return {"nodes": nodes, "edges": edges}
