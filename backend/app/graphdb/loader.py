"""Load the NetworkX-built KG into Memgraph via Cypher UNWIND batches."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from app.graphdb.client import MemgraphClient
from app.graphdb.schema import apply_schema, drop_all

log = logging.getLogger("icm.graphdb.loader")

_ID_PROPERTY = {"Incident": "incidentId", "Community": "communityId"}


def _id_prop(label):
    return _ID_PROPERTY.get(label, "name")


def _node_props(label, node_id, attrs):
    if label == "Incident":
        return {
            "incidentId": node_id,
            "title": attrs.get("title"),
            "severity": attrs.get("severity"),
            "status": attrs.get("status"),
            "service": attrs.get("service"),
            "region": attrs.get("region"),
            "team": attrs.get("team"),
            "rootCauseCategory": attrs.get("rootCauseCategory"),
            "mitigation": attrs.get("mitigation"),
            "createdAt": attrs.get("createdAt"),
            "impactedCustomers": attrs.get("impactedCustomers"),
        }
    if label == "Community":
        return {"communityId": node_id, "size": attrs.get("size"), "summary": attrs.get("summary")}
    return {"name": attrs.get("label") or node_id}


def _bucket_nodes(g):
    buckets = {}
    for n, d in g.nodes(data=True):
        label = d.get("type")
        if not label:
            continue
        buckets.setdefault(label, []).append(_node_props(label, n, d))
    return buckets


def _bucket_edges(g):
    buckets = {}
    seen = set()
    for u, v, data in g.edges(data=True):
        rel = data.get("relation")
        if not rel:
            continue
        u_attrs = g.nodes[u]
        v_attrs = g.nodes[v]
        u_label = u_attrs.get("type")
        v_label = v_attrs.get("type")
        if not u_label or not v_label:
            continue
        u_key = u if u_label in _ID_PROPERTY else (u_attrs.get("label") or u)
        v_key = v if v_label in _ID_PROPERTY else (v_attrs.get("label") or v)
        sig = (rel, str(u_key), str(v_key))
        if sig in seen:
            continue
        seen.add(sig)
        buckets.setdefault((u_label, v_label, rel), []).append({"src": u_key, "dst": v_key})
    return buckets


def _node_merge_cypher(label):
    if label == "Incident":
        return (
            "UNWIND $rows AS row "
            "MERGE (n:Incident {incidentId: row.incidentId}) "
            "SET n.title = row.title, n.severity = row.severity, n.status = row.status, "
            "n.service = row.service, n.region = row.region, n.team = row.team, "
            "n.rootCauseCategory = row.rootCauseCategory, n.mitigation = row.mitigation, "
            "n.createdAt = row.createdAt, n.impactedCustomers = row.impactedCustomers"
        )
    if label == "Community":
        return (
            "UNWIND $rows AS row "
            "MERGE (n:Community {communityId: row.communityId}) "
            "SET n.size = row.size, n.summary = row.summary"
        )
    return f"UNWIND $rows AS row MERGE (n:{label} {{name: row.name}})"


def _edge_merge_cypher(src_label, dst_label, rel):
    src_key = _id_prop(src_label)
    dst_key = _id_prop(dst_label)
    return (
        f"UNWIND $rows AS row "
        f"MATCH (a:{src_label} {{{src_key}: row.src}}), (b:{dst_label} {{{dst_key}: row.dst}}) "
        f"MERGE (a)-[:{rel}]->(b)"
    )


def _add_communities(communities, buckets):
    if not communities:
        return []
    buckets.setdefault("Community", []).extend([
        {"communityId": c["communityId"], "size": c.get("size"), "summary": c.get("summary")}
        for c in communities
    ])
    edges = []
    for c in communities:
        cid = c["communityId"]
        for iid in c.get("incidentIds", []) or []:
            edges.append({"src": iid, "dst": cid})
    return edges


def load_graph(client, g, *, wipe=True, communities=None):
    if wipe:
        drop_all(client)
    apply_schema(client)
    stats = {}
    node_buckets = _bucket_nodes(g)
    community_edges = _add_communities(communities, node_buckets)
    for label, rows in node_buckets.items():
        if not rows:
            continue
        n = client.write_many(_node_merge_cypher(label), rows)
        stats[f"nodes:{label}"] = n
    edge_buckets = _bucket_edges(g)
    for (src_label, dst_label, rel), rows in edge_buckets.items():
        if not rows:
            continue
        n = client.write_many(_edge_merge_cypher(src_label, dst_label, rel), rows)
        stats[f"edges:{src_label}-[{rel}]->{dst_label}"] = n
    if community_edges:
        n = client.write_many(_edge_merge_cypher("Incident", "Community", "IN_COMMUNITY"), community_edges)
        stats["edges:Incident-[IN_COMMUNITY]->Community"] = n
    return stats


def count_graph(client):
    rows = client.read("MATCH (n) RETURN count(n) AS c")
    nodes = rows[0]["c"] if rows else 0
    rows = client.read("MATCH ()-[r]->() RETURN count(r) AS c")
    edges = rows[0]["c"] if rows else 0
    return {"nodes": nodes, "edges": edges}