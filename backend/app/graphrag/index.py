"""Build the GraphRAG index from incidents.json.

Produces:
- data/index/graph.pickle        — NetworkX MultiDiGraph
- data/index/communities.json    — Louvain communities w/ LLM (or heuristic) summaries
- data/index/embeddings.npz      — incident + community embeddings
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import community as community_louvain  # python-louvain
import networkx as nx
import numpy as np

from app.config import Settings, get_settings

log = logging.getLogger("icm.graphrag.index")


# ---------- Graph construction ----------

NODE_TYPES = [
    "Incident", "Service", "Region", "Team", "Owner",
    "RootCauseCategory", "Mitigation", "Component",
]


def _nid(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def build_graph(incidents: list[dict[str, Any]]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for inc in incidents:
        iid = inc["incidentId"]
        g.add_node(
            iid,
            type="Incident",
            label=iid,
            title=inc["title"],
            description=inc["description"],
            severity=inc["severity"],
            status=inc["status"],
            service=inc["service"],
            region=inc["region"],
            team=inc["team"],
            owner=inc["owner"],
            createdAt=inc["createdAt"],
            rootCauseCategory=inc["rootCauseCategory"],
            mitigation=inc["mitigation"],
            impactedCustomers=inc["impactedCustomers"],
            tags=inc.get("tags", []),
        )
        svc = _nid("Service", inc["service"])
        reg = _nid("Region", inc["region"])
        team = _nid("Team", inc["team"])
        owner = _nid("Owner", inc["owner"])
        rc = _nid("RootCauseCategory", inc["rootCauseCategory"])
        mit = _nid("Mitigation", inc["mitigation"])

        for node_id, kind, label in [
            (svc, "Service", inc["service"]),
            (reg, "Region", inc["region"]),
            (team, "Team", inc["team"]),
            (owner, "Owner", inc["owner"]),
            (rc, "RootCauseCategory", inc["rootCauseCategory"]),
            (mit, "Mitigation", inc["mitigation"]),
        ]:
            if not g.has_node(node_id):
                g.add_node(node_id, type=kind, label=label)

        g.add_edge(iid, svc, relation="AFFECTS")
        g.add_edge(iid, reg, relation="LOCATED_IN")
        g.add_edge(iid, team, relation="OWNED_BY")
        g.add_edge(iid, owner, relation="ASSIGNED_TO")
        g.add_edge(iid, rc, relation="CAUSED_BY")
        g.add_edge(iid, mit, relation="MITIGATED_BY")

        for comp in inc.get("componentSignatures", []):
            cid = _nid("Component", comp)
            if not g.has_node(cid):
                g.add_node(cid, type="Component", label=comp)
            g.add_edge(iid, cid, relation="INVOLVES")

        for linked in inc.get("linkedIncidents", []):
            if linked != iid:
                g.add_edge(iid, linked, relation="RELATED_TO")

    return g


# ---------- Communities ----------

def detect_communities(g: nx.MultiDiGraph) -> dict[str, int]:
    """Louvain over an undirected projection restricted to Incident nodes
    connected via shared entities."""
    ug = nx.Graph()
    incident_nodes = [n for n, d in g.nodes(data=True) if d.get("type") == "Incident"]
    ug.add_nodes_from(incident_nodes)

    # Connect incidents that share a service+root-cause, or are RELATED_TO.
    by_key = defaultdict(list)
    for n in incident_nodes:
        d = g.nodes[n]
        by_key[(d["service"], d["rootCauseCategory"])].append(n)
    for group in by_key.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ug.add_edge(group[i], group[j], weight=1.0)
    for u, v, data in g.edges(data=True):
        if data.get("relation") == "RELATED_TO" and ug.has_node(u) and ug.has_node(v):
            if ug.has_edge(u, v):
                ug[u][v]["weight"] += 2.0
            else:
                ug.add_edge(u, v, weight=2.0)

    if ug.number_of_edges() == 0:
        return {n: 0 for n in incident_nodes}
    return community_louvain.best_partition(ug, random_state=42)


def _heuristic_summary(incidents: list[dict[str, Any]]) -> str:
    services = Counter(i["service"] for i in incidents)
    causes = Counter(i["rootCauseCategory"] for i in incidents)
    regions = Counter(i["region"] for i in incidents)
    sev_top = min(i["severity"] for i in incidents)
    total_impact = sum(i.get("impactedCustomers", 0) for i in incidents)
    top_services = ", ".join(f"{k} ({v})" for k, v in services.most_common(3))
    top_causes = ", ".join(f"{k} ({v})" for k, v in causes.most_common(3))
    top_regions = ", ".join(f"{k} ({v})" for k, v in regions.most_common(3))
    sample_titles = "; ".join(i["title"] for i in incidents[:3])
    return (
        f"Cluster of {len(incidents)} incidents. "
        f"Top services: {top_services}. Root causes: {top_causes}. "
        f"Regions: {top_regions}. Highest severity in cluster: Sev{sev_top}. "
        f"Total impacted customers (approx): {total_impact}. "
        f"Representative incidents: {sample_titles}."
    )


def build_communities(
    g: nx.MultiDiGraph,
    incidents: list[dict[str, Any]],
    partition: dict[str, int],
    *,
    llm_summary: bool = False,
) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    id_to_inc = {i["incidentId"]: i for i in incidents}
    for node, comm_id in partition.items():
        if node in id_to_inc:
            buckets[comm_id].append(id_to_inc[node])

    communities: list[dict[str, Any]] = []
    for comm_id, incs in sorted(buckets.items()):
        if len(incs) < 2:
            continue
        summary = _heuristic_summary(incs)
        if llm_summary:
            try:
                from app.graphrag.llm import summarize_community
                summary = summarize_community(incs) or summary
            except Exception as e:  # noqa: BLE001
                log.warning("LLM summary failed for community %s: %s", comm_id, e)
        communities.append({
            "communityId": f"C-{comm_id:03d}",
            "incidentIds": [i["incidentId"] for i in incs],
            "size": len(incs),
            "summary": summary,
            "topServices": [s for s, _ in Counter(i["service"] for i in incs).most_common(3)],
            "topRootCauses": [s for s, _ in Counter(i["rootCauseCategory"] for i in incs).most_common(3)],
            "topRegions": [s for s, _ in Counter(i["region"] for i in incs).most_common(3)],
        })
    communities.sort(key=lambda c: c["size"], reverse=True)
    return communities


# ---------- Embeddings ----------

def _fake_embedding(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic hashing embedding fallback (no network required)."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v


def _incident_text(inc: dict[str, Any]) -> str:
    return (
        f"{inc['incidentId']} | {inc['title']}\n"
        f"Service: {inc['service']} | Region: {inc['region']} | "
        f"Severity: Sev{inc['severity']} | Status: {inc['status']}\n"
        f"Root cause: {inc['rootCauseCategory']} | Mitigation: {inc['mitigation']}\n"
        f"{inc['description']}"
    )


def build_embeddings(
    incidents: list[dict[str, Any]],
    communities: list[dict[str, Any]],
    *,
    use_llm: bool = False,
) -> dict[str, np.ndarray | list[str]]:
    embed_fn = _fake_embedding
    if use_llm:
        try:
            from app.graphrag.llm import embed_texts
            inc_texts = [_incident_text(i) for i in incidents]
            comm_texts = [c["summary"] for c in communities]
            log.info("Embedding via Azure OpenAI (%d + %d)", len(inc_texts), len(comm_texts))
            inc_vecs = embed_texts(inc_texts)
            comm_vecs = embed_texts(comm_texts) if comm_texts else np.zeros((0, inc_vecs.shape[1]), dtype=np.float32)
            return {
                "incident_ids": np.array([i["incidentId"] for i in incidents]),
                "incident_embeddings": inc_vecs.astype(np.float32),
                "community_ids": np.array([c["communityId"] for c in communities]),
                "community_embeddings": comm_vecs.astype(np.float32),
            }
        except Exception as e:  # noqa: BLE001
            log.warning("Azure OpenAI embed failed, falling back to deterministic: %s", e)

    inc_vecs = np.stack([embed_fn(_incident_text(i)) for i in incidents])
    comm_vecs = (
        np.stack([embed_fn(c["summary"]) for c in communities])
        if communities else np.zeros((0, 256), dtype=np.float32)
    )
    return {
        "incident_ids": np.array([i["incidentId"] for i in incidents]),
        "incident_embeddings": inc_vecs,
        "community_ids": np.array([c["communityId"] for c in communities]),
        "community_embeddings": comm_vecs,
    }


# ---------- Pipeline ----------

def run(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    data_file = settings.data_path / "incidents.json"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Missing {data_file}. Run `python -m scripts.gen_incidents` first."
        )
    incidents = json.loads(data_file.read_text(encoding="utf-8"))
    log.info("Loaded %d incidents", len(incidents))

    g = build_graph(incidents)
    log.info("Graph: %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges())

    partition = detect_communities(g)
    communities = build_communities(
        g, incidents, partition, llm_summary=settings.use_llm_for_index and settings.has_azure_openai
    )
    log.info("Communities: %d", len(communities))

    embeddings = build_embeddings(
        incidents, communities,
        use_llm=settings.use_llm_for_index and settings.has_azure_openai,
    )

    idx = settings.index_path
    idx.mkdir(parents=True, exist_ok=True)
    with (idx / "graph.pickle").open("wb") as fh:
        pickle.dump(g, fh)
    (idx / "communities.json").write_text(
        json.dumps(communities, indent=2), encoding="utf-8"
    )
    np.savez(idx / "embeddings.npz", **embeddings)  # type: ignore[arg-type]

    return {
        "incidents": len(incidents),
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "communities": len(communities),
        "index_dir": str(idx),
    }
