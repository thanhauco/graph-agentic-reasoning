"""Memgraph-backed graph database layer.

Exposes a thin async-safe client plus typed Cypher query helpers. The agent's
retrieval primitives dispatch here when ``settings.graph_backend == 'memgraph'``.
"""

from app.graphdb.client import MemgraphClient, get_client, is_memgraph_enabled
from app.graphdb import queries

__all__ = ["MemgraphClient", "get_client", "is_memgraph_enabled", "queries"]
