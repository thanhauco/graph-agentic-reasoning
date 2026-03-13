"""Cypher DDL for the IcM knowledge graph in Memgraph.

Memgraph supports label-property indexes and existence/uniqueness constraints.
Run this once after the container is up (the loader does it automatically).
"""

from __future__ import annotations

from app.graphdb.client import MemgraphClient

SCHEMA_STATEMENTS: list[str] = [
    # Uniqueness on identifier fields.
    "CREATE CONSTRAINT ON (n:Incident) ASSERT n.incidentId IS UNIQUE;",
    "CREATE CONSTRAINT ON (n:Service) ASSERT n.name IS UNIQUE;",
    "CREATE CONSTRAINT ON (n:Region) ASSERT n.name IS UNIQUE;",
    "CREATE CONSTRAINT ON (n:Team) ASSERT n.name IS UNIQUE;",
    "CREATE CONSTRAINT ON (n:RootCauseCategory) ASSERT n.name IS UNIQUE;",
    "CREATE CONSTRAINT ON (n:Owner) ASSERT n.name IS UNIQUE;",
    "CREATE CONSTRAINT ON (n:Component) ASSERT n.name IS UNIQUE;",
    "CREATE CONSTRAINT ON (n:Mitigation) ASSERT n.name IS UNIQUE;",
    "CREATE CONSTRAINT ON (n:Community) ASSERT n.communityId IS UNIQUE;",
    # Label-property indexes for fast lookups.
    "CREATE INDEX ON :Incident(incidentId);",
    "CREATE INDEX ON :Incident(severity);",
    "CREATE INDEX ON :Incident(createdAt);",
    "CREATE INDEX ON :Incident(service);",
    "CREATE INDEX ON :Incident(region);",
    "CREATE INDEX ON :Service(name);",
    "CREATE INDEX ON :Region(name);",
    "CREATE INDEX ON :Team(name);",
    "CREATE INDEX ON :RootCauseCategory(name);",
    "CREATE INDEX ON :Owner(name);",
    "CREATE INDEX ON :Component(name);",
    "CREATE INDEX ON :Mitigation(name);",
    "CREATE INDEX ON :Community(communityId);",
]


def apply_schema(client: MemgraphClient) -> None:
    for stmt in SCHEMA_STATEMENTS:
        try:
            client.write(stmt)
        except Exception:  # noqa: BLE001
            # Memgraph raises if constraint/index already exists; ignore.
            pass


def drop_all(client: MemgraphClient) -> None:
    client.write("MATCH (n) DETACH DELETE n;")
