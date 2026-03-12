"""Bolt client for Memgraph (Cypher-compatible with the Neo4j driver).

Memgraph speaks the Bolt protocol, so we reuse ``neo4j`` as the client. We keep
a single module-level driver with a connection pool and expose small helpers:
``ping``, ``read``, ``write``. Failures are swallowed into ``is_available()``
so the rest of the app can gracefully fall back to NetworkX.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Iterable

from app.config import Settings, get_settings

log = logging.getLogger("icm.graphdb.client")


class MemgraphClient:
    def __init__(self, url: str, user: str = "", password: str = "") -> None:
        self.url = url
        self.user = user
        self.password = password
        self._driver = None
        self._available: bool | None = None

    # ---- lifecycle ----
    def _ensure_driver(self):
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import GraphDatabase

            auth = (self.user, self.password) if self.user else None
            self._driver = GraphDatabase.driver(self.url, auth=auth)
        except Exception as e:  # noqa: BLE001
            log.warning("Memgraph driver init failed: %s", e)
            self._driver = None
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:  # noqa: BLE001
                pass
            self._driver = None
            self._available = None

    # ---- probes ----
    def ping(self) -> bool:
        drv = self._ensure_driver()
        if drv is None:
            self._available = False
            return False
        try:
            with drv.session() as s:
                s.run("RETURN 1 AS ok").consume()
            self._available = True
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("Memgraph ping failed: %s", e)
            self._available = False
            return False

    def is_available(self) -> bool:
        if self._available is None:
            return self.ping()
        return self._available

    # ---- execution ----
    def read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        drv = self._ensure_driver()
        if drv is None:
            return []
        with drv.session() as s:
            result = s.run(cypher, **params)
            return [r.data() for r in result]

    def write(self, cypher: str, **params: Any) -> None:
        drv = self._ensure_driver()
        if drv is None:
            return
        with drv.session() as s:
            s.run(cypher, **params).consume()

    def write_many(self, cypher: str, rows: Iterable[dict[str, Any]], *, batch: int = 500) -> int:
        """Batched UNWIND write. Cypher must accept ``$rows`` list parameter."""
        drv = self._ensure_driver()
        if drv is None:
            return 0
        total = 0
        buffer: list[dict[str, Any]] = []
        with drv.session() as s:
            for row in rows:
                buffer.append(row)
                if len(buffer) >= batch:
                    s.run(cypher, rows=buffer).consume()
                    total += len(buffer)
                    buffer = []
            if buffer:
                s.run(cypher, rows=buffer).consume()
                total += len(buffer)
        return total


@lru_cache(maxsize=1)
def get_client() -> MemgraphClient:
    s: Settings = get_settings()
    return MemgraphClient(s.memgraph_url, s.memgraph_user, s.memgraph_password)


def is_memgraph_enabled() -> bool:
    """True iff configured backend is memgraph AND the driver can reach it."""
    s = get_settings()
    if s.graph_backend.lower() != "memgraph":
        return False
    return get_client().is_available()
