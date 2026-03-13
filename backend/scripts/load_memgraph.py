"""Mirror the local NetworkX knowledge graph into Memgraph.

Usage (from backend/):
    python -m scripts.load_memgraph                  # full reload
    python -m scripts.load_memgraph --if-empty       # only load when Memgraph is empty
    python -m scripts.load_memgraph --no-wipe        # MERGE without dropping
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import get_settings
from app.graphdb.client import MemgraphClient, get_client
from app.graphdb.loader import count_graph, load_graph
from app.state import AppState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("icm.load_memgraph")


def main() -> int:
    parser = argparse.ArgumentParser(description="Load NetworkX graph into Memgraph")
    parser.add_argument("--if-empty", action="store_true",
                        help="Skip loading when Memgraph already has nodes.")
    parser.add_argument("--no-wipe", action="store_true",
                        help="MERGE into existing data instead of DETACH DELETE first.")
    args = parser.parse_args()

    settings = get_settings()
    log.info("Loading local KG from %s", settings.index_path)
    store = AppState.load(settings)
    if store.graph.number_of_nodes() == 0:
        log.error("Local index is empty — run `python -m scripts.build_index` first.")
        return 2

    client: MemgraphClient = get_client()
    if not client.ping():
        log.error("Cannot reach Memgraph at %s", settings.memgraph_url)
        return 3

    if args.if_empty:
        before = count_graph(client)
        if before["nodes"] > 0:
            log.info("Memgraph already has %s nodes / %s edges — skipping.",
                     before["nodes"], before["edges"])
            return 0

    log.info("Pushing graph into Memgraph (wipe=%s) ...", not args.no_wipe)
    stats = load_graph(client, store.graph, wipe=not args.no_wipe,
                       communities=getattr(store, "communities", None))
    for k, v in stats.items():
        log.info("  %-28s %d", k, v)
    after = count_graph(client)
    log.info("Memgraph now: %s nodes, %s edges", after["nodes"], after["edges"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
