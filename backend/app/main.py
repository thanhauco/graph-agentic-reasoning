from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import agent as agent_routes
from app.routes import chat as chat_routes
from app.routes import graph as graph_routes
from app.state import AppState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("icm.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("Loading index from %s", settings.index_path)
    app.state.store = AppState.load(settings)
    log.info(
        "Loaded %d incidents, graph nodes=%d edges=%d communities=%d",
        len(app.state.store.incidents),
        app.state.store.graph.number_of_nodes(),
        app.state.store.graph.number_of_edges(),
        len(app.state.store.communities),
    )
    if settings.graph_backend.lower() == "memgraph":
        try:
            from app.graphdb import get_client
            from app.graphdb.loader import count_graph, load_graph

            client = get_client()
            if client.ping():
                counts = count_graph(client)
                if counts["nodes"] == 0 and app.state.store.graph.number_of_nodes() > 0:
                    log.info("Memgraph empty — loading %d nodes from local index",
                             app.state.store.graph.number_of_nodes())
                    load_graph(client, app.state.store.graph, wipe=False)
                    counts = count_graph(client)
                log.info("Memgraph reachable: %s nodes / %s edges", counts["nodes"], counts["edges"])
            else:
                log.warning("Memgraph configured but unreachable at %s — falling back to NetworkX",
                            settings.memgraph_url)
        except Exception as e:  # noqa: BLE001
            log.warning("Memgraph init failed: %s — falling back to NetworkX", e)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="IcM GraphRAG Agentic API",
        version="0.1.0",
        description="Agentic GraphRAG over synthetic 2026 Azure incidents",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        from app.graphdb import is_memgraph_enabled
        return {
            "ok": True,
            "service": "icm-graphrag",
            "graphBackend": settings.graph_backend,
            "memgraphReachable": is_memgraph_enabled(),
        }

    app.include_router(graph_routes.router, prefix="/api")
    app.include_router(chat_routes.router, prefix="/api")
    app.include_router(agent_routes.router, prefix="/api")
    return app


app = create_app()
