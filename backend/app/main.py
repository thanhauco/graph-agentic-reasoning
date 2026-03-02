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
        return {"ok": True, "service": "icm-graphrag"}

    app.include_router(graph_routes.router, prefix="/api")
    app.include_router(chat_routes.router, prefix="/api")
    app.include_router(agent_routes.router, prefix="/api")
    return app


app = create_app()
