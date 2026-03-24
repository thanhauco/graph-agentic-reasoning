from __future__ import annotations

import json

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import run_agent

router = APIRouter(tags=["chat"])


class ChatHistoryTurn(BaseModel):
    question: str
    answer: str | None = None
    citations: list[str] = []
    matchIds: list[str] = []


class ChatRequest(BaseModel):
    question: str
    selectedId: str | None = None
    history: list[ChatHistoryTurn] = []


@router.post("/chat/stream")
async def chat_stream(req: Request, body: ChatRequest):
    store = req.app.state.store

    async def event_gen():
        async for ev in run_agent(
            store,
            body.question,
            history=[t.model_dump() for t in body.history],
            selected_id=body.selectedId,
        ):
            yield {"event": ev["type"], "data": json.dumps(ev)}

    return EventSourceResponse(event_gen())


@router.get("/chat/ping")
def ping() -> dict:
    return {"ok": True}
