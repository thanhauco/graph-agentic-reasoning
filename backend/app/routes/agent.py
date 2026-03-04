from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["agent"])


@router.get("/agent/trace/{session_id}")
def trace(req: Request, session_id: str) -> dict:
    store = req.app.state.store
    trace = store.sessions.get(session_id)
    if not trace:
        raise HTTPException(404, "Session not found")
    return {"sessionId": session_id, "events": trace}


@router.get("/agent/sessions")
def sessions(req: Request) -> dict:
    store = req.app.state.store
    return {
        "sessions": [
            {"sessionId": sid, "events": len(tr)} for sid, tr in list(store.sessions.items())[-20:]
        ]
    }
