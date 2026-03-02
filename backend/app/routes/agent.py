from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["agent"])


@router.get("/agent/ping")
def ping() -> dict:
    return {"ok": True, "note": "agent trace wired in Phase 4"}
