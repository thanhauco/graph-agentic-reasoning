from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["chat"])


@router.get("/chat/ping")
def ping() -> dict:
    return {"ok": True, "note": "chat stream wired in Phase 4"}
