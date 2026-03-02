from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["graph"])


@router.get("/stats")
def stats() -> dict:
    return {"ok": True, "note": "populated after Phase 2"}
