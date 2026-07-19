"""Highlights reel endpoint -- see backend/highlights.py for the Query API extraction."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import highlights as highlights_module

router = APIRouter()


@router.get("/{run_id}")
def get_highlights(run_id: str) -> list[dict[str, object]]:
    try:
        return highlights_module.build_highlights(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
