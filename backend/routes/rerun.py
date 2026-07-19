"""Exposes the embeddable Rerun web-viewer URL (cam0/cam1 + arm joints + text log)."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/viewer_url")
def viewer_url(request: Request) -> dict[str, str]:
    return {"viewer_url": request.app.state.rerun_endpoints.viewer_url}
