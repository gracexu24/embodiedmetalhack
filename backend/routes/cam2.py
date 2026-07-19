"""Camera 3 (reference-house scan): live preview + the "Done" button.

Color detection itself is intentionally NOT implemented here -- see the TODO in
scan() below. That logic already exists as detect_model_house() in human_builder.py
on origin/main, which is deliberately not merged into this branch yet. Once it is,
swap the stub block in scan() for a real call and this endpoint's response shape
(door/wall/roof colors) is already what the frontend expects.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, StreamingResponse

from ..camera_hub import CameraHub, CameraUnavailableError

router = APIRouter()

SCANS_DIR = Path(__file__).resolve().parent.parent / "scans"

_MJPEG_BOUNDARY = "frame"


@router.get("/preview")
def preview(request: Request) -> StreamingResponse:
    hub: CameraHub = request.app.state.camera_hub
    return StreamingResponse(
        hub.jpeg_stream(),
        media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
    )


@router.post("/scan")
def scan(request: Request) -> dict[str, object]:
    """The "Done" button target: capture the current cam2 frame."""
    hub: CameraHub = request.app.state.camera_hub
    try:
        frame = hub.latest_frame()
    except CameraUnavailableError as exc:
        return {"status": "camera_unavailable", "error": str(exc)}

    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    scan_id = f"{int(time.time() * 1000)}"
    image_path = SCANS_DIR / f"{scan_id}.jpg"
    cv2.imwrite(str(image_path), frame)

    # TODO(human_builder): once origin/main's human_builder.py is merged, replace
    # this stub with:
    #   from human_builder import detect_model_house
    #   house_request = detect_model_house(frame, request.app.state.config["verification"])
    #   detected = {"door": house_request.door.value, "wall": house_request.wall.value,
    #               "roof": house_request.roof.value}
    detected = None

    return {
        "status": "captured",
        "scan_id": scan_id,
        "image_url": f"/api/cam2/scans/{scan_id}.jpg",
        "detected": detected,
        "note": "Color detection pending human_builder.py merge (see TODO in backend/routes/cam2.py).",
    }


@router.get("/scans/{scan_id}.jpg")
def get_scan(scan_id: str) -> FileResponse:
    return FileResponse(SCANS_DIR / f"{scan_id}.jpg")
