"""Camera 3 (reference-house scan): live preview + the "Done" / Build this button."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, StreamingResponse

from ..build_runner import BuildAlreadyRunningError, BuildRunner
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
    """Capture camera3, detect colors, and store the request (voice: Build this)."""
    hub: CameraHub = request.app.state.camera_hub
    runner: BuildRunner = request.app.state.build_runner
    print("[ui] Build This: capturing camera3 scan", flush=True)
    try:
        frame = hub.latest_frame()
    except CameraUnavailableError as exc:
        print(f"[ui] scan failed: {exc}", flush=True)
        return {"status": "camera_unavailable", "error": str(exc)}

    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    scan_id = f"{int(time.time() * 1000)}"
    image_path = SCANS_DIR / f"{scan_id}.jpg"
    cv2.imwrite(str(image_path), frame)
    print(f"[ui] scan saved to {image_path} (calibrate bands against this image)", flush=True)

    try:
        house_request = runner.detect_from_frame(frame)
        stored = runner.set_request(house_request)
    except (BuildAlreadyRunningError, RuntimeError, ValueError) as exc:
        print(f"[ui] scan detection failed: {exc}", flush=True)
        return {
            "status": "captured",
            "scan_id": scan_id,
            "image_url": f"/api/cam2/scans/{scan_id}.jpg",
            "detected": None,
            "error": str(exc),
        }

    return {
        "status": "captured",
        "scan_id": scan_id,
        "image_url": f"/api/cam2/scans/{scan_id}.jpg",
        "detected": {
            "door": house_request.door.value,
            "wall": house_request.wall.value,
            "roof": house_request.roof.value,
        },
        "request_sentence": stored["request_sentence"],
        "run_id": stored["run_id"],
    }


@router.get("/scans/{scan_id}.jpg")
def get_scan(scan_id: str) -> FileResponse:
    return FileResponse(SCANS_DIR / f"{scan_id}.jpg")
