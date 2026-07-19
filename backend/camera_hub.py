"""Owns cam2 (the reference/model-house scan camera).

cam0 and cam1 stay exclusively owned by house_builder.verifier.PlacementVerifier
during a build -- this hub never touches them, so there's no device contention.
cam2 is a separate, dedicated camera pointed at a human-built reference house.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

_PREVIEW_FPS = 15


class CameraUnavailableError(RuntimeError):
    """Raised when cam2 could not be opened or has not produced a frame yet."""


class CameraHub:
    def __init__(self, camera_config: dict[str, Any] | None) -> None:
        self._config = camera_config or {}
        self._capture: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = False

    def start(self) -> None:
        if not self._config:
            log.warning("No cam2 entry in config.yaml; reference-scan camera disabled.")
            return
        try:
            capture = cv2.VideoCapture(int(self._config["index"]))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self._config["width"]))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._config["height"]))
            capture.set(cv2.CAP_PROP_FPS, int(self._config["fps"]))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Could not open cam2 at index {self._config['index']}.")
        except Exception:
            log.warning(
                "cam2 unavailable; the reference-scan panel will report camera-unavailable.",
                exc_info=True,
            )
            return
        self._capture = capture
        self.available = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()

    def _capture_loop(self) -> None:
        assert self._capture is not None
        while not self._stop_event.is_set():
            ok, frame = self._capture.read()
            if ok and frame is not None:
                with self._lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.05)

    def latest_frame(self) -> np.ndarray:
        """Return a copy of the most recently captured frame (BGR, as cv2 reads it)."""
        with self._lock:
            if self._latest_frame is None:
                raise CameraUnavailableError("cam2 has not produced a frame yet.")
            return self._latest_frame.copy()

    def jpeg_stream(self) -> Iterator[bytes]:
        """Yield multipart/x-mixed-replace JPEG chunks for an MJPEG <img> preview."""
        boundary = b"--frame"
        interval = 1.0 / _PREVIEW_FPS
        while not self._stop_event.is_set():
            try:
                frame = self.latest_frame()
            except CameraUnavailableError:
                time.sleep(interval)
                continue
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            )
            time.sleep(interval)
