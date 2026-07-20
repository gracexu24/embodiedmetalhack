#!/usr/bin/env python3
"""Minimal human-builder UI: laptop camera preview + one Detect button.

Run:
  PYTHONPATH=src:. python human_builder_ui.py
  # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from human_builder import detect_model_house, request_to_sentence

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Human Builder</title>
  <style>
    :root { color-scheme: light; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin: 24px; max-width: 720px; }
    h1 { margin: 0 0 8px; font-size: 24px; }
    p { color: #555; margin: 0 0 16px; }
    img { width: 100%; background: #111; border-radius: 8px; display: block; }
    button {
      margin-top: 16px; padding: 10px 18px; font-size: 16px; cursor: pointer;
      border-radius: 8px; border: 1px solid #222; background: #111; color: #fff;
    }
    button:disabled { opacity: 0.5; cursor: wait; }
    #out {
      margin-top: 16px; padding: 12px; border-radius: 8px; background: #f4f4f5;
      font-family: ui-monospace, monospace; white-space: pre-wrap; min-height: 2.5em;
    }
    .err { color: #b42318; }
    .ok { color: #067647; }
  </style>
</head>
<body>
  <h1>Human Builder</h1>
  <p>Live camera preview. Click Detect to parse door / wall / roof colors into a harness sentence.</p>
  <img id="preview" src="/preview" alt="Camera preview" />
  <button id="btn" onclick="detect()">Detect house</button>
  <div id="out">Waiting for detect…</div>
  <script>
    async function detect() {
      const btn = document.getElementById('btn');
      const out = document.getElementById('out');
      btn.disabled = true;
      out.className = '';
      out.textContent = 'Detecting…';
      try {
        const res = await fetch('/detect', { method: 'POST' });
        const data = await res.json();
        if (data.error) {
          out.className = 'err';
          out.textContent = data.error;
        } else {
          out.className = 'ok';
          out.textContent = data.sentence;
        }
      } catch (err) {
        out.className = 'err';
        out.textContent = String(err);
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""


class CameraFeed:
    def __init__(self, index: int, width: int, height: int) -> None:
        self.index = index
        self.capture = cv2.VideoCapture(index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera index {index}.")
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # Wait briefly for the first frame.
        for _ in range(50):
            if self.latest() is not None:
                break
            time.sleep(0.05)

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.capture.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.05)

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def mjpeg(self):
        while not self._stop.is_set():
            frame = self.latest()
            if frame is None:
                time.sleep(0.05)
                continue
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            )
            time.sleep(1.0 / 15.0)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.capture.release()


def load_human_builder_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "human_builder" not in config:
        raise ValueError(f"{path} must define a human_builder section.")
    return config["human_builder"]


def create_app(camera: CameraFeed, hb_config: dict[str, Any]) -> FastAPI:
    app = FastAPI(title="Human Builder UI")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/preview")
    def preview() -> StreamingResponse:
        return StreamingResponse(
            camera.mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/detect")
    def detect() -> dict[str, str]:
        frame = camera.latest()
        if frame is None:
            return {"error": "No camera frame yet."}
        try:
            request = detect_model_house(frame, hb_config)
            sentence = request_to_sentence(request)
            print(f"[human_builder_ui] {sentence}", flush=True)
            return {"sentence": sentence}
        except ValueError as exc:
            print(f"[human_builder_ui] detect failed: {exc}", flush=True)
            return {"error": str(exc)}

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple human-builder camera UI.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    hb_config = load_human_builder_config(args.config)
    camera = CameraFeed(args.camera_index, 640, 480)
    print(
        f"[human_builder_ui] camera index={args.camera_index} "
        f"-> http://{args.host}:{args.port}/",
        flush=True,
    )
    app = create_app(camera, hb_config)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
