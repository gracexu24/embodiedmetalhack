"""Highlights reel: uses Rerun's Query API to pull key moments back out of a
completed build's recording, each paired with the nearest cam1 thumbnail.

Column names below (TextLog:text, Image:buffer, Image:format, harness_step) were
confirmed against rerun-sdk 0.34.1 by logging a probe recording and inspecting
Chunk.to_record_batch() -- see rerun.experimental.RrdReader / LazyStore / Chunk.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rerun.experimental as rre

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"

_WATCHED_ENTITIES = {
    "/harness/verification": "verification",
    "/harness/instruction": "instruction",
}
_THUMBNAIL_ENTITY = "/harness/cameras/cam1"
_THUMBNAIL_CHANNELS = 3  # cam0/cam1 frames are always logged as RGB uint8 (verifier.py)


def recording_path(run_id: str) -> Path:
    return RECORDINGS_DIR / f"{run_id}.rrd"


def build_highlights(run_id: str) -> list[dict[str, Any]]:
    path = recording_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"No recording found for run {run_id!r}.")

    # .stream() (not .store()) because the file sink writes without a footer/manifest
    # while the recording is not explicitly finalized -- .store()'s lazy, index-based
    # reader requires one. .stream().to_chunks() reads the chunks directly instead.
    chunks = rre.RrdReader(path).stream().to_chunks()

    events: list[dict[str, Any]] = []
    thumbnails: dict[int, str] = {}

    for chunk in chunks:
        entity_path = chunk.entity_path
        if entity_path in _WATCHED_ENTITIES:
            kind = _WATCHED_ENTITIES[entity_path]
            batch = chunk.to_record_batch().to_pydict()
            for step, text in zip(batch["harness_step"], batch["TextLog:text"], strict=True):
                events.append({"step": step, "kind": kind, "label": text[0]})
        elif entity_path == _THUMBNAIL_ENTITY:
            batch = chunk.to_record_batch().to_pydict()
            for step, buffer, image_format in zip(
                batch["harness_step"], batch["Image:buffer"], batch["Image:format"], strict=True
            ):
                thumbnail = _encode_thumbnail(buffer[0], image_format[0])
                if thumbnail is not None:
                    thumbnails[step] = thumbnail

    for event in events:
        event["thumbnail_base64"] = _nearest_thumbnail(thumbnails, event["step"])

    # Verification pass/fail moments are the most "highlight-worthy"; surface them
    # first, then instructions, preserving chronological order within each group.
    events.sort(key=lambda event: (0 if event["kind"] == "verification" else 1, event["step"]))
    return events


def _encode_thumbnail(buffer: list[int], image_format: dict[str, Any]) -> str | None:
    try:
        rgb = np.asarray(buffer, dtype=np.uint8).reshape(
            image_format["height"], image_format["width"], _THUMBNAIL_CHANNELS
        )
    except ValueError:
        return None
    ok, encoded = cv2.imencode(".jpg", rgb[:, :, ::-1])
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _nearest_thumbnail(thumbnails: dict[int, str], step: int) -> str | None:
    if not thumbnails:
        return None
    nearest_step = min(thumbnails, key=lambda candidate: abs(candidate - step))
    return thumbnails[nearest_step]
