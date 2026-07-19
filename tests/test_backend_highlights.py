"""Tests for backend.highlights, the Rerun Query-API-powered highlights reel.

Builds a small synthetic .rrd (matching the entity paths/timeline house_builder
itself logs to -- see rr_time.py, state_machine.py, verifier.py) rather than
running a full build, so this stays fast and independent of hardware/mock mode.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
import rerun as rr

from backend import highlights


@pytest.fixture
def synthetic_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(highlights, "RECORDINGS_DIR", tmp_path)
    run_id = "test-run"
    path = tmp_path / f"{run_id}.rrd"

    rr.init("house_builder_test", spawn=False)
    rr.set_sinks(rr.FileSink(str(path)))
    try:
        rr.set_time("harness_step", sequence=0)
        rr.log("/harness/verification", rr.TextLog("red door: PASS"))

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:] = (0, 0, 255)  # RGB
        rr.set_time("harness_step", sequence=1)
        rr.log("/harness/cameras/cam1", rr.Image(frame))

        rr.set_time("harness_step", sequence=2)
        rr.log("/harness/instruction", rr.TextLog("Pick up the red door block."))

        rr.set_time("harness_step", sequence=3)
        rr.log("/harness/verification", rr.TextLog("yellow wall: FAIL (block moved)"))
    finally:
        rr.disconnect()  # detach the file sink so later tests' rr.log calls don't reopen it

    return run_id


def test_build_highlights_ranks_verification_before_instruction(synthetic_recording: str) -> None:
    events = highlights.build_highlights(synthetic_recording)

    kinds_and_labels = [(event["kind"], event["label"]) for event in events]
    assert kinds_and_labels == [
        ("verification", "red door: PASS"),
        ("verification", "yellow wall: FAIL (block moved)"),
        ("instruction", "Pick up the red door block."),
    ]


def test_build_highlights_attaches_nearest_thumbnail(synthetic_recording: str) -> None:
    events = highlights.build_highlights(synthetic_recording)

    assert all(event["thumbnail_base64"] is not None for event in events)
    decoded = base64.b64decode(events[0]["thumbnail_base64"])
    image = cv2.imdecode(np.frombuffer(decoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (4, 4, 3)
    # Logged as RGB (0, 0, 255); cv2 decodes back to BGR, so blue should dominate.
    assert image[:, :, 0].mean() > image[:, :, 2].mean()


def test_build_highlights_missing_recording_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(highlights, "RECORDINGS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        highlights.build_highlights("does-not-exist")
