#!/usr/bin/env python3
"""Read a human-built model house from camera3 and emit a harness request."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from house_builder.models import ALLOWED_COLORS, Color, HouseRequest, Layer
from house_builder.verifier import COLOR_HSV_RANGES


def detect_model_house(
    frame: np.ndarray,
    verification_config: dict[str, Any],
) -> HouseRequest:
    """Detect the allowed color in each calibrated camera3 layer band."""
    height, width = frame.shape[:2]
    print(f"[calib] camera3 frame: {width}x{height}", flush=True)
    detected: dict[Layer, Color] = {}
    for layer in Layer:
        region = verification_config["height_regions"][layer.value]
        scores = {
            color: _color_occupancy(frame, color, region)
            for color in ALLOWED_COLORS[layer]
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_color, best_score = ranked[0]
        minimum = float(verification_config.get("min_color_occupancy", 0.01))
        margin = float(verification_config.get("min_color_margin", 0.005))
        score_text = ", ".join(f"{color.value}={score:.4f}" for color, score in ranked)
        print(
            f"[calib] {layer.value} band y={region['min_y']}..{region['max_y']}: "
            f"{score_text} (min_occupancy={minimum}, min_margin={margin})",
            flush=True,
        )
        if best_score < minimum:
            raise ValueError(
                f"No allowed {layer.value} color was visible in its camera3 height band."
            )
        if len(ranked) > 1 and best_score - ranked[1][1] < margin:
            raise ValueError(f"Ambiguous colors in the {layer.value} height band.")
        print(f"[calib] {layer.value} -> {best_color.value}", flush=True)
        detected[layer] = best_color

    return HouseRequest(
        door=detected[Layer.DOOR],
        wall=detected[Layer.WALL],
        roof=detected[Layer.ROOF],
    )


def request_to_sentence(request: HouseRequest) -> str:
    """Format a detected model as input accepted by run.py."""
    return (
        f"Build a house with a {request.door.value} door, "
        f"{request.wall.value} walls, and a {request.roof.value} roof."
    )


def _color_occupancy(
    frame: np.ndarray,
    color: Color,
    region: dict[str, int],
) -> float:
    min_y = max(0, int(region["min_y"]))
    max_y = min(frame.shape[0], int(region["max_y"]))
    if min_y >= max_y:
        raise ValueError(f"Invalid height region: {region}")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cropped = hsv[min_y:max_y, :]
    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    for lower, upper in COLOR_HSV_RANGES[color]:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(cropped, np.asarray(lower), np.asarray(upper)),
        )
    return float(np.count_nonzero(mask)) / float(mask.size)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "human_builder" not in config or "cameras" not in config:
        raise ValueError(f"{path} must define human_builder and cameras sections.")
    return config


def _capture_camera3(camera_config: dict[str, Any], warmup_frames: int) -> np.ndarray:
    camera = cv2.VideoCapture(int(camera_config["index"]))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config["width"]))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config["height"]))
    camera.set(cv2.CAP_PROP_FPS, int(camera_config["fps"]))
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open camera3.")
    actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[calib] opened camera3 index={camera_config['index']} "
        f"requested={camera_config['width']}x{camera_config['height']} "
        f"actual={actual_w}x{actual_h} (warmup={warmup_frames} frames)",
        flush=True,
    )
    try:
        frame: np.ndarray | None = None
        for _ in range(max(1, warmup_frames)):
            ok, captured = camera.read()
            if not ok or captured is None:
                raise RuntimeError("Could not read camera3.")
            frame = captured
        if frame is None:
            raise RuntimeError("camera3 returned no frame.")
        return frame
    finally:
        camera.release()


def capture_model_house(
    config: dict[str, Any],
    warmup_frames: int = 10,
) -> HouseRequest:
    """Capture camera3 and return the detected model-house request."""
    if not config.get("features", {}).get("human_builder", True):
        raise RuntimeError(
            "Human builder is disabled. Use the UI sentence or color input."
        )
    frame = _capture_camera3(
        config["cameras"]["camera3"],
        warmup_frames,
    )
    return detect_model_house(frame, config["human_builder"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a camera view of a model house into a harness sentence."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--image", type=Path, help="Use an image instead of live camera3")
    parser.add_argument("--warmup-frames", type=int, default=10)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise ValueError(f"Could not read image {args.image}.")
        else:
            request = capture_model_house(config, args.warmup_frames)
            frame = None
        if frame is not None:
            request = detect_model_house(frame, config["human_builder"])
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Could not read model house: {exc}")
        return 1

    print(request_to_sentence(request))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
