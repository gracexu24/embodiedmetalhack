"""Color-stack verification using only the side camera."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import rerun as rr

from .models import Block, Color, Layer, VerificationResult
from .rr_time import log_step

COLOR_HSV_RANGES = {
    Color.RED: [((0, 90, 70), (10, 255, 255)), ((170, 90, 70), (179, 255, 255))],
    Color.YELLOW: [((20, 80, 80), (40, 255, 255))],
    Color.BLUE: [((90, 70, 50), (135, 255, 255))],
    Color.GREEN: [((41, 60, 50), (85, 255, 255))],
}

DEFAULT_CAMERA_ENTITY_PATHS = {"cam0": "/harness/cameras/cam0", "cam1": "/harness/cameras/cam1"}


class PlacementVerifier:
    """Verify that each requested color is stacked above the previous color."""

    def __init__(
        self,
        config: dict[str, Any],
        camera_config: dict[str, Any] | None = None,
        *,
        camera_readers: dict[str, Callable[[], np.ndarray]] | None = None,
        camera_entity_paths: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.camera_config = camera_config or {}
        self.calls: list[Layer] = []
        self._placed_blocks: list[Block] = []
        self._cameras: dict[str, Any] = {}
        # camera_readers lets a caller source cam0/cam1 frames from something other
        # than an index-addressed cv2.VideoCapture -- e.g. RealSense wrappers.
        self._camera_readers = dict(camera_readers or {})
        self._camera_entity_paths = dict(camera_entity_paths or DEFAULT_CAMERA_ENTITY_PATHS)

    def verify(self, expected_block: Block, layer_index: int) -> VerificationResult:
        """Verify one layer from cam1 using color and calibrated height bands."""
        self.calls.append(expected_block.layer)

        expected_order = [Layer.DOOR, Layer.WALL, Layer.ROOF]
        if (
            layer_index not in range(3)
            or expected_order[layer_index] is not expected_block.layer
            or layer_index != len(self._placed_blocks)
        ):
            result = VerificationResult(
                False, False, False, False, False, "Layer does not match the verified stack."
            )
            self._log_result(expected_block, result)
            return result

        self._open_camera("cam1")
        frame_count = int(self.config["stability_frames"])
        current_centers: list[tuple[float, float]] = []
        support_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        current_region = self.config["height_regions"][expected_block.layer.value]

        print(
            f"[calib] verifying {expected_block.color.value} {expected_block.layer.value} "
            f"in cam1 band y={current_region['min_y']}..{current_region['max_y']} "
            f"over {frame_count} frames "
            f"(target_x={self.config['target_x']}, "
            f"max_center_error_px={self.config['max_center_error_px']})",
            flush=True,
        )
        for frame_index in range(frame_count):
            frame = self._read("cam1")
            current = self._detect_color(frame, expected_block.color, current_region)
            if current is None:
                print(
                    f"[calib]   frame {frame_index + 1}/{frame_count}: "
                    f"{expected_block.color.value} not detected in band",
                    flush=True,
                )
                continue
            support_text = ""
            if self._placed_blocks:
                support_block = self._placed_blocks[-1]
                support_region = self.config["height_regions"][support_block.layer.value]
                support = self._detect_color(frame, support_block.color, support_region)
                if support is not None:
                    support_pairs.append((current, support))
                    support_text = (
                        f", support {support_block.color.value} at "
                        f"({support[0]:.1f}, {support[1]:.1f}), "
                        f"x offset {abs(current[0] - support[0]):.1f}px"
                    )
                else:
                    support_text = f", support {support_block.color.value} NOT detected"
            print(
                f"[calib]   frame {frame_index + 1}/{frame_count}: centroid "
                f"({current[0]:.1f}, {current[1]:.1f}), "
                f"x error {abs(current[0] - float(self.config['target_x'])):.1f}px"
                f"{support_text}",
                flush=True,
            )
            current_centers.append(current)

        correct_block = len(current_centers) == frame_count
        correct_height = correct_block
        centered = self._centered(current_centers[-1] if current_centers else None)
        supported = not self._placed_blocks or (
            len(support_pairs) == frame_count
            and all(self._is_on_top(current, support) for current, support in support_pairs)
        )
        correct_position = centered and supported
        stable = self._is_stable(current_centers)
        checks = {
            f"{expected_block.color.value} block not visible in its height band": correct_block,
            "block is not centered above its support": correct_position,
            "block is outside its calibrated height band": correct_height,
            "block moved during verification": stable,
        }
        reasons = [reason for reason, passed in checks.items() if not passed]
        success = correct_block and correct_position and correct_height and stable
        print(
            f"[calib] result {expected_block.color.value} {expected_block.layer.value}: "
            f"{'PASS' if success else 'FAIL'} "
            f"(visible {len(current_centers)}/{frame_count} frames, "
            f"centered={centered}, supported={supported}, stable={stable})",
            flush=True,
        )
        if reasons:
            print(f"[calib]   failed checks: {'; '.join(reasons)}", flush=True)
        result = VerificationResult(
            success,
            correct_block,
            correct_position,
            correct_height,
            stable,
            "; ".join(reasons),
        )
        if success:
            self._placed_blocks.append(expected_block)
        self._log_result(expected_block, result)
        return result

    def close(self) -> None:
        """Release camera handles we opened ourselves and clear the verified stack."""
        for camera in self._cameras.values():
            camera.release()
        self._cameras.clear()
        self._placed_blocks.clear()

    def camera_observations(self) -> dict[str, np.ndarray]:
        """Return both camera frames for the policy; verification itself uses cam1."""
        self._open_camera("cam0")
        self._open_camera("cam1")
        return {"cam0": self._read("cam0"), "cam1": self._read("cam1")}

    def _log_result(self, expected_block: Block, result: VerificationResult) -> None:
        log_step()
        prefix = "/harness/verification"
        rr.log(
            prefix,
            rr.TextLog(
                f"{expected_block.color.value} {expected_block.layer.value}: "
                f"{'PASS' if result.success else 'FAIL'}"
                + (f" ({result.reason})" if result.reason else "")
            ),
        )
        for field in ("success", "correct_block", "correct_position", "correct_height", "stable"):
            rr.log(f"{prefix}/{field}", rr.Scalars(1.0 if getattr(result, field) else 0.0))

    def _open_camera(self, name: str) -> None:
        if name in self._camera_readers or name in self._cameras:
            return
        settings = self.camera_config[name]
        camera = cv2.VideoCapture(int(settings["index"]))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(settings["width"]))
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(settings["height"]))
        camera.set(cv2.CAP_PROP_FPS, int(settings["fps"]))
        if not camera.isOpened():
            camera.release()
            raise RuntimeError(f"Could not open required camera {name}.")
        actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"[calib] opened {name} index={settings['index']} "
            f"requested={settings['width']}x{settings['height']} actual={actual_w}x{actual_h}",
            flush=True,
        )
        self._cameras[name] = camera

    def _read(self, name: str) -> np.ndarray:
        if name in self._camera_readers:
            frame = self._camera_readers[name]()
        else:
            ok, frame = self._cameras[name].read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read required camera {name}.")
        log_step()
        rr.log(
            self._camera_entity_paths.get(name, f"/harness/cameras/{name}"),
            rr.Image(frame[:, :, ::-1]),
        )
        return frame

    def _detect_color(
        self,
        frame: np.ndarray,
        color: Color,
        region: dict[str, int],
    ) -> tuple[float, float] | None:
        min_y = max(0, int(region["min_y"]))
        max_y = min(frame.shape[0], int(region["max_y"]))
        if min_y >= max_y:
            raise ValueError(f"Invalid height region: {region}")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower, upper in COLOR_HSV_RANGES[color]:
            mask = cv2.bitwise_or(
                mask, cv2.inRange(hsv, np.asarray(lower), np.asarray(upper))
            )
        region_mask = np.zeros_like(mask)
        region_mask[min_y:max_y, :] = mask[min_y:max_y, :]
        contours, _ = cv2.findContours(
            region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        region_area = float((max_y - min_y) * frame.shape[1])
        occupancy = cv2.contourArea(contour) / region_area
        if occupancy < float(self.config.get("min_color_occupancy", 0.01)):
            return None
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        return (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )

    def _centered(self, center: tuple[float, float] | None) -> bool:
        return center is not None and abs(
            center[0] - float(self.config["target_x"])
        ) <= float(self.config["max_center_error_px"])

    def _is_on_top(
        self,
        current: tuple[float, float],
        support: tuple[float, float],
    ) -> bool:
        aligned = abs(current[0] - support[0]) <= float(
            self.config["max_stack_alignment_error_px"]
        )
        return aligned and current[1] < support[1]

    def _is_stable(self, centers: list[tuple[float, float]]) -> bool:
        required = int(self.config["stability_frames"])
        if len(centers) < required:
            return False
        points = np.asarray(centers[-required:])
        movement = np.linalg.norm(points - points[0], axis=1)
        return bool(
            float(np.max(movement))
            < float(self.config["max_stability_movement_px"])
        )
