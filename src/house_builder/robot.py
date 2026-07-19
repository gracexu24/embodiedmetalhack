"""SO-101 robot interface, backed by so100-hackathon's Feetech driver.

Reuses the exact calibration-based interface exercised all session in that repo's
``replay_episode.py``/``deploy_policy.py`` (auto-detect the calibrated follower by USB
serial id, read/write through ``calibrations/<usb_id>.json``) rather than going through
LeRobot's own robot classes -- this is the path that's actually been proven working on
real hardware.

Run this with so100-hackathon's pixi env on PYTHONPATH so ``so100_hackathon`` is
importable (that package is already editable-installed inside the pixi env)::

    export PYTHONPATH=/path/to/embodiedmetalhack/src
    cd /path/to/so100-hackathon
    pixi run python /path/to/embodiedmetalhack/run.py "Build a house with ..." \\
        --config /path/to/embodiedmetalhack/config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from so100_hackathon.calibration import (
    MotorCalibration,
    load_arm_kind,
    load_arm_ranges,
    load_calibration,
)
from so100_hackathon.feetech import FeetechBus, detect_arm_ports, usb_id_from_port


class Robot:
    """Minimal robot interface used by the builder and policy."""

    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def move_home(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def get_observation(self) -> dict[str, Any]:
        raise NotImplementedError

    def send_action(self, action: object) -> None:
        raise NotImplementedError


class SO101Robot(Robot):
    """Drives the follower arm through so100-hackathon's Feetech bus directly."""

    def __init__(self, config: dict[str, Any]) -> None:
        raw_port = config.get("port")  # None/missing/placeholder -> auto-detect the follower
        self.port_hint = None if not raw_port or "REPLACE" in str(raw_port) else str(raw_port)
        self.calibration_dir = Path(str(config.get("calibration_dir", "calibrations")))
        self.home_pose = [float(v) for v in config["home_pose"]]
        # Same safety net as deploy_policy.py's --max-step-deg: caps how far a single
        # send_action/move_home step can move any joint, regardless of what the caller asks.
        self.max_step_deg = float(config.get("max_step_deg", 10.0))
        self.home_ramp_steps = int(config.get("home_ramp_steps", 60))
        # Mirrors deploy_policy.py's --dry-run: torque never enabled, no goal ever written
        # to the bus -- only logged. Opening the port and reading telemetry still happens,
        # so a dry run's logged "current" state is real, not faked.
        self.dry_run = bool(config.get("dry_run", False))
        self._bus: FeetechBus | None = None
        self._calibration: list[MotorCalibration] = []
        self._range_min: list[int] = []
        self._range_max: list[int] = []
        self._connected = False
        self.log = logging.getLogger(__name__)

    def connect(self) -> None:
        ports = (self.port_hint,) if self.port_hint else detect_arm_ports()
        if not ports:
            raise RuntimeError(
                "No SO-100/101 arms found (no /dev/cu.usbmodem* ports); set robot.port explicitly."
            )

        candidates: list[tuple[str, str, list[MotorCalibration], tuple[list[int], list[int]]]] = []
        for candidate in ports:
            usb_id = usb_id_from_port(candidate)
            calibration_path = self.calibration_dir / f"{usb_id}.json"
            kind = load_arm_kind(calibration_path)
            if kind == "leader" and self.port_hint is None:
                continue
            try:
                calibration = load_calibration(calibration_path)
            except (FileNotFoundError, KeyError):
                continue
            ranges = load_arm_ranges(calibration_path)
            if ranges is None:
                raise RuntimeError(
                    f"{usb_id}: calibration has no range_min/range_max -- "
                    "recalibrate: pixi run calibrate-so100 follower"
                )
            candidates.append((candidate, usb_id, calibration, ranges))
        if not candidates:
            raise RuntimeError(
                "No calibrated follower arm found (pixi run calibrate-so100 follower)."
            )
        if len(candidates) > 1:
            names = ", ".join(c[1] for c in candidates)
            raise RuntimeError(
                f"Multiple follower candidates ({names}); set robot.port to disambiguate."
            )

        chosen_port, _usb_id, calibration, (range_min, range_max) = candidates[0]
        bus = FeetechBus(chosen_port)
        if self.dry_run:
            self.log.info("dry run: torque stays off, no goals will be written")
        else:
            bus.set_torque(False)
            bus.configure_follower_control()
            bus.set_torque(True)
        self._bus = bus
        self._calibration = calibration
        self._range_min = range_min
        self._range_max = range_max
        self._connected = True

    def disconnect(self) -> None:
        if self._bus is not None:
            try:
                self._bus.set_torque(False)
            finally:
                self._bus.close()
        self._connected = False

    def move_home(self) -> None:
        """Ramp gently to the configured home pose rather than jump there."""
        self._require_connected()
        start = self.get_observation()["joint_state"]
        for step in range(self.home_ramp_steps):
            blend = (step + 1) / self.home_ramp_steps
            target = [s + blend * (h - s) for s, h in zip(start, self.home_pose, strict=True)]
            self._drive_to(target, clamp_to_current=False)

    def stop(self) -> None:
        if self._bus is not None:
            self._bus.set_torque(False)

    def get_observation(self) -> dict[str, Any]:
        self._require_connected()
        assert self._bus is not None
        telemetry = self._bus.read_telemetry()
        joint_state = [
            calib.calibrated_from_raw(t.position_raw)
            for calib, t in zip(self._calibration, telemetry, strict=True)
        ]
        return {"joint_state": joint_state}

    def send_action(self, action: object) -> None:
        """One absolute joint-pose command, clamped to --max-step-deg like deploy_policy.py."""
        self._require_connected()
        target = [float(v) for v in action]  # type: ignore[attr-defined]
        self._drive_to(target, clamp_to_current=True)

    def _drive_to(self, target: list[float], *, clamp_to_current: bool) -> None:
        assert self._bus is not None
        if clamp_to_current:
            current = self.get_observation()["joint_state"]
            target = [
                min(max(t, c - self.max_step_deg), c + self.max_step_deg)
                for t, c in zip(target, current, strict=True)
            ]
        if self.dry_run:
            self.log.info("dry run goal: %s", ["%+.1f" % v for v in target])
            return
        goals_raw = [
            min(max(calib.raw_from_calibrated(target[i]), self._range_min[i]), self._range_max[i])
            for i, calib in enumerate(self._calibration)
        ]
        self._bus.sync_write_goal(goals_raw)

    def _require_connected(self) -> None:
        if not self._connected or self._bus is None:
            raise RuntimeError("SO-101 is not connected.")
