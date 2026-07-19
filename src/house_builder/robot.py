"""Mock and LeRobot-backed SO-101 robot implementations."""

from typing import Any

import rerun as rr

from .rr_time import log_step

# Canonical SO101 follower joint order (verified against lerobot.robots.so_follower.
# SO101Follower's FeetechMotorsBus construction: motors are registered in exactly
# this order, ids 1-6).
SO101_MOTOR_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _log_joint_state(entity_path: str, joint_values: dict[str, float]) -> None:
    log_step()
    for joint_name, value in joint_values.items():
        rr.log(f"{entity_path}/{joint_name}", rr.Scalars(value))


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


class MockRobot(Robot):
    """In-memory robot that records lifecycle and actions."""

    def __init__(self) -> None:
        self.connected = False
        self.stopped = False
        self.disconnected = False
        self.home_count = 0
        self.actions: list[object] = []

    def connect(self) -> None:
        self.connected = True
        self.disconnected = False
        self.stopped = False

    def disconnect(self) -> None:
        self.connected = False
        self.disconnected = True

    def move_home(self) -> None:
        self._require_connected()
        self.home_count += 1

    def stop(self) -> None:
        self.stopped = True

    def get_observation(self) -> dict[str, Any]:
        self._require_connected()
        joint_state = [0.0] * 6
        _log_joint_state("/harness/mock_arm", dict(zip(SO101_MOTOR_ORDER, joint_state)))
        return {"joint_state": joint_state}

    def send_action(self, action: object) -> None:
        self._require_connected()
        self.actions.append(action)
        if isinstance(action, dict):
            _log_joint_state(
                "/harness/mock_arm",
                {k.removesuffix(".pos"): v for k, v in action.items() if isinstance(v, (int, float))},
            )

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("Mock robot is not connected.")


class SO101Robot(Robot):
    """LeRobot integration boundary for an SO-101 follower arm."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.port = str(config["port"])
        self.robot_id = str(config["id"])
        self.home_pose = list(config["home_pose"])
        self._robot: Any = None
        self._connected = False

    def connect(self) -> None:
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is not installed. Install the version used by your SO-101, "
                "then adapt house_builder/robot.py."
            ) from exc

        if len(self.home_pose) != len(SO101_MOTOR_ORDER):
            raise RuntimeError(
                f"config.yaml robot.home_pose must have {len(SO101_MOTOR_ORDER)} values "
                f"(one per {SO101_MOTOR_ORDER}), got {len(self.home_pose)}."
            )

        config = SO101FollowerConfig(port=self.port, id=self.robot_id)
        self._robot = SO101Follower(config)
        # calibrate=False: this harness assumes the arm was already calibrated via
        # `lerobot-calibrate --robot.type=so101_follower ...` before running the build.
        self._robot.connect(calibrate=False)
        self._connected = True

    def disconnect(self) -> None:
        try:
            if self._robot is not None:
                self._robot.disconnect()
        finally:
            self._connected = False

    def move_home(self) -> None:
        self._require_connected()
        action = {f"{name}.pos": value for name, value in zip(SO101_MOTOR_ORDER, self.home_pose)}
        sent = self._robot.send_action(action)
        _log_joint_state("/harness/arm", {k.removesuffix(".pos"): v for k, v in sent.items()})

    def stop(self) -> None:
        # SO101Follower has no dedicated stop() method; cut motor torque directly via
        # the underlying FeetechMotorsBus so the arm goes limp immediately rather than
        # continuing to hold or chase its last commanded goal position.
        if self._robot is not None and self._connected:
            self._robot.bus.disable_torque()

    def get_observation(self) -> dict[str, Any]:
        self._require_connected()
        raw = self._robot.get_observation()
        joint_state = [float(raw[f"{name}.pos"]) for name in SO101_MOTOR_ORDER]
        _log_joint_state("/harness/arm", dict(zip(SO101_MOTOR_ORDER, joint_state)))
        return {"joint_state": joint_state}

    def send_action(self, action: object) -> None:
        self._require_connected()
        if not isinstance(action, dict):
            raise TypeError(f"SO101Robot.send_action expects a dict, got {type(action)!r}")
        sent = self._robot.send_action(action)
        _log_joint_state("/harness/arm", {k.removesuffix(".pos"): v for k, v in sent.items()})

    def _require_connected(self) -> None:
        if not self._connected or self._robot is None:
            raise RuntimeError("SO-101 is not connected.")
