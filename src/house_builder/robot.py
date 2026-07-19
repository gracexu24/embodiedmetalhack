"""SO-101 robot interface and LeRobot integration boundary."""

from typing import Any


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
    """LeRobot integration boundary for an SO-101 follower arm."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.port = str(config["port"])
        self.robot_id = str(config["id"])
        self.home_pose = list(config["home_pose"])
        self._robot: Any = None
        self._connected = False

    def connect(self) -> None:
        try:
            import lerobot  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is not installed. Install the version used by your SO-101, "
                "then adapt house_builder/robot.py."
            ) from exc

        # ADAPT TO INSTALLED LEROBOT VERSION
        # Construct the documented SO-101 follower configuration using self.port and
        # self.robot_id, create the robot, and call its documented connection method.
        raise RuntimeError(
            "LeRobot was found, but SO-101 construction varies by version. "
            "Adapt the marked section in house_builder/robot.py."
        )

    def disconnect(self) -> None:
        try:
            if self._robot is not None:
                # ADAPT TO INSTALLED LEROBOT VERSION
                disconnect = getattr(self._robot, "disconnect", None)
                if disconnect is not None:
                    disconnect()
        finally:
            self._connected = False

    def move_home(self) -> None:
        self._require_connected()
        # ADAPT TO INSTALLED LEROBOT VERSION
        # Send self.home_pose through the installed version's documented action API.
        raise RuntimeError("Adapt SO-101 home motion in house_builder/robot.py.")

    def stop(self) -> None:
        if self._robot is not None:
            # ADAPT TO INSTALLED LEROBOT VERSION
            stop = getattr(self._robot, "stop", None)
            if stop is not None:
                stop()

    def get_observation(self) -> dict[str, Any]:
        self._require_connected()
        # ADAPT TO INSTALLED LEROBOT VERSION
        raise RuntimeError("Adapt SO-101 observations in house_builder/robot.py.")

    def send_action(self, action: object) -> None:
        self._require_connected()
        # ADAPT TO INSTALLED LEROBOT VERSION
        raise RuntimeError("Adapt SO-101 action sending in house_builder/robot.py.")

    def _require_connected(self) -> None:
        if not self._connected or self._robot is None:
            raise RuntimeError("SO-101 is not connected.")
