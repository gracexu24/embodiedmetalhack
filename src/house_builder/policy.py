"""MolmoAct2 policy interface and LeRobot integration boundary."""

from collections.abc import Callable
from typing import Any

from .robot import Robot


class Policy:
    """Minimal policy interface: one short instruction per call."""

    def load(self) -> None:
        raise NotImplementedError

    def run_instruction(self, instruction: str, duration_seconds: float) -> bool:
        raise NotImplementedError


class MolmoAct2Policy(Policy):
    """Version-dependent MolmoAct2 integration kept in one module."""

    def __init__(
        self,
        config: dict[str, Any],
        robot: Robot,
        camera_observations: Callable[[], dict[str, Any]],
    ) -> None:
        self.checkpoint = str(config["local_checkpoint"] or config["checkpoint"])
        self.device = str(config.get("device", "cuda"))
        self.robot = robot
        self.camera_observations = camera_observations
        self._policy: Any = None

    def load(self) -> None:
        if self.device == "cuda":
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("PyTorch is required to run MolmoAct2.") from exc
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "MolmoAct2 is configured for CUDA, but CUDA is unavailable. "
                    "Install a CUDA-enabled PyTorch environment."
                )

        try:
            import lerobot  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is not installed. Install a compatible release, then adapt "
                "house_builder/policy.py."
            ) from exc

        # ADAPT TO INSTALLED LEROBOT VERSION
        # Load self.checkpoint with the installed release's documented MolmoAct2 API.
        # The model input must contain cam0, cam1, current joint state, and exactly
        # one short language instruction.
        raise RuntimeError(
            "MolmoAct2 loading varies by LeRobot version. "
            "Adapt the marked section in house_builder/policy.py."
        )

    def run_instruction(self, instruction: str, duration_seconds: float) -> bool:
        if self._policy is None:
            raise RuntimeError("MolmoAct2 policy has not been loaded.")
        if duration_seconds <= 0:
            raise ValueError("Skill duration must be positive.")

        observations = self.camera_observations()
        observations["joint_state"] = self.robot.get_observation()["joint_state"]
        observations["instruction"] = instruction

        # ADAPT TO INSTALLED LEROBOT VERSION
        # Repeatedly request bounded action chunks from self._policy for no longer
        # than duration_seconds, and forward each chunk with self.robot.send_action.
        # Return True only when the skill rollout completes without a safety stop.
        raise RuntimeError(
            "Adapt MolmoAct2 rollout execution in house_builder/policy.py."
        )
