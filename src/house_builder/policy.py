"""Mock and MolmoAct2 policy implementations."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import rerun as rr

from .robot import SO101_MOTOR_ORDER, Robot
from .rr_time import log_step

# ASSUMPTIONS -- none of these are verifiable without downloading the real
# `lerobot/MolmoAct2-SO100_101-LeRobot` checkpoint and running an actual GPU rollout,
# which this environment couldn't do (config loading hit a draccus/argparse bug under
# this environment's Python version -- see the integration notes). Confirm all three
# against the real checkpoint before trusting a live rollout; override via
# policy.image_keys / policy.action_names in config.yaml if they differ:
#
# 1. image_keys: built from lerobot's "observation.images.<name>" convention applied
#    to this harness's own cam0/cam1 naming.
# 2. action_names: this checkpoint's name implies it was trained on a plain SO101
#    follower action space -- SO101_MOTOR_ORDER from robot.py, verified against the
#    installed lerobot's SO101Follower motor bus construction.
# 3. Channel order: frames are converted BGR -> RGB before being handed to the model
#    (the near-universal convention for vision-language backbones). If this
#    checkpoint's training pipeline fed raw BGR frames instead, drop the conversion.
DEFAULT_IMAGE_KEYS = ["observation.images.cam0", "observation.images.cam1"]
DEFAULT_ACTION_NAMES = [f"{name}.pos" for name in SO101_MOTOR_ORDER]


class Policy:
    """Minimal policy interface: one short instruction per call."""

    def load(self) -> None:
        raise NotImplementedError

    def run_instruction(self, instruction: str, duration_seconds: float) -> bool:
        raise NotImplementedError


class MockPolicy(Policy):
    """Policy that records instructions and returns configured outcomes."""

    def __init__(
        self,
        outcomes: list[bool] | None = None,
        raise_on_call: int | None = None,
    ) -> None:
        self.loaded = False
        self.instructions: list[str] = []
        self.outcomes = list(outcomes or [])
        self.raise_on_call = raise_on_call

    def load(self) -> None:
        self.loaded = True

    def run_instruction(self, instruction: str, duration_seconds: float) -> bool:
        if not self.loaded:
            raise RuntimeError("Mock policy has not been loaded.")
        if duration_seconds <= 0:
            raise ValueError("Skill duration must be positive.")
        call_index = len(self.instructions)
        self.instructions.append(instruction)
        log_step()
        rr.log("/harness/instruction", rr.TextLog(instruction))
        if self.raise_on_call == call_index:
            log_step()
            rr.log("/harness/instruction", rr.TextLog("  -> FAILED: injected exception"))
            raise RuntimeError("Injected mock policy exception.")
        outcome = self.outcomes.pop(0) if self.outcomes else True
        log_step()
        rr.log("/harness/instruction", rr.TextLog(f"  -> {'ok' if outcome else 'FAILED'}"))
        return outcome


class MolmoAct2Policy(Policy):
    """Loads a lerobot.policies.molmoact2.MolmoAct2Policy checkpoint and rolls it out
    one short instruction at a time, using lerobot's own predict_action helper.
    """

    def __init__(
        self,
        config: dict[str, Any],
        robot: Robot,
        camera_observations: Callable[[], dict[str, Any]],
    ) -> None:
        self.checkpoint = str(config["local_checkpoint"] or config["checkpoint"])
        self.device = str(config.get("device", "cuda"))
        self.image_keys = list(config.get("image_keys") or DEFAULT_IMAGE_KEYS)
        self.action_names = list(config.get("action_names") or DEFAULT_ACTION_NAMES)
        self.robot = robot
        self.camera_observations = camera_observations
        self._policy: Any = None
        self._preprocessor: Any = None
        self._postprocessor: Any = None

    def load(self) -> None:
        if self.device == "cuda":
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("PyTorch is required to run MolmoAct2.") from exc
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "MolmoAct2 is configured for CUDA, but CUDA is unavailable. "
                    "Use --mock or install a CUDA-enabled PyTorch environment."
                )

        try:
            from lerobot.policies.molmoact2 import (
                MolmoAct2Policy as _MolmoAct2Policy,
            )
            from lerobot.policies.molmoact2 import make_molmoact2_pre_post_processors
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is not installed, or this release doesn't ship "
                "lerobot.policies.molmoact2. Install a compatible release, then adapt "
                "house_builder/policy.py."
            ) from exc

        self._policy = _MolmoAct2Policy.from_pretrained(self.checkpoint)
        self._policy.to(self.device)
        self._policy.eval()
        self._preprocessor, self._postprocessor = make_molmoact2_pre_post_processors(
            self._policy.config
        )

    def run_instruction(self, instruction: str, duration_seconds: float) -> bool:
        if self._policy is None:
            raise RuntimeError("MolmoAct2 policy has not been loaded.")
        if duration_seconds <= 0:
            raise ValueError("Skill duration must be positive.")

        import torch
        from lerobot.common.control_utils import predict_action
        from lerobot.policies.utils import make_robot_action
        from lerobot.utils.constants import ACTION

        log_step()
        rr.log("/harness/instruction", rr.TextLog(instruction))

        self._policy.reset()
        ds_features = {ACTION: {"names": self.action_names}}
        device = torch.device(self.device)
        deadline = time.perf_counter() + duration_seconds

        try:
            while time.perf_counter() < deadline:
                cameras = self.camera_observations()  # {"cam0": bgr, "cam1": bgr}
                observation = {
                    self.image_keys[0]: np.ascontiguousarray(cameras["cam0"][:, :, ::-1]),
                    self.image_keys[1]: np.ascontiguousarray(cameras["cam1"][:, :, ::-1]),
                    "observation.state": np.asarray(
                        self.robot.get_observation()["joint_state"], dtype=np.float32
                    ),
                }
                action = predict_action(
                    observation,
                    self._policy,
                    device,
                    self._preprocessor,
                    self._postprocessor,
                    use_amp=False,
                    task=instruction,
                    robot_type="so101_follower",
                )
                action_dict = make_robot_action(action, ds_features)
                self.robot.send_action(action_dict)
        except Exception:
            log_step()
            rr.log("/harness/instruction", rr.TextLog("  -> FAILED: rollout raised an exception"))
            raise

        log_step()
        rr.log("/harness/instruction", rr.TextLog("  -> ok"))
        return True
