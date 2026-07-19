"""MolmoAct2 policy interface: calls a Modal-hosted /act endpoint.

Speaks the exact JSON contract so100-hackathon's tools/apps/deploy_policy.py and
tools/apps/policy_server_modal_molmoact2.py already use, so this is a drop-in client for
either the public zero-shot checkpoint's server or a fine-tuned one deployed the same way:

    POST {"instruction": str, "state": [6 floats], "images": {"top": <b64 jpeg>, "side": ...}}
        -> {"actions": [[6 floats], ...]}   # an absolute-joint-pose action chunk

All safety clamping (per-joint max-step-per-tick) lives in SO101Robot.send_action, not
here -- this module only handles talking to the model.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from .robot import Robot

# cam0 = overhead camera, cam1 = side camera -- matches so100-hackathon's own convention
# (deploy_policy.py's --camera-names default, and camera 0/1 detection order all session).
CAMERA_KEY_MAP = {"cam0": "top", "cam1": "side"}


class Policy:
    """Minimal policy interface: one short instruction per call."""

    def load(self) -> None:
        raise NotImplementedError

    def run_instruction(self, instruction: str, duration_seconds: float) -> bool:
        raise NotImplementedError


class MolmoAct2Policy(Policy):
    """Calls a Modal-hosted MolmoAct2 /act endpoint instead of loading the model locally.

    Config (the ``policy`` section of config.yaml)::

        policy:
          server: https://<your-modal-endpoint>/act
          fps: 30.0                # control rate actions are sent at
          execute_steps: 24        # actions consumed per chunk before re-observing
          jpeg_quality: 85
          skill_duration_seconds: 10
    """

    def __init__(
        self,
        config: dict[str, Any],
        robot: Robot,
        camera_observations: Callable[[], dict[str, Any]],
    ) -> None:
        self.server = str(config["server"])
        self.fps = float(config.get("fps", 30.0))
        self.execute_steps = int(config.get("execute_steps", 24))
        self.jpeg_quality = int(config.get("jpeg_quality", 85))
        self.robot = robot
        self.camera_observations = camera_observations

    def load(self) -> None:
        """No local model to load -- inference runs on Modal. A GET (not POST) so this
        doesn't trigger a real inference call or wait through a cold start; it only
        proves the endpoint exists and routes, so a typo'd URL fails fast instead of
        mid-skill."""
        try:
            urllib.request.urlopen(self.server, timeout=10)
        except urllib.error.HTTPError:
            pass  # any HTTP response (even 404/405) proves the endpoint is reachable
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach the Modal policy server at {self.server}: {exc}"
            ) from exc

    def run_instruction(self, instruction: str, duration_seconds: float) -> bool:
        """Run the policy loop for duration_seconds and return True on a clean finish.

        The return value is NOT a task-success signal -- MolmoAct2 has no built-in
        stopping criterion; it never reports "done", it just keeps emitting motion for as
        long as it's asked to run. True here only means the loop completed without an
        error (server unreachable, malformed response); it says nothing about whether the
        skill actually worked. HouseBuilder treats the camera verifier as the only source
        of ground truth about placement -- see _run_until_verified in builder.py.
        """
        if duration_seconds <= 0:
            raise ValueError("Skill duration must be positive.")

        period = 1.0 / self.fps
        started = time.monotonic()
        while time.monotonic() - started < duration_seconds:
            state = list(self.robot.get_observation()["joint_state"])
            images = self.camera_observations()
            chunk = self._query_actions(instruction, state, images)

            next_tick = time.monotonic()
            for action in chunk[: self.execute_steps]:
                if time.monotonic() - started >= duration_seconds:
                    break
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                next_tick += period
                self.robot.send_action(action)
        return True

    def _query_actions(
        self, instruction: str, state: list[float], images: dict[str, Any]
    ) -> list[list[float]]:
        # verifier.camera_observations() keys images "cam0"/"cam1" (its own naming, tied to
        # config.yaml's cameras section); the /act contract (and the observation.images.top/
        # .side keys the finetuned server logs on load) speaks "top"/"side" instead -- cam0 is
        # the overhead camera, cam1 the side one, matching so100-hackathon's own convention.
        renamed = {CAMERA_KEY_MAP.get(name, name): frame for name, frame in images.items()}
        payload = {
            "instruction": instruction,
            "state": state,
            "images": {name: self._encode_jpeg(frame) for name, frame in renamed.items()},
        }
        request = urllib.request.Request(
            self.server,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach the Modal policy server at {self.server}: {exc}"
            ) from exc
        if "error" in body:
            raise RuntimeError(f"Policy server error: {body['error']}")
        actions = body["actions"]
        if not isinstance(actions, list) or not actions or len(actions[0]) != len(state):
            raise RuntimeError(f"Policy server returned malformed actions: {actions!r}")
        return actions

    def _encode_jpeg(self, frame: np.ndarray) -> str:
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed for a camera frame.")
        return base64.b64encode(encoded.tobytes()).decode()
