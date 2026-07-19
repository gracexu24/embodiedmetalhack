#!/usr/bin/env python3
"""Simulate a full build with fake robot/policy/verifier -- no hardware, no Modal calls.

Prints every trigger point live: when the policy gets called, when verification runs
(and how many times, if a layer needs retries), and what actually decides whether the
build moves on or stops. Same HouseBuilder orchestration exercised in
tests/test_builder.py, just runnable standalone with visible output instead of asserted
silently.

Sequential, not parallel: each check_interval_seconds slice of policy execution is
followed by robot.move_home() and verifier.verify(), one at a time -- never at the same
time. Verification only happens once the arm has stopped and moved out of frame.

    python simulate.py                                  # everything succeeds first try
    python simulate.py --retry-layer wall --retry-failures 2   # wall needs 3 checks
    python simulate.py --fail-layer roof                # roof never verifies, times out
    python simulate.py --crash-layer door               # the policy itself raises

--fail-layer times out after retrying every check_interval_seconds until
skill_duration_seconds elapses (the real retry-until-timeout behavior); --crash-layer
raises immediately, since that -- not a False return from run_instruction, which is
never a meaningful signal for MolmoAct2 -- is the actual way a policy failure surfaces.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from house_builder.builder import HouseBuilder  # noqa: E402
from house_builder.models import (  # noqa: E402
    Block,
    Color,
    HouseRequest,
    Layer,
    VerificationResult,
)
from house_builder.planner import create_build_plan  # noqa: E402
from house_builder.policy import Policy  # noqa: E402
from house_builder.robot import Robot  # noqa: E402
from house_builder.verifier import PlacementVerifier  # noqa: E402


class FakeRobot(Robot):
    def connect(self) -> None:
        print("[robot] connect()")

    def disconnect(self) -> None:
        print("[robot] disconnect()")

    def move_home(self) -> None:
        print("[robot] move_home()")

    def stop(self) -> None:
        print("[robot] stop() -- torque off")

    def get_observation(self) -> dict[str, object]:
        return {"joint_state": [0.0] * 6}

    def send_action(self, action: object) -> None:
        del action


class FakePolicy(Policy):
    """Simulates the Modal round-trip. Its return value is never a success/failure
    signal (see run_instruction's real docstring in policy.py) -- a policy-level
    failure can only surface as an exception, which is what --crash-layer simulates.

    instruction_layers maps each planned instruction string back to its Layer -- the
    instruction text itself never contains the words "door"/"wall"/"roof" (it's phrased
    in colors and shapes, e.g. "stack it on the first red block"), so this can't be
    derived by inspecting the instruction string alone."""

    def __init__(self, crash_layer: Layer | None, instruction_layers: dict[str, Layer]) -> None:
        self.crash_layer = crash_layer
        self.instruction_layers = instruction_layers

    def load(self) -> None:
        print("[policy] load() -- would be a Modal reachability check")

    def run_instruction(self, instruction: str, duration_seconds: float) -> bool:
        print(f"[policy] run_instruction({instruction!r}, {duration_seconds}s)")
        layer = self.instruction_layers[instruction]
        if layer is self.crash_layer:
            raise RuntimeError(f"Simulated policy crash on the {layer.value} layer.")
        return True


class FakeVerifier(PlacementVerifier):
    """Simulates the cam1 color-stack check without touching any real camera."""

    def __init__(
        self, fail_layer: Layer | None, retry_layer: Layer | None, retry_failures: int
    ) -> None:
        self.fail_layer = fail_layer
        self.retry_layer = retry_layer
        self.retry_failures = retry_failures
        self.calls: list[Layer] = []

    def verify(self, expected_block: Block, layer_index: int) -> VerificationResult:
        self.calls.append(expected_block.layer)
        attempt = self.calls.count(expected_block.layer)
        label = f"{expected_block.color.value} {expected_block.layer.value}"
        print(f"[verify] checking {label} (attempt {attempt})")

        if expected_block.layer is self.fail_layer:
            result = VerificationResult(
                False, True, False, True, False, "Simulated: never verifies."
            )
        elif expected_block.layer is self.retry_layer and attempt <= self.retry_failures:
            result = VerificationResult(
                False, True, False, True, False, "Simulated transient failure."
            )
        else:
            result = VerificationResult(True, True, True, True, True)

        print(f"[verify] -> success={result.success}")
        return result

    def close(self) -> None:
        print("[verify] close()")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fail-layer", choices=["door", "wall", "roof"], default=None)
    parser.add_argument("--crash-layer", choices=["door", "wall", "roof"], default=None)
    parser.add_argument("--retry-layer", choices=["door", "wall", "roof"], default=None)
    parser.add_argument("--retry-failures", type=int, default=1)
    parser.add_argument("--duration", type=float, default=10.0, help="skill_duration_seconds")
    parser.add_argument("--check-interval", type=float, default=3.0, help="check_interval_seconds")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    fail_layer = Layer(args.fail_layer) if args.fail_layer else None
    crash_layer = Layer(args.crash_layer) if args.crash_layer else None
    retry_layer = Layer(args.retry_layer) if args.retry_layer else None

    request = HouseRequest(Color.RED, Color.YELLOW, Color.BLUE)
    instruction_layers = {step.instruction: step.block.layer for step in create_build_plan(request)}

    robot = FakeRobot()
    policy = FakePolicy(crash_layer, instruction_layers)
    verifier = FakeVerifier(fail_layer, retry_layer, args.retry_failures)
    builder = HouseBuilder(robot, policy, verifier, args.duration, args.check_interval)

    door, wall, roof = request.door.value, request.wall.value, request.roof.value
    print(f"\nRequest: door={door}, wall={wall}, roof={roof}\n")

    try:
        result = builder.build(request)
    except Exception as exc:
        # Matches run.py's own top-level handling: a policy crash propagates all the way
        # out of build(), and this is where a real run would catch and report it.
        print(f"\nBuild aborted safely: {exc}")
        print(f"Final state: {builder.state_machine.current.value}")
        return 1

    print(f"\nFinal state: {builder.state_machine.current.value}")
    print(f"State history: {' -> '.join(s.value for s in builder.state_machine.history)}")
    completed = [layer.value for layer in result.completed_layers]
    print(f"Result: success={result.success}, completed={completed}")
    print(f"Message: {result.message}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
