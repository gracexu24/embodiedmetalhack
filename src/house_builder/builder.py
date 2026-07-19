"""Readable three-layer build loop."""

import logging
from collections.abc import Callable

from .models import Block, BuildResult, HouseRequest, Layer
from .planner import create_build_plan
from .policy import Policy
from .robot import Robot
from .state_machine import BuildState, BuildStateMachine
from .verifier import PlacementVerifier


class HouseBuilder:
    def __init__(
        self,
        robot: Robot,
        policy: Policy,
        verifier: PlacementVerifier,
        duration_seconds: float = 10.0,
        on_state_change: Callable[[BuildState, BuildState], None] | None = None,
    ) -> None:
        self.robot = robot
        self.policy = policy
        self.verifier = verifier
        self.duration_seconds = duration_seconds
        self.on_state_change = on_state_change
        self.state_machine = BuildStateMachine(on_transition=on_state_change)
        self.log = logging.getLogger(__name__)

    def build(self, request: HouseRequest) -> BuildResult:
        """Build door, wall, and roof, stopping at the first failure."""
        plan = create_build_plan(request)
        completed: list[Layer] = []
        self.log.info("Parsed house request: %s", request)
        self.state_machine = BuildStateMachine(on_transition=self.on_state_change)

        try:
            self.state_machine.transition(BuildState.CONNECTING)
            self.robot.connect()
            self.policy.load()
            self.state_machine.transition(BuildState.HOMING)
            self.robot.move_home()

            for layer_index, step in enumerate(plan):
                layer = step.block.layer
                self.state_machine.transition(BuildState.PICKING)
                self.log.info("Building %s layer", layer.value)
                self.log.info("Pick instruction: %s", step.pick_instruction)
                if not self.policy.run_instruction(
                    step.pick_instruction,
                    self.duration_seconds,
                ):
                    return self._failure(
                        completed,
                        step.block,
                        "The policy failed to pick the requested block.",
                    )

                self.state_machine.transition(BuildState.PLACING)
                self.log.info("Place instruction: %s", step.place_instruction)
                if not self.policy.run_instruction(
                    step.place_instruction,
                    self.duration_seconds,
                ):
                    return self._failure(
                        completed,
                        step.block,
                        "The policy failed to place the held block.",
                    )

                self.robot.move_home()
                self.state_machine.transition(BuildState.VERIFYING)
                verification = self.verifier.verify(step.block, layer_index)
                self.log.info("Verification result: %s", verification)
                if not verification.success:
                    return self._failure(
                        completed,
                        step.block,
                        verification.reason or "Placement verification failed.",
                    )
                completed.append(layer)

            self.robot.move_home()
            self.state_machine.transition(BuildState.COMPLETED)
            result = BuildResult(
                True,
                completed,
                None,
                "House completed successfully.",
            )
            self.log.info("Final result: %s", result)
            return result
        except (Exception, KeyboardInterrupt):
            self.state_machine.fail()
            raise
        finally:
            try:
                self.robot.stop()
            finally:
                self.verifier.close()
                self.robot.disconnect()

    def _failure(
        self,
        completed: list[Layer],
        expected_block: Block,
        reason: str,
    ) -> BuildResult:
        self.state_machine.fail()
        self.robot.stop()
        message = (
            f"{expected_block.layer.value.capitalize()} layer failed; expected "
            f"{expected_block.color.value} {expected_block.layer.value}: {reason} "
            "Stop and have a human reset the structure."
        )
        result = BuildResult(
            False,
            list(completed),
            expected_block.layer,
            message,
        )
        self.log.error("Final result: %s", result)
        return result


def recover_failed_placement(expected_block: Block) -> bool:
    """Future autonomous recovery hook; the first version always requires a human."""
    del expected_block
    return False
