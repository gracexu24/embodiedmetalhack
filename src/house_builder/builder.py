"""Readable three-layer build loop."""

import logging
from collections.abc import Callable

from .models import Block, BuildResult, BuildStep, HouseRequest, Layer, VerificationResult
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
        check_interval_seconds: float = 3.0,
        on_state_change: Callable[[BuildState, BuildState], None] | None = None,
    ) -> None:
        self.robot = robot
        self.policy = policy
        self.verifier = verifier
        self.duration_seconds = duration_seconds
        # MolmoAct2 has no built-in stopping signal -- it never reports "task complete",
        # it just keeps emitting motion for as long as it's asked to run. So instead of
        # running blind for the full duration_seconds and checking once at the end, run
        # in check_interval_seconds slices and re-verify after each one, stopping as soon
        # as the block is actually in place rather than always burning the full budget.
        self.check_interval_seconds = check_interval_seconds
        self.on_state_change = on_state_change
        self.state_machine = BuildStateMachine(on_transition=on_state_change)
        self.log = logging.getLogger(__name__)
        self._plan: list[BuildStep] = []
        self._completed: list[Layer] = []
        self._failed_layer: Layer | None = None
        self._session_open = False

    def build(self, request: HouseRequest) -> BuildResult:
        """Build all three layers in one call."""
        try:
            self.prepare(request)
            result = BuildResult(False, message="No layers were built.")
            for layer in Layer:
                result = self.build_layer(layer)
                if not result.success:
                    return result
            return result
        finally:
            self.close()

    def prepare(self, request: HouseRequest) -> None:
        """Connect and prepare a staged build without moving a block."""
        if self._session_open:
            raise RuntimeError("A build session is already active.")
        self._plan = create_build_plan(request)
        self._completed = []
        self._failed_layer = None
        self.state_machine = BuildStateMachine(on_transition=self.on_state_change)
        self.log.info("Prepared house request: %s", request)
        self.state_machine.transition(BuildState.CONNECTING)
        self._session_open = True
        try:
            self.robot.connect()
            self.policy.load()
            self.state_machine.transition(BuildState.HOMING)
            self.robot.move_home()
        except (Exception, KeyboardInterrupt):
            self.state_machine.fail()
            self.close()
            raise

    def build_layer(self, layer: Layer) -> BuildResult:
        """Build and verify the next requested layer in an active session."""
        if not self._session_open:
            raise RuntimeError("Prepare a build before building a layer.")
        if self._failed_layer is not None:
            raise RuntimeError(
                f"{self._failed_layer.value.capitalize()} failed. "
                "Reset the failed placement and retry the last step."
            )
        return self._build_layer(layer)

    def retry_last_step(self) -> BuildResult:
        """Retry the failed layer after a human has reset its placement."""
        if not self._session_open or self._failed_layer is None:
            raise RuntimeError("There is no failed step to retry.")
        layer = self._failed_layer
        self._failed_layer = None
        self.log.info("Human reset confirmed; retrying %s layer", layer.value)
        self.robot.enable()
        self.robot.move_home()
        return self._build_layer(layer)

    def _build_layer(self, layer: Layer) -> BuildResult:
        """Execute one new or explicitly retried layer."""
        layer_index = len(self._completed)
        if layer_index >= len(self._plan):
            raise RuntimeError("The house is already complete.")
        step = self._plan[layer_index]
        if step.block.layer is not layer:
            raise ValueError(
                f"Build {step.block.layer.value} before {layer.value}."
            )

        try:
            self.state_machine.transition(BuildState.EXECUTING)
            self.log.info("Building %s layer", layer.value)
            self.log.info("MolmoAct2 instruction: %s", step.instruction)
            verification = self._run_until_verified(step, layer_index)
            self.log.info("Verification result: %s", verification)
            if not verification.success:
                return self._failure(
                    self._completed,
                    step.block,
                    verification.reason or "Placement verification failed.",
                )

            self._completed.append(layer)
            if len(self._completed) == len(self._plan):
                self.robot.move_home()
                self.state_machine.transition(BuildState.COMPLETED)
                result = BuildResult(
                    True,
                    list(self._completed),
                    None,
                    "House completed successfully.",
                )
                self.log.info("Final result: %s", result)
                self.close()
                return result

            return BuildResult(
                True,
                list(self._completed),
                None,
                f"{layer.value.capitalize()} layer completed.",
            )
        except (Exception, KeyboardInterrupt):
            self.state_machine.fail()
            self.close()
            raise

    def _run_until_verified(self, step: BuildStep, layer_index: int) -> VerificationResult:
        """Run the policy in check_interval_seconds slices, re-verifying after each one,
        stopping as soon as verification succeeds instead of always running the full
        duration_seconds. self.policy.run_instruction's own return value isn't a
        meaningful success signal for MolmoAct2 (see run_instruction's docstring) -- ground
        truth here is entirely what the camera verifier reports."""
        elapsed = 0.0
        verification = VerificationResult(False, False, False, False, False, "No attempt made.")
        while elapsed < self.duration_seconds:
            chunk_duration = min(self.check_interval_seconds, self.duration_seconds - elapsed)
            self.policy.run_instruction(step.instruction, chunk_duration)
            elapsed += chunk_duration

            self.robot.move_home()
            self.state_machine.transition(BuildState.VERIFYING)
            verification = self.verifier.verify(step.block, layer_index)
            if verification.success or elapsed >= self.duration_seconds:
                return verification
            self.state_machine.transition(BuildState.EXECUTING)
        return verification

    @property
    def completed_layers(self) -> list[Layer]:
        return list(self._completed)

    @property
    def session_active(self) -> bool:
        return self._session_open

    @property
    def failed_layer(self) -> Layer | None:
        return self._failed_layer

    def close(self) -> None:
        """Stop and disconnect an active build session."""
        if not self._session_open:
            return
        self._session_open = False
        self._failed_layer = None
        try:
            self.robot.stop()
        finally:
            try:
                self.verifier.close()
            finally:
                self.robot.disconnect()

    def _failure(
        self,
        completed: list[Layer],
        expected_block: Block,
        reason: str,
    ) -> BuildResult:
        self._failed_layer = expected_block.layer
        self.state_machine.fail()
        self.robot.stop()
        message = (
            f"{expected_block.layer.value.capitalize()} layer failed; expected "
            f"{expected_block.color.value} {expected_block.layer.value}: {reason} "
            'Have a human remove the failed placement, then say "retry last step".'
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
