"""Readable three-layer build loop."""

import logging
import threading
from collections.abc import Callable
from typing import Any

from .models import Block, BuildResult, BuildStep, HouseRequest, Layer
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
        on_highlight: Callable[[str, str, Any], None] | None = None,
    ) -> None:
        self.robot = robot
        self.policy = policy
        self.verifier = verifier
        self.duration_seconds = duration_seconds
        # Called at key build moments (layer started, layer placed) with
        # (kind, label, cam1_frame_or_None) so the UI can build a live highlights reel.
        self.on_highlight = on_highlight
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
        # Set by pause() (from the UI/another thread) to stop the currently-running task
        # loop after its current policy chunk. The operator -- not verification -- decides
        # when a task is done: run continuously, pause, reset by hand, continue to the next.
        self._pause_event = threading.Event()
        # Set by request_stop() to abort the running task loop (the Stop button). Distinct
        # from pause: a stop aborts the layer (not marked complete) and the session is then
        # closed. Signalled out-of-band from the API thread so the single build worker --
        # busy in the policy loop -- actually breaks free (a queued stop would never run).
        self._stop_event = threading.Event()

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
            # No auto-homing: the policy runs from the arm's current pose (like
            # deploy_policy.py). Homing to a fixed pose is an operator action done between
            # tasks (robot.move_home), never automatic -- that was the start-of-run lurch.
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
        return self._build_layer(layer)

    def _build_layer(self, layer: Layer) -> BuildResult:
        """Execute one new or explicitly retried layer.

        Layers may be built in any order: each is an independent pick-and-place with no
        physical dependency, so the requested layer's plan step is looked up directly
        rather than forced to follow door -> wall -> roof. The house is COMPLETED once all
        planned layers are done, whatever order they were built in.
        """
        if layer in self._completed:
            raise RuntimeError(f"{layer.value.capitalize()} layer is already built.")
        step = next((s for s in self._plan if s.block.layer is layer), None)
        if step is None:
            raise RuntimeError(f"No {layer.value} layer in the current plan.")

        try:
            self.state_machine.transition(BuildState.EXECUTING)
            # A previous task's pause released torque (robot.stop) so the operator could
            # reset the scene by hand; re-enable it before this layer drives the arm.
            # Servos re-hold their current pose, and send_action clamps to max_step_deg,
            # so there is no jump. No-op on the first layer (torque already on).
            self.robot.enable()
            self.log.info("Building %s layer", layer.value)
            self.log.info("MolmoAct2 instruction: %s", step.instruction)
            self._emit_highlight("instruction", step.instruction)
            # Run this task's policy continuously until the operator pauses (no visual
            # verification, no auto-home). Returning means the operator judged the task
            # done and paused it; they reset the scene by hand before the next task.
            if not self._run_current_task(step):
                # Stop was requested mid-task: abort without marking the layer complete.
                # close() (torque off + disconnect) is driven by BuildRunner._stop once
                # this worker frees; the session is torn down, not advanced.
                return BuildResult(
                    False,
                    list(self._completed),
                    None,
                    "Build stopped.",
                )

            self._completed.append(layer)
            self._emit_highlight("completed", f"{layer.value.capitalize()} placed")
            if len(self._completed) == len(self._plan):
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

    def _run_current_task(self, step: BuildStep) -> bool:
        """Drive the policy for this task's instruction continuously until the operator
        pauses or stops. MolmoAct2 has no built-in "done" signal, so instead of a fixed
        duration or camera verification, we loop the policy in short check_interval_seconds
        chunks -- short only so pause()/stop are noticed promptly between chunks -- until
        _pause_event or _stop_event is set (from the UI).

        Returns True if the operator paused (the task is done -- reset by hand, build the
        next layer) or False if a stop was requested (abort the layer and tear down)."""
        self._pause_event.clear()
        should_stop = lambda: self._pause_event.is_set() or self._stop_event.is_set()  # noqa: E731
        while not self._pause_event.is_set() and not self._stop_event.is_set():
            self.policy.run_instruction(
                step.instruction, self.check_interval_seconds, should_stop=should_stop
            )
        if self._stop_event.is_set():
            # Stop: leave torque handling to close() (called by BuildRunner._stop), which
            # sets torque off and disconnects. Don't touch the bus here.
            self.log.info("Stop requested; aborting task: %s", step.instruction)
            return False
        # Operator paused: release torque so the arm goes slack and the scene can be reset
        # by hand. Done here on the build thread (after the policy loop has stopped) to
        # avoid touching the Feetech bus while a chunk is mid-write. The next layer turns
        # torque back on -- see _build_layer's robot.enable().
        self.robot.stop()
        self.log.info(
            "Task paused by operator; torque released for manual reset: %s", step.instruction
        )
        return True

    def request_stop(self) -> None:
        """Signal the running task loop to abort after its current policy chunk. Thread-safe
        -- called from the UI/API thread while the build loop runs on the single worker, so
        the worker breaks free and BuildRunner._stop's close() can actually run."""
        self._stop_event.set()

    def _emit_highlight(self, kind: str, label: str) -> None:
        """Push a live highlight (with a fresh cam1 frame) to on_highlight, if wired.

        Best-effort: a camera hiccup or callback error must never break the build, so a
        missing frame is passed as None and any failure is logged and swallowed."""
        if self.on_highlight is None:
            return
        frame: Any = None
        try:
            frame = self.verifier.camera_observations().get("cam1")
        except Exception:
            self.log.warning("Could not capture cam1 thumbnail for highlight", exc_info=True)
        try:
            self.on_highlight(kind, label, frame)
        except Exception:
            self.log.warning("on_highlight callback failed", exc_info=True)

    def pause(self) -> None:
        """Signal the running task loop to stop after its current policy chunk. Thread-safe
        -- called from the UI/API thread while the build loop runs on another."""
        self._pause_event.set()

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
