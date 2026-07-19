"""Small state machine for the linear build lifecycle."""

import logging
from enum import Enum

import rerun as rr

from .rr_time import log_step


class BuildState(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    HOMING = "homing"
    PICKING = "picking"
    PLACING = "placing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


_ALLOWED: dict[BuildState, set[BuildState]] = {
    BuildState.IDLE: {BuildState.CONNECTING},
    BuildState.CONNECTING: {BuildState.HOMING, BuildState.FAILED},
    BuildState.HOMING: {BuildState.PICKING, BuildState.FAILED},
    BuildState.PICKING: {BuildState.PLACING, BuildState.FAILED},
    BuildState.PLACING: {BuildState.VERIFYING, BuildState.FAILED},
    BuildState.VERIFYING: {
        BuildState.PICKING,
        BuildState.COMPLETED,
        BuildState.FAILED,
    },
    BuildState.COMPLETED: set(),
    BuildState.FAILED: set(),
}


class BuildStateMachine:
    """Track and validate the few states used by a three-layer build."""

    def __init__(self) -> None:
        self.current = BuildState.IDLE
        self.history = [BuildState.IDLE]
        self.log = logging.getLogger(__name__)

    def transition(self, next_state: BuildState) -> None:
        if next_state not in _ALLOWED[self.current]:
            raise RuntimeError(
                f"Invalid build transition: {self.current.value} -> {next_state.value}"
            )
        self.log.info("State: %s -> %s", self.current.value, next_state.value)
        log_step()
        rr.log("/harness/state", rr.TextLog(f"{self.current.value} -> {next_state.value}"))
        self.current = next_state
        self.history.append(next_state)

    def fail(self) -> None:
        """Move to FAILED unless the build already ended."""
        if self.current not in {BuildState.COMPLETED, BuildState.FAILED}:
            self.transition(BuildState.FAILED)
