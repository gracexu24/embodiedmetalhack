"""Tests for the on_transition/on_state_change hook added for the web dashboard's
WebSocket status stream (backend/build_runner.py)."""

from house_builder.builder import HouseBuilder
from house_builder.models import Block, Color, HouseRequest, Layer, VerificationResult
from house_builder.policy import Policy
from house_builder.robot import Robot
from house_builder.state_machine import BuildState, BuildStateMachine
from house_builder.verifier import PlacementVerifier

REQUEST = HouseRequest(Color.RED, Color.YELLOW, Color.BLUE)


class RecordingRobot(Robot):
    def __init__(self) -> None:
        self.connected = False
        self.disconnected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnected = True

    def move_home(self) -> None:
        return

    def stop(self) -> None:
        return

    def enable(self) -> None:
        return

    def get_observation(self) -> dict[str, object]:
        return {"joint_state": [0.0] * 6}

    def send_action(self, action: object) -> None:
        del action


class RecordingPolicy(Policy):
    def load(self) -> None:
        return

    def run_instruction(self, instruction: str, duration_seconds: float) -> bool:
        del instruction, duration_seconds
        return True


class StubVerifier(PlacementVerifier):
    def __init__(self) -> None:
        self.calls: list[Layer] = []

    def verify(self, expected_block: Block, layer_index: int) -> VerificationResult:
        del layer_index
        self.calls.append(expected_block.layer)
        return VerificationResult(True, True, True, True, True)

    def close(self) -> None:
        return


def test_state_machine_on_transition_fires_with_previous_and_next() -> None:
    seen: list[tuple[BuildState, BuildState]] = []
    machine = BuildStateMachine(on_transition=lambda prev, nxt: seen.append((prev, nxt)))

    machine.transition(BuildState.CONNECTING)
    machine.transition(BuildState.HOMING)

    assert seen == [
        (BuildState.IDLE, BuildState.CONNECTING),
        (BuildState.CONNECTING, BuildState.HOMING),
    ]


def test_state_machine_without_callback_still_works() -> None:
    machine = BuildStateMachine()
    machine.transition(BuildState.CONNECTING)
    assert machine.current is BuildState.CONNECTING


def test_house_builder_forwards_on_state_change_through_full_build() -> None:
    seen: list[str] = []
    builder = HouseBuilder(
        RecordingRobot(),
        RecordingPolicy(),
        StubVerifier(),
        on_state_change=lambda prev, nxt: seen.append(nxt.value),
    )

    result = builder.build(REQUEST)

    assert result.success
    assert seen[0] == "connecting"
    assert seen[-1] == "completed"
    assert seen.count("verifying") == 3
