"""Tests for the on_transition/on_state_change hook added for the web dashboard's
WebSocket status stream (backend/build_runner.py)."""

from house_builder.builder import HouseBuilder
from house_builder.models import Color, HouseRequest
from house_builder.policy import MockPolicy
from house_builder.robot import MockRobot
from house_builder.state_machine import BuildState, BuildStateMachine
from house_builder.verifier import PlacementVerifier

REQUEST = HouseRequest(Color.RED, Color.YELLOW, Color.BLUE)


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
    robot = MockRobot()
    verifier = PlacementVerifier({}, mock=True)
    builder = HouseBuilder(
        robot,
        MockPolicy(),
        verifier,
        on_state_change=lambda prev, nxt: seen.append(nxt.value),
    )

    result = builder.build(REQUEST)

    assert result.success
    assert seen[0] == "connecting"
    assert seen[-1] == "completed"
    assert seen.count("verifying") == 3
