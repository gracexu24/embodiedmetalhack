import pytest

from house_builder.builder import HouseBuilder
from house_builder.models import Color, HouseRequest, Layer
from house_builder.planner import create_build_plan
from house_builder.policy import MockPolicy
from house_builder.robot import MockRobot
from house_builder.state_machine import BuildState, BuildStateMachine
from house_builder.verifier import PlacementVerifier

REQUEST = HouseRequest(Color.RED, Color.YELLOW, Color.BLUE)


def make_builder(
    *,
    fail_once: Layer | None = None,
    fail_always: Layer | None = None,
    policy: MockPolicy | None = None,
) -> tuple[HouseBuilder, MockRobot, MockPolicy, PlacementVerifier]:
    robot = MockRobot()
    selected_policy = policy or MockPolicy()
    verifier = PlacementVerifier(
        {},
        mock=True,
        fail_once=fail_once,
        fail_always=fail_always,
    )
    builder = HouseBuilder(robot, selected_policy, verifier)
    return builder, robot, selected_policy, verifier


def test_successful_three_layer_build() -> None:
    builder, robot, _, verifier = make_builder()
    result = builder.build(REQUEST)
    assert result.success
    assert result.completed_layers == [Layer.DOOR, Layer.WALL, Layer.ROOF]
    assert verifier.calls == [Layer.DOOR, Layer.WALL, Layer.ROOF]
    assert robot.disconnected
    assert builder.state_machine.current is BuildState.COMPLETED
    assert builder.state_machine.history.count(BuildState.VERIFYING) == 3


def test_door_verification_failure_stops_immediately() -> None:
    builder, robot, policy, verifier = make_builder(fail_once=Layer.DOOR)
    result = builder.build(REQUEST)
    assert not result.success
    assert result.failed_layer is Layer.DOOR
    assert verifier.calls == [Layer.DOOR]
    assert len(policy.instructions) == 2
    assert robot.stopped
    assert builder.state_machine.current is BuildState.FAILED


def test_wall_failure_prevents_roof_execution() -> None:
    builder, _, policy, verifier = make_builder(fail_always=Layer.WALL)
    result = builder.build(REQUEST)
    assert not result.success
    assert result.failed_layer is Layer.WALL
    assert verifier.calls == [Layer.DOOR, Layer.WALL]
    assert len(policy.instructions) == 4
    assert not any("roof" in instruction for instruction in policy.instructions)


def test_roof_failure_is_reported() -> None:
    builder, _, policy, _ = make_builder(fail_once=Layer.ROOF)
    result = builder.build(REQUEST)
    assert not result.success
    assert result.failed_layer is Layer.ROOF
    assert "blue roof" in result.message
    assert len(policy.instructions) == 6


def test_robot_disconnects_after_exception() -> None:
    policy = MockPolicy(raise_on_call=1)
    builder, robot, _, _ = make_builder(policy=policy)
    with pytest.raises(RuntimeError, match="Injected"):
        builder.build(REQUEST)
    assert robot.stopped
    assert robot.disconnected
    assert builder.state_machine.current is BuildState.FAILED


def test_policy_instructions_execute_in_order() -> None:
    builder, _, policy, _ = make_builder()
    result = builder.build(REQUEST)
    expected = [
        instruction
        for step in create_build_plan(REQUEST)
        for instruction in (step.pick_instruction, step.place_instruction)
    ]
    assert result.success
    assert policy.instructions == expected


def test_state_machine_rejects_invalid_transition() -> None:
    machine = BuildStateMachine()
    with pytest.raises(RuntimeError, match="idle -> verifying"):
        machine.transition(BuildState.VERIFYING)
