import pytest

from house_builder.builder import HouseBuilder
from house_builder.models import Block, Color, HouseRequest, Layer, VerificationResult
from house_builder.planner import create_build_plan
from house_builder.policy import Policy
from house_builder.robot import Robot
from house_builder.state_machine import BuildState, BuildStateMachine
from house_builder.verifier import PlacementVerifier

REQUEST = HouseRequest(Color.RED, Color.YELLOW, Color.BLUE)


class RecordingRobot(Robot):
    def __init__(self) -> None:
        self.connected = False
        self.stopped = False
        self.disconnected = False
        self.home_count = 0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnected = True

    def move_home(self) -> None:
        self.home_count += 1

    def stop(self) -> None:
        self.stopped = True

    def get_observation(self) -> dict[str, object]:
        return {"joint_state": [0.0] * 6}

    def send_action(self, action: object) -> None:
        del action


class RecordingPolicy(Policy):
    def __init__(self, raise_on_call: int | None = None) -> None:
        self.instructions: list[str] = []
        self.raise_on_call = raise_on_call

    def load(self) -> None:
        pass

    def run_instruction(self, instruction: str, duration_seconds: float) -> bool:
        del duration_seconds
        call_index = len(self.instructions)
        self.instructions.append(instruction)
        if self.raise_on_call == call_index:
            raise RuntimeError("Injected policy exception.")
        return True


class StubVerifier(PlacementVerifier):
    def __init__(
        self,
        fail_layer: Layer | None = None,
        retry_layer: Layer | None = None,
        retry_failures: int = 0,
    ) -> None:
        self.fail_layer = fail_layer
        self.retry_layer = retry_layer
        self.retry_failures = retry_failures
        self.calls: list[Layer] = []

    def verify(self, expected_block: Block, layer_index: int) -> VerificationResult:
        del layer_index
        self.calls.append(expected_block.layer)
        if expected_block.layer is self.fail_layer:
            return VerificationResult(
                False, True, False, True, False, "Injected verification failure."
            )
        if expected_block.layer is self.retry_layer:
            attempt = self.calls.count(self.retry_layer)
            if attempt <= self.retry_failures:
                return VerificationResult(
                    False, True, False, True, False, "Simulated transient failure."
                )
        return VerificationResult(True, True, True, True, True)

    def close(self) -> None:
        pass


def make_builder(
    *,
    fail_layer: Layer | None = None,
    policy: RecordingPolicy | None = None,
) -> tuple[HouseBuilder, RecordingRobot, RecordingPolicy, StubVerifier]:
    robot = RecordingRobot()
    selected_policy = policy or RecordingPolicy()
    verifier = StubVerifier(fail_layer)
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


def test_staged_build_waits_for_each_layer_command() -> None:
    builder, robot, policy, _ = make_builder()
    builder.prepare(REQUEST)

    door = builder.build_layer(Layer.DOOR)
    assert door.completed_layers == [Layer.DOOR]
    assert builder.session_active
    assert len(policy.instructions) == 1

    wall = builder.build_layer(Layer.WALL)
    assert wall.completed_layers == [Layer.DOOR, Layer.WALL]
    assert builder.session_active
    assert len(policy.instructions) == 2

    roof = builder.build_layer(Layer.ROOF)
    assert roof.success
    assert not builder.session_active
    assert robot.disconnected
    assert len(policy.instructions) == 3


def test_staged_build_rejects_wall_before_door() -> None:
    builder, _, _, _ = make_builder()
    builder.prepare(REQUEST)
    try:
        with pytest.raises(ValueError, match="Build door before wall"):
            builder.build_layer(Layer.WALL)
    finally:
        builder.close()


def test_door_verification_failure_stops_immediately() -> None:
    # A persistently-failing layer retries every check_interval_seconds until
    # duration_seconds runs out, not just once: default 10.0 / 3.0 -> checks at
    # elapsed 3, 6, 9, 10 -> 4 attempts before giving up.
    builder, robot, policy, verifier = make_builder(fail_layer=Layer.DOOR)
    result = builder.build(REQUEST)
    assert not result.success
    assert result.failed_layer is Layer.DOOR
    assert verifier.calls == [Layer.DOOR] * 4
    assert len(policy.instructions) == 4
    assert robot.stopped
    assert builder.state_machine.current is BuildState.FAILED


def test_wall_failure_prevents_roof_execution() -> None:
    builder, _, policy, verifier = make_builder(fail_layer=Layer.WALL)
    result = builder.build(REQUEST)
    assert not result.success
    assert result.failed_layer is Layer.WALL
    assert verifier.calls == [Layer.DOOR] + [Layer.WALL] * 4
    assert len(policy.instructions) == 1 + 4
    assert not any("triangle" in instruction for instruction in policy.instructions)


def test_roof_failure_is_reported() -> None:
    builder, _, policy, _ = make_builder(fail_layer=Layer.ROOF)
    result = builder.build(REQUEST)
    assert not result.success
    assert result.failed_layer is Layer.ROOF
    assert "blue roof" in result.message
    assert len(policy.instructions) == 1 + 1 + 4


def test_layer_retries_until_verified_then_moves_on() -> None:
    """The core reason for retrying: MolmoAct2 has no stopping signal, so a layer that
    fails its first couple of checks should keep going and succeed once verification
    actually passes, instead of stopping on the first failed check."""
    robot = RecordingRobot()
    policy = RecordingPolicy()
    verifier = StubVerifier(retry_layer=Layer.DOOR, retry_failures=2)
    builder = HouseBuilder(robot, policy, verifier)
    result = builder.build(REQUEST)
    assert result.success
    assert result.completed_layers == [Layer.DOOR, Layer.WALL, Layer.ROOF]
    # DOOR: 2 failed checks + 1 successful one; WALL and ROOF succeed on their first check.
    assert verifier.calls == [Layer.DOOR] * 3 + [Layer.WALL, Layer.ROOF]
    assert len(policy.instructions) == 3 + 1 + 1


def test_robot_disconnects_after_exception() -> None:
    policy = RecordingPolicy(raise_on_call=1)
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
        step.instruction for step in create_build_plan(REQUEST)
    ]
    assert result.success
    assert policy.instructions == expected


def test_state_machine_rejects_invalid_transition() -> None:
    machine = BuildStateMachine()
    with pytest.raises(RuntimeError, match="idle -> verifying"):
        machine.transition(BuildState.VERIFYING)
