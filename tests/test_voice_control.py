import pytest

from house_builder.models import BuildResult, Color, HouseRequest, Layer
from voice_control import VoiceBuildController, VoiceCommand, parse_voice_command


class FakeBuilder:
    def __init__(self, fail_on: Layer | None = None) -> None:
        self.session_active = False
        self.completed_layers: list[Layer] = []
        self.failed_layer: Layer | None = None
        self.prepared: HouseRequest | None = None
        self.closed = False
        self.fail_on = fail_on
        self.retry_count = 0

    def prepare(self, request: HouseRequest) -> None:
        self.prepared = request
        self.session_active = True

    def build_layer(self, layer: Layer) -> BuildResult:
        if self.failed_layer is not None:
            raise RuntimeError("Reset the failed placement and retry the last step.")
        expected = list(Layer)[len(self.completed_layers)]
        if layer is not expected:
            raise ValueError(f"Expected {expected.value}.")
        if layer is self.fail_on:
            self.failed_layer = layer
            return BuildResult(
                False,
                list(self.completed_layers),
                layer,
                f"{layer.value.capitalize()} failed.",
            )
        self.completed_layers.append(layer)
        if layer is Layer.ROOF:
            self.session_active = False
            return BuildResult(True, list(self.completed_layers), message="House completed.")
        return BuildResult(
            True,
            list(self.completed_layers),
            message=f"{layer.value.capitalize()} completed.",
        )

    def retry_last_step(self) -> BuildResult:
        if self.failed_layer is None:
            raise RuntimeError("There is no failed step to retry.")
        layer = self.failed_layer
        self.failed_layer = None
        self.fail_on = None
        self.retry_count += 1
        return self.build_layer(layer)

    def close(self) -> None:
        self.closed = True
        self.session_active = False
        self.failed_layer = None


@pytest.mark.parametrize(
    ("transcript", "command"),
    [
        ("Build this!", VoiceCommand.BUILD_THIS),
        ("start", VoiceCommand.START),
        ("build the wall", VoiceCommand.BUILD_WALL),
        ("BUILD ROOF", VoiceCommand.BUILD_ROOF),
        ("retry last step", VoiceCommand.RETRY_LAST_STEP),
        ("retry the last step", VoiceCommand.RETRY_LAST_STEP),
        ("retry", VoiceCommand.RETRY_LAST_STEP),
        ("stop", VoiceCommand.STOP),
        ("something else", VoiceCommand.UNKNOWN),
    ],
)
def test_parse_voice_command(transcript: str, command: VoiceCommand) -> None:
    assert parse_voice_command(transcript) is command


def test_voice_commands_stage_the_build_in_order() -> None:
    request = HouseRequest(Color.RED, Color.GREEN, Color.BLUE)
    builder = FakeBuilder()
    output: list[str] = []
    controller = VoiceBuildController(builder, lambda: request, output.append)  # type: ignore[arg-type]

    assert controller.handle("build this")
    assert builder.prepared is None
    assert controller.handle("start")
    assert builder.prepared == request
    assert builder.completed_layers == [Layer.DOOR]
    assert controller.handle("build wall")
    assert builder.completed_layers == [Layer.DOOR, Layer.WALL]
    assert controller.handle("build roof")
    assert builder.completed_layers == [Layer.DOOR, Layer.WALL, Layer.ROOF]
    assert any("Build a house with a red door" in line for line in output)


def test_start_requires_build_this() -> None:
    builder = FakeBuilder()
    output: list[str] = []
    controller = VoiceBuildController(  # type: ignore[arg-type]
        builder,
        lambda: HouseRequest(Color.BLUE, Color.YELLOW, Color.RED),
        output.append,
    )

    assert controller.handle("start")
    assert builder.completed_layers == []
    assert output[-1] == 'Say "build this" before starting.'


def test_retry_last_step_after_failed_wall() -> None:
    request = HouseRequest(Color.RED, Color.GREEN, Color.BLUE)
    builder = FakeBuilder(fail_on=Layer.WALL)
    output: list[str] = []
    controller = VoiceBuildController(builder, lambda: request, output.append)  # type: ignore[arg-type]

    assert controller.handle("build this")
    assert controller.handle("start")
    assert builder.completed_layers == [Layer.DOOR]
    assert controller.handle("build wall")
    assert builder.failed_layer is Layer.WALL
    assert builder.completed_layers == [Layer.DOOR]

    assert controller.handle("build roof")
    assert output[-1] == 'Reset the failed placement, then say "retry last step".'

    assert controller.handle("retry last step")
    assert builder.retry_count == 1
    assert builder.failed_layer is None
    assert builder.completed_layers == [Layer.DOOR, Layer.WALL]
    assert any('Say "build roof" to continue.' in line for line in output)


def test_retry_last_step_with_nothing_failed() -> None:
    builder = FakeBuilder()
    output: list[str] = []
    controller = VoiceBuildController(  # type: ignore[arg-type]
        builder,
        lambda: HouseRequest(Color.BLUE, Color.YELLOW, Color.RED),
        output.append,
    )

    assert controller.handle("retry last step")
    assert output[-1] == "There is no failed step to retry."
