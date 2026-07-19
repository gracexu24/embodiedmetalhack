import pytest

from house_builder.models import BuildResult, Color, HouseRequest, Layer
from voice_control import VoiceBuildController, VoiceCommand, parse_voice_command


class FakeBuilder:
    def __init__(self) -> None:
        self.session_active = False
        self.completed_layers: list[Layer] = []
        self.prepared: HouseRequest | None = None
        self.closed = False

    def prepare(self, request: HouseRequest) -> None:
        self.prepared = request
        self.session_active = True

    def build_layer(self, layer: Layer) -> BuildResult:
        expected = list(Layer)[len(self.completed_layers)]
        if layer is not expected:
            raise ValueError(f"Expected {expected.value}.")
        self.completed_layers.append(layer)
        if layer is Layer.ROOF:
            self.session_active = False
            return BuildResult(True, list(self.completed_layers), message="House completed.")
        return BuildResult(
            True,
            list(self.completed_layers),
            message=f"{layer.value.capitalize()} completed.",
        )

    def close(self) -> None:
        self.closed = True
        self.session_active = False


@pytest.mark.parametrize(
    ("transcript", "command"),
    [
        ("Build this!", VoiceCommand.BUILD_THIS),
        ("start", VoiceCommand.START),
        ("build the wall", VoiceCommand.BUILD_WALL),
        ("BUILD ROOF", VoiceCommand.BUILD_ROOF),
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
