"""Small domain models for the three-layer house."""

from dataclasses import dataclass, field
from enum import Enum


class Color(str, Enum):
    RED = "red"
    YELLOW = "yellow"
    BLUE = "blue"
    GREEN = "green"


class Layer(str, Enum):
    DOOR = "door"
    WALL = "wall"
    ROOF = "roof"


ALLOWED_COLORS: dict[Layer, frozenset[Color]] = {
    Layer.DOOR: frozenset({Color.RED, Color.BLUE}),
    Layer.WALL: frozenset({Color.YELLOW, Color.GREEN}),
    Layer.ROOF: frozenset({Color.RED, Color.BLUE}),
}


def _validate_layer_color(layer: Layer, color: Color) -> None:
    if color not in ALLOWED_COLORS[layer]:
        allowed = " or ".join(sorted(item.value for item in ALLOWED_COLORS[layer]))
        raise ValueError(f"{layer.value} color must be {allowed}, got {color.value}.")


@dataclass(frozen=True)
class HouseRequest:
    door: Color
    wall: Color
    roof: Color

    def __post_init__(self) -> None:
        _validate_layer_color(Layer.DOOR, self.door)
        _validate_layer_color(Layer.WALL, self.wall)
        _validate_layer_color(Layer.ROOF, self.roof)


@dataclass(frozen=True)
class Block:
    layer: Layer
    color: Color

    def __post_init__(self) -> None:
        _validate_layer_color(self.layer, self.color)


@dataclass(frozen=True)
class BuildStep:
    block: Block
    instruction: str


@dataclass
class VerificationResult:
    success: bool
    correct_block: bool
    correct_position: bool
    correct_height: bool
    stable: bool
    reason: str = ""


@dataclass
class BuildResult:
    success: bool
    completed_layers: list[Layer] = field(default_factory=list)
    failed_layer: Layer | None = None
    message: str = ""
