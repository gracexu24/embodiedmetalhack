"""Small domain models for the three-layer house."""

from dataclasses import dataclass, field
from enum import Enum


class Color(str, Enum):
    RED = "red"
    YELLOW = "yellow"
    BLUE = "blue"


class Layer(str, Enum):
    DOOR = "door"
    WALL = "wall"
    ROOF = "roof"


@dataclass(frozen=True)
class HouseRequest:
    door: Color
    wall: Color
    roof: Color


@dataclass(frozen=True)
class Block:
    layer: Layer
    color: Color


@dataclass(frozen=True)
class BuildStep:
    block: Block
    pick_instruction: str
    place_instruction: str


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
