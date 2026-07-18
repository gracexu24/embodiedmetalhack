import pytest

from house_builder.models import Color, HouseRequest
from house_builder.parser import parse_house_request


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        (
            "red door, yellow wall, blue roof",
            HouseRequest(Color.RED, Color.YELLOW, Color.BLUE),
        ),
        (
            "blue roof, red walls, yellow door",
            HouseRequest(Color.YELLOW, Color.RED, Color.BLUE),
        ),
        (
            "door blue, wall red, roof yellow",
            HouseRequest(Color.BLUE, Color.RED, Color.YELLOW),
        ),
        (
            "BUILD: a BLUE-DOOR; YELLOW-WALL, RED-ROOF house!",
            HouseRequest(Color.BLUE, Color.YELLOW, Color.RED),
        ),
        (
            "I want a yellow roof with blue walls and a red door.",
            HouseRequest(Color.RED, Color.BLUE, Color.YELLOW),
        ),
    ],
)
def test_parse_supported_sentences(sentence: str, expected: HouseRequest) -> None:
    assert parse_house_request(sentence) == expected


def test_missing_roof() -> None:
    with pytest.raises(ValueError, match="Missing: roof"):
        parse_house_request("red door and yellow walls")


def test_conflicting_door_colors() -> None:
    with pytest.raises(ValueError, match="Conflicting colors for door"):
        parse_house_request("red door, blue door, yellow wall, red roof")


def test_unsupported_color() -> None:
    with pytest.raises(ValueError, match="Unsupported color 'green' for door"):
        parse_house_request("green door, yellow wall, blue roof")
