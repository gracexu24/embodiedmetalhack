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
            "blue roof, green walls, blue door",
            HouseRequest(Color.BLUE, Color.GREEN, Color.BLUE),
        ),
        (
            "door blue, wall green, roof red",
            HouseRequest(Color.BLUE, Color.GREEN, Color.RED),
        ),
        (
            "BUILD: a BLUE-DOOR; YELLOW-WALL, RED-ROOF house!",
            HouseRequest(Color.BLUE, Color.YELLOW, Color.RED),
        ),
        (
            "I want a red roof with green walls and a red door.",
            HouseRequest(Color.RED, Color.GREEN, Color.RED),
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
    with pytest.raises(ValueError, match="Unsupported color 'orange' for door"):
        parse_house_request("orange door, yellow wall, blue roof")


@pytest.mark.parametrize(
    ("sentence", "message"),
    [
        ("yellow door, green wall, red roof", "door color must be blue or red"),
        ("red door, blue wall, red roof", "wall color must be green or yellow"),
        ("red door, yellow wall, green roof", "roof color must be blue or red"),
    ],
)
def test_rejects_color_in_wrong_layer(sentence: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_house_request(sentence)
