"""Deterministic parser for natural-language house descriptions."""

import re

from .models import Color, HouseRequest, Layer

_COLOR_NAMES = {color.value for color in Color}
_LAYER_NAMES = {
    "door": Layer.DOOR,
    "wall": Layer.WALL,
    "walls": Layer.WALL,
    "roof": Layer.ROOF,
}
_UNSUPPORTED_COLORS = {
    "black",
    "brown",
    "green",
    "grey",
    "gray",
    "orange",
    "pink",
    "purple",
    "white",
}


def parse_house_request(sentence: str) -> HouseRequest:
    """Parse one unambiguous color for the door, wall, and roof."""
    normalized = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
    if not normalized:
        raise ValueError("The house description is empty.")

    tokens = normalized.split()
    matches: dict[Layer, set[Color]] = {layer: set() for layer in Layer}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        third_token = tokens[index + 2] if index + 2 < len(tokens) else ""

        if token in _UNSUPPORTED_COLORS and next_token in _LAYER_NAMES:
            layer = _LAYER_NAMES[next_token]
            raise ValueError(f"Unsupported color {token!r} for {layer.value}.")
        if token in _LAYER_NAMES and next_token in _UNSUPPORTED_COLORS:
            layer = _LAYER_NAMES[token]
            raise ValueError(f"Unsupported color {next_token!r} for {layer.value}.")
        if (
            token in _LAYER_NAMES
            and next_token == "is"
            and third_token in _UNSUPPORTED_COLORS
        ):
            layer = _LAYER_NAMES[token]
            raise ValueError(f"Unsupported color {third_token!r} for {layer.value}.")

        if token in _COLOR_NAMES and next_token in _LAYER_NAMES:
            matches[_LAYER_NAMES[next_token]].add(Color(token))
            index += 2
            continue
        if token in _LAYER_NAMES and next_token in _COLOR_NAMES:
            matches[_LAYER_NAMES[token]].add(Color(next_token))
            index += 2
            continue
        if (
            token in _LAYER_NAMES
            and next_token == "is"
            and third_token in _COLOR_NAMES
        ):
            matches[_LAYER_NAMES[token]].add(Color(third_token))
            index += 3
            continue
        index += 1

    colors_by_layer: dict[Layer, Color] = {}
    for layer, layer_matches in matches.items():
        if len(layer_matches) > 1:
            names = ", ".join(sorted(color.value for color in layer_matches))
            raise ValueError(f"Conflicting colors for {layer.value}: {names}.")
        if layer_matches:
            colors_by_layer[layer] = layer_matches.pop()

    missing = [layer.value for layer in Layer if layer not in colors_by_layer]
    if missing:
        raise ValueError(
            "Could not determine one color for each required layer. "
            f"Missing: {', '.join(missing)}."
        )

    return HouseRequest(
        door=colors_by_layer[Layer.DOOR],
        wall=colors_by_layer[Layer.WALL],
        roof=colors_by_layer[Layer.ROOF],
    )
