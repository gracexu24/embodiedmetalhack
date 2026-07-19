import pytest

from house_builder.models import Color, HouseRequest, Layer
from house_builder.planner import create_build_plan


def test_plan_has_three_layers_in_order() -> None:
    plan = create_build_plan(HouseRequest(Color.RED, Color.YELLOW, Color.BLUE))
    assert len(plan) == 3
    assert [step.block.layer for step in plan] == [
        Layer.DOOR,
        Layer.WALL,
        Layer.ROOF,
    ]
    assert [step.block.color for step in plan] == [
        Color.RED,
        Color.YELLOW,
        Color.BLUE,
    ]


def test_plan_uses_one_combined_instruction_per_layer() -> None:
    plan = create_build_plan(HouseRequest(Color.RED, Color.YELLOW, Color.BLUE))
    assert plan[0].instruction == (
        "Pick up the red block and place it on the black rectangle."
    )
    assert plan[1].instruction == (
        "Pick up the yellow block and stack it on the first red block."
    )
    assert plan[2].instruction == (
        "Pick up the blue triangle block and stack it on the second yellow block."
    )


@pytest.mark.parametrize("door", [Color.RED, Color.BLUE])
@pytest.mark.parametrize("wall", [Color.YELLOW, Color.GREEN])
@pytest.mark.parametrize("roof", [Color.RED, Color.BLUE])
def test_instructions_cover_every_physical_combination(
    door: Color,
    wall: Color,
    roof: Color,
) -> None:
    plan = create_build_plan(HouseRequest(door, wall, roof))
    assert plan[0].instruction == (
        f"Pick up the {door.value} block and place it on the black rectangle."
    )
    assert plan[1].instruction == (
        f"Pick up the {wall.value} block and stack it on the first {door.value} block."
    )
    assert plan[2].instruction == (
        f"Pick up the {roof.value} triangle block and stack it "
        f"on the second {wall.value} block."
    )
