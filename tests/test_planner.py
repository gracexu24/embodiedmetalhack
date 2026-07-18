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


def test_plan_instructions_are_short_and_specific() -> None:
    plan = create_build_plan(HouseRequest(Color.RED, Color.YELLOW, Color.BLUE))
    assert plan[0].pick_instruction == "Pick up the red door block."
    assert plan[0].place_instruction == (
        "Place the held red door block in the house foundation position."
    )
    assert plan[1].pick_instruction == "Pick up the yellow wall block."
    assert plan[1].place_instruction == (
        "Stack the held yellow wall block directly on top of the door block."
    )
    assert plan[2].pick_instruction == "Pick up the blue roof block."
    assert plan[2].place_instruction == (
        "Stack the held blue roof block directly on top of the wall block."
    )
