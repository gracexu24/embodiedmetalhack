"""Turn a house request into three combined MolmoAct2 skills."""

from .models import Block, BuildStep, HouseRequest, Layer


def create_build_plan(request: HouseRequest) -> list[BuildStep]:
    """Return one combined pick-and-place instruction per layer."""
    door = Block(Layer.DOOR, request.door)
    wall = Block(Layer.WALL, request.wall)
    roof = Block(Layer.ROOF, request.roof)
    return [
        BuildStep(
            door,
            (
                f"Pick up the {door.color.value} block and place it "
                "on the black rectangle."
            ),
        ),
        BuildStep(
            wall,
            (
                f"Pick up the {wall.color.value} block and stack it "
                f"on the first {door.color.value} block."
            ),
        ),
        BuildStep(
            roof,
            (
                f"Pick up the {roof.color.value} triangle block and stack it "
                f"on the second {wall.color.value} block."
            ),
        ),
    ]
