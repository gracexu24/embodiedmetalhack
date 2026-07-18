"""Turn a house request into three ordered pick-and-place steps."""

from .models import Block, BuildStep, HouseRequest, Layer


def create_build_plan(request: HouseRequest) -> list[BuildStep]:
    """Return door, wall, and roof steps in safe stacking order."""
    door = Block(Layer.DOOR, request.door)
    wall = Block(Layer.WALL, request.wall)
    roof = Block(Layer.ROOF, request.roof)
    return [
        BuildStep(
            door,
            f"Pick up the {door.color.value} door block.",
            f"Place the held {door.color.value} door block in the house foundation position.",
        ),
        BuildStep(
            wall,
            f"Pick up the {wall.color.value} wall block.",
            (
                f"Stack the held {wall.color.value} wall block "
                "directly on top of the door block."
            ),
        ),
        BuildStep(
            roof,
            f"Pick up the {roof.color.value} roof block.",
            (
                f"Stack the held {roof.color.value} roof block "
                "directly on top of the wall block."
            ),
        ),
    ]
