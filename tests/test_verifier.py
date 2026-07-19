import cv2
import numpy as np

from house_builder.models import Block, Color, Layer
from house_builder.verifier import PlacementVerifier

CONFIG = {
    "target_x": 320,
    "max_center_error_px": 25,
    "max_stack_alignment_error_px": 25,
    "stability_frames": 2,
    "max_stability_movement_px": 5,
    "min_color_occupancy": 0.01,
    "height_regions": {
        "door": {"min_y": 300, "max_y": 380},
        "wall": {"min_y": 220, "max_y": 300},
        "roof": {"min_y": 140, "max_y": 220},
    },
}

_BGR = {
    Color.RED: (0, 0, 255),
    Color.YELLOW: (0, 255, 255),
    Color.BLUE: (255, 0, 0),
    Color.GREEN: (0, 255, 0),
}


class FrameVerifier(PlacementVerifier):
    def __init__(self, frames: list[np.ndarray]) -> None:
        super().__init__(CONFIG)
        self.frames = iter(frames)

    def _open_camera(self, name: str) -> None:
        assert name == "cam1"

    def _read(self, name: str) -> np.ndarray:
        assert name == "cam1"
        return next(self.frames)


def frame_with_stack(
    blocks: list[Block],
    positions: dict[Layer, int] | None = None,
) -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    positions = positions or {}
    ranges = {
        Layer.DOOR: (310, 370),
        Layer.WALL: (230, 290),
        Layer.ROOF: (150, 210),
    }
    for block in blocks:
        min_y, max_y = ranges[block.layer]
        x = positions.get(block.layer, 320)
        cv2.rectangle(frame, (x - 20, min_y), (x + 20, max_y), _BGR[block.color], -1)
    return frame


def test_cam1_verifies_each_color_above_previous_color() -> None:
    door = Block(Layer.DOOR, Color.RED)
    wall = Block(Layer.WALL, Color.GREEN)
    roof = Block(Layer.ROOF, Color.BLUE)
    frames = (
        [frame_with_stack([door])] * 2
        + [frame_with_stack([door, wall])] * 2
        + [frame_with_stack([door, wall, roof])] * 2
    )
    verifier = FrameVerifier(frames)

    assert verifier.verify(door, 0).success
    assert verifier.verify(wall, 1).success
    assert verifier.verify(roof, 2).success


def test_cam1_rejects_color_not_aligned_with_support() -> None:
    door = Block(Layer.DOOR, Color.RED)
    wall = Block(Layer.WALL, Color.YELLOW)
    frames = [frame_with_stack([door])] * 2 + [
        frame_with_stack([door, wall], {Layer.DOOR: 380, Layer.WALL: 320})
    ] * 2
    verifier = FrameVerifier(frames)

    assert verifier.verify(door, 0).success
    result = verifier.verify(wall, 1)
    assert not result.success
    assert "centered above" in result.reason
