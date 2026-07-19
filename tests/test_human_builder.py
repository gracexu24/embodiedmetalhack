import cv2
import numpy as np

from house_builder.models import Color, HouseRequest
from human_builder import detect_model_house, request_to_sentence

CONFIG = {
    "min_color_occupancy": 0.01,
    "min_color_margin": 0.005,
    "height_regions": {
        "door": {"min_y": 300, "max_y": 380},
        "wall": {"min_y": 220, "max_y": 300},
        "roof": {"min_y": 140, "max_y": 220},
    },
}


def test_camera_model_house_becomes_harness_sentence() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (300, 310), (340, 370), (0, 0, 255), -1)
    cv2.rectangle(frame, (300, 230), (340, 290), (0, 255, 0), -1)
    cv2.rectangle(frame, (300, 150), (340, 210), (255, 0, 0), -1)

    request = detect_model_house(frame, CONFIG)

    assert request == HouseRequest(Color.RED, Color.GREEN, Color.BLUE)
    assert request_to_sentence(request) == (
        "Build a house with a red door, green walls, and a blue roof."
    )
