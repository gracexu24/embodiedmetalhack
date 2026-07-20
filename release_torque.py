"""Release torque on the SO-101 follower -- a panic/relax utility.

A hard-killed backend (kill -9) never runs SO101Robot.disconnect(), so the Feetech
servos stay energized with nothing connected to them. Run this to connect to the
follower and turn torque OFF so the arm can be moved by hand.

    .venv/bin/python release_torque.py                 # uses ./config.yaml
    .venv/bin/python release_torque.py --config path/to/config.yaml

Requires so100_hackathon importable (it is inside this repo's .venv).
"""

from __future__ import annotations

import argparse

import yaml

from house_builder.robot import SO101Robot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    robot = SO101Robot(config["robot"])
    print("Connecting to follower...", flush=True)
    robot.connect()  # briefly holds current pose; writes no motion goals
    try:
        robot.stop()  # set_torque(False)
    finally:
        robot.disconnect()  # set_torque(False) + close the bus
    print("Torque released -- the arm is now free to move by hand.", flush=True)


if __name__ == "__main__":
    main()
