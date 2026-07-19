#!/usr/bin/env python3
"""Command-line entry point for one requested house."""

import argparse
import logging
from pathlib import Path
from typing import Any

import rerun as rr
import yaml

from house_builder.builder import HouseBuilder
from house_builder.parser import parse_house_request
from house_builder.policy import MolmoAct2Policy
from house_builder.robot import SO101Robot
from house_builder.rr_blueprint import build_blueprint
from house_builder.verifier import PlacementVerifier


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    for key in ("robot", "policy", "cameras", "verification"):
        if key not in config:
            raise ValueError(f"{path} is missing the {key!r} section.")
    features = config.get("features", {})
    config["verification"]["enabled"] = bool(
        features.get("camera_verification", True)
    )
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a three-block SO-101 house.")
    parser.add_argument("request", help="Natural-language description of the desired house")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--no-viewer", action="store_true", help="Don't spawn the Rerun viewer (headless runs)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rr.init("house_builder", spawn=not args.no_viewer, default_blueprint=build_blueprint())

    try:
        request = parse_house_request(args.request)
    except ValueError as exc:
        print(f"Could not parse house request: {exc}")
        print(
            "\nRequired:\n- door\n- wall\n- roof\n\nExample:\n"
            "Build a house with a red door, yellow walls, and a blue roof."
        )
        return 2

    print(
        "Parsed house:\n"
        f"  door: {request.door.value}\n"
        f"  wall: {request.wall.value}\n"
        f"  roof: {request.roof.value}\n",
        flush=True,
    )

    try:
        config = load_config(args.config)
        robot = SO101Robot(config["robot"])
        verifier = PlacementVerifier(
            config["verification"],
            config["cameras"],
        )
        policy = MolmoAct2Policy(
            config["policy"],
            robot,
            verifier.camera_observations,
        )
        builder = HouseBuilder(
            robot,
            policy,
            verifier,
            float(config["policy"]["skill_duration_seconds"]),
            float(config["policy"].get("check_interval_seconds", 3.0)),
        )
        result = builder.build(request)
    except KeyboardInterrupt:
        print("\nBuild interrupted. Robot stop and disconnect were requested.")
        return 130
    except Exception as exc:
        print(f"Build aborted safely: {exc}")
        return 1

    print(result.message)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
