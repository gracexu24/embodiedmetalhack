"""Small YAML config loader for the backend, mirroring run.py's loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
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
