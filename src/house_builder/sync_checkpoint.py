"""Download a trained MolmoAct2 checkpoint from a Modal Volume to local disk, then
report the local path to put in house_builder/config.yaml's policy.local_checkpoint.

Fill in MODAL_VOLUME_NAME and MODAL_CHECKPOINT_PATH once your Modal training job's
volume name and output path are settled -- same "placeholder now, fill in later"
pattern as LEADER_PORT/FOLLOWER_PORT in robot/teleop_config.py. Requires Modal auth
already set up locally (`modal token set` / MODAL_TOKEN_ID+MODAL_TOKEN_SECRET).

NOT independently testable without real Modal credentials and a real volume, so this
hasn't been run end-to-end -- verified against the installed `modal` SDK's actual
method signatures (Volume.from_name, .iterdir, .read_file_into_fileobj), not guessed.
"""
from __future__ import annotations

import argparse
import posixpath
from pathlib import Path

import modal
from modal.types import FileEntryType

MODAL_VOLUME_NAME = None  # e.g. "molmoact2-checkpoints" -- fill in once training is set up
MODAL_CHECKPOINT_PATH = None  # e.g. "/run-2026-07-18/checkpoint-final" -- path *inside* the volume
DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent / "checkpoints"


def sync_checkpoint(
    volume_name: str,
    remote_path: str,
    local_dir: Path,
    environment_name: str | None = None,
) -> Path:
    """Downloads every file under `remote_path` in the named Modal Volume into
    `local_dir`, preserving the directory structure, and returns the local path
    corresponding to `remote_path` (what to put in policy.local_checkpoint).
    """
    volume = modal.Volume.from_name(volume_name, environment_name=environment_name)
    local_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for entry in volume.iterdir(remote_path, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        # entry.path semantics (volume-absolute vs. relative to remote_path) aren't
        # verifiable without a live volume -- posixpath.relpath degrades gracefully
        # either way, but double check the printed destinations look right the first
        # time you run this.
        relative = posixpath.relpath(entry.path, remote_path)
        local_path = local_dir / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            volume.read_file_into_fileobj(entry.path, f)
        downloaded += 1
        print(f"  {entry.path} -> {local_path} ({entry.size} bytes)")

    if downloaded == 0:
        raise RuntimeError(
            f"No files found under {remote_path!r} in volume {volume_name!r}. "
            "Check the volume name and path."
        )
    print(f"Downloaded {downloaded} file(s) to {local_dir}")
    return local_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default=MODAL_VOLUME_NAME, help="Modal Volume name")
    parser.add_argument("--path", default=MODAL_CHECKPOINT_PATH, help="Path inside the volume")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument(
        "--environment",
        default=None,
        help="Modal environment name, if not default",
    )
    args = parser.parse_args()

    if not args.volume or not args.path:
        parser.error(
            "Set MODAL_VOLUME_NAME/MODAL_CHECKPOINT_PATH at the top of this file, "
            "or pass --volume/--path explicitly."
        )

    local_path = sync_checkpoint(args.volume, args.path, args.local_dir, args.environment)
    print(f"\nSet house_builder/config.yaml's policy.local_checkpoint to:\n  {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
