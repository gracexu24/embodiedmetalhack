"""Run a trained MolmoAct2 policy on the follower arm (the Deploy step, closed loop).

Observe (joint state + camera frames) -> POST to a MolmoAct2 inference server -> execute
the returned action chunk on the follower -> observe again::

    pixi run deploy-policy -- --task "pick up the ball" --server http://<gpu-box>:8080/act --dry-run
    pixi run deploy-policy -- --task "pick up the ball" --server http://<gpu-box>:8080/act

The server is a thin wrapper around the checkpoint's ``predict_action`` -- see
``tools/apps/policy_server_molmoact2.py`` for a reference implementation to run on the
GPU box. The contract is one JSON POST per observation::

    {"instruction": str, "state": [6 floats], "images": {"top": <base64 jpeg>, "side": ...}}
        -> {"actions": [[6 floats], ...]}    # an absolute-joint-pose action chunk

Safety: every step is clamped to the follower's calibrated range *and* to
``--max-step-deg`` per joint per tick, so one bad prediction cannot slam the arm. Public
checkpoints were trained on the LeRobot v2.1 joint convention and WILL command wild poses
on a v3-calibrated arm -- deploy checkpoints fine-tuned on your own export, and always do
a ``--dry-run`` first (predictions stream to the viewer, the arm never moves). Ctrl-C
releases torque. The rollout streams to the so100-server live proxy, so the Deploy page
viewer shows exactly what the policy is doing.

Every live rollout is also *recorded as an episode* -- same take machinery as
record-episode, default dataset ``molmoact2_eval``, the instruction as its task and the
tag ``Needs review``. Evaluation runs land in the catalog next to your teleop data:
review them on Refine, and the good ones can even be exported and trained on. Pass
``--dataset ""`` to skip recording; dry runs are never recorded.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import tyro

os.environ.setdefault("RERUN_INSECURE_SKIP_HOST_CHECK", "1")

import rerun as rr  # noqa: E402 - the env var above must be set before use
from replay_episode import (  # noqa: E402  # pyrefly: ignore[missing-import] - sibling script, on sys.path when run as tools/apps/deploy_policy.py
    Follower,
    drive_to,
    open_follower,
    read_calibrated,
)

from so100_hackathon.calibration import DEFAULT_MOTOR_NAMES  # noqa: E402
from so100_hackathon.cameras import detect_camera_indices  # noqa: E402
from so100_hackathon.takes import (  # noqa: E402
    APP_ID,
    begin_take,
    episode_path,
    finish_take,
    next_episode,
    optimize_rrd,
    register_rrd,
    sanitize_name,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GRIPPER_INDEX = DEFAULT_MOTOR_NAMES.index("gripper")


class GripperLatch:
    """Schmitt-trigger hysteresis for --gripper-binary. A plain per-tick sigmoid has no
    memory: a raw prediction hovering near --gripper-threshold (e.g. right after a grasp,
    which sits near a decision boundary, not deep in "closed" territory) can flip the
    decision on the very next inference call from noise alone -- grab, then immediately
    let go, with no real intent behind the release. Once latched, this requires the
    prediction to clearly cross into the OTHER side (past threshold +/- hysteresis/2)
    before flipping, so noise near the boundary can't undo a grasp -- while a real,
    sustained crossing (e.g. actually releasing once over the cup) still goes through."""

    def __init__(self) -> None:
        self.is_open: bool | None = None  # unknown until the first clear reading

    def decide(self, value: float, threshold: float, hysteresis: float) -> float:
        half = hysteresis / 2.0
        if value > threshold + half:
            self.is_open = True
        elif value < threshold - half:
            self.is_open = False
        elif self.is_open is None:  # first reading landed inside the deadband: pick a side
            self.is_open = value > threshold
        return 100.0 if self.is_open else 0.0


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("localhost", port)) == 0


def open_cameras(indices: tuple[int, ...] | None, names: tuple[str, ...]) -> dict[str, tuple[int, cv2.VideoCapture]]:
    """Semantic camera name -> (index, opened capture), in cam-index order (like the export)."""
    if indices is None:
        indices = detect_camera_indices()
    if len(indices) < len(names):
        raise SystemExit(f"found {len(indices)} camera(s) but need {len(names)} ({', '.join(names)}); pass --cameras explicitly")
    if len(indices) > len(names):
        print(f"ignoring extra camera(s) {indices[len(names) :]} -- the policy sees {', '.join(names)}")
    cameras: dict[str, tuple[int, cv2.VideoCapture]] = {}
    for name, index in zip(names, indices, strict=False):
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise SystemExit(f"camera {index} ({name}) failed to open")
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"camera {index} -> {name}")
        cameras[name] = (index, capture)
    return cameras


def observe_cameras(cameras: dict[str, tuple[int, cv2.VideoCapture]], jpeg_quality: int, rec: rr.RecordingStream) -> dict[str, bytes]:
    """Grab one fresh JPEG per camera and log it (under ``camera/cam<N>``, the recording
    convention -- so a recorded rollout episode exports exactly like a teleop one)."""
    jpegs: dict[str, bytes] = {}
    for name, (index, capture) in cameras.items():
        capture.grab()  # drop a possibly stale buffered frame
        ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"camera '{name}' stopped delivering frames")
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise SystemExit(f"camera '{name}': JPEG encoding failed")
        jpegs[name] = encoded.tobytes()
        rec.set_time("time", timestamp=time.time())
        rec.log(f"camera/cam{index}", rr.EncodedImage(contents=jpegs[name], media_type="image/jpeg"))
    return jpegs


def query_actions(server: str, task: str, state: list[float], jpegs: dict[str, bytes], action_mode: str | None) -> np.ndarray:
    """One inference round-trip: observation in, an (N, joints) action chunk out."""
    payload: dict[str, object] = {
        "instruction": task,
        "state": state,
        "images": {name: base64.b64encode(data).decode() for name, data in jpegs.items()},
    }
    if action_mode is not None:
        payload["action_mode"] = action_mode
    request = urllib.request.Request(server, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read())
    except urllib.error.URLError as error:
        raise SystemExit(f"cannot reach the policy server at {server} ({error}) -- is policy_server_molmoact2.py running?") from None
    actions = np.asarray(body["actions"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != len(state):
        raise SystemExit(f"server returned actions of shape {actions.shape}, expected (N, {len(state)})")
    print(f"  chunk of {len(actions)} actions in {time.monotonic() - started:.2f}s")
    return actions


def log_prediction(follower: Follower, target: np.ndarray, current: list[float], rec: rr.RecordingStream) -> None:
    """Dry-run counterpart of drive_to: log what *would* be commanded, drive nothing."""
    rec.set_time("time", timestamp=time.time())
    rec.log(f"{follower.name}/goal", rr.Scalars([float(v) for v in target]))
    rec.log(f"{follower.name}/position", rr.Scalars(current))


def debug_chunk(follower: Follower, motor_names: list[str], state: list[float], chunk: np.ndarray, rec: rr.RecordingStream) -> None:
    """Print what the server actually predicted for this observation, and log the full raw
    chunk (every step, not just the executed/clamped ones) on its own timeline so it's
    inspectable in the viewer independent of what the arm ends up doing."""
    current = np.asarray(state)
    first_delta = chunk[0] - current
    drift = chunk[-1] - chunk[0]
    print(f"  debug: current   {', '.join(f'{n}={v:+.1f}' for n, v in zip(motor_names, current, strict=True))}")
    print(f"  debug: chunk[0]  {', '.join(f'{n}={v:+.1f}' for n, v in zip(motor_names, chunk[0], strict=True))}")
    print(f"  debug: 1st delta {', '.join(f'{n}={v:+.1f}' for n, v in zip(motor_names, first_delta, strict=True))}")
    print(f"  debug: chunk drift (last-first) {', '.join(f'{n}={v:+.1f}' for n, v in zip(motor_names, drift, strict=True))}")
    for i, step in enumerate(chunk):
        rec.set_time("chunk_step", sequence=i)
        rec.log(f"{follower.name}/chunk_preview", rr.Scalars([float(v) for v in step]))


@dataclasses.dataclass
class Config:
    task: str
    """Natural-language instruction for the policy -- the same kind of sentence you typed
    while recording (e.g. "pick up the ball and place it in the box")."""

    server: str = "http://localhost:8080/act"
    """MolmoAct2 inference server endpoint (see tools/apps/policy_server_molmoact2.py)."""

    action_mode: str | None = None
    """continuous (flow-matching, smoother) or discrete (action tokens, more decisive
    per-action commitment). None leaves it to the server's own default. Sent per-request,
    so switching modes doesn't need a server redeploy -- but note --gripper-threshold was
    tuned against one mode's characteristic value range and likely needs re-tuning after
    switching (get fresh --debug numbers, don't assume the old threshold still applies)."""

    fps: float = 30.0
    """Control rate at which chunk actions are written to the arm."""

    execute_steps: int = 24
    """Actions executed from each chunk before re-observing. The server's chunk horizon is
    30; each inference round-trip costs ~0.5-1s regardless, so executing only a handful of
    steps per call mostly buys idle time, not safety -- the arm stutters (move a beat,
    freeze, repeat) instead of moving continuously. Leaving a handful unexecuted still
    hides the *next* call's latency; going all the way to 30 would reintroduce the stutter
    at the chunk boundary instead of within it."""

    max_step_deg: float = 10.0
    """Per-joint, per-tick motion cap in calibrated units. The safety net against a bad
    prediction; raise it only once the rollout looks sane."""

    gripper_binary: bool = False
    """Latch the gripper to fully closed (0%) or fully open (100%) instead of passing the
    model's raw continuous prediction straight through -- for a hesitant/partial
    prediction that doesn't close enough to actually grip. Uses hysteresis (GripperLatch),
    not a memoryless per-tick threshold: a raw prediction hovering near --gripper-threshold
    (e.g. right after a grasp) can't flip the decision by itself, so it won't grab-then-
    immediately-release from noise alone. Still ramps at the same --max-step-deg rate as
    every other joint -- it just commits to an extreme instead of settling on a soft
    in-between value. Use --debug first to see what range the model's raw gripper
    predictions actually fall in before trusting the default --gripper-threshold -- it's
    assumed to roughly track this repo's own 0-100 calibrated-percent convention, not
    confirmed exactly."""

    gripper_threshold: float = 18.0
    """Calibrated-percent midpoint (0=closed, 100=open) --gripper-binary latches around.
    NOT the halfway point of 0-100: the checkpoint's own training stats show its typical
    gripper prediction rarely exceeds ~45 even when "as open as it gets". 18 sits between
    the checkpoint's median (~9, mostly-closed) and its usual open-ish tail (~30-45). Tune
    from your own --debug numbers if this still isn't right for your task."""

    gripper_hysteresis: float = 8.0
    """--gripper-binary's deadband width in percent, centered on --gripper-threshold: the
    prediction has to clear threshold +/- hysteresis/2 to flip the latch, not just tip past
    the threshold itself. Wider = more resistant to noise-driven flip-flopping (grab then
    immediately let go) but slower to react to a real, intended release."""

    seconds: float = 0.0
    """Stop after this many seconds (0 = run until Ctrl-C)."""

    dry_run: bool = False
    """Query the policy and stream its predictions to the viewer without moving the arm
    (torque stays off). Always do this first with a new checkpoint."""

    debug: bool = False
    """Print a per-chunk diagnostic (current state, first predicted pose, per-joint delta,
    and how far the chunk drifts by its last step) and log the full raw chunk to the
    viewer under ``<follower>/chunk_preview`` (its own timeline, so you can inspect what
    the server actually predicted independent of what got executed/clamped)."""

    cameras: tuple[int, ...] | None = None
    """Camera indices to use, in --camera-names order. Default: auto-detect."""

    camera_names: tuple[str, ...] = ("top", "side")
    """Semantic camera names sent to the server; must match the keys the checkpoint was
    trained on (the export's default is top, side)."""

    jpeg_quality: int = 85
    """JPEG quality of the frames sent to the server."""

    ramp_seconds: float = 2.0
    """How long the follower takes to glide to the first predicted pose."""

    port: str | None = None
    """Serial port of the follower. Default: the plugged-in arm whose calibration says "follower"."""

    calibration_dir: Path = Path("calibrations")
    """Directory of <usb_id>.json calibrations."""

    proxy_port: int = 9876
    """so100-server live proxy port (the rollout streams there for the viewer)."""

    dataset: str = "molmoact2_eval"
    """Record each live rollout as an episode of this catalog dataset (the loop closer:
    evaluation runs land next to your teleop data, ready for Refine). Pass ``--dataset ""``
    to disable; dry runs are never recorded."""

    tag: str = "Needs review"
    """Curation tag stamped on the rollout episode."""

    recordings_dir: Path = REPO_ROOT / "recordings"
    """Folder rollout episodes are written to, as ``<dataset>/<episode>.rrd``."""

    catalog_port: int = 51234
    """so100-server catalog port; if reachable, rollout episodes are registered on stop."""


def main(config: Config) -> None:
    follower = open_follower(config.calibration_dir, config.port)
    cameras = open_cameras(config.cameras, config.camera_names)

    proxy_uri = f"rerun+http://localhost:{config.proxy_port}/proxy" if _port_open(config.proxy_port) else None
    record = bool(config.dataset) and not config.dry_run
    take_path: Path | None = None
    if record:
        episode = next_episode(config.recordings_dir, config.dataset)
        take_path = episode_path(config.recordings_dir, config.dataset, episode)
        rec = begin_take(take_path, episode=episode, dataset=sanitize_name(config.dataset), task=config.task, proxy_uri=proxy_uri)
        print(f"recording:  {take_path} (episode '{episode}')")
    else:
        rec = rr.RecordingStream(APP_ID, recording_id=f"deploy-{time.strftime('%H%M%S')}")
        rec.connect_grpc(url=proxy_uri or f"rerun+http://localhost:{config.proxy_port}/proxy")
    motor_names = [calib.motor_name for calib in follower.calibration]
    rec.log(f"{follower.name}/goal", rr.SeriesLines(names=[f"{name} goal" for name in motor_names]), static=True)
    rec.log(f"{follower.name}/position", rr.SeriesLines(names=motor_names), static=True)
    if config.debug:
        rec.log(f"{follower.name}/chunk_preview", rr.SeriesLines(names=[f"{name} chunk" for name in motor_names]), static=True)

    period = 1.0 / config.fps
    started = time.monotonic()
    gripper_latch = GripperLatch()  # persists across chunks: the whole point of hysteresis
    try:
        if not config.dry_run:
            follower.bus.set_torque(False)
            follower.bus.configure_follower_control()
            follower.bus.set_torque(True)
            print("torque ON -- keep a hand near the arm; Ctrl-C stops and releases it")
        print(f"task: {config.task!r} (first inference may take a while if the server is warming up)")

        first_chunk = True
        while config.seconds <= 0 or time.monotonic() - started < config.seconds:
            state = read_calibrated(follower)
            jpegs = observe_cameras(cameras, config.jpeg_quality, rec)
            chunk = query_actions(config.server, config.task, state, jpegs, config.action_mode)
            if config.debug:
                debug_chunk(follower, motor_names, state, chunk, rec)  # the model's raw prediction, before any gripper reshaping

            if first_chunk and not config.dry_run:
                # Glide from wherever the arm is to the first predicted pose.
                start_pose = np.asarray(read_calibrated(follower))
                ramp_target = np.array(chunk[0], dtype=np.float64)
                if config.gripper_binary:
                    ramp_target[GRIPPER_INDEX] = gripper_latch.decide(chunk[0][GRIPPER_INDEX], config.gripper_threshold, config.gripper_hysteresis)
                ramp_steps = max(2, int(config.ramp_seconds * config.fps))
                for step in range(ramp_steps):
                    blend = (step + 1) / ramp_steps
                    drive_to(follower, start_pose + blend * (ramp_target - start_pose), rec)
                    time.sleep(config.ramp_seconds / ramp_steps)
            first_chunk = False

            next_tick = time.monotonic()
            for raw_action in chunk[: config.execute_steps]:
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                next_tick += period
                current = read_calibrated(follower)
                action = np.array(raw_action, dtype=np.float64)
                if config.gripper_binary:
                    # Sequential, in temporal order: this is what gives the latch memory.
                    action[GRIPPER_INDEX] = gripper_latch.decide(raw_action[GRIPPER_INDEX], config.gripper_threshold, config.gripper_hysteresis)
                # The per-tick cap: never move any joint further than --max-step-deg at once.
                target = np.clip(action, np.asarray(current) - config.max_step_deg, np.asarray(current) + config.max_step_deg)
                if config.dry_run:
                    log_prediction(follower, target, current, rec)
                else:
                    drive_to(follower, target, rec)
        print("time limit reached")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        if not config.dry_run:
            try:
                follower.bus.set_torque(False)
                print("torque released")
            except (RuntimeError, OSError) as error:
                print(f"FAILED to release torque ({error}) -- power-cycle the arm to free it")
        follower.bus.close()
        for _, capture in cameras.values():
            capture.release()
        if record and take_path is not None:
            finish_take(rec, dataset=sanitize_name(config.dataset), task=config.task, tag=config.tag, proxy_uri=proxy_uri)
            optimize_rrd(take_path)
            if _port_open(config.catalog_port):
                registration = register_rrd(f"rerun+http://localhost:{config.catalog_port}", sanitize_name(config.dataset), take_path)
                print(f"registered: dataset '{registration['dataset']}', segments {registration['segment_ids']}")
            else:
                print(f"saved:      {take_path} (no catalog on port {config.catalog_port}; the next `pixi run so100-server` start registers it)")


if __name__ == "__main__":
    main(tyro.cli(Config))
