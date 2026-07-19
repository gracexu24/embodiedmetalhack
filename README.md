# SO-101 Three-Block House Builder

<p align="center">
  <img src="assets/block_house.png" alt="Three-block house: red door, yellow wall, blue roof" width="420">
</p>

A small Python harness that turns one natural-language request into short MolmoAct2 skills for a
Seeed Studio SO-101. It parses three colors, plans three layers, and for each layer runs the
combined pick-and-place instruction in short slices, re-verifying after each one and retrying
until the placement is confirmed or the time budget runs out. A layer that never verifies stops
the whole build; a human resets the structure before trying again.

## House and physical layout

Every house is exactly `door` (bottom), `wall` (middle), and `roof` (top). Colors are restricted
by physical layer:

- door: red or blue
- wall: yellow or green
- roof: red or blue

The six available blocks are laid out in layer groups:

```text
Door blocks        Wall blocks          Roof blocks
Red door  Blue door | Yellow wall  Green wall | Red roof  Blue roof
```

Color order may vary within each group so the policy learns block identity instead of one fixed
coordinate.

## Example requests

```text
Build a house with a red door, yellow walls, and a blue roof.
Make the door blue, the walls green, and the roof red.
I want a blue roof with green walls and a red door.
Create a blue-door, yellow-wall, red-roof house.
```

Parsing is deterministic, case-insensitive, punctuation-tolerant, accepts `wall`/`walls`, and
allows either `red door` or `door red`. Missing, conflicting, and unsupported colors are errors.

## Architecture

```text
"Red door, green walls, blue roof"
                 ↓
          Sentence parser
                 ↓
     [red door, green wall, blue roof]
                 ↓
         Three-step planner
                 ↓
      Small build state machine
                 ↓
       Pick and place door
                 ↓
             Verify
                 ↓
       Pick and stack wall
                 ↓
             Verify
                 ↓
       Pick and stack roof
                 ↓
             Verify
                 ↓
          Completed house
```

The small state machine validates the linear lifecycle:

```text
IDLE → CONNECTING → HOMING
                    ↓
             EXECUTING → VERIFYING
                 ↑           │
                 └───────────┘
                    ↓
               COMPLETED
```

The `EXECUTING ⇄ VERIFYING` loop runs more than once per layer when needed: each pass runs the
policy for `check_interval_seconds`, then re-verifies; if it isn't confirmed yet and time remains
within `skill_duration_seconds`, it loops back to `EXECUTING` instead of giving up after one look.
Only running out of that time budget (or a policy/runtime exception) moves to `FAILED`.
Autonomous recovery *after* a `FAILED` build remains intentionally out of scope — a human resets
the structure and the whole build restarts from the door layer.

## Repository structure

```text
so101-house-builder/
├── README.md
├── requirements.txt
├── pyproject.toml
├── config.yaml
├── human_builder.py
├── run.py
├── voice_control.py
├── simulate.py
├── backend/                 # FastAPI dashboard API
├── frontend/                # React/Vite UI
├── src/house_builder/
│   ├── __init__.py
│   ├── models.py
│   ├── parser.py
│   ├── planner.py
│   ├── robot.py
│   ├── policy.py
│   ├── state_machine.py
│   ├── verifier.py
│   ├── builder.py
│   ├── rr_time.py
│   ├── rr_blueprint.py
│   └── sync_checkpoint.py
└── tests/
```

## Setup

Complete setup runs top to bottom. Sections 1–3 are enough to run tests and the parser/planner
without hardware; sections 4–8 are required before moving the real arm.

### 1. Prerequisites

- Python 3.11 or newer.
- A Seeed Studio SO-101 follower arm on a serial port (for real builds).
- Three cameras: overhead (`cam0`), side (`cam1`), and a model-house camera (`camera3`).
- A clone of [so100-hackathon](https://github.com/mission-robotics-ai/so100-hackathon) with its
  pixi environment installed (`pixi install`) -- `src/house_builder/robot.py` drives the arm
  through that repo's own Feetech driver and calibration files, so it must be importable at run
  time (see "Running", below). No local GPU or LeRobot install is needed for *this* repo: the
  MolmoAct2 policy itself runs on a Modal-hosted endpoint (see `policy_server_modal_molmoact2.py`
  in so100-hackathon), and `policy.py` just calls it over HTTP.
- On macOS, PortAudio for microphone input:

```bash
brew install portaudio
```

On Debian/Ubuntu:

```bash
sudo apt-get install portaudio19-dev
```

### 2. Clone and create a virtual environment

```bash
cd so101-house-builder
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install the package

```bash
# Everything (recommended for development): runtime + dev tools + voice
python -m pip install -e ".[dev,voice,web]"

# Runtime only
python -m pip install -e .
```

Dependency groups:

- **runtime** — `numpy`, `opencv-python` (HSV color segmentation), `PyYAML`.
- **dev** — `pytest`, `ruff`, `mypy`.
- **voice** — `SpeechRecognition`, `PyAudio` (microphone input for `voice_control.py`).

Verify the install with no hardware attached:

```bash
pytest
ruff check .
mypy src/house_builder
```

### 4. Configure the robot

`src/house_builder/robot.py`'s `SO101Robot` drives the arm through so100-hackathon's own Feetech
bus and calibration files (the same code path exercised by that repo's `replay_episode.py` /
`deploy_policy.py`), not through a separate LeRobot robot class. Edit the `robot` section of
`config.yaml`:

```yaml
robot:
  port: null                    # null -> auto-detect the calibrated follower by USB serial id;
                                 # set explicitly only if multiple calibrated followers are plugged
                                 # in at once
  calibration_dir: /absolute/path/to/so100-hackathon/calibrations
  home_pose: [0, 0, 0, 0, 0, 0] # calibrated middle pose, gripper closed -- already a safe default
                                 # under this calibration convention, not a placeholder
  max_step_deg: 10.0             # per-joint, per-tick motion cap -- the safety net against a bad
                                 # prediction
  dry_run: true                  # torque never enables, no goal is ever written -- only logged.
                                 # Flip to false only after watching a dry run look sane.
```

Validate joint ordering, units, gripper behavior, stop behavior, and home motion at low speed
before any policy rollout -- start with `dry_run: true`.

### 5. Configure the policy (Modal-hosted MolmoAct2)

`src/house_builder/policy.py`'s `MolmoAct2Policy` doesn't load a model locally -- it calls a
Modal-hosted `/act` HTTP endpoint (see so100-hackathon's `tools/apps/policy_server_modal_molmoact2.py`,
or a fine-tuned variant deployed the same way). Deploy that server first (`modal deploy ...` from
so100-hackathon), then point this repo at it:

```yaml
policy:
  server: https://<your-modal-endpoint>          # ends in .modal.run
  fps: 30.0                    # control rate actions are sent at
  execute_steps: 24            # actions consumed per chunk before re-observing
  jpeg_quality: 85
  skill_duration_seconds: 10   # total time budget per layer
  check_interval_seconds: 3.0  # how often to re-verify during that budget (see Architecture)
```

Each policy call receives one short instruction plus `cam0`/`cam1` frames (sent as `top`/`side` in
the `/act` payload) and the current joint state. `policy.load()` is a cheap reachability check
(GET, not POST), not a real inference call -- it won't trigger a slow cold start, it just fails
fast on a typo'd URL instead of failing mid-skill.

### 6. Connect and configure cameras

| Camera    | Role                                                              | Config key       |
| --------- | ----------------------------------------------------------------- | ---------------- |
| `cam0`    | policy observation only                                           | `cameras.cam0`   |
| `cam1`    | color, horizontal alignment, layer height, support, stability     | `cameras.cam1`   |
| `camera3` | human-built model-house input for `human_builder.py`              | `cameras.camera3`|

Set the correct OS capture `index` for each camera in `config.yaml`. All default to 640×480 at
30 FPS.

### Optional camera features

The two camera-dependent checks can be disabled independently:

```yaml
features:
  camera_verification: false  # skip post-placement cam1 checks
  human_builder: false        # disable camera3 model-house scan and UI panel
```

With `human_builder: false`, prepare builds from the dashboard's sentence or color inputs.
Both modes generate the canonical phrase
`Build a house with a <door> door, <wall> walls, and a <roof> roof.` and feed it through the
same parser and staged pipeline. With `camera_verification: false`, each completed policy
instruction is accepted without visual confirmation. `cam0` and `cam1` are still required as
MolmoAct2 policy observations; this switch removes verification, not policy camera input.

### 7. Calibrate vision (required before real hardware)

Every pixel value in `config.yaml` is an example and must be calibrated after the cameras and jig
are rigidly mounted. Because image Y grows downward, the door band has the largest Y values and
the roof the smallest; keep the three bands ordered and non-overlapping.

1. Build a correct reference stack (door, wall, roof).
2. Capture a still from each camera (for `camera3`: `python human_builder.py --image saved.jpg`,
   or any screenshot).
3. Read the pixel Y-range each layer occupies and set `min_y`/`max_y`.

Calibrate these blocks:

- `verification.height_regions` — layer bands seen by `cam1`.
- `verification.target_x`, `max_center_error_px`, `max_stack_alignment_error_px`,
  `stability_frames`, `max_stability_movement_px`, `min_color_occupancy`, `min_color_margin`.
- `human_builder.height_regions` and its `min_color_occupancy` / `min_color_margin` for `camera3`.

### 8. Microphone (voice control)

Install the `voice` extra (section 3) and PortAudio (section 1). Microphone mode uses the Google
recognizer from `SpeechRecognition`, so it requires network access. To find a device index or
debug without a mic, use `--text` (section "Voice-controlled staged build").

### Running: `so100_hackathon` must be importable

`robot.py` imports `so100_hackathon.feetech`/`so100_hackathon.calibration` directly, so any
command below that touches real hardware (`run.py`, `voice_control.py` -- not `simulate.py`, which
uses fakes and needs neither) has to run where that package resolves. It's already
editable-installed inside so100-hackathon's own pixi environment, so the simplest way is to run
from there with this repo's `src/` also on `PYTHONPATH`:

```bash
export PYTHONPATH=/absolute/path/to/embodiedmetalhack/src
cd /absolute/path/to/so100-hackathon
pixi run python /absolute/path/to/embodiedmetalhack/run.py "Build a house with ..." \
    --config /absolute/path/to/embodiedmetalhack/config.yaml
```

### Quick start

```bash
# 1. Install
python -m pip install -e ".[dev,voice]"

# 2. (no hardware) confirm the toolchain
pytest && ruff check . && mypy src/house_builder

# 3. (no hardware, no Modal) simulate the full build loop
python simulate.py
python simulate.py --retry-layer wall --retry-failures 2   # see a layer retry then succeed
python simulate.py --fail-layer roof                        # see a layer time out and fail

# 4. One-shot build from a typed request (needs the PYTHONPATH + pixi env above)
python run.py "Build a house with a red door, yellow walls, and a blue roof."

# 5. One-shot build from a model-house image
python run.py "$(python human_builder.py --image model_house.jpg)"

# 6. Voice-driven staged build
python voice_control.py            # or: python voice_control.py --text
```

## Human-built model input

`human_builder.py` reads a human-built model house from `camera3`, detects the dominant allowed
color in each calibrated layer band, and emits a sentence accepted by the harness:

```bash
python human_builder.py
# Build a house with a red door, green walls, and a blue roof.
```

It can also process a saved side-camera image:

```bash
python human_builder.py --image model_house.jpg
```

The voice controller invokes this scan when it hears `Build this`. You can still feed a saved
image directly into the one-shot harness:

```bash
python run.py "$(python human_builder.py --image model_house.jpg)"
```

The detector rejects missing or ambiguous colors rather than guessing.

## Voice-controlled staged build

Start the microphone controller:

```bash
python voice_control.py
```

It recognizes these commands:

- **`Build this`** — captures `camera3`, parses the model house, and stores the resulting build
  request. The robot does not move yet.
- **`start`** — connects and homes the robot, then executes and verifies the door layer.
- **`build wall`** — executes only after the door passed verification.
- **`build roof`** — executes only after the wall passed verification, then safely closes the
  completed session.
- **`retry last step`** — after a layer fails, remove the failed placement by hand, then say this
  to re-run only that layer. Aliases: `retry the last step`, `retry`.
- **`stop`** — stops and disconnects safely.

If a layer fails, later layer commands are rejected until you retry or stop. Recognition uses the
Google recognizer provided by `SpeechRecognition`, so microphone mode requires network access. For
setup and debugging without speech recognition, type the same commands:

```bash
python voice_control.py --text
```

## Web dashboard

The FastAPI + React dashboard mirrors the staged voice commands with clickable buttons.

```bash
python -m pip install -e ".[dev,voice,web]"
cd frontend && npm install && npm run dev   # http://localhost:5173
# in another terminal:
uvicorn backend.main:app --reload --port 8000
```

UI buttons: **Build This**, **Start**, **Build Wall**, **Build Roof**, **Retry Last Step**, **Stop**.
Reference scan uses `camera3` and `human_builder.detect_model_house`. Live cam0/cam1 monitoring
uses the embedded Rerun web viewer.

## Cam1 color-stack verification

Verification uses no ArUco markers. It segments red, yellow, blue, and green pixels from `cam1` and
searches only inside the calibrated vertical band for the current layer. For a wall or roof, it
also finds the previously verified color in the band directly below and checks that the new
centroid is above and horizontally aligned with that support.

Separate height bands allow same-color stacks to be checked, but color-only vision cannot prove
physical shape identity. It verifies that the requested color occupies the expected layer, not
that a same-colored block is definitely the door, wall, or roof shape.

## Placement verification

All four checks must pass:

- requested color visible in the current `cam1` height band
- centroid near the fixed horizontal construction target
- wall/roof centroid above and aligned with the previously verified support color
- centroid movement below the stability threshold over several `cam1` frames

No full 3D reconstruction is required. A failed check reports the expected block and reason,
stops the robot, and prevents later layers from running.

## Running

```bash
python run.py \
  "Build a house with a red door, yellow walls, and a blue roof." \
  --config config.yaml
```

(See "Running: `so100_hackathon` must be importable" above for the `PYTHONPATH` + pixi env this
actually needs.)

The resulting three MolmoAct2 skills are:

```text
Pick up the red block and place it on the black rectangle.
Pick up the yellow block and stack it on the first red block.
Pick up the blue triangle block and stack it on the second yellow block.
```

Ctrl+C and exceptions use the same `finally` cleanup to stop and disconnect the robot and close
cameras.

## Testing

```bash
pytest
ruff check .
mypy src/house_builder
```

Tests cover parser variants and errors, exact planning, valid and invalid state transitions,
successful building, each layer failure, prevention of later execution, instruction order, and
cleanup after exceptions.

## Safety

- This research harness is not a certified safety controller.
- Use a physical emergency stop, guarding, independent motion limits, and low initial speed.
- Keep people outside the workspace while torque is enabled.
- Test stop, home, disconnect, joint ordering, and action scaling before policy rollouts.
- Never stack above a failed verification or use uncalibrated example camera coordinates.

## Future autonomous recovery

Retrying *within* a layer is already automatic (`check_interval_seconds`, see Architecture) --
this section is about recovery *after* a layer has actually failed (timed out without verifying).
`recover_failed_placement()` in `builder.py` is the single future extension point for that case.
It currently returns `False`: a human must reset any failed structure, and the next build starts
over from the door layer -- there's no way to resume mid-build at the failed layer yet. Add
recovery only after removal, restacking, and collapse detection are independently validated.

## Fine-tuning data

Collect episodes using the same combined language labels sent during building:

```text
Pick up the red block and place it on the black rectangle.
Pick up the blue block and place it on the black rectangle.

Pick up the yellow block and stack it on the first red block.
Pick up the yellow block and stack it on the first blue block.
Pick up the green block and stack it on the first red block.
Pick up the green block and stack it on the first blue block.

Pick up the red triangle block and stack it on the second green block.
Pick up the red triangle block and stack it on the second yellow block.
Pick up the blue triangle block and stack it on the second green block.
Pick up the blue triangle block and stack it on the second yellow block.
```

Keep `Door blocks | Wall blocks | Roof blocks` grouping, but vary color order and valid positions
within each group. Record synchronized `cam0`, `cam1`, joint state, actions, gripper state,
instruction, and success metadata. Combine all ten skills into one multitask MolmoAct2
fine-tuning dataset with separate language labels.
