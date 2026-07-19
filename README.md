# SO-101 Three-Block House Builder

A small Python harness that turns one natural-language request into short MolmoAct2 skills for a
Seeed Studio SO-101. It parses three colors, plans three layers, runs one combined pick-and-place
instruction per layer, verifies each placement, and stops at the first failure.

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

Any policy, verification, or runtime failure moves to `FAILED`. Autonomous recovery remains
intentionally out of scope.

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
├── src/house_builder/
│   ├── __init__.py
│   ├── models.py
│   ├── parser.py
│   ├── planner.py
│   ├── robot.py
│   ├── policy.py
│   ├── state_machine.py
│   ├── verifier.py
│   └── builder.py
└── tests/
    ├── test_parser.py
    ├── test_planner.py
    ├── test_builder.py
    ├── test_human_builder.py
    ├── test_verifier.py
    └── test_voice_control.py
```

## Setup

Complete setup runs top to bottom. Sections 1–3 are enough to run tests and the parser/planner
without hardware; sections 4–8 are required before moving the real arm.

### 1. Prerequisites

- Python 3.11 or newer.
- A Seeed Studio SO-101 follower arm on a serial port (for real builds).
- Three cameras: overhead (`cam0`), side (`cam1`), and a model-house camera (`camera3`).
- A CUDA-capable NVIDIA GPU if you run the real MolmoAct2 policy (`policy.device: cuda`).
- The LeRobot release used by your SO-101, installed separately (see section 5).
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
python -m pip install -e ".[dev,voice]"

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

Edit the `robot` section of `config.yaml` with your hardware values (the defaults are
placeholders, not working values):

```yaml
robot:
  type: so101_follower
  port: /dev/REPLACE_WITH_SO101_PORT   # e.g. /dev/ttyACM0 (Linux) or /dev/tty.usbmodem* (macOS)
  id: replace_with_robot_id
  home_pose: [0, 0, 0, 0, 0, 0]        # calibrate to a safe home pose
```

Validate joint ordering, units, gripper behavior, stop behavior, and home motion at low speed
before any policy rollout.

### 5. Install and adapt LeRobot + MolmoAct2

Install the LeRobot release your SO-101 setup uses (follow its own instructions), then adapt the
two integration boundaries. All version-specific imports live in exactly two files:

- `src/house_builder/robot.py` — SO-101 connect, home, stop, observation, action.
- `src/house_builder/policy.py` — MolmoAct2 load and rollout.

Both mark the spots to edit with:

```python
# ADAPT TO INSTALLED LEROBOT VERSION
```

Until adapted, those sections raise clear errors instead of guessing unstable APIs. Configure the
policy in `config.yaml`:

```yaml
policy:
  checkpoint: lerobot/MolmoAct2-SO100_101-LeRobot
  local_checkpoint: null        # set a path to use a local fine-tuned checkpoint
  device: cuda                  # CUDA is checked at load time
  skill_duration_seconds: 10
```

Each policy call receives one short instruction plus `cam0`, `cam1`, and current joint state.

### 6. Connect and configure cameras

| Camera    | Role                                                              | Config key       |
| --------- | ----------------------------------------------------------------- | ---------------- |
| `cam0`    | policy observation only                                           | `cameras.cam0`   |
| `cam1`    | color, horizontal alignment, layer height, support, stability     | `cameras.cam1`   |
| `camera3` | human-built model-house input for `human_builder.py`              | `cameras.camera3`|

Set the correct OS capture `index` for each camera in `config.yaml`. All default to 640×480 at
30 FPS.

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

### Quick start

```bash
# 1. Install
python -m pip install -e ".[dev,voice]"

# 2. (no hardware) confirm the toolchain
pytest && ruff check . && mypy src/house_builder

# 3. One-shot build from a typed request
python run.py "Build a house with a red door, yellow walls, and a blue roof."

# 4. One-shot build from a model-house image
python run.py "$(python human_builder.py --image model_house.jpg)"

# 5. Voice-driven staged build
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
- **`stop`** — stops and disconnects safely.

Recognition uses the Google recognizer provided by `SpeechRecognition`, so microphone mode
requires network access. For setup and debugging without speech recognition, type the same
commands:

```bash
python voice_control.py --text
```

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
  "Build a house with a red door, yellow walls, and a blue roof."
```

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

`recover_failed_placement()` in `builder.py` is the single future extension point. It currently
returns `False`: a human must reset any failed structure. Add recovery only after removal,
restacking, and collapse detection are independently validated.

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
