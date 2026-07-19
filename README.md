# SO-101 Three-Block House Builder

A small Python harness that turns one natural-language request into short MolmoAct2 skills for a
Seeed Studio SO-101. It parses three colors, plans three layers, runs one pick and one placement
instruction per layer, verifies each placement, and stops at the first failure.

Mock mode works without a robot, cameras, CUDA, LeRobot, or MolmoAct2.

## House and physical layout

Every house is exactly `door` (bottom), `wall` (middle), and `roof` (top). Each is red, yellow,
or blue. The nine available blocks are laid out in layer groups:

```text
Door blocks                       Wall blocks                       Roof blocks
Red door  Yellow door  Blue door | Red wall  Yellow wall  Blue wall | Red roof  Yellow roof  Blue roof
```

Color order may vary within each group so the policy learns block identity instead of one fixed
coordinate.

## Example requests

```text
Build a house with a red door, yellow walls, and a blue roof.
Make the door blue, the walls red, and the roof yellow.
I want a yellow roof with blue walls and a red door.
Create a blue-door, yellow-wall, red-roof house.
```

Parsing is deterministic, case-insensitive, punctuation-tolerant, accepts `wall`/`walls`, and
allows either `red door` or `door red`. Missing, conflicting, and unsupported colors are errors.

## Architecture

```text
"Red door, yellow walls, blue roof"
                 ↓
          Sentence parser
                 ↓
     [red door, yellow wall, blue roof]
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
          PICKING → PLACING → VERIFYING
             ↑                    │
             └────────────────────┘
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
├── run.py
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
    └── test_verifier.py
```

## Installation

Python 3.11 or newer is required.

```bash
cd so101-house-builder
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`opencv-python` provides the HSV color segmentation used by verification.

## Mock mode

```bash
python run.py --mock \
  "Build a house with a blue door, red walls, and a yellow roof."
```

The mock robot records lifecycle calls, the mock policy records all six instructions, and the
mock verifier can succeed, fail one layer once, or permanently fail one layer.

## Real SO-101 configuration

Set `mock_mode: false` in `config.yaml`, then replace the robot port, ID, and home pose. Validate
joint ordering, units, gripper behavior, stop behavior, and home motion at low speed. The values
in the repository are placeholders, not hardware defaults.

## MolmoAct2 adapter notes

The default checkpoint is `lerobot/MolmoAct2-SO100_101-LeRobot`.
`policy.local_checkpoint` can select a local fine-tuned checkpoint. `policy.py` checks CUDA and
contains all policy-side LeRobot imports. `robot.py` contains all SO-101-side LeRobot imports.

Both files are now wired against `lerobot==0.6.0`'s real API (`SO101Follower`/`SO101FollowerConfig`
for the arm, `MolmoAct2Policy.from_pretrained` + `make_molmoact2_pre_post_processors` +
`lerobot.common.control_utils.predict_action` for the policy) instead of the placeholder
`# ADAPT TO INSTALLED LEROBOT VERSION` stubs. Two things in `policy.py` remain assumptions rather
than verified facts, because verifying them needs a live GPU rollout against the actual trained
checkpoint: the camera observation key names (`image_keys`) and the action output name/order
(`action_names`). Both default to reasonable values (see the comments in `config.yaml`) and can be
overridden there if your checkpoint differs. `robot.py`'s `stop()` now really cuts motor torque via
`bus.disable_torque()` rather than being a no-op.

Every policy call receives one short instruction, `cam0`, `cam1`, and current joint state.

## Rerun visualization

`run.py` now spawns a [Rerun](https://rerun.io) viewer by default (`--no-viewer` for headless runs)
showing, in real time: build-state transitions, each pick/place instruction and its outcome, arm
joint positions, both camera feeds, and every verification check with pass/fail reasons. This works
in `--mock` mode too — no hardware needed to see it. See `src/house_builder/rr_time.py` (the shared
`harness_step` timeline every module logs against) and `src/house_builder/rr_blueprint.py` (the
default viewer layout).

## Syncing a trained checkpoint from Modal

`src/house_builder/sync_checkpoint.py` downloads a trained checkpoint from a Modal Volume to local
disk and prints the path to put in `config.yaml`'s `policy.local_checkpoint`. Fill in
`MODAL_VOLUME_NAME` and `MODAL_CHECKPOINT_PATH` at the top of the file once your Modal training
job's volume name and output path are settled, then:

```bash
python -m pip install ".[checkpoint-sync]"
python src/house_builder/sync_checkpoint.py
```

Requires Modal auth already set up locally (`modal token set`). Not independently testable without
real Modal credentials and a real volume -- it's written against the installed `modal` SDK's actual
method signatures (`Volume.from_name`, `.iterdir`, `.read_file_into_fileobj`), not guessed, but
hasn't been run against a live volume. Double check the printed source -> destination paths the
first time you run it.

## Cameras

- `cam0`: policy observation only
- `cam1`: color, horizontal alignment, layer height, support color, and stability

Both default to 640×480 at 30 FPS. All target coordinates, height bands, and thresholds in
`config.yaml` are examples requiring calibration after rigid camera mounting.

## Cam1 color-stack verification

Verification uses no ArUco markers. It segments red, yellow, and blue pixels from `cam1` and
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

The resulting six skills are:

```text
Pick up the red door block.
Place the held red door block in the house foundation position.
Pick up the yellow wall block.
Stack the held yellow wall block directly on top of the door block.
Pick up the blue roof block.
Stack the held blue roof block directly on top of the wall block.
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

Collect short episodes with one language label each:

```text
Pick up the red door block.
Pick up the yellow door block.
Pick up the blue door block.

Pick up the red wall block.
Pick up the yellow wall block.
Pick up the blue wall block.

Pick up the red roof block.
Pick up the yellow roof block.
Pick up the blue roof block.

Place the held door block in the house foundation position.
Stack the held wall block directly on top of the door block.
Stack the held roof block directly on top of the wall block.
```

Keep `Door blocks | Wall blocks | Roof blocks` grouping, but vary color order and valid positions
within each group. Collect fixed-foundation door placement separately from wall-on-door and
roof-on-wall stacking. Record synchronized `cam0`, `cam1`, joint state, actions, gripper state,
instruction, and success metadata. Combine all skills into one multitask MolmoAct2 fine-tuning
dataset with separate language labels.
