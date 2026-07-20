# SO-101 Rob the Builder

<p align="center">
  <img src="assets/block_house.png" alt="Three-block house: door, wall, and roof" width="420">
</p>

Natural-language house building for a [Seeed Studio SO-101](https://www.seeedstudio.com/) arm:
parse three layer colors → plan three pick-and-place skills → run them on a Modal-hosted policy
(MolmoAct2, ACT, …). Builds are **operator-paced** — the policy runs until you **Pause** (UI /
voice), then you reset the scene by hand and continue. Optional cam1 HSV verification can be
enabled in config; by default Pause is the done signal. No auto-home (policy starts from the
current pose).

This repo is the control layer. Drivers, calibration, and Modal train/serve scripts live in
vendored [so100-hackathon](https://github.com/jaidevshriram/so100-hackathon)
(`third_party/so100-hackathon/`; upstream also
[mission-robotics-ai/so100-hackathon](https://github.com/mission-robotics-ai/so100-hackathon)).

## Demo

<p align="center">
  <img src="assets/demos/workspace_setup.png" alt="SO-101 workspace with overhead and side cameras" width="720">
  <br>
  <em>SO-101 follower, overhead (cam0) + side (cam1), colored blocks on the build mat.</em>
</p>

Stacking demos (both clips):

- [demo_build.mp4](assets/demos/demo_build.mp4)
- [demo_policy.mp4](assets/demos/demo_policy.mp4)

<video src="assets/demos/demo_build.mp4" controls width="720"></video>
<video src="assets/demos/demo_policy.mp4" controls width="720"></video>

## Task

| Layer | Position | Colors |
| ----- | -------- | ------ |
| door  | bottom   | red, blue |
| wall  | middle   | yellow, green |
| roof  | top      | red, blue |

```text
Door blocks          Wall blocks            Roof blocks
Red door  Blue door | Yellow wall  Green wall | Red roof  Blue roof
```

Vary color order within each group so the policy learns identity, not one fixed coordinate.

**Example request → planner skills**

```text
Build a house with a red door, yellow walls, and a blue roof.
→ Pick up the red block and place it on the black rectangle.
→ Pick up the yellow block and stack it on the first red block.
→ Pick up the blue triangle block and stack it on the second yellow block.
```

Other accepted phrasings (deterministic, case-insensitive):  
`Make the door blue, the walls green, and the roof red.` ·  
`I want a blue roof with green walls and a red door.` ·  
`Create a blue-door, yellow-wall, red-roof house.`  
Accepts `wall`/`walls` and `red door` or `door red`. Missing, conflicting, or unsupported colors
are errors.

```text
request → parser → planner → state machine
  door ──Pause──► wall ──Pause──► roof → COMPLETED
                    └── FAILED → retry after hand reset

IDLE → CONNECTING → EXECUTING ⇄ Pause / next layer → COMPLETED
                         └→ FAILED → retry last step
```

On Pause, torque is released for a hand reset; the next layer re-enables torque.

## Repository

```text
embodiedmetalhack/
├── run.py / voice_control.py
├── simulate.py / disarm.py / release_torque.py
├── config.yaml  requirements.txt  pyproject.toml
├── assets/demos/          # workspace photo + demo videos
├── backend/               # FastAPI dashboard (:8000)
├── frontend/              # React/Vite UI (:5173)
├── third_party/so100-hackathon/   # Feetech driver, calibrations, Modal apps
├── src/house_builder/     # models, parser, planner, robot, policy,
│                          # state_machine, verifier, builder, rr_*, sync_checkpoint
└── tests/
```

## Setup

Sections 1–3: tests / offline. 4–7: real arm.

### 1. Prerequisites

- Python 3.11+
- SO-101 follower on serial (for real builds)
- Cameras: overhead `cam0`, side `cam1`
- Vendored driver: `cd third_party/so100-hackathon && pixi install` (once, for hardware)
- No local GPU — policies are Modal HTTP endpoints
- Voice mic: `brew install portaudio` or `sudo apt-get install portaudio19-dev`

### 2–3. Install

```bash
git clone https://github.com/gracexu24/embodiedmetalhack.git
cd embodiedmetalhack
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev,voice,web]"   # or: pip install -e .
pytest && ruff check . && mypy src/house_builder
```

| Extra | Packages |
| ----- | -------- |
| runtime | `numpy`, `opencv-python`, `PyYAML`, `rerun-sdk` |
| dev | `pytest`, `ruff`, `mypy` |
| voice | `SpeechRecognition`, `PyAudio` |
| web | `fastapi`, `uvicorn`, `websockets` |

### 4–5. Robot and policy (`config.yaml`)

```yaml
robot:
  port: null                    # null = auto-detect calibrated follower USB serial
  calibration_dir: third_party/so100-hackathon/calibrations
  home_pose: [0, 0, 0, 0, 0, 0]
  max_step_deg: 10.0
  dry_run: true                 # flip false only after a sane dry run

policy:
  server: https://jaidevtrumpet--molmoact2-jags-lora-act.modal.run
  fps: 30.0
  execute_steps: 24
  jpeg_quality: 85
  skill_duration_seconds: 10
  check_interval_seconds: 3.0   # chunk length so Pause/Stop are noticed
```

Validate joint order, gripper sense, stop, and home at low speed before policy rollouts.
Each policy call sends the instruction, `cam0`/`cam1` (as `top`/`side`), and joint state.
`policy.load()` is a reachability check, not a cold-start inference call.
Servers are defined under `third_party/so100-hackathon/tools/apps/`.

### 6. Cameras and features

| Camera | Role | Config |
| ------ | ---- | ------ |
| `cam0` | policy observation (overhead / top) | `cameras.cam0` |
| `cam1` | policy observation (side); optional HSV verify | `cameras.cam1` |

Defaults: 640×480 @ 30 FPS — set OS capture `index` values.

```yaml
features:
  camera_verification: false  # HSV verifier; build loop is operator-paced by default
```

Prepare builds from a typed sentence or the dashboard color pickers — both produce  
`Build a house with a <door> door, <wall> walls, and a <roof> roof.`

**Optional cam1 verification:** when enabled, segments red/yellow/blue/green in calibrated
height bands (no ArUco). Wall/roof also check the support color below. Color-only vision
confirms color in band, not physical shape identity. After cameras are fixed, set
`verification.height_regions` (`min_y`/`max_y`; Y grows downward — door largest Y, roof
smallest) and related thresholds in `config.yaml`.

### 7. Hardware Python path

`robot.py` imports `so100_hackathon` from the vendored tree. Real arm motion needs that pixi
env + this repo on `PYTHONPATH` (`simulate.py` and tests do not):

```bash
export REPO=$(pwd) PYTHONPATH=$REPO/src
cd third_party/so100-hackathon && pixi install   # once
pixi run python "$REPO/run.py" "Build a house with a red door, yellow walls, and a blue roof." \
  --config "$REPO/config.yaml"
```

## Run

```bash
python simulate.py                                                              # offline fakes
python run.py "Build a house with a red door, yellow walls, and a blue roof."   # needs pixi
python voice_control.py            # mic (needs network for Google recognizer)
python voice_control.py --text
```

**Dashboard**

```bash
uvicorn backend.main:app --reload --port 8000          # terminal 1
cd frontend && npm install && npm run dev              # terminal 2 → :5173
```

Buttons mirror voice. Live monitoring uses the embedded Rerun viewer.

| Command | Effect |
| ------- | ------ |
| `Build this` | Store the typed / color-picked request; arm idle |
| `start` | Connect / load policy; run door until Pause |
| `build wall` / `build roof` | Run that layer until Pause |
| `Pause` (UI) | Stop policy loop; release torque for hand reset |
| `retry last step` | After failure + hand reset, re-run that layer (`retry` ok) |
| `stop` | Abort and disconnect |

**Panic** (servos left on after a crash):

```bash
python release_torque.py
# from third_party/so100-hackathon: pixi run python "$REPO/disarm.py"
curl -X POST -H 'Content-Type: application/json' \
  -d '{"command":"stop"}' http://localhost:8000/api/build/command
```

## Dataset

[JaidevShriram/JAGS_v0_testing](https://huggingface.co/datasets/JaidevShriram/JAGS_v0_testing)
(Apache-2.0) — LeRobot `so100_follower`:

| Field | Value |
| ----- | ----- |
| Scale | ~253 episodes / 245,934 frames @ 30 FPS (full card) |
| Cameras | `observation.images.top`, `.side` (720×1280) |
| Action / state | 6-DoF: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper |
| Tasks | 10 language-conditioned pick-and-place skills |

Hub checkpoint: [JaidevShriram/molmoact2-jags-ae](https://huggingface.co/JaidevShriram/molmoact2-jags-ae).

**Fine-tune labels** (same strings the planner emits — keep door \| wall \| roof grouping, vary
color order; record cam0, cam1, joints, actions, gripper, instruction, success):

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

## Inference

Modal-hosted `/act` servers, driven from so100-hackathon (`pixi run deploy-policy`) or this
harness (`policy.server`).

| Policy | Example task | Server |
| ------ | ------------ | ------ |
| MolmoAct2 LoRA VLM | Pick up the blue block and place it on the second yellow rectangle | `https://jaidevtrumpet--molmoact2-jags-lora-act.modal.run` |
| ACT | Pick up the red block and place it on the second yellow block | `https://jaidevtrumpet--act-jags-pick-red-block-act.modal.run` |
| MolmoAct2 Action Expert | Pick up the yellow block and place it on the second red block | `https://jaidevtrumpet--molmoact2-jags-ae-act.modal.run` |
| MolmoAct2 SO-101 scale | — | `https://jaidevtrumpet--molmoact2-so101-scale-act.modal.run` |
| Pi0.5 / SmolVLA API | — | `https://jaidevtrumpet--lerobot-pi05-smolvla-training-trained-policy-api.modal.run` |
| ACT train predict | — | `https://jaidevtrumpet--so101-act-train-actserver-predict.modal.run` |

```bash
# From so100-hackathon after pixi install
pixi run deploy-policy -- \
  --task "Pick up the blue block and place it on the second yellow rectangle" \
  --server https://jaidevtrumpet--molmoact2-jags-lora-act.modal.run

pixi run deploy-policy -- \
  --task "Pick up the red block and place it on the second yellow block" \
  --server https://jaidevtrumpet--act-jags-pick-red-block-act.modal.run \
  --execute-steps 10

pixi run deploy-policy -- \
  --task "Pick up the yellow block and place it on the second red block" \
  --server https://jaidevtrumpet--molmoact2-jags-ae-act.modal.run
```

| Train job | Policy | Data slice | Steps |
| --------- | ------ | ---------- | ----- |
| `molmoact2-jags-ep0` | MolmoAct2 | JAGS, config `episodes: [0]` (1 of 31) | 5,000 |
| `act_JAGS_v0_testing` (`so101-act-train`) | ACT | all 31 eps / 21,946 frames | 20,000 |

## Training

Not in this harness — use so100-hackathon / LeRobot / AllenAI, deploy to Modal, set
`policy.server`. Prefer base **`allenai/MolmoAct2-SO100_101`** for SO-100/101.

**Modal LoRA (so100-hackathon)**

```bash
GPU_COUNT=2 PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
pixi run modal run --detach tools/apps/finetune_modal_molmoact2_lora.py \
  --dataset-repo-id JaidevShriram/JAGS_v0_testing
```

Logs: [ap-hMXYyB6VFhyMCqMgO1ufuA](https://modal.com/apps/jaidevtrumpet/main/ap-hMXYyB6VFhyMCqMgO1ufuA?activeTab=logs)

**Torchrun / multi-GPU MolmoAct2 (reference)**

```bash
export EXP_NAME="molmoact2-my-robot-lora"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes="${NNODES:-1}" --nproc-per-node=8 \
  --node_rank="${RANK:-0}" --master_addr="${ADDR:-127.0.0.1}" --master_port="${PORT:-29415}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  my_robot \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --max_duration=50000 \
  --device_batch_size=2 \
  --global_batch_size=64 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=200 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/finetune/${EXP_NAME}" \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=true \
  --ft_action_expert=true \
  --ft_embedding=lm_head \
  --lora_enable=true \
  --lora_rank=64 \
  --llm_learning_rate=5e-5 \
  --vit_learning_rate=5e-5 \
  --connector_learning_rate=5e-5 \
  --action_expert_learning_rate=5e-5
```

Start from `allenai/MolmoAct2-SO100_101` when your entrypoint supports it.

**Pi0.5 / SmolVLA (separate B300)**

```bash
modal run modal_app.py::train \
  --dataset-url "https://huggingface.co/datasets/JaidevShriram/JAGS_v0_testing"
```

API: `https://jaidevtrumpet--lerobot-pi05-smolvla-training-trained-policy-api.modal.run` ·  
Logs: [ap-ldE0fOOUstQbPuHDS2iDOY](https://modal.com/apps/jaidevtrumpet/main/ap-ldE0fOOUstQbPuHDS2iDOY?activeTab=logs)

**ACT**

Predict: `https://jaidevtrumpet--so101-act-train-actserver-predict.modal.run` ·  
Logs: [ap-iSSFTN4p717iOvZwSjFnwH](https://modal.com/apps/jaidevtrumpet/main/ap-iSSFTN4p717iOvZwSjFnwH?activeTab=logs) ·  
Companion: [sheanrahman192/hackathonjustACT](https://github.com/sheanrahman192/hackathonjustACT)

## Links

| Resource | URL |
| -------- | --- |
| This harness | https://github.com/gracexu24/embodiedmetalhack |
| so100-hackathon (primary) | https://github.com/jaidevshriram/so100-hackathon |
| Upstream so100-hackathon | https://github.com/mission-robotics-ai/so100-hackathon |
| ACT fork | https://github.com/sheanrahman192/hackathonjustACT |
| Dataset | https://huggingface.co/datasets/JaidevShriram/JAGS_v0_testing |
| MolmoAct2 AE checkpoint | https://huggingface.co/JaidevShriram/molmoact2-jags-ae |

Serve URLs are listed under [Inference](#inference).

## Credits

Vendored SO-100 stack under **MIT / Apache-2.0** from so100-hackathon
([Rerun.io](https://rerun.io) / mission-robotics / Jaidev Shriram) — see
`third_party/so100-hackathon/LICENSE-*`. Source only is vendored; regenerate `.pixi/`,
`recordings/`, `datasets/` with `pixi install`. Dataset and Modal jobs: Jaidev Shriram &
collaborators. This repo is the house-builder layer on top.
