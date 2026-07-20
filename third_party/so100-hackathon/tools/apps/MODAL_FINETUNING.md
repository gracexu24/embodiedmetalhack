# Fine-tuning on Modal (ACT + MolmoAct2)

A runbook for training LeRobot policies on your own [Modal](https://modal.com) GPU account
straight from a Hugging Face dataset — an alternative to `pixi run finetune` (which trains on
New Theory's GPUs from your local recording catalog). Two policies are wired up:

| Task | Script | Policy | GPU | Notes |
| --- | --- | --- | --- | --- |
| `pixi run finetune-modal-act` | `finetune_modal_act.py` | ACT | T4 | Small ResNet+transformer, trains from scratch |
| `pixi run finetune-modal-molmoact2` | `finetune_modal_molmoact2.py` | MolmoAct2 (action-expert-only) | H100 | Ai2 VLA, fine-tunes from a released checkpoint |

Both submit a job to Modal's cloud, build their own container image (LeRobot + torch + ffmpeg),
train, and optionally push the result to the HF Hub. Only the lightweight `modal` client runs
locally — no torch/lerobot needed in the local pixi env. Checkpoints land in the
`so100-lerobot-checkpoints` Modal Volume under `/checkpoints/<job_name>`.

## One-time setup

```bash
# 1. modal is in pyproject.toml's [tool.pixi.pypi-dependencies], so this installs the client:
pixi install

# 2. Link your Modal account (opens a browser):
pixi run modal setup

# 3. Store an HF *write* token as a Modal secret (the training container reads $HF_TOKEN
#    from it to download gated checkpoints and push results). Reuses the token already
#    cached locally from `huggingface-cli login` / newt login:
pixi run modal secret create huggingface-secret HF_TOKEN="$(cat ~/.cache/huggingface/token)"
```

## Running

```bash
# ACT, defaults (20k steps, push to JaidevShriram/act-test-run-data):
pixi run finetune-modal-act

# MolmoAct2, defaults (5k steps, action-expert-only, push to JaidevShriram/molmoact2-so100-action-expert):
pixi run finetune-modal-molmoact2

# Quick end-to-end smoke test — proves the pipeline runs and a checkpoint saves, ~a few minutes:
pixi run finetune-modal-molmoact2 -- --steps 20 --save-freq 20 --job-name molmoact2-so100-smoketest --no-push-to-hub

# Any local_entrypoint arg is overridable after `--`:
pixi run finetune-modal-act -- --dataset-repo-id <hf/dataset> --steps 40000 --no-push-to-hub
```

Watch a run at the `modal.com/apps/...` URL printed on launch, or `pixi run modal app list`.
Stream logs from an already-running app (e.g. after closing the launching terminal):
`pixi run modal app logs <app-id>`. Stopping needs `--yes` (no TTY):
`pixi run modal app stop <app-id> --yes`.

### Why `--detach` + periodic checkpoints (learned the hard way)

Both pixi tasks run `modal run --detach`. **This matters:** a plain `modal run` is foreground —
the remote job's lifetime is tied to the local client's network connection, so a DNS blip or a
closed laptop lid **kills the whole training run**. A ~2.5h ACT run was lost exactly this way (a
local `StreamTerminatedError: Connection lost`), and because `save_freq` was left at the default
(only checkpointing at the very end), *nothing* had been saved — the run rewound to zero. The two
fixes, now baked in:

- **`--detach`** — the job runs server-side and survives a local disconnect. Follow it with
  `modal app logs <app-id>`; re-attaching costs nothing.
- **Frequent `--save_freq`** (ACT defaults to 2000) with each checkpoint committed to the Volume
  as it lands, plus `--resume`. If a run does die, restart with the same `--job-name` and
  `--resume` to continue from the last committed step instead of starting over.

## The dataset this was built against

`JaidevShriram/Test_Run_Data` — a LeRobot v3 dataset exported by this repo's own pipeline:
- **1 episode, 1564 frames, 30 fps** (a single ~52s take — enough to smoke-test the plumbing,
  not to get a working policy).
- 6-dim `action` + `observation.state` (SO-101 joints: shoulder_pan/lift, elbow_flex,
  wrist_flex/roll, gripper).
- **One camera**, `observation.images.top` (720×1280, av1). This single-camera shape drove
  several of the MolmoAct2 config choices below.

## MolmoAct2 gotchas (learned the hard way)

MolmoAct2 is Ai2's open Action-Reasoning VLA
([docs](https://huggingface.co/docs/lerobot/en/molmoact2)), `--policy.type=molmoact2` in
LeRobot — **not** New Theory proprietary. Getting it to train took four attempts; each error
and its fix:

1. **`Cannot specify both --policy.path and --policy.type`** — these are mutually exclusive.
   `--policy.path` restores a LeRobot-saved checkpoint *including its policy type*, so passing
   `--policy.type` too is an error. → Removed `--policy.type`.

2. **`Feature mismatch … Missing: [cam0, cam1] … Extra: [observation.images.top]`** — the
   released `lerobot/MolmoAct2-SO100_101-LeRobot` checkpoint has its **config frozen to two
   cameras**, and `--policy.path` restores that config wholesale (only batch_size / steps /
   LoRA flags / action_mode are documented as overridable afterward — image keys are not). Our
   dataset has one camera. → Switched to the **generalist base checkpoint** `allenai/MolmoAct2`
   via **`--policy.checkpoint_path`**, which *builds the policy config fresh* from the dataset +
   CLI options, so `--policy.image_keys=["observation.images.top"]` actually takes effect.
   (This route needs `--policy.type=molmoact2` back — see the two-routes note below.)
   **Trade-off:** we lose the SO-100/101 checkpoint's baked-in joint-convention correction
   (`joint_signs`/`joint_offsets` for an old calibration convention) and its SO-100-specific
   pretraining. Fine for a smoke test; revisit with a 2-camera recording to use the specialist.

3. **`MolmoAct2 action gripper values are not under [-1, 1]. Please set normalize_gripper=True`**
   — our gripper channel is in raw units, and MolmoAct2's continuous action head validates the
   [-1, 1] range. Default is `normalize_gripper=false`. → Added `--policy.normalize_gripper=true`.

4. **Trains.** 578M learnable params (action-expert-only), effective batch size 16 on the A10G.

Two supporting choices in the command:
- `--policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'`
  — MolmoAct2 defaults to *quantile* normalization, which needs the dataset pre-processed by
  `augment_dataset_quantile_stats.py`. Mean/std sidesteps that extra step.
- `--policy.chunk_size=10 --policy.n_action_steps=10 --policy.num_flow_timesteps=8
  --policy.gradient_checkpointing=true` — the doc's recommended small-dataset fine-tune settings.

### `--policy.path` vs `--policy.checkpoint_path` (the key distinction)

- **`--policy.path`** = resume from a **LeRobot-saved** checkpoint (local dir or Hub). Restores
  the saved config, weights, processor, and norm stats together. **Cannot** pass `--policy.type`;
  limited config overrides.
- **`--policy.checkpoint_path`** = initialize from an **original MolmoAct2 HF** checkpoint
  (e.g. `allenai/MolmoAct2`). Builds a fresh policy config/processor from the dataset metadata +
  your `--policy.*` options. **Requires** `--policy.type=molmoact2`.

## GPU sizing (from the LeRobot MolmoAct2 docs)

| Mode | Peak GPU memory (bs=8→32) | Fits |
| --- | --- | --- |
| Inference (bs=1) | ~12 GiB | T4 |
| Fine-tune, action-expert only | ~16–21 GiB | A10G |
| Fine-tune, LoRA VLM | ~20–41 GiB | A100-40GB |
| Fine-tune, full model | ~48–60 GiB | H100 |

Those figures assume **gradient checkpointing on**, which is how the LeRobot docs measured them.
This repo now runs action-expert-only on an **H100 with checkpointing off** (see below): measured
**23.15 GiB** at batch 16 -- comfortable on 80 GiB, and just over what the A10G's 24 GiB could
have safely held, which is why the GPU and the checkpointing flag move together.

ACT is tiny (~52M params) and fits a T4 comfortably.

### CPU sizing matters as much as the GPU

Modal's default reservation is ~0.125 cores / 128MiB. The MolmoAct2 dataloader does pyav video
decode plus image transforms on every batch, so on the default the A10G sat at roughly 0%
utilization between brief 50-70% spikes -- GPU memory steady at ~13GiB (model loaded and fine),
just starved of input. `finetune_modal_molmoact2.py` therefore reserves cores and workers
explicitly, as four constants at the top of the file (`CPU_CORES`, `DATALOADER_WORKERS`,
`PREFETCH_FACTOR`, `MEMORY_MIB`).

**Measured, same dataset/steps/batch run side by side (2026-07-18, 500 steps, batch 16):**

| | A10G, ckpt on, 8 cores / 6 workers | H100, ckpt off, 16 cores / 12 workers |
| --- | --- | --- |
| step time | 3.62 s | **0.57 s (6.4x)** |
| `updt_s` (GPU compute) | 3.444 | 0.433 |
| `data_s` (dataloader wait) | 0.202 (~5.5% of step) | 0.230 (~35% of step) |
| `mem_gb` | 15.00 / 24 | 23.15 / 80 |
| default 5k-step run | ~5 h -- **exceeds the 4 h timeout** | ~47 min |

Notes on reading that table:

- The 6.4x is the *compound* of the GPU swap and checkpointing-off; the two weren't measured
  separately. It also settles cost: the H100 bills ~4x the A10G per hour but finishes 6.4x
  sooner -- cheaper per run, not a tradeoff.
- `data_s` barely moved while compute shrank 8x, so the dataloader wait that was noise on the
  A10G is now ~1/3 of every H100 step. **The next speedup lever is the dataloader, not a bigger
  GPU** -- and per the ceiling note below, likely pre-decoded frames rather than more cores.
- `mem_gb` 23.15 with checkpointing off confirms that shape would not fit the A10G's 24 GiB:
  the GPU and the checkpointing flag genuinely had to move together.

**Still unmeasured:** the pre-fix baseline (the original starvation was never timed before the
CPU reservation landed), and the 8-vs-16-core / 6-vs-12-worker deltas in isolation.

Those four constants are one unit -- change them together:

- **Workers stay *below* cores.** `pin_memory=True` puts a batch-copying thread on the main
  process next to the optimizer loop; workers == cores starves exactly that thread.
- **Memory tracks `workers x prefetch`.** That product is the in-flight batch count (~5GiB was
  observed at 4x4). `persistent_workers` defaults True, so those buffers live for the whole run,
  and `pin_memory` makes them page-locked -- bursting past a soft request on a loaded host means
  an OOM kill, not swapping.
- **Reservations are billed** at the higher of request or usage, for the job's whole wall-clock.

- **The GPU sets the deadline.** Cores are sized against step time, not in the abstract: a
  faster GPU eats batches faster and gives the dataloader less wall-clock to produce each one.
  The H100 move is why `CPU_CORES` went 8 → 16 in the same change.

Watch GPU utilization on the Modal app dashboard after a change -- the goal is pinned-high rather
than spiky. If it goes back to spiky on the H100, the dataloader has become the bottleneck again
and the next lever is more cores, not a bigger GPU. There is a ceiling to that: pyav decoding
720p video has a per-worker throughput limit, and past some point the real fix is pre-decoding
frames or exporting the dataset at a lower resolution rather than buying more cores.

If the step counter freezes with both CPU and GPU flat and no traceback, suspect a forked pyav
decoder wedging a dataloader worker rather than a slow GPU: lower `DATALOADER_WORKERS`, or try
`--dataset.video_backend=torchcodec`.

(`finetune_modal_act.py` still runs on the default reservation. ACT decodes the same video, so
it likely has the same starvation, but it's a T4 job that hasn't been profiled -- measure before
paying for cores there.)

## Deploying a trained MolmoAct2 (not done here)

`pyproject.toml` has a `serve-modal-molmoact2` task pointing at
`tools/apps/policy_server_modal_molmoact2.py` — a Modal serve endpoint for `deploy-policy` to
hit. That serving/deployment work is owned by a **separate session** and is out of scope for
this finetuning runbook; see there for its status.
