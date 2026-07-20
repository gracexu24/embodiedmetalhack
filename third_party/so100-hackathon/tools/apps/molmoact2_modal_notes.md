# MolmoAct2 on Modal — testing notes

Working notes for testing a MolmoAct2 policy on the SO-101 via a Modal-hosted inference
server. Not part of the canonical course (`pixi run learn`) — a scratch reference for
picking this back up later.

## How commands reach the arm (3 layers)

1. **Raw primitive** — `tools/apps/replay_episode.py:102` `drive_to`: clamps a target pose
   to the calibrated range and calls `follower.bus.sync_write_goal(goals_raw)`. One
   absolute joint-pose write per tick. `read_calibrated` (line 97) reads current state back.
2. **Replay a recorded trajectory** (no policy):
   `pixi run replay-episode -- --dataset my_task --episode episode_01`
3. **Live policy loop** — `tools/apps/deploy_policy.py`: observe (joint state + camera
   JPEGs) → POST to an inference server → `drive_to` the returned action chunk → repeat.
   Reuses `replay_episode.py`'s `open_follower`/`drive_to`/`read_calibrated`. Safety: every
   step clamped to `--max-step-deg` per tick; `--dry-run` streams predictions to the
   viewer without moving anything.

`/act` contract `deploy_policy.py` speaks: POST
`{"instruction", "state", "images": {name: b64 jpeg}}` → `{"actions": [[...], ...]}`.

## What's built

- `tools/apps/policy_server_molmoact2.py` — reference `/act` server, run directly on a
  GPU box (`pip install torch transformers`, not a pixi env). Default checkpoint
  `allenai/MolmoAct2-SO100_101`.
- `tools/apps/policy_server_modal_molmoact2.py` — same model-loading/`predict_action`
  logic, ported to a Modal `A10G` so no GPU box is needed. Two entrypoints:
  - `modal run tools/apps/policy_server_modal_molmoact2.py --task "..."` — local smoke
    test, placeholder image + zeroed state, just confirms the checkpoint loads and shapes
    line up.
  - `modal deploy tools/apps/policy_server_modal_molmoact2.py` — persistent `/act` HTTP
    endpoint, drop-in `--server` for `deploy_policy.py`.
- `pixi run serve-modal-molmoact2` — `modal serve` (temporary URL, for iterating).
- `tools/apps/finetune_modal_molmoact2.py` — already existed: fine-tunes the
  action-expert-only on a Modal A10G, starting from the **generalist** `allenai/MolmoAct2`
  checkpoint (via `--policy.checkpoint_path`), *not* from `lerobot/MolmoAct2-SO100_101-LeRobot`
  — that one's config is frozen to two cameras and rejects this repo's single-camera datasets.
  See `MODAL_FINETUNING.md` for why. **This matters for the joint convention below:** a
  fine-tune from the generalist base does *not* inherit the SO-100/101 checkpoint's baked-in
  correction, so `NEEDS_JOINT_FIX` must stay `True` when serving it.

## Running the actual test ("move the green block near the yellow block")

```bash
pixi run modal setup                                      # once, browser auth
modal deploy tools/apps/policy_server_modal_molmoact2.py   # prints a URL ending in /act

pixi run deploy-policy -- --task "move the green block near the yellow block" \
    --server https://<printed-url> --dry-run               # ALWAYS dry-run first
# if predictions look sane (small, smooth per-joint deltas, not wild jumps):
pixi run deploy-policy -- --task "move the green block near the yellow block" \
    --server https://<printed-url>
```

## Open risk / caveat

Default checkpoint is the public `allenai/MolmoAct2-SO100_101`. `deploy_policy.py`'s own
docstring warns public checkpoints used the old joint convention and can command wrong
poses on a v3-calibrated arm — confirmed by LeRobot's own MolmoAct2 docs: the
joint-convention fix (`joint_signs=[1,-1,1,1,1,1]`, `joint_offsets=[0,90,90,0,0,0]`) lives
in the **LeRobot processor pipeline** wrapping `lerobot/MolmoAct2-SO100_101-LeRobot`, not
in the raw HF `transformers`/`trust_remote_code` path this server (and the original
`policy_server_molmoact2.py`) uses. `--dry-run` is not optional on the first try with this
default checkpoint.

Safer path if dry-run looks off: fine-tune on your own recordings
(`pixi run finetune-modal-molmoact2`) and point `CHECKPOINT` in
`policy_server_modal_molmoact2.py` at that result instead, then `modal deploy` again.

## Possible next steps (not done yet)

- Test with real camera frames (`--image-path` on the local smoke test, or just go
  straight to `deploy_policy.py --dry-run` with the actual arm + cameras).
- If joint-convention risk bites: port the server to LeRobot's own policy-loading API
  (`PreTrainedPolicy.from_pretrained("lerobot/MolmoAct2-SO100_101-LeRobot")`) instead of
  the raw `transformers` path, to get the processor-level joint fix. LeRobot's own
  `lerobot-rollout` CLI does this end-to-end already but expects to run on a machine
  physically attached to the robot with a local GPU — not directly usable with Modal
  without the same client/server split this repo already uses.
