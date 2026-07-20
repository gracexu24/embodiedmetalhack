"""Fine-tune MolmoAct2 on Modal B200s in one of three tuning modes, from a LeRobot dataset.

A ``--mode``-parameterized generalization of ``finetune_modal_molmoact2_lora.py``: same
SO-100/101 specialist checkpoint (``allenai/MolmoAct2-SO100_101``), same two-camera /
B200 / multi-GPU infra, but the *tuning mode* is a flag so full-vs-LoRA-vs-action-expert
runs come from one script instead of three::

    # action-expert-only (lightest; VLM frozen, only the action expert trains)
    GPU_COUNT=2 pixi run modal run --detach tools/apps/finetune_modal_molmoact2_modes.py \
        --mode action_expert --job-name molmoact2-jags-ae \
        --policy-repo-id JaidevShriram/molmoact2-jags-ae

    # LoRA-VLM r64 + full action expert (the tuned default of the sibling script)
    GPU_COUNT=2 pixi run modal run --detach tools/apps/finetune_modal_molmoact2_modes.py \
        --mode lora --job-name molmoact2-jags-lora

    # full fine-tune (everything trainable; heaviest, most overfit-prone on small data)
    GPU_COUNT=2 pixi run modal run --detach tools/apps/finetune_modal_molmoact2_modes.py \
        --mode full --job-name molmoact2-jags-full \
        --policy-repo-id JaidevShriram/molmoact2-jags-full

The three modes and how they map to lerobot 0.6.0 policy flags (all share
``action_mode=continuous``; each uses distinct learning rates and gradient-checkpointing):

- **action_expert** -- ``train_action_expert_only=true``. Freezes the VLM; trains only the
  action expert. Cheapest (~16-21GiB per the docs), checkpointing off. LeRobot docs
  recommend this (or LoRA) for small datasets. LR: action-expert 5e-5.
- **lora** -- ``enable_lora_vlm=true`` + ``lora_rank=64``, action expert fully trainable
  (``enable_lora_action_expert=false``). Middle ground. All LR groups 5e-5 (Ai2's recipe).
  Checkpointing off (measured 138.9GiB on B200).
- **full** -- ``enable_lora_vlm=false``, nothing frozen. Heaviest: the whole VLA's optimizer
  step. Gradient checkpointing **on** (memory) -- expect ~1.35x slower steps. Per-group
  LRs from the docs: VLM 1e-5, ViT 5e-6, connector 5e-6, action-expert 5e-5. The docs put
  the LoRA-vs-full boundary at ~200 demos, so on a ~250-episode set full FT is the
  overfit-prone extreme -- run it to compare, not because it's the safe choice.

Everything else (checkpoint, both cameras, MEAN_STD norm, num_flow_timesteps=8,
normalize_gripper, HF cache on the Volume, per-2-min checkpoint commits, GPU_COUNT ->
accelerate) is copied from the LoRA sibling; see its docstring and MODAL_FINETUNING.md for
the measured reasoning. Needs the ``huggingface-secret``, ``wandb-secret`` (or
``--wandb-enable false``), and the ``so100-lerobot-checkpoints`` Volume. lerobot refuses an
existing ``output_dir`` -- use a fresh ``--job-name`` per run.
"""

from __future__ import annotations

import os

import modal

# Settable so a run is identifiable in `modal app list` on a shared workspace (many
# look-alike "train-molm..." apps otherwise). Set MODAL_APP_NAME to something distinctly
# yours so teammates don't `modal app stop` the wrong one.
app = modal.App(os.environ.get("MODAL_APP_NAME", "train-molmoact2-so101-modes"))

# Read at import time on the launching machine -- fixes the decorator's GPU request. Also
# passed to train() as an arg, since the remote re-import would see 1. Cap at 2 (team
# decision). action_expert is light enough for GPU_COUNT=1 to halve its cost.
GPU_COUNT = int(os.environ.get("GPU_COUNT", "1"))
GPU_MODEL = "B200"

# Tuning-mode -> the policy flags that differ between modes. Everything NOT here is shared
# and lives in the base command below. Each mode also picks its own gradient_checkpointing.
MODE_FLAGS = {
    "action_expert": [
        "--policy.train_action_expert_only=true",
        "--policy.optimizer_action_expert_lr=5e-5",
    ],
    "lora": [
        "--policy.enable_lora_vlm=true",
        "--policy.enable_lora_action_expert=false",
        "--policy.lora_rank=64",
        "--policy.optimizer_lr=5e-5",
        "--policy.optimizer_vit_lr=5e-5",
        "--policy.optimizer_connector_lr=5e-5",
        "--policy.optimizer_action_expert_lr=5e-5",
    ],
    "full": [
        "--policy.enable_lora_vlm=false",
        "--policy.optimizer_lr=1e-5",
        "--policy.optimizer_vit_lr=5e-6",
        "--policy.optimizer_connector_lr=5e-6",
        "--policy.optimizer_action_expert_lr=5e-5",
    ],
}
# Full FT trains the whole VLA -> checkpointing on for memory headroom on the B200. The
# other two measured comfortably under 192GiB with it off (~1.35x faster).
MODE_GRAD_CHECKPOINTING = {"action_expert": "false", "lora": "false", "full": "true"}


def run_banner(*, mode, dataset_repo_id, checkpoint_path, job_name, steps, batch_size,
               gpu_count, chunk_size, n_action_steps, episodes, wandb_enable, wandb_project,
               push_to_hub, policy_repo_id) -> str:
    run_type = "SMOKE TEST" if (steps <= 100 or "smoke" in job_name) else "TRAINING RUN"
    return "\n".join([
        "=" * 72,
        f"  MolmoAct2 fine-tune -- mode={mode.upper()} -- {run_type}",
        f"  dataset   : {dataset_repo_id} (episodes: {episodes or 'all'})",
        f"  base ckpt : {checkpoint_path}",
        f"  gpu       : {GPU_MODEL} x {gpu_count}  (global batch {batch_size * gpu_count})",
        f"  steps     : {steps:,}  (chunk {chunk_size}, exec {n_action_steps}/chunk, continuous)",
        f"  job/wandb : {job_name}  ({'wandb ' + wandb_project if wandb_enable else 'wandb OFF'})",
        f"  output    : /checkpoints/{job_name}  (serve: checkpoints/last)",
        f"  hub push  : {policy_repo_id if push_to_hub else 'OFF'}",
        "=" * 72,
    ])


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("lerobot[molmoact2,training]==0.6.0", "wandb")
    .env({"HF_HOME": "/checkpoints/hf-cache"})
)

checkpoints = modal.Volume.from_name("so100-lerobot-checkpoints", create_if_missing=True)

# Dataloader sizing per training process (== per GPU under accelerate); see the LoRA
# sibling for the measured reasoning (two 720p av1 streams make decode the step's latency
# floor; workers stay below cores because pin_memory adds a page-locked copy thread).
CPU_CORES_PER_GPU = 24.0
DATALOADER_WORKERS = 18
PREFETCH_FACTOR = 4
MEMORY_MIB_PER_GPU = 65536


@app.function(
    image=image,
    gpu=f"{GPU_MODEL}:{GPU_COUNT}",
    cpu=min(CPU_CORES_PER_GPU * GPU_COUNT, 64.0),
    memory=int(MEMORY_MIB_PER_GPU * min(GPU_COUNT, 3)),  # MiB; decode buffers, not weights
    timeout=int(60 * 60 * 6.5),
    volumes={"/checkpoints": checkpoints},
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
)
def train(
    mode: str,
    dataset_repo_id: str,
    policy_repo_id: str,
    checkpoint_path: str,
    job_name: str,
    steps: int,
    batch_size: int,
    push_to_hub: bool,
    save_freq: int,
    image_keys: str,
    chunk_size: int,
    n_action_steps: int,
    episodes: str | None,
    wandb_enable: bool,
    wandb_project: str,
    gpu_count: int,
) -> None:
    import subprocess
    import threading

    print(run_banner(mode=mode, dataset_repo_id=dataset_repo_id, checkpoint_path=checkpoint_path,
                     job_name=job_name, steps=steps, batch_size=batch_size, gpu_count=gpu_count,
                     chunk_size=chunk_size, n_action_steps=n_action_steps, episodes=episodes,
                     wandb_enable=wandb_enable, wandb_project=wandb_project, push_to_hub=push_to_hub,
                     policy_repo_id=policy_repo_id), flush=True)

    output_dir = f"/checkpoints/{job_name}"
    if gpu_count > 1:
        launcher = ["accelerate", "launch", f"--num_processes={gpu_count}",
                    "--mixed_precision=bf16", "-m", "lerobot.scripts.lerobot_train"]
    else:
        launcher = ["lerobot-train"]

    command = launcher + [
        f"--dataset.repo_id={dataset_repo_id}",
        "--dataset.video_backend=pyav",
        "--dataset.image_transforms.enable=true",
        "--policy.type=molmoact2",
        f"--policy.checkpoint_path={checkpoint_path}",
        "--policy.device=cuda",
        "--policy.action_mode=continuous",
        *MODE_FLAGS[mode],  # the tuning-mode-specific flags + learning rates
        f"--policy.image_keys={image_keys}",
        f"--policy.chunk_size={chunk_size}",
        f"--policy.n_action_steps={n_action_steps}",
        "--policy.num_flow_timesteps=8",
        "--policy.normalize_gripper=true",
        "--policy.model_dtype=bfloat16",
        f"--policy.gradient_checkpointing={MODE_GRAD_CHECKPOINTING[mode]}",
        "--policy.setup_type=single so100/so101 robotic arm in molmoact2",
        "--policy.control_mode=absolute joint pose",
        f"--policy.scheduler_decay_steps={steps}",
        '--policy.normalization_mapping={"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}',
        f"--policy.push_to_hub={'true' if push_to_hub else 'false'}",
        f"--policy.repo_id={policy_repo_id}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--num_workers={DATALOADER_WORKERS}",
        f"--prefetch_factor={PREFETCH_FACTOR}",
        "--log_freq=20",
        f"--wandb.enable={'true' if wandb_enable else 'false'}",
        "--save_checkpoint=true",
        f"--save_freq={save_freq}",
    ]
    if wandb_enable:
        command.append(f"--wandb.project={wandb_project}")
    if episodes:
        command.append(f"--dataset.episodes={episodes}")
    print("running:", " ".join(command))

    # Commit the Volume every ~2 min so each checkpoint is reachable/durable soon after it
    # lands (Volume writes are invisible outside the container until committed).
    stop_committing = threading.Event()

    def commit_periodically() -> None:
        while not stop_committing.wait(120):
            checkpoints.commit()

    threading.Thread(target=commit_periodically, daemon=True).start()
    try:
        subprocess.run(command, check=True)
    finally:
        stop_committing.set()
        checkpoints.commit()


@app.local_entrypoint()
def main(
    mode: str = "action_expert",
    dataset_repo_id: str = "JaidevShriram/JAGS_v0_testing",
    policy_repo_id: str = "JaidevShriram/molmoact2-jags",
    checkpoint_path: str = "allenai/MolmoAct2-SO100_101",
    job_name: str = "molmoact2-jags-modes",
    steps: int | None = None,
    batch_size: int = 16,
    push_to_hub: bool = True,
    save_freq: int = 1_000,
    image_keys: str = '["observation.images.top", "observation.images.side"]',
    chunk_size: int = 10,
    n_action_steps: int = 10,
    episodes: str | None = None,
    wandb_enable: bool = True,
    wandb_project: str = "so100-hackathon",
) -> None:
    """
    mode: Tuning mode -- "action_expert" (VLM frozen, cheapest), "lora" (LoRA-VLM r64 +
        full expert), or "full" (everything trainable, heaviest/overfit-prone).
    dataset_repo_id: HF Hub LeRobot v3 dataset. Default: your JAGS_v0_testing.
    policy_repo_id: HF repo the finished policy pushes to. Use a distinct one per mode/run
        so runs don't overwrite each other.
    checkpoint_path: Starting MolmoAct2 checkpoint. SO-100/101 specialist (two-camera).
    job_name: Volume checkpoint subfolder + W&B run name. MUST be unique per run (lerobot
        refuses an existing output_dir).
    steps: Global optimizer steps (default 12,000; also the LR-decay horizon). full mode is
        ~1.35x slower per step -- consider fewer steps or it may approach the 6.5h timeout.
    batch_size: Per-GPU batch. Global batch = batch_size x GPU_COUNT.
    push_to_hub: Push the finished policy to policy_repo_id.
    save_freq: Checkpoint every N steps (also saves the final step).
    image_keys: JSON list of camera streams, in processor order. Default = both JAGS cams.
    chunk_size: Action horizon. 10 is the SO-100/LIBERO-recipe value (matches the ckpt).
    n_action_steps: Actions executed per predicted chunk before re-querying. Must be <=
        chunk_size. 10 = execute the whole chunk (the recipe default); this is mainly a
        rollout knob, changeable at deploy without retraining.
    episodes: Optional JSON list of episode indices. Default: all episodes.
    wandb_enable: Log to W&B (needs the wandb-secret). Pass --wandb-enable false to skip.
    wandb_project: W&B project name.
    """
    if mode not in MODE_FLAGS:
        raise SystemExit(f"--mode must be one of {sorted(MODE_FLAGS)}; got {mode!r}")
    if n_action_steps > chunk_size:
        raise SystemExit(f"--n-action-steps ({n_action_steps}) cannot exceed --chunk-size ({chunk_size})")
    if steps is None:
        steps = 12_000

    print(run_banner(mode=mode, dataset_repo_id=dataset_repo_id, checkpoint_path=checkpoint_path,
                     job_name=job_name, steps=steps, batch_size=batch_size, gpu_count=GPU_COUNT,
                     chunk_size=chunk_size, n_action_steps=n_action_steps, episodes=episodes,
                     wandb_enable=wandb_enable, wandb_project=wandb_project, push_to_hub=push_to_hub,
                     policy_repo_id=policy_repo_id))
    # .spawn() (fire-and-forget), NOT .remote() (blocking): a blocking call from a local
    # entrypoint is tethered to the client, so --detach + a client disconnect cancels it
    # mid-run (observed). .spawn() submits the job and returns a handle immediately, so the
    # remote function runs fully independently and survives the client going away.
    handle = train.spawn(  # pyrefly: ignore[invalid-param-spec] - modal's .spawn() wrapper
        mode=mode,
        dataset_repo_id=dataset_repo_id,
        policy_repo_id=policy_repo_id,
        checkpoint_path=checkpoint_path,
        job_name=job_name,
        steps=steps,
        batch_size=batch_size,
        push_to_hub=push_to_hub,
        save_freq=save_freq,
        image_keys=image_keys,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        episodes=episodes,
        wandb_enable=wandb_enable,
        wandb_project=wandb_project,
        gpu_count=GPU_COUNT,
    )
    print(f"spawned train function (call id {handle.object_id}); it runs independently of "
          f"this client. Follow it with: modal app logs <app-id>")
