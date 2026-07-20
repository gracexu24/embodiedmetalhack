"""Fine-tune MolmoAct2 (action-expert-only) on a Modal GPU from a Hugging Face LeRobot dataset.

Ai2's MolmoAct2 (https://huggingface.co/docs/lerobot/en/molmoact2), ported into LeRobot as
``--policy.type=molmoact2`` -- an open policy, not New Theory's. This starts from the
generalist ``allenai/MolmoAct2`` HF checkpoint (via ``--policy.checkpoint_path``, which
builds the policy config fresh from the dataset + CLI options) and fine-tunes only the
action expert -- the mode LeRobot's own docs recommend for small datasets (~16-21GiB peak
memory, fits a single A10G), versus full fine-tuning (~48-60GiB, H100-class).

Not started from the released ``lerobot/MolmoAct2-SO100_101-LeRobot`` checkpoint: that one
restores its saved config wholesale via ``--policy.path`` (only batch_size/steps/LoRA
flags/action_mode are documented as overridable afterward), and its config is frozen to two
cameras (cam0, cam1) -- it rejects a single-camera dataset like ours (``observation.images.top``)
with a feature-mismatch error. The trade-off going generalist: we lose that checkpoint's
baked-in SO-100/101 joint-convention correction (a joint_signs/joint_offsets fix for an old
calibration convention) and its SO-100-specific pretraining. Revisit if you get a 2-camera
recording and want the specialist checkpoint instead::

    pixi run finetune-modal-molmoact2 -- --dataset-repo-id JaidevShriram/Test_Run_Data

See ``finetune_modal_act.py`` for the lighter ACT alternative and shared one-time Modal
setup (``modal setup``, the ``huggingface-secret``) -- both scripts reuse the same
``so100-lerobot-checkpoints`` Volume under different ``job_name`` subfolders.

lerobot 0.6.0 (not this repo's pinned 0.4.2 -- MolmoAct2 is too new for that release) with
its ``molmoact2`` extra, requires Python >=3.12; that only affects the remote container
image, not this repo's local pixi envs.
"""

from __future__ import annotations

import modal

app = modal.App("so100-molmoact2-finetune")

image = modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg").pip_install("lerobot[molmoact2,training]==0.6.0")

checkpoints = modal.Volume.from_name("so100-lerobot-checkpoints", create_if_missing=True)


# Container sizing for the dataloader. Modal's default reservation is a ~0.125-core sliver and
# 128MiB, which starves this job: the dataloader does pyav video decode + image transforms on
# every batch, so on the default the A10G sat near 0% utilization between brief 50-70% spikes,
# waiting on CPU rather than computing. With the reservation in place, measured data_s (dataloader
# wait) was 0.202s/step on the A10G shape -- ~5.5% of the step, i.e. no longer the bottleneck.
# See MODAL_FINETUNING.md ("CPU sizing matters as much as the GPU") for the full measured table.
#
# These four are one unit; changing any of them without the others reintroduces a problem the
# others were sized around, which is why they live here rather than inline at their use sites.
# Scaled with the GPU below, not independent of it. A faster GPU consumes batches faster, so
# the dataloader has proportionally less wall-clock to produce each one: on the A10G a measured
# 3.57s/step gave 6 workers 3.57s to decode 16 samples, while an H100 step needs the same 16
# samples in a fraction of that. Moving to a faster GPU without raising these just relocates the
# starvation to a more expensive machine.
CPU_CORES = 16.0
# Deliberately below CPU_CORES. lerobot builds the DataLoader with pin_memory=True (device is
# cuda), so a host-side thread copies every batch into page-locked memory *on the main process*,
# alongside the optimizer/GPU-feed loop. Workers == cores leaves that thread scavenging spare
# host capacity -- the exact reliance this reservation exists to remove.
DATALOADER_WORKERS = 12
# lerobot's own default. In-flight host buffers = DATALOADER_WORKERS * PREFETCH_FACTOR batches,
# so this is the other half of the memory sizing below; raise it and MEMORY_MIB must follow.
PREFETCH_FACTOR = 4
# ~5GiB was observed at 4 workers x 4 prefetch (16 batches in flight), so 12 x 4 = 48 batches
# extrapolates to ~15GiB; this is ~2x that. Headroom matters more than the average here:
# persistent_workers defaults True in lerobot 0.6.0 (buffers live for the whole run, not freed
# between epochs) and pin_memory makes them page-locked, so bursting past a soft request on a
# loaded host ends in an OOM kill mid-run rather than swapping.
MEMORY_MIB = 32768


# cpu/memory are *reservations*: bursting above them is allowed, but Modal bills the higher of
# request or actual usage, for the job's whole wall-clock (including model load and checkpoint
# upload, where the dataloader is idle). The H100 bills ~4x the A10G per hour but measured 6.4x
# faster, so this shape is cheaper per run as well as faster.
# If this shape is not schedulable on the account, scale CPU_CORES and DATALOADER_WORKERS down
# together rather than dropping only one.
#
# GPU, measured side by side (2026-07-18, 500 steps, batch 16, this dataset): A10G with
# gradient checkpointing on = 3.62s/step; H100 with checkpointing off = 0.57s/step (6.4x,
# compound of both changes). H100 memory measured 23.15GiB -- checkpointing-off would not have
# fit the A10G's 24GiB, which is why the GPU and the flag in train() below move together.
# On this shape data_s is ~35% of the step, so the next lever is the dataloader (likely
# pre-decoded frames), not a bigger GPU.
#
# Timeout stays 4h. On the A10G it was too short for the default run (5000 x 3.62s = ~5h,
# killed near step 4000); on the H100 the same run measures ~47min, comfortably inside. If you
# revert to gpu="A10G", raise this to 60 * 60 * 6 first and set gradient_checkpointing back on.
@app.function(
    image=image,
    gpu="H100",
    cpu=CPU_CORES,
    memory=MEMORY_MIB,  # MiB
    timeout=60 * 60 * 4,
    volumes={"/checkpoints": checkpoints},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(
    dataset_repo_id: str,
    policy_repo_id: str,
    checkpoint_path: str,
    job_name: str,
    steps: int,
    batch_size: int,
    push_to_hub: bool,
    save_freq: int,
) -> None:
    import subprocess

    output_dir = f"/checkpoints/{job_name}"
    command = [
        "lerobot-train",
        f"--dataset.repo_id={dataset_repo_id}",
        "--dataset.video_backend=pyav",
        "--dataset.image_transforms.enable=true",
        "--policy.type=molmoact2",
        f"--policy.checkpoint_path={checkpoint_path}",
        "--policy.device=cuda",
        "--policy.action_mode=continuous",
        "--policy.train_action_expert_only=true",
        '--policy.image_keys=["observation.images.top"]',
        "--policy.chunk_size=10",
        "--policy.n_action_steps=10",
        "--policy.num_flow_timesteps=8",
        # Our gripper channel is in raw units, not scaled to [-1, 1]; molmoact2's continuous
        # action head validates that range, so let it fold the gripper into normalization.
        "--policy.normalize_gripper=true",
        "--policy.model_dtype=bfloat16",
        # Off because the H100 above has 80GiB. Checkpointing trades ~30% step time for memory by
        # discarding activations and recomputing them in backward; that trade was mandatory at
        # 24GiB (~13GiB resident before activations) and is not at 80GiB. Set back to true if you
        # move this job to any GPU smaller than an H100, or it will OOM rather than run slowly.
        "--policy.gradient_checkpointing=false",
        "--policy.setup_type=so100 robotic arm",
        "--policy.control_mode=absolute joint pose",
        # Our dataset hasn't been through augment_dataset_quantile_stats.py (molmoact2's
        # default normalization needs it); mean/std sidesteps that extra preprocessing step.
        '--policy.normalization_mapping={"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}',
        f"--policy.push_to_hub={'true' if push_to_hub else 'false'}",
        f"--policy.repo_id={policy_repo_id}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        # Sized with the container: see CPU_CORES / DATALOADER_WORKERS / PREFETCH_FACTOR above.
        # More workers than reserved cores just adds contention. Note these fork pyav decoders,
        # which hold non-fork-safe internal state -- if the step counter ever freezes with both
        # CPU and GPU flat and no traceback, suspect a wedged worker and try a lower count or
        # --dataset.video_backend=torchcodec before assuming the GPU is at fault.
        f"--num_workers={DATALOADER_WORKERS}",
        f"--prefetch_factor={PREFETCH_FACTOR}",
        "--wandb.enable=false",
        "--save_checkpoint=true",
        f"--save_freq={save_freq}",
    ]
    print("running:", " ".join(command))
    subprocess.run(command, check=True)
    checkpoints.commit()


@app.local_entrypoint()
def main(
    dataset_repo_id: str = "JaidevShriram/Test_Run_Data",
    policy_repo_id: str = "JaidevShriram/molmoact2-so100-action-expert",
    checkpoint_path: str = "allenai/MolmoAct2",
    job_name: str = "molmoact2-so100-action-expert",
    steps: int = 5_000,
    batch_size: int = 16,
    push_to_hub: bool = True,
    save_freq: int = 1_000,
) -> None:
    """
    dataset_repo_id: HF Hub LeRobot v3 dataset to train on.
    policy_repo_id: Where the trained policy is pushed on the HF Hub (only used if push_to_hub).
    checkpoint_path: Starting MolmoAct2 HF checkpoint (--policy.checkpoint_path).
    job_name: Run name -- also the checkpoint subfolder under the Modal Volume.
    steps: Training steps. JaidevShriram/Test_Run_Data is a single ~52s episode (1564 frames),
        so treat a first run as a pipeline smoke test rather than a production policy. For a
        quick end-to-end check that the pipeline runs and a checkpoint saves, use e.g.
        `--steps 20 --save-freq 20 --job-name molmoact2-so100-smoketest --no-push-to-hub`.
    batch_size: Training batch size (16 fits action-expert-only fine-tuning on an A10G).
    push_to_hub: Push the finished policy to policy_repo_id on the HF Hub.
    save_freq: Checkpoint every N steps (lerobot also saves on the final step). Keep this <=
        steps on a short verification run so a checkpoint actually lands.
    """
    train.remote(  # pyrefly: ignore[invalid-param-spec] - modal's .remote() wrapper, same as policy_server_modal_molmoact2.py
        dataset_repo_id=dataset_repo_id,
        policy_repo_id=policy_repo_id,
        checkpoint_path=checkpoint_path,
        job_name=job_name,
        steps=steps,
        batch_size=batch_size,
        push_to_hub=push_to_hub,
        save_freq=save_freq,
    )
