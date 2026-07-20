"""Fine-tune an ACT policy on a Modal GPU from a Hugging Face LeRobot dataset.

An alternative to ``pixi run finetune`` (which trains on New Theory's GPUs): this trains
on your own Modal account instead, straight from a dataset already on the HF Hub -- no
local recordings or export step needed::

    pixi run finetune-modal-act -- --dataset-repo-id JaidevShriram/Test_Run_Data

The remote container builds its own image (lerobot + torch + ffmpeg) independent of this
repo's pixi envs; only the lightweight ``modal`` client runs locally to submit the job.
Checkpoints land in the ``so100-lerobot-checkpoints`` Modal Volume under
``/checkpoints/<job-name>``, and (by default) the finished policy is pushed to
``--policy-repo-id`` on the HF Hub.

One-time setup, in this env (``pixi install`` already added the ``modal`` package):

    pixi run modal setup                                          # browser auth
    pixi run modal secret create huggingface-secret HF_TOKEN=<your HF write token>

``modal run tools/apps/finetune_modal_act.py --help`` shows this same CLI without pixi's
task wrapper. See ``finetune_modal_molmoact2.py`` for the MolmoAct2 alternative.
"""

from __future__ import annotations

import modal

app = modal.App("so100-lerobot-finetune")

# lerobot pulls its own torch; on Modal's Linux/CUDA images plain `pip install torch` is
# already a CUDA-enabled wheel, so no extra index url is needed.
image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("lerobot==0.4.2")

checkpoints = modal.Volume.from_name("so100-lerobot-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 4,
    volumes={"/checkpoints": checkpoints},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(
    dataset_repo_id: str,
    policy_repo_id: str,
    job_name: str,
    steps: int,
    batch_size: int,
    push_to_hub: bool,
    save_freq: int,
    resume: bool,
) -> None:
    import subprocess

    output_dir = f"/checkpoints/{job_name}"
    command = [
        "lerobot-train",
        f"--dataset.repo_id={dataset_repo_id}",
        "--policy.type=act",
        f"--policy.repo_id={policy_repo_id}",
        f"--policy.push_to_hub={'true' if push_to_hub else 'false'}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        "--policy.device=cuda",
        "--wandb.enable=false",
        "--save_checkpoint=true",
        # Periodic checkpoints (not just the final one) so an interruption rewinds to the
        # last save instead of losing the whole run. Each save is committed to the Volume
        # below; re-run with --resume to continue from output_dir's latest checkpoint.
        f"--save_freq={save_freq}",
        f"--resume={'true' if resume else 'false'}",
    ]
    print("running:", " ".join(command))
    # Commit each new checkpoint to the Volume as it lands, so progress survives even if the
    # container is killed mid-run (a resume then picks up from the last committed step).
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
        if "Checkpoint policy after step" in line:
            checkpoints.commit()
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)
    checkpoints.commit()


@app.local_entrypoint()
def main(
    dataset_repo_id: str = "JaidevShriram/Test_Run_Data",
    policy_repo_id: str = "JaidevShriram/act-test-run-data",
    job_name: str = "test-run-data-act",
    steps: int = 20_000,
    batch_size: int = 8,
    push_to_hub: bool = True,
    save_freq: int = 2_000,
    resume: bool = False,
) -> None:
    """
    dataset_repo_id: HF Hub LeRobot v3 dataset to train on.
    policy_repo_id: Where the trained policy is pushed on the HF Hub (only used if push_to_hub).
    job_name: Run name -- also the checkpoint subfolder under the Modal Volume.
    steps: Training steps. JaidevShriram/Test_Run_Data is a single ~52s episode (1564 frames),
        so treat a first run as a pipeline smoke test rather than a production policy -- rerun
        with more steps (and more recorded episodes) once the plumbing is confirmed working.
    batch_size: Training batch size.
    push_to_hub: Push the finished policy to policy_repo_id on the HF Hub.
    save_freq: Checkpoint every N steps (committed to the Volume as each lands). Keeps an
        interruption from losing more than the last N steps.
    resume: Continue from the latest checkpoint already in the Volume for this job_name.
    """
    train.remote(  # pyrefly: ignore[invalid-param-spec] - modal's .remote() wrapper, same as policy_server_modal_molmoact2.py
        dataset_repo_id=dataset_repo_id,
        policy_repo_id=policy_repo_id,
        job_name=job_name,
        steps=steps,
        batch_size=batch_size,
        push_to_hub=push_to_hub,
        save_freq=save_freq,
        resume=resume,
    )
