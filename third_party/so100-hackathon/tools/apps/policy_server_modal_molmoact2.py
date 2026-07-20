"""Serve a MolmoAct2 checkpoint for the SO-100/101 as a Modal-hosted inference endpoint, so
you can test what a policy is capable of -- e.g. "move the green block near the yellow
block" -- without owning a GPU box yourself.

Speaks the same JSON ``/act`` contract as ``policy_server_molmoact2.py`` (same
checkpoint-loading and ``predict_action`` call -- see that file's docstring for the
contract and the joint-convention caveat), so it's a drop-in ``--server`` for
``deploy_policy.py``::

    pixi run modal setup                                     # once, browser auth
    modal deploy tools/apps/policy_server_modal_molmoact2.py  # prints a URL ending in /act

    pixi run deploy-policy -- --task "move the green block near the yellow block" \\
        --server https://<printed-url> --dry-run              # ALWAYS dry-run a new checkpoint first
    pixi run deploy-policy -- --task "move the green block near the yellow block" \\
        --server https://<printed-url>                        # drop --dry-run once that looks sane

Or smoke-test the model alone, no arm/server needed (a placeholder image and zeroed state
just prove the checkpoint loads and returns a chunk of the right shape -- it says nothing
about real capability, which needs real camera frames from ``deploy_policy.py``):

    modal run tools/apps/policy_server_modal_molmoact2.py --task "pick up the ball"

The default checkpoint is the public ``allenai/MolmoAct2-SO100_101`` and needs no HF
token. To serve your own fine-tune (from ``finetune_modal_molmoact2.py``) instead, edit
CHECKPOINT below and add the same ``huggingface-secret`` (see ``finetune_modal_act.py``)
if the repo is private, then ``modal deploy`` again -- and also set NEEDS_JOINT_FIX to
False: a fine-tune starting from lerobot/MolmoAct2-SO100_101-LeRobot is trained on this
repo's own joint convention directly and does NOT need the correction below; applying it
twice would be as wrong as not applying it once.
"""

from __future__ import annotations

import modal

app = modal.App("so100-molmoact2-serve")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "transformers", "accelerate", "pillow", "numpy", "fastapi[standard]", "einops", "torchvision", "requests"
)

checkpoints_volume = modal.Volume.from_name("so100-lerobot-checkpoints", create_if_missing=True)

CHECKPOINT = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
NUM_STEPS = 10
# The checkpoint's own config.json has "action_mode": "both" (a discrete action-token
# head, num_action_tokens=2048, alongside the continuous flow-matching one) -- but calling
# inference_action_mode="discrete" through this trust_remote_code path reproducibly fails:
# `ValueError: inference_action_mode='discrete' requires an `action_tokenizer` input`,
# which we don't have a confirmed way to construct/load for this checkpoint. Don't set
# this to "discrete" (or pass --action-mode discrete from deploy_policy.py) until that's
# actually solved -- it WILL 500 and crash a live run (deploy_policy.py exits cleanly on a
# server error, so the arm is left safe, but the run is lost).
# Just the fallback if a request doesn't specify one -- deploy_policy.py's --action-mode
# sends it per-request, so switching modes doesn't need a redeploy.
INFERENCE_ACTION_MODE = "continuous"

# The public checkpoint was trained on data using the pre-LeRobot-0.5.0 SO-100/101 joint
# convention. LeRobot's own docs for lerobot/MolmoAct2-SO100_101-LeRobot (the corrected
# port of this same checkpoint) document the fix: flip shoulder_lift's sign, and shift
# shoulder_lift + elbow_flex by 90 degrees. Order matches this repo's motor order
# (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper).
# state -> model:   model_state  = JOINT_SIGNS * arm_state + JOINT_OFFSETS
# model -> action:  arm_action   = JOINT_SIGNS * (model_action - JOINT_OFFSETS)
JOINT_SIGNS = (1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
JOINT_OFFSETS = (0.0, 90.0, 90.0, 0.0, 0.0, 0.0)
NEEDS_JOINT_FIX = True


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/checkpoints": checkpoints_volume},
    scaledown_window=300,  # keep the model warm for 5 idle minutes between requests
    timeout=600,
)
class MolmoAct2Server:
    @modal.enter()
    def load(self) -> None:
        import torch  # pyrefly: ignore[missing-import] - GPU container only, not a pixi env
        from transformers import AutoModelForImageTextToText, AutoProcessor  # pyrefly: ignore[missing-import] - GPU container only

        print(f"loading {CHECKPOINT} (bf16) ...")
        self.processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model = AutoModelForImageTextToText.from_pretrained(CHECKPOINT, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
        print("model ready")

    def _predict(self, instruction: str, state: list[float], jpegs: list[bytes], action_mode: str) -> list[list[float]]:
        import io

        import numpy as np
        import torch  # pyrefly: ignore[missing-import] - GPU container only, not a pixi env
        from PIL import Image

        images = [Image.open(io.BytesIO(data)).convert("RGB") for data in jpegs]
        signs = np.asarray(JOINT_SIGNS, dtype=np.float32)
        offsets = np.asarray(JOINT_OFFSETS, dtype=np.float32)
        arm_state = np.asarray(state, dtype=np.float32)
        model_state = signs * arm_state + offsets if NEEDS_JOINT_FIX else arm_state
        with torch.inference_mode():
            output = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=instruction,
                state=model_state,
                norm_tag=NORM_TAG,
                inference_action_mode=action_mode,
                num_steps=NUM_STEPS,
            )
        # predict_action returns a MolmoAct2ActionOutput, not a plain array; .actions is
        # (batch, num_steps, joints) -- batch is always 1 here (one observation in).
        model_actions = output.actions[0].to(torch.float32).cpu().numpy()
        arm_actions = signs * (model_actions - offsets) if NEEDS_JOINT_FIX else model_actions
        return arm_actions.tolist()

    @modal.method()
    def predict(self, instruction: str, state: list[float], jpegs: list[bytes], action_mode: str = INFERENCE_ACTION_MODE) -> list[list[float]]:
        """Direct (non-HTTP) entrypoint, used by the local smoke test below."""
        return self._predict(instruction, state, jpegs, action_mode)

    @modal.fastapi_endpoint(method="POST")
    def act(self, payload: dict) -> dict:
        import base64

        from fastapi.responses import JSONResponse  # pyrefly: ignore[missing-import] - GPU container only, not a pixi env

        try:
            jpegs = [base64.b64decode(data) for data in payload["images"].values()]
            action_mode = payload.get("action_mode", INFERENCE_ACTION_MODE)
            actions = self._predict(payload["instruction"], payload["state"], jpegs, action_mode)
            return {"actions": actions}
        except Exception as error:  # surface the failure to the client instead of a hung arm loop
            return JSONResponse(status_code=500, content={"error": f"{type(error).__name__}: {error}"})


@app.local_entrypoint()
def main(task: str = "pick up the ball", image_path: str | None = None, action_mode: str = INFERENCE_ACTION_MODE) -> None:
    """Smoke test: loads the checkpoint on Modal and prints one predicted action chunk.

    Without --image-path this sends a solid-gray placeholder frame and a zeroed 6-dof
    state -- enough to confirm the checkpoint loads and the shapes line up, nothing more.
    Pass a real JPEG (e.g. a frame saved from `pixi run log-so100`) for a more meaningful
    look at what the model predicts for a given instruction.
    """
    import io

    from PIL import Image

    if image_path:
        with open(image_path, "rb") as file:
            jpeg = file.read()
    else:
        buffer = io.BytesIO()
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(buffer, format="JPEG")
        jpeg = buffer.getvalue()

    actions: list[list[float]] = MolmoAct2Server().predict.remote(task, [0.0] * 6, [jpeg, jpeg], action_mode)  # pyrefly: ignore[invalid-param-spec, bad-assignment]
    print(f"task: {task!r}")
    print(f"predicted chunk: {len(actions)} steps, {len(actions[0])} joints each")
    for step in actions:
        print("  " + ", ".join(f"{v:+.1f}" for v in step))
