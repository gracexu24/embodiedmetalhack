"""Serve the ``molmoact2-jags-lora`` LoRA fine-tune from the ``so100-lerobot-checkpoints``
Volume as a Modal ``/act`` endpoint.

Adapted from ``policy_server_modal_molmoact2_finetune.py`` (currently only on the
``modal-dataloader-cpu-reservation`` branch, not yet merged to main) -- same LeRobot-native
loading approach, just pointed at this LoRA job's checkpoint instead of the scale run's.
lerobot-train saves a **LeRobot-format** checkpoint (policy ``config.json`` +
``model.safetensors`` + ``policy_preprocessor``/``policy_postprocessor`` files) regardless of
tuning mode (action-expert/LoRA/full) -- ``config.json`` encodes the LoRA settings
(``enable_lora_vlm``, ``lora_rank``, ...) so ``MolmoAct2Policy.from_pretrained`` rebuilds the
LoRA-adapted architecture before loading weights, the same call as any other mode.

Speaks the same JSON ``/act`` contract, so it's a drop-in ``--server`` for ``deploy_policy.py``::

    modal deploy tools/apps/policy_server_modal_molmoact2_lora.py   # prints a URL ending in /act

    pixi run deploy-policy -- --task "Pick up the red block and place it on the black rectangle" \\
        --server https://<printed-url> --dry-run    # ALWAYS dry-run a new checkpoint first

Smoke test without an arm (placeholder frames + zeroed state; proves loading and shapes only)::

    modal run tools/apps/policy_server_modal_molmoact2_lora.py

CHECKPOINT_DIR is pinned to step 005000, not the in-progress ``checkpoints/last`` -- the
molmoact2-jags-lora training run (app jaidev-molmoact2-lora) was still going (past step 5000 of
12000) when this was written. Bump the step number for a later checkpoint once training
progresses further or finishes; re-`modal deploy` to pick it up. No joint-convention correction
is applied here, deliberately: a fine-tune trained on this repo's own exports learns this repo's
convention directly -- applying the public-checkpoint fix on top would corrupt every pose (see
``policy_server_modal_molmoact2.py``).

Camera frames are matched to the policy's cameras BY NAME: for each train-time image key
``observation.images.<name>`` the request must carry ``images[<name>]``. Extra cameras in the
request are ignored; missing ones are a 400 with the expected names in the error.
"""

from __future__ import annotations

import modal

# Distinct from the already-deployed "serve-molmoact2-so101-scale" app, which serves a
# different checkpoint -- this workspace accumulates many look-alike apps.
app = modal.App("serve-molmoact2-jags-lora")

# lerobot's molmoact2 extra brings torch/transformers pinned to versions the checkpoint was
# trained with. HF_HOME on the Volume persists the ~11GB allenai/MolmoAct2 base download
# (LeRobot restores the fine-tune by first instantiating that base) across cold starts.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("lerobot[molmoact2]==0.6.0", "fastapi[standard]")
    .env({"HF_HOME": "/checkpoints/hf-cache"})
)

checkpoints_volume = modal.Volume.from_name("so100-lerobot-checkpoints", create_if_missing=True)

CHECKPOINT_DIR = "/checkpoints/molmoact2-jags-lora/checkpoints/005000/pretrained_model"
NUM_STEPS = 10  # flow-matching denoise steps at inference (matches the training-time chunk setup)


@app.cls(
    image=image,
    gpu="A10G",  # inference at batch 1 is ~12GiB per the LeRobot docs; no need for the training H100
    volumes={"/checkpoints": checkpoints_volume},
    scaledown_window=300,
    timeout=600,
)
class MolmoAct2FinetuneServer:
    @modal.enter()
    def load(self) -> None:
        import torch  # pyrefly: ignore[missing-import] - GPU container only, not a pixi env
        from lerobot.policies.factory import make_pre_post_processors  # pyrefly: ignore[missing-import] - GPU container only
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy  # pyrefly: ignore[missing-import] - GPU container only

        self.volume = checkpoints_volume
        self.volume.reload()  # pick up checkpoints committed after this container booted
        print(f"loading LeRobot checkpoint {CHECKPOINT_DIR} ...")
        self.policy = MolmoAct2Policy.from_pretrained(CHECKPOINT_DIR)
        self.policy.to("cuda").eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(self.policy.config, pretrained_path=CHECKPOINT_DIR)
        self.image_keys: list[str] = list(self.policy.config.image_keys)
        self.torch = torch
        print(f"policy ready; cameras expected (in order): {self.image_keys}")

    def _predict(self, instruction: str, state: list[float], images_by_name: dict[str, bytes]) -> list[list[float]]:
        import io

        import numpy as np
        from PIL import Image

        torch = self.torch
        observation: dict = {}
        for key in self.image_keys:
            name = key.rsplit(".", 1)[-1]
            if name not in images_by_name:
                raise ValueError(f"missing camera {name!r}: this checkpoint expects images named {[k.rsplit('.', 1)[-1] for k in self.image_keys]}")
            frame = np.asarray(Image.open(io.BytesIO(images_by_name[name])).convert("RGB"))
            # channel-first float32 in [0,1] with a batch dim -- the same shape
            # lerobot's own control loop feeds the preprocessor pipeline.
            observation[key] = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(torch.float32).div_(255.0).to("cuda")
        observation["observation.state"] = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to("cuda")
        observation["task"] = instruction

        with torch.inference_mode():
            batch = self.preprocessor(observation)
            chunk = self.policy.predict_action_chunk(batch, inference_action_mode="continuous", num_steps=NUM_STEPS)
            chunk = self.postprocessor(chunk)
        return chunk[0].to(torch.float32).cpu().numpy().tolist()

    @modal.method()
    def predict(self, instruction: str, state: list[float], images_by_name: dict[str, bytes]) -> list[list[float]]:
        """Direct (non-HTTP) entrypoint, used by the local smoke test below."""
        return self._predict(instruction, state, images_by_name)

    # label pins the subdomain: https://<workspace>--molmoact2-jags-lora-act.modal.run
    # -- a URL that says what it serves, instead of Modal's auto "<app>-<class>-<hash>".
    @modal.fastapi_endpoint(method="POST", label="molmoact2-jags-lora-act")
    def act(self, payload: dict) -> dict:
        import base64

        from fastapi.responses import JSONResponse  # pyrefly: ignore[missing-import] - GPU container only, not a pixi env

        try:
            images = {name: base64.b64decode(data) for name, data in payload["images"].items()}
            actions = self._predict(payload["instruction"], payload["state"], images)
            return {"actions": actions}
        except Exception as error:  # surface the failure to the client instead of a hung arm loop
            return JSONResponse(status_code=500, content={"error": f"{type(error).__name__}: {error}"})


@app.local_entrypoint()
def main(task: str = "Pick up the red block and place it on the black rectangle") -> None:
    """Smoke test: loads CHECKPOINT_DIR on Modal and prints one predicted action chunk.

    Sends a solid-gray frame for every camera the checkpoint expects and a zeroed 6-dof
    state -- enough to confirm the LeRobot loading path, the processor pipelines, and the
    output shape. It says nothing about real capability; that needs deploy_policy.py.
    """
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1280, 720), color=(128, 128, 128)).save(buffer, format="JPEG")
    jpeg = buffer.getvalue()

    # Offer a frame under every camera name this repo's exports use; the server picks the
    # ones its checkpoint needs and ignores the rest.
    images = {"top": jpeg, "side": jpeg}
    actions: list[list[float]] = MolmoAct2FinetuneServer().predict.remote(task, [0.0] * 6, images)  # pyrefly: ignore[invalid-param-spec, bad-assignment]
    print(f"task: {task!r}")
    print(f"predicted chunk: {len(actions)} steps, {len(actions[0])} joints each")
    for step in actions:
        print("  " + ", ".join(f"{value:+.1f}" for value in step))
