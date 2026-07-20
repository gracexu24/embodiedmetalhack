"""Serve the finished ACT fine-tune (job JAGS_v0_testing_task_pick-up-the-red-block-and-place-it-on-th)
from the ``so101-checkpoints`` Volume as a Modal ``/act`` endpoint.

Same pattern as ``policy_server_modal_molmoact2_lora.py``/``_finetune.py`` (LeRobot-native
loading: ``config.json`` + ``model.safetensors`` + processor files, so the exact train-time
pipeline runs at inference), just swapped to ``ACTPolicy``. ACT's ``predict_action_chunk``
takes no extra kwargs (unlike MolmoAct2's ``inference_action_mode``/``num_steps``) -- it's
just ``policy.predict_action_chunk(batch)``.

Speaks the same JSON ``/act`` contract, so it's a drop-in ``--server`` for ``deploy_policy.py``::

    modal deploy tools/apps/policy_server_modal_act.py   # prints a URL ending in /act

    pixi run deploy-policy -- --task "Pick up the red block and place it on the black rectangle" \\
        --server https://<printed-url> --dry-run   # ALWAYS dry-run a new checkpoint first

Smoke test without an arm (placeholder frames + zeroed state; proves loading and shapes only)::

    modal run tools/apps/policy_server_modal_act.py

Training finished at step 10000/10000 (this repo's own convention: 31 episodes, one task, no
joint-convention correction needed -- trained directly on this repo's own calibrated-degree
export, same as the MolmoAct2 fine-tunes). ACT's own config sets chunk_size=n_action_steps=100:
it commits to a full 100-step chunk per inference call rather than a shorter receding-horizon
slice -- deploy_policy.py's --execute-steps just uses however many of those 100 it's given.
"""

from __future__ import annotations

import modal

app = modal.App("serve-act-jags-pick-red-block")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("lerobot==0.6.0", "fastapi[standard]")
    .env({"HF_HOME": "/checkpoints/hf-cache"})
)

checkpoints_volume = modal.Volume.from_name("so101-checkpoints", create_if_missing=False)

CHECKPOINT_DIR = "/checkpoints/JAGS_v0_testing_task_pick-up-the-red-block-and-place-it-on-th/checkpoints/010000/pretrained_model"


@app.cls(
    image=image,
    gpu="A10G",  # ACT (ResNet18 backbone + small transformer) is far lighter than MolmoAct2/pi05
    volumes={"/checkpoints": checkpoints_volume},
    scaledown_window=300,
    timeout=600,
)
class ACTServer:
    @modal.enter()
    def load(self) -> None:
        import torch  # pyrefly: ignore[missing-import] - GPU container only, not a pixi env
        from lerobot.policies.act.modeling_act import ACTPolicy  # pyrefly: ignore[missing-import] - GPU container only
        from lerobot.policies.factory import make_pre_post_processors  # pyrefly: ignore[missing-import] - GPU container only

        self.volume = checkpoints_volume
        self.volume.reload()
        print(f"loading LeRobot checkpoint {CHECKPOINT_DIR} ...")
        self.policy = ACTPolicy.from_pretrained(CHECKPOINT_DIR)
        self.policy.to("cuda").eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(self.policy.config, pretrained_path=CHECKPOINT_DIR)
        # image_keys isn't a universal PreTrainedConfig field (MolmoAct2Config happens to
        # have one; ACTConfig/PI05Config don't) -- input_features is, across policy types.
        self.image_keys: list[str] = [key for key in self.policy.config.input_features if key.startswith("observation.images.")]
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
                expected = [k.rsplit(".", 1)[-1] for k in self.image_keys]
                raise ValueError(f"missing camera {name!r}: this checkpoint expects images named {expected}")
            frame = np.asarray(Image.open(io.BytesIO(images_by_name[name])).convert("RGB"))
            observation[key] = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(torch.float32).div_(255.0).to("cuda")
        observation["observation.state"] = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to("cuda")
        observation["task"] = instruction

        with torch.inference_mode():
            batch = self.preprocessor(observation)
            chunk = self.policy.predict_action_chunk(batch)
            chunk = self.postprocessor(chunk)
        return chunk[0].to(torch.float32).cpu().numpy().tolist()

    @modal.method()
    def predict(self, instruction: str, state: list[float], images_by_name: dict[str, bytes]) -> list[list[float]]:
        """Direct (non-HTTP) entrypoint, used by the local smoke test below."""
        return self._predict(instruction, state, images_by_name)

    @modal.fastapi_endpoint(method="POST", label="act-jags-pick-red-block-act")
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
    """Smoke test: loads CHECKPOINT_DIR on Modal and prints one predicted action chunk."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1280, 720), color=(128, 128, 128)).save(buffer, format="JPEG")
    jpeg = buffer.getvalue()

    images = {"top": jpeg, "side": jpeg}
    actions: list[list[float]] = ACTServer().predict.remote(task, [0.0] * 6, images)  # pyrefly: ignore[invalid-param-spec, bad-assignment]
    print(f"task: {task!r}")
    print(f"predicted chunk: {len(actions)} steps, {len(actions[0])} joints each")
    for step in actions[:5]:
        print("  " + ", ".join(f"{value:+.1f}" for value in step))
