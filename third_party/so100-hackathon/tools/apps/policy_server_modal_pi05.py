"""Serve the in-progress pi0.5 fine-tune from the ``lerobot-pi05-smolvla-data`` Volume as a
Modal ``/act`` endpoint.

Same LeRobot-native loading pattern as the MolmoAct2/ACT servers, swapped to ``PI05Policy``.
``predict_action_chunk`` takes no extra kwargs -- ``num_inference_steps`` (10, per this
checkpoint's config) is read automatically from the saved config, not passed at call time.

Speaks the same JSON ``/act`` contract, so it's a drop-in ``--server`` for ``deploy_policy.py``::

    modal deploy tools/apps/policy_server_modal_pi05.py   # prints a URL ending in /act

    pixi run deploy-policy -- --task "Pick up the red block and place it on the black rectangle" \\
        --server https://<printed-url> --dry-run   # ALWAYS dry-run a new checkpoint first

Smoke test without an arm (placeholder frames + zeroed state; proves loading and shapes only)::

    modal run tools/apps/policy_server_modal_pi05.py

CAVEAT -- training was only ~50% done (step 15412/30824, job
jaidevshriram-jags-v0-testing-20260719-110707-pi05) when this was pinned: predictions may be
noticeably worse than a finished run. Bump CHECKPOINT_DIR's step number as training progresses
(check `modal volume ls lerobot-pi05-smolvla-data runs/.../pi05/checkpoints`), then re-deploy.

CAVEAT (resolved) -- this checkpoint's saved config has compile_model=true,
compile_mode="max-autotune". That mode's kernel-autotuning search hit repeated shared-memory
OOM errors on the A10G ("No valid triton configs... Hardware limit:101376") for many
candidate kernels, taking long enough (several minutes, sometimes never converging cleanly)
that a second request could arrive mid-compile and get the container killed before the first
ever finished -- an actual instability, not just slowness. Fixed by passing
compile_model=False to from_pretrained, overriding the saved config: max-autotune is meant for
high-throughput repeated calls with a cached compiled graph, not a serverless container that
cold-starts fresh every time. The load()-time warm-up forward pass is kept anyway so any
first-call latency (now just ordinary eager-mode inference, no compile) happens during
container startup rather than on your first real request.
"""

from __future__ import annotations

import modal

app = modal.App("serve-pi05-jags")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("lerobot[pi]==0.6.0", "fastapi[standard]")
    .env({"HF_HOME": "/data/hf"})
)

data_volume = modal.Volume.from_name("lerobot-pi05-smolvla-data", create_if_missing=False)

CHECKPOINT_DIR = "/data/runs/jaidevshriram-jags-v0-testing-20260719-110707/pi05/checkpoints/015412/pretrained_model"


def _placeholder_jpeg() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1280, 720), color=(128, 128, 128)).save(buffer, format="JPEG")
    return buffer.getvalue()


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/data": data_volume},
    scaledown_window=300,
    timeout=900,  # generous: torch.compile's max-autotune can make the first call slow
)
class PI05Server:
    @modal.enter()
    def load(self) -> None:
        import torch  # pyrefly: ignore[missing-import] - GPU container only, not a pixi env
        from lerobot.policies.factory import make_pre_post_processors  # pyrefly: ignore[missing-import] - GPU container only
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # pyrefly: ignore[missing-import] - GPU container only

        self.volume = data_volume
        self.volume.reload()
        print(f"loading LeRobot checkpoint {CHECKPOINT_DIR} ...")
        # compile_model=False overrides the checkpoint's saved compile_model=true,
        # compile_mode="max-autotune": that mode's kernel-autotuning search was hitting
        # shared-memory OOM errors on the A10G for many candidate configs (see this file's
        # git history), taking long enough that a second request could arrive mid-compile
        # and get the container killed before it ever finished. max-autotune is meant for
        # high-throughput repeated calls with a cached compiled graph, not a serverless
        # container that cold-starts fresh every time -- eager execution is slower per call
        # but actually reliable here.
        self.policy = PI05Policy.from_pretrained(CHECKPOINT_DIR, compile_model=False)
        self.policy.to("cuda").eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(self.policy.config, pretrained_path=CHECKPOINT_DIR)
        # image_keys isn't a universal PreTrainedConfig field (MolmoAct2Config happens to
        # have one; PI05Config/ACTConfig don't) -- input_features is, across policy types.
        self.image_keys: list[str] = [key for key in self.policy.config.input_features if key.startswith("observation.images.")]
        self.torch = torch
        print(f"policy ready; cameras expected (in order): {self.image_keys}")

        # Warm up torch.compile's max-autotune search HERE, during container startup
        # (which has no request-facing timeout), not on the first real HTTP call. Modal's
        # web endpoint returns a 303-redirect-to-poll for slow calls, and this checkpoint's
        # compile_model=true made the first inference take long enough to hit that -- a
        # plain HTTP client (including deploy_policy.py's urllib POST) doesn't know how to
        # follow that continuation, so it would see a redirect instead of real JSON.
        print("warming up (torch.compile autotune) with a dummy forward pass ...")
        placeholder = {name.rsplit(".", 1)[-1]: _placeholder_jpeg() for name in self.image_keys}
        self._predict("warmup", [0.0] * 6, placeholder)
        print("warm-up done; ready for real requests")

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

    @modal.fastapi_endpoint(method="POST", label="pi05-jags-act")
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

    The warm-up in load() means this smoke test itself should be fast even on a cold
    container -- the slow autotune already happened before it started accepting calls.
    """
    jpeg = _placeholder_jpeg()
    images = {"top": jpeg, "side": jpeg}
    actions: list[list[float]] = PI05Server().predict.remote(task, [0.0] * 6, images)  # pyrefly: ignore[invalid-param-spec, bad-assignment]
    print(f"task: {task!r}")
    print(f"predicted chunk: {len(actions)} steps, {len(actions[0])} joints each")
    for step in actions[:5]:
        print("  " + ", ".join(f"{value:+.1f}" for value in step))
