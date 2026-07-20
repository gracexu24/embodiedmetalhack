"""Reference MolmoAct2 inference server for deploy_policy.py.

Runs on the GPU box (NOT in a pixi environment -- it needs torch + a CUDA card)::

    pip install torch transformers pillow numpy
    python policy_server_molmoact2.py --checkpoint <your-hf-user>/molmoact2_my_task --port 8080

Serves the tiny JSON contract deploy_policy.py speaks: POST /act with
``{"instruction", "state", "images": {name: <base64 jpeg>}}``, returns
``{"actions": [[...], ...]}`` -- one absolute-joint-pose chunk per request.

The model call is the Hugging Face ``predict_action`` API; if your checkpoint's model
card documents different arguments (e.g. a different ``norm_tag``), adjust here -- see
https://github.com/allenai/molmoact2/tree/main/examples for the upstream servers this
mirrors.

The public ``allenai/MolmoAct2-SO100_101`` checkpoint was trained on the pre-LeRobot-0.5.0
SO-100/101 joint convention (see LeRobot's ``lerobot/MolmoAct2-SO100_101-LeRobot`` docs,
"Joint frame transform (SO-100/101 zero-shot)"): flip shoulder_lift's sign, shift
shoulder_lift + elbow_flex by 90 degrees. ``--joint-fix`` applies this (default on); turn
it off with ``--no-joint-fix`` once you're serving your own fine-tune (from
finetune_modal_molmoact2.py), which is trained on this repo's convention directly and
does NOT need the correction -- applying it twice would be as wrong as not applying it.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Order matches this repo's motor order: shoulder_pan, shoulder_lift, elbow_flex,
# wrist_flex, wrist_roll, gripper. See the joint-fix docstring note above.
JOINT_SIGNS = (1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
JOINT_OFFSETS = (0.0, 90.0, 90.0, 0.0, 0.0, 0.0)


def load_model(checkpoint: str, device: str):
    import torch  # pyrefly: ignore[missing-import] - GPU box only, not a pixi env
    from transformers import AutoModelForImageTextToText, AutoProcessor  # pyrefly: ignore[missing-import] - GPU box only

    print(f"loading {checkpoint} on {device} (bf16) ...")
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    print("model ready")
    return processor, model


def make_handler(processor, model, norm_tag: str, num_steps: int, joint_fix: bool, action_mode: str):
    import numpy as np
    import torch  # pyrefly: ignore[missing-import] - GPU box only, not a pixi env
    from PIL import Image  # pyrefly: ignore[missing-import] - GPU box only

    signs = np.asarray(JOINT_SIGNS, dtype=np.float32)
    offsets = np.asarray(JOINT_OFFSETS, dtype=np.float32)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server API
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                images = [Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB") for data in payload["images"].values()]
                state = np.asarray(payload["state"], dtype=np.float32)
                if joint_fix:
                    state = signs * state + offsets
                with torch.inference_mode():
                    output = model.predict_action(
                        processor=processor,
                        images=images,
                        task=payload["instruction"],
                        state=state,
                        norm_tag=norm_tag,
                        inference_action_mode=action_mode,
                        num_steps=num_steps,
                    )
                # predict_action returns a MolmoAct2ActionOutput, not a plain array; .actions
                # is (batch, num_steps, joints) -- batch is always 1 here (one observation in).
                actions = output.actions[0].to(torch.float32).cpu().numpy()
                if joint_fix:
                    actions = signs * (actions - offsets)
                body = json.dumps({"actions": actions.tolist()}).encode()
                self.send_response(200)
            except Exception as error:  # surface the failure to the client instead of a hung arm loop
                body = json.dumps({"error": f"{type(error).__name__}: {error}"}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="allenai/MolmoAct2-SO100_101", help="HF repo id or local path of the fine-tuned checkpoint")
    parser.add_argument("--norm-tag", default="so100_so101_molmoact2", help="normalization tag the checkpoint was trained with")
    parser.add_argument("--num-steps", type=int, default=10, help="flow-matching denoising steps (--action-mode continuous only)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--joint-fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the old-convention joint-frame correction (on by default; turn off for your own fine-tune)",
    )
    parser.add_argument(
        "--action-mode",
        choices=["continuous", "discrete"],
        default="continuous",
        help="continuous (flow-matching, works) or discrete (action tokens -- config.json says action_mode=both, "
        "but this trust_remote_code path reproducibly raises 'requires an action_tokenizer input'; unsolved, don't use yet)",
    )
    args = parser.parse_args()

    processor, model = load_model(args.checkpoint, args.device)
    server = ThreadingHTTPServer(
        ("0.0.0.0", args.port), make_handler(processor, model, args.norm_tag, args.num_steps, args.joint_fix, args.action_mode)
    )
    print(f"serving /act on port {args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
