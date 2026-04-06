"""LitServe API server for optimized SDXL inference.

Three deployment presets tuned from benchmark results:
  - speed:    LCM 4 steps — 6.7× faster, no quality loss
  - balanced: LCM 8 steps + TinyVAE — 5.5× faster, 25% less VRAM
  - quality:  DeepCache interval=2 — 1.7× faster, zero quality degradation

Usage:
    python server/serve.py --preset speed --port 8000

    curl -X POST http://localhost:8000/predict \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "a photo of an astronaut riding a horse on mars"}'
"""

import base64
import io
import time
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import litserve as ls

from sdxl_opt.pipeline import CompressionConfig, load_pipeline, generate_images
from sdxl_opt.utils import seed_everything, setup_logging

logger = logging.getLogger("sdxl_opt")

PRESETS = {
    "speed": CompressionConfig(
        name="speed", dtype="fp16", use_lcm_lora=True,
        num_inference_steps=4, guidance_scale=1.5,
    ),
    "balanced": CompressionConfig(
        name="balanced", dtype="fp16", use_lcm_lora=True,
        use_tiny_vae=True, num_inference_steps=8, guidance_scale=1.5,
    ),
    "quality": CompressionConfig(
        name="quality", dtype="fp16", use_deepcache=True,
        deepcache_interval=2, num_inference_steps=50, guidance_scale=7.5,
    ),
}


class SDXLServingAPI(ls.LitAPI):
    def __init__(self, preset="speed"):
        super().__init__()
        self.preset = preset

    def setup(self, device):
        self.config = PRESETS[self.preset]
        logger.info(f"Loading SDXL with '{self.preset}' preset...")
        self.pipe = load_pipeline(self.config)
        gen = seed_everything(0)
        generate_images(self.pipe, self.config, ["warmup"],
                        generator=gen, height=512, width=512)
        torch.cuda.synchronize()
        logger.info("Server ready.")

    def decode_request(self, request):
        return {
            "prompt": request["prompt"],
            "seed": request.get("seed", 42),
            "height": min(request.get("height", 1024), 1024),
            "width": min(request.get("width", 1024), 1024),
        }

    def predict(self, inputs):
        gen = seed_everything(inputs["seed"])
        t0 = time.perf_counter()
        imgs = generate_images(self.pipe, self.config, [inputs["prompt"]],
                               generator=gen, height=inputs["height"],
                               width=inputs["width"])
        return {"image": imgs[0], "latency_s": time.perf_counter() - t0}

    def encode_response(self, output):
        buf = io.BytesIO()
        output["image"].save(buf, format="PNG")
        return {
            "image_base64": base64.b64encode(buf.getvalue()).decode(),
            "latency_s": round(output["latency_s"], 3),
            "preset": self.preset,
            "steps": self.config.num_inference_steps,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="speed")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    setup_logging("INFO")
    server = ls.LitServer(SDXLServingAPI(preset=args.preset),
                          accelerator="gpu", devices=1, timeout=300)
    server.run(port=args.port)
