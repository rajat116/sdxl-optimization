"""
LitServe API server for optimized SDXL inference.

Three deployment presets:
  - speed:    LCM 4 steps — maximum throughput
  - balanced: LCM 8 steps + TinyVAE — good quality, low memory
  - quality:  DeepCache — minimal quality loss, moderate speedup

Usage:
    python server/serve.py --preset balanced --port 8000

    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"prompt": "a photo of an astronaut riding a horse on mars"}'
"""

import argparse
import base64
import gc
import io
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import litserve as ls

from sdxl_opt.pipeline import CompressionConfig, load_pipeline, generate_images
from sdxl_opt.utils import seed_everything, setup_logging

logger = logging.getLogger("sdxl_opt")

# ---------------------------------------------------------------------------
# Deployment presets (tuned from benchmark results on T4)
# ---------------------------------------------------------------------------
PRESETS = {
    "speed": CompressionConfig(
        name="speed",
        dtype="fp16",
        use_lcm_lora=True,
        num_inference_steps=4,
        guidance_scale=1.5,
    ),
    "balanced": CompressionConfig(
        name="balanced",
        dtype="fp16",
        use_lcm_lora=True,
        use_tiny_vae=True,
        num_inference_steps=8,
        guidance_scale=1.5,
    ),
    "quality": CompressionConfig(
        name="quality",
        dtype="fp16",
        use_deepcache=True,
        deepcache_interval=2,
        num_inference_steps=50,
        guidance_scale=7.5,
    ),
}

PRESET_INFO = {
    "speed": {"description": "Maximum throughput (~0.85s)", "steps": 4, "expected_speedup": "6.7×"},
    "balanced": {"description": "Good quality + low memory (~1.0s)", "steps": 8, "expected_speedup": "5.5×"},
    "quality": {"description": "Minimal quality loss (~3.4s)", "steps": 50, "expected_speedup": "1.7×"},
}


class SDXLServingAPI(ls.LitAPI):
    """LitServe API wrapping the optimized SDXL pipeline."""

    def __init__(self, preset: str = "balanced"):
        super().__init__()
        self.preset = preset

    def setup(self, device: str) -> None:
        """Load the pipeline and run warm-up inference."""
        self.config = PRESETS[self.preset]
        logger.info(f"Loading SDXL with preset: {self.preset}")
        self.pipe = load_pipeline(self.config)

        # Warm up
        logger.info("Running warm-up inference...")
        gen = seed_everything(0)
        generate_images(
            self.pipe, self.config, ["warmup"],
            generator=gen, height=512, width=512,
        )
        torch.cuda.synchronize()
        logger.info("Server ready.")

    def decode_request(self, request: dict) -> dict:
        """Validate and parse incoming request."""
        prompt = request.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Request must include a non-empty 'prompt' string")

        return {
            "prompt": prompt.strip(),
            "negative_prompt": request.get("negative_prompt", ""),
            "seed": request.get("seed", 42),
            "height": min(request.get("height", 1024), 1024),
            "width": min(request.get("width", 1024), 1024),
        }

    def predict(self, inputs: dict) -> dict:
        """Run inference and return generated image."""
        gen = seed_everything(inputs["seed"])

        t0 = time.perf_counter()
        images = generate_images(
            self.pipe, self.config, [inputs["prompt"]],
            generator=gen,
            height=inputs["height"],
            width=inputs["width"],
        )
        latency = time.perf_counter() - t0

        return {"image": images[0], "latency_s": latency}

    def encode_response(self, output: dict) -> dict:
        """Encode PIL image as base64 PNG."""
        buf = io.BytesIO()
        output["image"].save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "image_base64": b64,
            "latency_s": round(output["latency_s"], 3),
            "preset": self.preset,
            "preset_info": PRESET_INFO[self.preset],
            "format": "png",
        }


def main():
    parser = argparse.ArgumentParser(description="SDXL Optimized API Server")
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="balanced",
        help="Compression preset to use",
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    setup_logging("INFO")

    api = SDXLServingAPI(preset=args.preset)
    server = ls.LitServer(
        api,
        accelerator="gpu",
        devices=1,
        timeout=300,
    )
    server.run(port=args.port)


if __name__ == "__main__":
    main()
