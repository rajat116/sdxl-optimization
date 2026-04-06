#!/usr/bin/env python3
"""
LitServe API server for optimized SDXL inference.

Supports three deployment presets:
  - speed:    Maximum throughput (~4 steps, full compression stack)
  - balanced: Good quality/speed trade-off (~8 steps)
  - quality:  Minimal quality loss (~50 steps, caching + compile only)

Usage:
    python server/serve.py --preset balanced --port 8000
    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"prompt": "a photo of an astronaut riding a horse"}'
"""

import argparse
import base64
import io
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import litserve as ls
import torch

from sdxl_opt.pipeline import CompressionConfig, load_pipeline, generate_images
from sdxl_opt.utils import seed_everything, setup_logging

logger = logging.getLogger("sdxl_opt")

# ---------------------------------------------------------------------------
# Deployment presets
# ---------------------------------------------------------------------------
PRESETS: dict[str, CompressionConfig] = {
    "speed": CompressionConfig(
        name="preset_speed",
        dtype="fp16",
        use_lcm_lora=True,
        use_torch_compile=True,
        compile_mode="reduce-overhead",
        use_tiny_vae=True,
        use_deepcache=True,
        deepcache_interval=3,
        num_inference_steps=4,
        guidance_scale=1.5,
    ),
    "balanced": CompressionConfig(
        name="preset_balanced",
        dtype="fp16",
        use_lcm_lora=True,
        use_deepcache=True,
        deepcache_interval=2,
        num_inference_steps=8,
        guidance_scale=1.5,
    ),
    "quality": CompressionConfig(
        name="preset_quality",
        dtype="fp16",
        use_deepcache=True,
        deepcache_interval=2,
        use_torch_compile=True,
        compile_mode="reduce-overhead",
        num_inference_steps=50,
        guidance_scale=7.5,
    ),
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

        # Warm up (critical for torch.compile)
        logger.info("Running warm-up inference...")
        gen = seed_everything(0)
        generate_images(
            self.pipe, self.config, ["warmup"], generator=gen, height=512, width=512
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
            "num_steps": request.get("num_steps", self.config.num_inference_steps),
            "guidance_scale": request.get("guidance_scale", self.config.guidance_scale),
        }

    def predict(self, inputs: dict) -> dict:
        """Run inference and return generated image."""
        gen = seed_everything(inputs["seed"])

        # Override steps/guidance if provided
        config = CompressionConfig(
            **{
                **vars(self.config),
                "num_inference_steps": inputs["num_steps"],
                "guidance_scale": inputs["guidance_scale"],
            }
        )

        t0 = time.perf_counter()
        images = generate_images(
            self.pipe, config, [inputs["prompt"]],
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
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    setup_logging("INFO")

    api = SDXLServingAPI(preset=args.preset)
    server = ls.LitServer(
        api,
        accelerator="gpu",
        devices=1,
        workers_per_device=args.workers,
        timeout=120,
    )
    server.run(port=args.port)


if __name__ == "__main__":
    main()
