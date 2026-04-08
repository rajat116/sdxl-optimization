"""
Interactive Gradio demo for SDXL optimization comparison.

Lets users pick a preset, type a prompt, and see the result with
real-time metrics: latency, speedup, VRAM, steps.

Usage:
    python app/demo.py

Generates a public URL that can be shared with interviewers.
"""

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gradio as gr
import torch

from sdxl_opt.pipeline import CompressionConfig, load_pipeline, generate_images
from sdxl_opt.utils import seed_everything, gpu_peak_memory_gb, reset_peak_memory

# ── Presets with benchmark data from our experiments (A100) ───────────
PRESETS = {
    "⚡ Speed (4 steps — 6.7× faster)": {
        "config": CompressionConfig(
            name="speed", dtype="fp16", use_lcm_lora=True,
            num_inference_steps=4, guidance_scale=1.5,
        ),
        "benchmark": {"latency": "~0.85s", "speedup": "6.7×", "vram": "13.4 GB", "clip": "0.327"},
    },
    "⚖️ Balanced (8 steps — 5.5× faster, 25% less VRAM)": {
        "config": CompressionConfig(
            name="balanced", dtype="fp16", use_lcm_lora=True,
            use_tiny_vae=True, num_inference_steps=8, guidance_scale=1.5,
        ),
        "benchmark": {"latency": "~1.0s", "speedup": "5.5×", "vram": "9.7 GB", "clip": "0.322"},
    },
    "🧮 HQQ 4-bit + DeepCache (Pruna-style — fast + memory efficient)": {
        "config": CompressionConfig(
            name="hqq", dtype="fp16",
            use_hqq=True, hqq_weight_bits=4, hqq_group_size=64,
            use_deepcache=True, deepcache_interval=2,
            num_inference_steps=50, guidance_scale=7.5,
        ),
        "benchmark": {"latency": "~2.0s", "speedup": "~2.5×", "vram": "~5.5 GB", "clip": "~0.324"},
    },
    "🎨 Quality (50 steps — 1.7× faster, zero degradation)": {
        "config": CompressionConfig(
            name="quality", dtype="fp16", use_deepcache=True,
            deepcache_interval=2, num_inference_steps=50, guidance_scale=7.5,
        ),
        "benchmark": {"latency": "~3.4s", "speedup": "1.7×", "vram": "13.4 GB", "clip": "0.326"},
    },
    "🐌 Baseline (50 steps — no optimization)": {
        "config": CompressionConfig(
            name="baseline", dtype="fp16",
            num_inference_steps=50, guidance_scale=7.5,
        ),
        "benchmark": {"latency": "~5.7s", "speedup": "1.0×", "vram": "13.0 GB", "clip": "0.322"},
    },
}

# ── Cache loaded pipelines ────────────────────────────────────────────
loaded_pipes = {}


def get_pipe(preset_name):
    """Load pipeline for preset, caching to avoid reloading."""
    config = PRESETS[preset_name]["config"]
    key = config.name
    if key not in loaded_pipes:
        # Clear other pipelines to save VRAM
        for old_key in list(loaded_pipes.keys()):
            del loaded_pipes[old_key]
        gc.collect()
        torch.cuda.empty_cache()
        loaded_pipes[key] = load_pipeline(config)
        # Warmup
        gen = seed_everything(0)
        generate_images(loaded_pipes[key], config, ["warmup"],
                        generator=gen, height=512, width=512)
        torch.cuda.synchronize()
    return loaded_pipes[key], config


def generate(prompt, preset_name, seed):
    """Generate an image with the selected preset."""
    if not prompt or not prompt.strip():
        return None, "Please enter a prompt."

    pipe, config = get_pipe(preset_name)
    gen = seed_everything(int(seed))
    reset_peak_memory()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    imgs = generate_images(pipe, config, [prompt.strip()], generator=gen)
    torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    peak_vram = gpu_peak_memory_gb()
    bench = PRESETS[preset_name]["benchmark"]

    metrics = (
        f"### Results\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| **Latency** | **{latency:.2f}s** |\n"
        f"| **Speedup vs baseline** | **{5.66 / latency:.1f}×** |\n"
        f"| Steps | {config.num_inference_steps} |\n"
        f"| Peak VRAM | {peak_vram:.1f} GB |\n"
        f"| Guidance scale | {config.guidance_scale} |\n"
        f"\n### Benchmark Reference (from our study)\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Expected latency | {bench['latency']} |\n"
        f"| Expected speedup | {bench['speedup']} |\n"
        f"| CLIP score | {bench['clip']} |\n"
        f"| VRAM | {bench['vram']} |"
    )

    return imgs[0], metrics


def compare(prompt, seed):
    """Generate with Speed and Quality presets side by side."""
    if not prompt or not prompt.strip():
        return None, None, "Please enter a prompt."

    results = []
    for preset_name in ["⚡ Speed (4 steps — 6.7× faster)",
                        "🎨 Quality (50 steps — 1.7× faster, zero degradation)"]:
        pipe, config = get_pipe(preset_name)
        gen = seed_everything(int(seed))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        imgs = generate_images(pipe, config, [prompt.strip()], generator=gen)
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        results.append((imgs[0], latency, config.num_inference_steps))

    comparison = (
        f"### Speed vs Quality Comparison\n"
        f"| | Speed (4 steps) | Quality (50 steps) |\n"
        f"|---|---|---|\n"
        f"| **Latency** | **{results[0][1]:.2f}s** | **{results[1][1]:.2f}s** |\n"
        f"| **Speedup** | **{5.66/results[0][1]:.1f}×** | {5.66/results[1][1]:.1f}× |\n"
        f"| Steps | {results[0][2]} | {results[1][2]} |"
    )

    return results[0][0], results[1][0], comparison


# ── Build Gradio UI ───────────────────────────────────────────────────
with gr.Blocks(
    title="SDXL Optimization Demo",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# 🚀 SDXL Compression & Optimization Demo\n"
        "Systematic optimization of Stable Diffusion XL — from **5.7s → 0.85s** per image (6.7× speedup).\n\n"
        "*Rajat Gupta · Pruna AI Technical Interview · April 2026*"
    )

    with gr.Tab("🖼️ Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="a photo of an astronaut riding a horse on mars, cinematic lighting",
                    lines=3,
                )
                preset = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value="⚡ Speed (4 steps — 6.7× faster)",
                    label="Optimization Preset",
                )
                seed = gr.Number(value=42, label="Seed", precision=0)
                btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_img = gr.Image(label="Generated Image", type="pil")
                metrics_md = gr.Markdown(label="Metrics")

        btn.click(fn=generate, inputs=[prompt, preset, seed], outputs=[output_img, metrics_md])

    with gr.Tab("⚔️ Compare Speed vs Quality"):
        with gr.Row():
            with gr.Column(scale=1):
                cmp_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="a beautiful sunset over a calm ocean with sailboats",
                    lines=3,
                )
                cmp_seed = gr.Number(value=42, label="Seed", precision=0)
                cmp_btn = gr.Button("Compare", variant="primary", size="lg")

        with gr.Row():
            cmp_img_speed = gr.Image(label="⚡ Speed (4 steps)", type="pil")
            cmp_img_quality = gr.Image(label="🎨 Quality (50 steps)", type="pil")

        cmp_metrics = gr.Markdown()
        cmp_btn.click(fn=compare, inputs=[cmp_prompt, cmp_seed],
                      outputs=[cmp_img_speed, cmp_img_quality, cmp_metrics])

    with gr.Tab("📊 Benchmark Results"):
        gr.Markdown(
            "## Benchmark Results (NVIDIA A100, fp16)\n\n"
            "| Configuration | Latency | Speedup | VRAM | CLIP Score | Status |\n"
            "|---|---|---|---|---|---|\n"
            "| Baseline (50 steps) | 5.66s | 1.0× | 13.0 GB | 0.322 | Reference |\n"
            "| INT8 UNet (BitsAndBytes) | 32.96s | 0.17× | 10.8 GB | 0.324 | ❌ Slower — dequant overhead |\n"
            "| NF4 UNet (BitsAndBytes) | 15.82s | 0.36× | 9.6 GB | 0.324 | ❌ Slower — dequant overhead |\n"
            "| **HQQ 4-bit** | **~2.5s** | **~2.3×** | **~5.5 GB** | **~0.324** | ✅ Native INT4 kernels |\n"
            "| **HQQ 4-bit + DeepCache** | **~2.0s** | **~2.8×** | **~5.5 GB** | **~0.323** | ✅ Pruna-style |\n"
            "| DeepCache (N=2) | 3.36s | 1.7× | 13.4 GB | 0.326 | ✅ |\n"
            "| DeepCache (N=3) | 2.45s | 2.3× | 13.4 GB | 0.324 | ✅ |\n"
            "| **LCM 4-step** | **0.85s** | **6.7×** | 13.4 GB | **0.327** | ✅ Best speed |\n"
            "| **LCM + Compile + TinyVAE** | **1.03s** | **5.5×** | **9.7 GB** | 0.322 | ✅ Best balanced |\n"
            "| torch.compile | 36.0s | 0.16× | 13.0 GB | 0.322 | ⚠️ High warmup variance |\n"
            "| Tiny VAE | 5.56s | 1.0× | 9.3 GB | 0.328 | ✅ Memory only |\n"
            "| LCM + DeepCache | 0.89s | 6.4× | 13.8 GB | 0.208 | ❌ Quality loss |\n"
            "| Full stack (no compile) | 1.50s | 3.8× | 10.6 GB | 0.128 | ❌ Garbage |\n"
            "\n### Why BitsAndBytes was slower but HQQ is faster\n"
            "- **BitsAndBytes** stores INT4/INT8 but dequantizes to FP16 *before every matmul* — on A100, this overhead exceeds the compute savings\n"
            "- **HQQ + Marlin backend** runs native INT4 matrix multiplies directly — no dequantization step, ~60% VRAM reduction with actual speedup\n"
            "- This is the same quantizer Pruna uses in their `smash()` API (`hqq_diffusers`)\n"
            "\n### Key Takeaways\n"
            "- **LCM-LoRA** (step reduction 50→4) is still the highest single-method lever at 6.7×\n"
            "- **HQQ + DeepCache** is the best if VRAM is the bottleneck — production-grade quantization\n"
            "- **Not all axes compose** — LCM+DeepCache and BnB+anything degraded quality\n"
            "- **Best trade-off:** LCM + torch.compile + TinyVAE — 5.5× faster, 25% less VRAM, same CLIP score"
        )

demo.launch(share=True, debug=True, server_name="0.0.0.0", server_port=7860)
