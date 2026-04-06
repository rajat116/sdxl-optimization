"""Benchmarking: latency, peak VRAM, throughput with warm-up and multiple runs."""

import gc
import logging
import time
from dataclasses import dataclass

import pandas as pd
import torch
import numpy as np

from .pipeline import CompressionConfig, generate_images, load_pipeline
from .utils import (
    EVAL_PROMPTS,
    reset_peak_memory,
    gpu_peak_memory_gb,
    gpu_memory_allocated_gb,
    seed_everything,
)

logger = logging.getLogger("sdxl_opt")


@dataclass
class BenchmarkResult:
    config_name: str
    config_label: str
    num_steps: int
    # Latency (seconds per image)
    latencies: list[float]
    mean_latency_s: float
    std_latency_s: float
    # Memory
    peak_vram_gb: float
    allocated_vram_gb: float
    # Quality (filled later by evaluate module)
    clip_score: float | None = None
    clip_score_std: float | None = None
    # Derived
    throughput_img_per_min: float = 0.0

    def __post_init__(self):
        if self.mean_latency_s > 0:
            self.throughput_img_per_min = 60.0 / self.mean_latency_s

    def to_dict(self) -> dict:
        return {
            "config": self.config_name,
            "label": self.config_label,
            "steps": self.num_steps,
            "latency_mean_s": round(self.mean_latency_s, 3),
            "latency_std_s": round(self.std_latency_s, 3),
            "peak_vram_gb": round(self.peak_vram_gb, 2),
            "throughput_img_min": round(self.throughput_img_per_min, 2),
            "clip_score": round(self.clip_score, 4) if self.clip_score else None,
        }


def warmup(pipe, config: CompressionConfig, n_warmup: int = 2) -> None:
    """Run warm-up inferences to stabilize timings (esp. for torch.compile)."""
    logger.info(f"Warming up ({n_warmup} iterations)...")
    gen = seed_everything(0)
    for _ in range(n_warmup):
        generate_images(
            pipe, config, ["warmup prompt"], generator=gen, height=512, width=512
        )
    torch.cuda.synchronize()
    logger.info("Warm-up complete")


def benchmark_config(
    config: CompressionConfig,
    prompts: list[str] | None = None,
    n_runs: int = 3,
    n_warmup: int = 2,
    height: int = 1024,
    width: int = 1024,
) -> tuple[BenchmarkResult, list]:
    """
    Benchmark a single compression configuration.

    Returns (BenchmarkResult, list_of_generated_images_from_last_run).
    """
    if prompts is None:
        prompts = EVAL_PROMPTS[:4]  # Use 4 prompts by default

    logger.info(f"=== Benchmarking: {config.name} ({config.short_label()}) ===")

    # Load pipeline
    pipe = load_pipeline(config)

    # Warm up
    warmup(pipe, config, n_warmup)

    # Benchmark runs
    latencies = []
    last_images = []
    for run_idx in range(n_runs):
        gen = seed_everything(42)  # Same seed every run for fair comparison
        reset_peak_memory()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        imgs = generate_images(pipe, config, prompts, generator=gen, height=height, width=width)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        per_image = elapsed / len(prompts)
        latencies.append(per_image)
        last_images = imgs

        peak = gpu_peak_memory_gb()
        logger.info(
            f"  Run {run_idx + 1}/{n_runs}: {per_image:.2f}s/img, peak VRAM: {peak:.2f} GB"
        )

    result = BenchmarkResult(
        config_name=config.name,
        config_label=config.short_label(),
        num_steps=config.num_inference_steps,
        latencies=latencies,
        mean_latency_s=float(np.mean(latencies)),
        std_latency_s=float(np.std(latencies)),
        peak_vram_gb=gpu_peak_memory_gb(),
        allocated_vram_gb=gpu_memory_allocated_gb(),
    )

    logger.info(
        f"  => Mean: {result.mean_latency_s:.2f}s ± {result.std_latency_s:.2f}s | "
        f"Peak VRAM: {result.peak_vram_gb:.2f} GB | "
        f"Throughput: {result.throughput_img_per_min:.1f} img/min"
    )

    # Cleanup to free VRAM before next config
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return result, last_images


def benchmark_suite(
    configs: list[CompressionConfig],
    prompts: list[str] | None = None,
    n_runs: int = 3,
    n_warmup: int = 2,
    results_dir: str = "results",
) -> tuple[pd.DataFrame, dict[str, list]]:
    """
    Run benchmarks for all configs. Returns a DataFrame of results
    and a dict mapping config_name -> generated images.
    """
    from .utils import ensure_dir

    ensure_dir(results_dir)

    all_results = []
    all_images = {}

    for config in configs:
        result, images = benchmark_config(
            config, prompts=prompts, n_runs=n_runs, n_warmup=n_warmup
        )
        all_results.append(result.to_dict())
        all_images[config.name] = images

    df = pd.DataFrame(all_results)
    csv_path = f"{results_dir}/benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    return df, all_images
