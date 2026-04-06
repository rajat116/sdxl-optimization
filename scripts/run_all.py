#!/usr/bin/env python3
"""Run the full SDXL compression experiment suite from configs/experiments.yaml."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add src to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sdxl_opt.pipeline import CompressionConfig
from sdxl_opt.benchmark import benchmark_suite
from sdxl_opt.evaluate import compute_clip_scores, save_comparison_grid, save_individual_images
from sdxl_opt.pareto import plot_pareto, plot_speedup_bar, plot_memory_comparison
from sdxl_opt.utils import setup_logging, get_gpu_info, seed_everything

logger = logging.getLogger("sdxl_opt")


def load_configs(yaml_path: str) -> tuple[dict, list[CompressionConfig]]:
    """Parse YAML config into CompressionConfig objects."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    eval_settings = raw.get("eval", {})
    configs = []
    for entry in raw["configs"]:
        name = entry.pop("name")
        entry.pop("description", None)  # optional field, not in dataclass
        configs.append(CompressionConfig(name=name, **entry))

    return eval_settings, configs


def main():
    parser = argparse.ArgumentParser(description="SDXL Compression Experiment Suite")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Path to YAML config")
    parser.add_argument("--results-dir", default="results", help="Output directory")
    parser.add_argument("--n-runs", type=int, default=None, help="Override number of benchmark runs")
    parser.add_argument("--skip-eval", action="store_true", help="Skip CLIP score evaluation")
    parser.add_argument("--only", nargs="+", help="Only run these config names")
    args = parser.parse_args()

    setup_logging("INFO")

    # GPU info
    gpu = get_gpu_info()
    logger.info(f"GPU: {gpu}")

    # Load configs
    eval_settings, configs = load_configs(args.config)

    if args.only:
        configs = [c for c in configs if c.name in args.only]
        logger.info(f"Filtered to {len(configs)} configs: {[c.name for c in configs]}")

    n_runs = args.n_runs or eval_settings.get("n_runs", 3)
    prompts = eval_settings.get("prompts")

    logger.info(f"Running {len(configs)} configurations × {n_runs} runs")

    # --- Benchmark all configs ---
    df, all_images = benchmark_suite(
        configs,
        prompts=prompts,
        n_runs=n_runs,
        n_warmup=eval_settings.get("n_warmup", 2),
        results_dir=args.results_dir,
    )

    # --- CLIP score evaluation ---
    if not args.skip_eval and prompts:
        logger.info("Computing CLIP scores...")
        for config in configs:
            if config.name in all_images:
                imgs = all_images[config.name]
                mean, std, _ = compute_clip_scores(imgs, prompts[: len(imgs)])
                df.loc[df["config"] == config.name, "clip_score"] = mean
        df.to_csv(f"{args.results_dir}/benchmark_results.csv", index=False)

    # --- Save images ---
    for config_name, imgs in all_images.items():
        save_individual_images(imgs, config_name, args.results_dir)
    if prompts:
        save_comparison_grid(all_images, prompts, f"{args.results_dir}/comparison_grid.png")

    # --- Generate plots ---
    if "clip_score" in df.columns and df["clip_score"].notna().any():
        plot_pareto(df, f"{args.results_dir}/pareto_frontier.png")
    plot_speedup_bar(df, output_path=f"{args.results_dir}/speedup_bar.png")
    plot_memory_comparison(df, output_path=f"{args.results_dir}/memory_comparison.png")

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    logger.info(f"All results saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
