#!/usr/bin/env python3
"""Generate a markdown report from benchmark results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd


def main():
    results_dir = Path("results")
    csv_path = results_dir / "benchmark_results.csv"

    if not csv_path.exists():
        print("No results found. Run experiments first: make run")
        return

    df = pd.read_csv(csv_path)

    # Compute speedup
    baseline = df.loc[df["config"] == "baseline", "latency_mean_s"].values
    if len(baseline) > 0:
        df["speedup"] = (baseline[0] / df["latency_mean_s"]).round(2)

    report = []
    report.append("# SDXL Compression Study — Results Report\n")
    report.append("## Benchmark Results\n")
    report.append(df.to_markdown(index=False))
    report.append("\n")

    # Best configs
    report.append("## Recommendations\n")
    if "speedup" in df.columns:
        fastest = df.loc[df["speedup"].idxmax()]
        report.append(f"**Fastest:** {fastest['config']} ({fastest['speedup']}× speedup)\n")

    if "clip_score" in df.columns and df["clip_score"].notna().any():
        best_quality = df.loc[df["clip_score"].idxmax()]
        report.append(f"**Best quality:** {best_quality['config']} (CLIP: {best_quality['clip_score']:.4f})\n")

    lowest_mem = df.loc[df["peak_vram_gb"].idxmin()]
    report.append(f"**Lowest memory:** {lowest_mem['config']} ({lowest_mem['peak_vram_gb']:.1f} GB)\n")

    report_text = "\n".join(report)
    report_path = results_dir / "REPORT.md"
    report_path.write_text(report_text)
    print(report_text)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
