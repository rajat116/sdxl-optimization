# SDXL Optimization Study

A systematic compression and optimization study of [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), treating the model as a **system of components** (text encoders, UNet, VAE) and optimizing each axis independently before combining methods on a Pareto frontier.

> Built for Pruna AI Technical Interview — Rajat Gupta, April 2026

---

## Key Results (NVIDIA A100, fp16)

| Configuration | Latency (s) | Speedup | Peak VRAM (GB) | CLIP Score | Status |
|---|---|---|---|---|---|
| Baseline (50 steps) | 5.72 | 1.0× | 13.0 | 0.322 | Reference |
| INT8 UNet (BitsAndBytes) | 33.06 | 0.17× | 10.8 | 0.324 | ❌ Slower — dequant overhead |
| NF4 UNet (BitsAndBytes) | 15.95 | 0.36× | 9.6 | 0.324 | ❌ Slower — dequant overhead |
| **HQQ 4-bit** | **5.65** | **1.0×** | **11.3** | **0.317** | ✅ Same speed, 13% less VRAM |
| **HQQ 4-bit + DeepCache** | **3.31** | **1.7×** | **11.6** | **0.318** | ✅ |
| HQQ 4-bit + compile | 36.7 | 0.16× | 11.3 | 0.317 | ⚠️ High warmup variance |
| **HQQ + DeepCache (pruna)** | **5.07** | **1.1×** | **9.5** | **0.323** | ✅ 27% less VRAM |
| DeepCache (N=2) | 3.40 | 1.7× | 13.4 | 0.326 | ✅ |
| DeepCache (N=3) | 2.48 | 2.3× | 13.4 | 0.324 | ✅ |
| **LCM 8-step** | **1.36** | **4.2×** | 13.4 | 0.320 | ✅ |
| **LCM 4-step** | **0.876** | **6.5×** | 13.4 | **0.327** | ✅ Best speed |
| torch.compile | 35.9 | 0.16× | 13.0 | 0.322 | ⚠️ High warmup variance |
| Tiny VAE | 5.56 | 1.0× | 9.3 | 0.328 | ✅ Memory only |
| NF4 + DeepCache | 8.53 | 0.67× | 10.0 | 0.325 | ⚠️ Overhead negates caching |
| LCM + DeepCache | 0.891 | 6.4× | 13.8 | 0.208 | ❌ Quality collapse |
| **LCM + Compile + TinyVAE** | **1.26** | **4.5×** | **9.7** | 0.322 | ✅ Best balanced |
| Full stack (no compile) | 1.51 | 3.8× | 6.7 | 0.128 | ❌ Garbage output |

### Key Insights

- **LCM-LoRA** is the highest-leverage single optimization: 50 → 4 steps, **6.5× speedup**, no quality loss
- **BitsAndBytes quantization was slower** — dequantization overhead at every matmul dominates on A100
- **HQQ 4-bit** matches baseline speed but uses **13% less VRAM** — on smaller GPUs where VRAM is the constraint, this is the right choice. Combined with DeepCache it reaches **1.7× speedup + 11% VRAM reduction**
- **HQQ + DeepCache via pruna library** achieves **1.1× speedup + 27% VRAM reduction**
- **Not all axes compose cleanly** — LCM + DeepCache and full stack both degraded quality significantly
- **torch.compile** has huge warmup variance (std=43s!), not practical for low-batch inference
- **Best speed:** LCM 4-step — 6.5× faster, same quality
- **Best balanced:** LCM + compile + TinyVAE — 4.5× faster, 25% less VRAM, same CLIP score

### Combination Coverage

We tested each optimization axis independently, then explored stacked combinations. Not all combinations are compatible — this table shows what was tested and why some were skipped.

| Combination | Quant | Cache | Compile | LCM | TinyVAE | Tested | Result |
|---|---|---|---|---|---|---|---|
| NF4 + DeepCache | ✅ | ✅ | | | | ✅ | ⚠️ 0.67× — quantization overhead negated caching gains |
| LCM + DeepCache | | ✅ | | ✅ | | ✅ | ❌ CLIP 0.208 — quality collapsed |
| LCM + Compile + TinyVAE | | | ✅ | ✅ | ✅ | ✅ | ✅ **Best balanced: 5.5×, 9.7 GB** |
| Full stack (no compile) | ✅ | ✅ | | ✅ | ✅ | ✅ | ❌ CLIP 0.128 — garbage output |
| Compile + DeepCache | | ✅ | ✅ | | | ❌ | Skipped: CUDA graph conflict (torch.compile uses CUDA graphs which overwrite DeepCache's cached tensors) |
| NF4 + LCM | ✅ | | | ✅ | | ❌ | Not tested — quantization already showed negative speedup as a single axis |
| NF4 + LCM + Compile + TinyVAE | ✅ | | ✅ | ✅ | ✅ | ❌ | Not tested — same reason |

**Why no pruning?** Structured pruning of SDXL's UNet is not supported out-of-the-box in diffusers. It requires custom implementation: identifying which attention heads or conv channels to prune, applying masks, and fine-tuning to recover quality. This is a multi-day research effort, not suitable for a benchmark study — but a promising direction for future work (and exactly the kind of thing Pruna could automate).

### Deployment Presets

| Preset | Config | Latency | Speedup | VRAM | Use Case |
|---|---|---|---|---|---|
| 🚀 **Speed** | LCM 4-step | 0.876s | 6.5× | 13.4 GB | Real-time / interactive |
| ⚖️ **Balanced** | LCM + compile + TinyVAE | 1.26s | 4.5× | 9.7 GB | Production APIs |
| 🎨 **Quality** | DeepCache (N=2) | 3.40s | 1.7× | 13.4 GB | Quality-critical |
| 🧮 **Memory** | HQQ 4-bit + DeepCache | 5.07s | 1.1× | 9.5 GB | VRAM-constrained deployment |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SDXL Pipeline                        │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │Text Enc 1│   │Text Enc 2│   │   VAE    │            │
│  │ CLIP-L   │   │ OpenCLIP │   │ Decoder  │            │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘            │
│       │              │              │                   │
│       └──────┬───────┘              │                   │
│              ▼                      │                   │
│  ┌───────────────────┐              │                   │
│  │   UNet (2.6B)     │──────────────┘                   │
│  │   - CrossAttn     │                                  │
│  │   - SelfAttn      │  ◄── 85%+ of compute             │
│  │   - ResBlocks     │                                  │
│  └───────────────────┘                                  │
└─────────────────────────────────────────────────────────┘

Optimization axes:
  1. Quantization    → reduce precision per-component
  2. Caching         → skip redundant UNet computations
  3. Step reduction  → fewer denoising steps via distilled schedulers
  4. Compilation     → torch.compile graph optimization
  5. VAE swap        → lightweight decoder
  6. Stacking        → combine orthogonal methods
```

---

## Project Structure

```
sdxl-optimization/
├── src/sdxl_opt/
│   ├── pipeline.py        # Pipeline loader with compression configs
│   ├── benchmark.py       # Timing, memory, throughput measurement
│   ├── evaluate.py        # CLIP score, image quality metrics
│   ├── pareto.py          # Pareto frontier analysis & plots
│   └── utils.py           # Helpers, logging, seeds
├── app/
│   └── demo.py            # Gradio interactive demo (shareable URL)
├── server/
│   └── serve.py           # LitServe API server
├── configs/
│   └── experiments.yaml   # All experiment configurations
├── notebooks/
│   ├── 01_compression_study.ipynb   # Main experiment notebook (Colab A100)
│   └── 02_deploy_demo.ipynb        # LitServe + Gradio deployment demo
├── scripts/
│   ├── run_all.py         # Run full experiment suite
│   └── generate_report.py # Auto-generate results tables & plots
├── results/               # Benchmark CSV, plots, images
├── Makefile
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start

### Option A: Google Colab (Recommended)

**Benchmarks:** Open `notebooks/01_compression_study.ipynb` in Colab with A100 runtime.

**Deployment demo:** Open `notebooks/02_deploy_demo.ipynb` in Colab with A100 runtime.

### Option B: Local / Cloud

```bash
git clone https://github.com/rajat116/sdxl-optimization.git
cd sdxl-optimization
pip install -e ".[dev]"
pip install DeepCache "litserve==0.2.5" "gradio>=4.0.0"

# Run benchmarks
make run

# Start LitServe API
python server/serve.py --preset speed --port 8000

# Launch Gradio demo
python app/demo.py
```

---

## API Deployment

### LitServe API

The optimized model is served via [LitServe](https://github.com/Lightning-AI/LitServe) with configurable compression presets.

```bash
# Start server with speed preset
python server/serve.py --preset speed --port 8000

# Test inference
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a photo of an astronaut riding a horse on mars", "seed": 42}'
```

**Response:**
```json
{
  "image_base64": "<base64 PNG>",
  "latency_s": 0.856,
  "preset": "speed",
  "steps": 4
}
```

**Available presets:** `speed` | `balanced` | `quality`

### Interactive Gradio Demo

A full interactive UI with preset comparison and live metrics.

```bash
python app/demo.py
```

Features:
- **Generate tab** — pick preset, type prompt, see image + real-time metrics
- **Compare tab** — generate same prompt with Speed vs Quality side by side
- **Benchmark tab** — full results table from the compression study
- **Shareable public URL** — interviewers can try it live

---

## Approach & Philosophy

Most optimization studies apply one method and report a number. This study takes a **systems approach**:

1. **Profile first** — understand where time and memory are spent before optimizing
2. **Per-component analysis** — text encoders, UNet, and VAE have different computational profiles and respond differently to compression
3. **Orthogonal methods** — quantization, caching, step reduction, and compilation are largely independent axes; we explore them individually then stack
4. **Pareto-optimal selection** — there is no single "best" compression; we map the quality–speed–memory frontier and recommend configurations for different deployment scenarios
5. **Statistical rigor** — multiple runs with std dev, not single-shot numbers
6. **Negative results matter** — documenting what doesn't work (and why) is as valuable as the wins

## Compression Methods

| Method | Target | Mechanism | Result |
|---|---|---|---|
| INT8 quantization | UNet weights | Post-training 8-bit via BitsAndBytes | Memory ↓, Speed ↓ (dequant overhead) |
| NF4 quantization | UNet weights | 4-bit NormalFloat via BitsAndBytes | Memory ↓↓, Speed ↓ (dequant overhead) |
| **HQQ 4-bit** | UNet weights | Native INT4 kernels, no dequant | **Memory ↓ (13%), Speed neutral on A100** |
| DeepCache | UNet inference | Cache & reuse high-level features across steps | 1.7–2.3× speedup |
| LCM-LoRA | Scheduler + UNet | Distilled consistency model, fewer steps | **6.5× speedup** (best single method) |
| torch.compile | UNet | Graph capture & kernel fusion | High warmup variance, impractical for low-batch |
| Tiny VAE | VAE decoder | Smaller decoder architecture | Memory ↓, no speed gain |
| HQQ + DeepCache | UNet weights + inference | Native INT4 + skip redundant blocks | 1.7× speedup + 11% VRAM ↓ |
| HQQ + DeepCache (pruna) | UNet | hqq_diffusers + deepcache | 1.1× speedup + 27% VRAM ↓ |
| Stacking | Multiple | Combine orthogonal methods | Best: LCM + compile + TinyVAE = 4.5× |

## Future Work

- **Hardware sweep** — HQQ gives real speedup on smaller GPUs where VRAM is the bottleneck; A100's FP16 tensor cores are already near-optimal
- **Ternary quantization** — BitNet-style 1.58-bit weights for the UNet
- **Per-block mixed precision** — different UNet blocks have different sensitivity; quantize aggressively where quality impact is low
- **Speculative decoding for diffusion** — use a small model for early denoising steps, full model for final steps
- **HQQ on matched dependency versions** — version pinning in Colab limited performance; expect better results with aligned diffusers + transformers versions

---

## License

MIT
