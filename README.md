# SDXL Optimization Study

A systematic compression and optimization study of [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), treating the model as a **system of components** (text encoders, UNet, VAE) and optimizing each axis independently before combining methods on a Pareto frontier.

> Built for Pruna AI Technical Interview — Rajat Gupta, April 2026

---

## Key Results

| Configuration | Latency (s) | Peak VRAM (GB) | CLIP Score | Speedup |
|---|---|---|---|---|
| Baseline (fp32, 50 steps) | — | — | — | 1.0× |
| + torch.compile | — | — | — | — |
| + INT8 UNet | — | — | — | — |
| + NF4 UNet | — | — | — | — |
| + DeepCache (N=2) | — | — | — | — |
| + TGATE | — | — | — | — |
| + LCM-LoRA (8 steps) | — | — | — | — |
| + Tiny VAE decoder | — | — | — | — |
| **Best combo** | — | — | — | — |

*Results filled after running experiments on A100 (Colab Pro+)*

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
│  │   - SelfAttn      │  ◄── This is where 85%+ of      │
│  │   - ResBlocks     │      compute lives               │
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

## Project Structure

```
sdxl-optimization/
├── src/sdxl_opt/
│   ├── pipeline.py        # Pipeline loader with compression configs
│   ├── compress.py        # All compression method implementations
│   ├── benchmark.py       # Timing, memory, throughput measurement
│   ├── evaluate.py        # CLIP score, image quality metrics
│   ├── pareto.py          # Pareto frontier analysis
│   └── utils.py           # Helpers, logging, seeds
├── configs/
│   └── experiments.yaml   # All experiment configurations
├── notebooks/
│   ├── 01_compression_study.ipynb   # Main experiment notebook (Colab-ready)
│   └── 02_deploy_demo.ipynb        # LitServe deployment demo
├── scripts/
│   ├── run_all.py         # Run full experiment suite
│   └── generate_report.py # Auto-generate results tables & plots
├── server/
│   └── serve.py           # LitServe API server
├── results/               # Auto-populated with metrics & images
├── docs/
│   └── presentation.md    # Slide content
├── Makefile               # One-command execution
├── requirements.txt
└── pyproject.toml
```

## Quick Start

### Option A: Google Colab (Recommended)

Open `notebooks/01_compression_study.ipynb` in Colab with A100 runtime.

### Option B: Local / Cloud

```bash
git clone https://github.com/rajatgupta/sdxl-optimization.git
cd sdxl-optimization
pip install -e .
make run          # runs full experiment suite
make serve        # starts LitServe API
make report       # generates results report
```

## Approach & Philosophy

Most optimization studies apply one method and report a number. This study takes a **systems approach**:

1. **Profile first** — understand where time and memory are spent before optimizing
2. **Per-component analysis** — text encoders, UNet, and VAE have different computational profiles and respond differently to compression
3. **Orthogonal methods** — quantization, caching, step reduction, and compilation are largely independent axes; we explore them individually then stack
4. **Pareto-optimal selection** — there is no single "best" compression; we map the quality–speed–memory frontier and recommend configurations for different deployment scenarios
5. **Statistical rigor** — multiple runs with std dev, not single-shot numbers

## Compression Methods

| Method | Target | Mechanism | Expected Impact |
|---|---|---|---|
| INT8 quantization | UNet weights | Post-training weight-only quantization | ~1.5-2× memory reduction |
| NF4 quantization | UNet weights | 4-bit NormalFloat via BitsAndBytes | ~3-4× memory reduction |
| DeepCache | UNet inference | Cache & reuse high-level features across steps | ~1.5-2× speedup |
| TGATE | UNet cross-attention | Gate out cross-attention after early steps | ~1.2-1.5× speedup |
| LCM-LoRA | Scheduler + UNet | Distilled consistency model, fewer steps | ~5-8× speedup |
| torch.compile | Full pipeline | Graph capture & kernel fusion | ~1.2-1.5× speedup |
| Tiny VAE | VAE decoder | Smaller decoder architecture | Marginal latency + memory |

## API Deployment

The optimized model is served via [LitServe](https://github.com/Lightning-AI/LitServe) with:

- Async inference with request batching
- Health check endpoint
- Configurable compression preset (speed / balanced / quality)
- Proper error handling and input validation

```bash
# Start server
make serve

# Test inference
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a photo of an astronaut riding a horse on mars"}'
```

## License

MIT
