"""Shared utilities: seeding, GPU info, logging."""

import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("sdxl_opt")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def seed_everything(seed: int = 42) -> torch.Generator:
    """Set all random seeds and return a torch Generator for reproducible diffusion."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


@dataclass
class GPUInfo:
    name: str
    total_memory_gb: float
    compute_capability: tuple[int, int]

    def __repr__(self) -> str:
        return f"{self.name} ({self.total_memory_gb:.1f} GB, CC {self.compute_capability[0]}.{self.compute_capability[1]})"


def get_gpu_info() -> GPUInfo:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available")
    props = torch.cuda.get_device_properties(0)
    return GPUInfo(
        name=props.name,
        total_memory_gb=props.total_memory / 1e9,
        compute_capability=(props.major, props.minor),
    )


def gpu_memory_allocated_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9


def gpu_memory_reserved_gb() -> float:
    return torch.cuda.memory_reserved() / 1e9


def gpu_peak_memory_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9


def reset_peak_memory() -> None:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


@contextmanager
def track_peak_memory():
    """Context manager that yields a dict with peak_memory_gb after exit."""
    reset_peak_memory()
    result = {}
    yield result
    result["peak_memory_gb"] = gpu_peak_memory_gb()


@contextmanager
def timer():
    """Context manager that yields a dict with elapsed_s after exit."""
    result = {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield result
    torch.cuda.synchronize()
    result["elapsed_s"] = time.perf_counter() - start


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# Standard evaluation prompts — diverse enough to stress-test quality
EVAL_PROMPTS = [
    "a photo of an astronaut riding a horse on mars, high quality, detailed",
    "a beautiful sunset over a calm ocean with sailboats, oil painting style",
    "a corgi wearing a top hat and monocle, portrait photography",
    "a futuristic cityscape at night with neon lights reflecting on wet streets",
    "a plate of sushi on a wooden table, food photography, sharp focus",
    "an ancient library with towering bookshelves and golden light, fantasy art",
    "a red fox sitting in a snowy forest, wildlife photography",
    "abstract geometric patterns in vibrant colors, digital art",
]
