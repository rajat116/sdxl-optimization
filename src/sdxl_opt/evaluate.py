"""Evaluate image quality using CLIP score and visual comparison grids."""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("sdxl_opt")


def compute_clip_scores(
    images: list[Image.Image],
    prompts: list[str],
    model_name: str = "openai/clip-vit-large-patch14",
) -> tuple[float, float, list[float]]:
    """
    Compute CLIP score (cosine similarity between image and text embeddings).

    Returns (mean_score, std_score, per_image_scores).
    """
    from transformers import CLIPProcessor, CLIPModel

    logger.info(f"Computing CLIP scores with {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    scores = []
    for img, prompt in zip(images, prompts):
        inputs = processor(text=[prompt], images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Cosine similarity between image and text embeddings
            score = outputs.logits_per_image.item() / 100.0  # Normalize to ~[0, 1]
            scores.append(score)

    del model, processor
    torch.cuda.empty_cache()

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    logger.info(f"  CLIP score: {mean_score:.4f} ± {std_score:.4f}")

    return mean_score, std_score, scores


def save_comparison_grid(
    images_dict: dict[str, list[Image.Image]],
    prompts: list[str],
    output_path: str = "results/comparison_grid.png",
    max_prompts: int = 4,
) -> None:
    """
    Create a visual comparison grid: rows = prompts, columns = configs.
    """
    import matplotlib.pyplot as plt

    config_names = list(images_dict.keys())
    n_prompts = min(len(prompts), max_prompts)
    n_configs = len(config_names)

    fig, axes = plt.subplots(
        n_prompts, n_configs,
        figsize=(4 * n_configs, 4 * n_prompts),
        squeeze=False,
    )

    for col, cfg_name in enumerate(config_names):
        axes[0, col].set_title(cfg_name, fontsize=10, fontweight="bold")
        for row in range(n_prompts):
            ax = axes[row, col]
            if row < len(images_dict[cfg_name]):
                ax.imshow(images_dict[cfg_name][row])
            ax.axis("off")

    # Add prompt labels on the left
    for row in range(n_prompts):
        short_prompt = prompts[row][:50] + "..." if len(prompts[row]) > 50 else prompts[row]
        axes[row, 0].set_ylabel(short_prompt, fontsize=8, rotation=0, labelpad=120, va="center")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison grid saved to {output_path}")


def save_individual_images(
    images: list[Image.Image],
    config_name: str,
    output_dir: str = "results",
) -> list[str]:
    """Save generated images to disk. Returns list of file paths."""
    from .utils import ensure_dir

    img_dir = ensure_dir(f"{output_dir}/images/{config_name}")
    paths = []
    for i, img in enumerate(images):
        path = img_dir / f"sample_{i:02d}.png"
        img.save(path)
        paths.append(str(path))
    return paths
