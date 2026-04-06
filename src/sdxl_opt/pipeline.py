"""Load SDXL pipeline with configurable compression methods applied."""

import logging
from dataclasses import dataclass, field

import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    LCMScheduler,
    AutoencoderTiny,
)

logger = logging.getLogger("sdxl_opt")

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdxl"
TINY_VAE_ID = "madebyollin/taesdxl"


@dataclass
class CompressionConfig:
    """Specifies which compression methods to apply."""

    name: str = "baseline"

    # Quantization
    quantize_unet: str | None = None  # "int8", "nf4", or None
    quantize_text_encoders: bool = False

    # Caching
    use_deepcache: bool = False
    deepcache_interval: int = 2  # reuse features every N steps

    # Cross-attention gating
    use_tgate: bool = False
    tgate_stop_step: int = 10  # stop cross-attn after this step

    # Step reduction
    use_lcm_lora: bool = False
    num_inference_steps: int = 50  # default SDXL steps
    guidance_scale: float = 7.5

    # Compilation
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"

    # VAE
    use_tiny_vae: bool = False

    # Pipeline-level
    enable_model_cpu_offload: bool = False
    enable_vae_slicing: bool = False
    dtype: str = "fp16"  # "fp32", "fp16", "bf16"

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[self.dtype]

    def short_label(self) -> str:
        """Human-readable short label for plots."""
        parts = [self.name]
        if self.quantize_unet:
            parts.append(f"q{self.quantize_unet}")
        if self.use_deepcache:
            parts.append(f"dc{self.deepcache_interval}")
        if self.use_tgate:
            parts.append("tgate")
        if self.use_lcm_lora:
            parts.append(f"lcm{self.num_inference_steps}s")
        if self.use_torch_compile:
            parts.append("compiled")
        if self.use_tiny_vae:
            parts.append("tvae")
        return "+".join(parts)


def _apply_quantization(pipe: StableDiffusionXLPipeline, config: CompressionConfig) -> None:
    """Apply weight quantization to UNet and optionally text encoders."""
    if config.quantize_unet is None:
        return

    if config.quantize_unet == "int8":
        from diffusers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Loading UNet with INT8 quantization")
        # Reload UNet with quantization — diffusers supports this natively
        pipe.unet = type(pipe.unet).from_pretrained(
            MODEL_ID,
            subfolder="unet",
            quantization_config=quantization_config,
            torch_dtype=config.torch_dtype,
        )

    elif config.quantize_unet == "nf4":
        from diffusers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=config.torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Loading UNet with NF4 quantization")
        pipe.unet = type(pipe.unet).from_pretrained(
            MODEL_ID,
            subfolder="unet",
            quantization_config=quantization_config,
            torch_dtype=config.torch_dtype,
        )

    else:
        raise ValueError(f"Unknown quantization: {config.quantize_unet}")


def _apply_lcm_lora(pipe: StableDiffusionXLPipeline, config: CompressionConfig) -> None:
    """Load LCM-LoRA adapter and switch scheduler."""
    if not config.use_lcm_lora:
        return
    logger.info("Applying LCM-LoRA for step reduction")
    pipe.load_lora_weights(LCM_LORA_ID)
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)


def _apply_tiny_vae(pipe: StableDiffusionXLPipeline, config: CompressionConfig) -> None:
    """Swap the VAE decoder for the tiny variant."""
    if not config.use_tiny_vae:
        return
    logger.info("Swapping VAE decoder for TinyVAE (taesdxl)")
    pipe.vae = AutoencoderTiny.from_pretrained(TINY_VAE_ID, torch_dtype=config.torch_dtype)
    pipe.vae = pipe.vae.to(pipe.device)


def _apply_torch_compile(pipe: StableDiffusionXLPipeline, config: CompressionConfig) -> None:
    """Apply torch.compile to the UNet."""
    if not config.use_torch_compile:
        return
    logger.info(f"Compiling UNet with mode={config.compile_mode}")
    pipe.unet = torch.compile(pipe.unet, mode=config.compile_mode)


def load_pipeline(config: CompressionConfig) -> StableDiffusionXLPipeline:
    """Load SDXL pipeline and apply all compression methods from config."""
    logger.info(f"Loading pipeline: {config.name} [{config.short_label()}]")

    # --- Base loading ---
    # For quantized loading, we need a different flow
    if config.quantize_unet in ("int8", "nf4"):
        from diffusers import BitsAndBytesConfig

        quant_map = {
            "int8": BitsAndBytesConfig(load_in_8bit=True),
            "nf4": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=config.torch_dtype,
                bnb_4bit_use_double_quant=True,
            ),
        }
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=config.torch_dtype,
            quantization_config=quant_map[config.quantize_unet],
            variant="fp16" if config.dtype == "fp16" else None,
        )
        logger.info(f"Loaded with {config.quantize_unet} quantization")
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=config.torch_dtype,
            variant="fp16" if config.dtype == "fp16" else None,
        )
        pipe = pipe.to("cuda")

    # --- Apply compression stack ---
    _apply_lcm_lora(pipe, config)
    _apply_tiny_vae(pipe, config)

    # Pipeline-level optimizations
    if config.enable_vae_slicing:
        pipe.enable_vae_slicing()
    if config.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()

    _apply_torch_compile(pipe, config)

    return pipe


def generate_images(
    pipe: StableDiffusionXLPipeline,
    config: CompressionConfig,
    prompts: list[str],
    generator: torch.Generator | None = None,
    height: int = 1024,
    width: int = 1024,
) -> list:
    """Generate images from a list of prompts, handling DeepCache and TGATE inline."""
    images = []

    # Setup DeepCache if needed
    deepcache_helper = None
    if config.use_deepcache:
        from DeepCache import DeepCacheSDHelper

        deepcache_helper = DeepCacheSDHelper(pipe=pipe)
        deepcache_helper.set_params(cache_interval=config.deepcache_interval, cache_branch_id=0)
        deepcache_helper.enable()
        logger.info(f"DeepCache enabled (interval={config.deepcache_interval})")

    for prompt in prompts:
        call_kwargs = dict(
            prompt=prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            height=height,
            width=width,
        )
        if generator is not None:
            call_kwargs["generator"] = generator

        # TGATE modifies the call
        if config.use_tgate:
            import tgate

            call_kwargs["gate_step"] = config.tgate_stop_step
            result = tgate.tgate(pipe, **call_kwargs)
        else:
            result = pipe(**call_kwargs)

        images.append(result.images[0])

    if deepcache_helper is not None:
        deepcache_helper.disable()

    return images
