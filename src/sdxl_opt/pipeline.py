"""Load SDXL pipeline with configurable compression methods applied."""

import logging
from dataclasses import dataclass

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    LCMScheduler,
    AutoencoderTiny,
    BitsAndBytesConfig,
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
    deepcache_interval: int = 2

    # Cross-attention gating
    use_tgate: bool = False
    tgate_stop_step: int = 10

    # Step reduction
    use_lcm_lora: bool = False
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    # Compilation
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"

    # VAE
    use_tiny_vae: bool = False

    # Pipeline-level
    enable_model_cpu_offload: bool = False
    enable_vae_slicing: bool = False
    dtype: str = "fp16"

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


def load_pipeline(config: CompressionConfig):
    """Load SDXL pipeline and apply all compression methods from config."""
    logger.info(f"Loading pipeline: {config.name} [{config.short_label()}]")

    if config.quantize_unet:
        # -----------------------------------------------------------
        # Load UNet SEPARATELY with quantization, then pass to pipeline.
        # This is the reliable approach — avoids PipelineQuantizationConfig
        # issues across diffusers versions.
        # -----------------------------------------------------------
        if config.quantize_unet == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif config.quantize_unet == "nf4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=config.torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            raise ValueError(f"Unknown quantization: {config.quantize_unet}")

        logger.info(f"Loading UNet with {config.quantize_unet} quantization")
        unet = UNet2DConditionModel.from_pretrained(
            MODEL_ID,
            subfolder="unet",
            quantization_config=bnb_config,
            torch_dtype=config.torch_dtype,
        )

        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            unet=unet,
            torch_dtype=config.torch_dtype,
            variant="fp16" if config.dtype == "fp16" else None,
        )
        # Move non-quantized components to CUDA
        # (quantized UNet is already on CUDA via bitsandbytes)
        pipe.text_encoder = pipe.text_encoder.to("cuda")
        pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda")
        pipe.vae = pipe.vae.to("cuda")

    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=config.torch_dtype,
            variant="fp16" if config.dtype == "fp16" else None,
        ).to("cuda")

    # --- LCM-LoRA ---
    if config.use_lcm_lora:
        logger.info("Applying LCM-LoRA for step reduction")
        pipe.load_lora_weights(LCM_LORA_ID)
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # --- Tiny VAE ---
    if config.use_tiny_vae:
        logger.info("Swapping VAE decoder for TinyVAE (taesdxl)")
        pipe.vae = AutoencoderTiny.from_pretrained(TINY_VAE_ID, torch_dtype=config.torch_dtype)
        pipe.vae = pipe.vae.to("cuda")

    # --- Pipeline-level optimizations ---
    if config.enable_vae_slicing:
        pipe.enable_vae_slicing()
    if config.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()

    # --- torch.compile ---
    if config.use_torch_compile:
        logger.info(f"Compiling UNet with mode={config.compile_mode}")
        pipe.unet = torch.compile(pipe.unet, mode=config.compile_mode)

    return pipe


def generate_images(
    pipe,
    config: CompressionConfig,
    prompts: list[str],
    generator=None,
    height: int = 1024,
    width: int = 1024,
) -> list:
    """Generate images from a list of prompts, handling DeepCache inline."""
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

        result = pipe(**call_kwargs)
        images.append(result.images[0])

    if deepcache_helper is not None:
        deepcache_helper.disable()

    return images
