"""
GenColorBench T2I Model Configurations and Pipeline Loading.

Supports: SD3, SD3.5, FLUX, PixArt, SANA, HunyuanDiT
"""

import os
import torch
from typing import Dict, Any, Optional, Tuple
from PIL import Image

# =============================================================================
# Cache Directory Configuration
# =============================================================================

CACHE_DIR = "/data/144-1/users/mabutt/gencolorbench_v4/cache"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================================================
# Model Configurations
# =============================================================================

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # FLUX models
    "flux-dev": {
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "pipeline": "FluxPipeline",
        "dtype": "bfloat16",
        "default_steps": 28,
        "default_guidance": 3.5,
        "max_sequence_length": 512,
    },
    "flux-schnell": {
        "repo_id": "black-forest-labs/FLUX.1-schnell",
        "pipeline": "FluxPipeline",
        "dtype": "bfloat16",
        "default_steps": 4,
        "default_guidance": 0.0,
        "max_sequence_length": 256,
    },
    
    # SD3 models
    "sd3": {
        "repo_id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "pipeline": "StableDiffusion3Pipeline",
        "dtype": "float16",
        "default_steps": 28,
        "default_guidance": 7.0,
    },
    "sd3.5-medium": {
        "repo_id": "stabilityai/stable-diffusion-3.5-medium",
        "pipeline": "StableDiffusion3Pipeline",
        "dtype": "bfloat16",
        "default_steps": 40,
        "default_guidance": 4.5,
    },
    "sd3.5-large": {
        "repo_id": "stabilityai/stable-diffusion-3.5-large",
        "pipeline": "StableDiffusion3Pipeline",
        "dtype": "bfloat16",
        "default_steps": 40,
        "default_guidance": 4.5,
    },
    "sd3.5-large-turbo": {
        "repo_id": "stabilityai/stable-diffusion-3.5-large-turbo",
        "pipeline": "StableDiffusion3Pipeline",
        "dtype": "bfloat16",
        "default_steps": 4,
        "default_guidance": 0.0,
    },
    
    # PixArt models
    "pixart-alpha": {
        "repo_id": "PixArt-alpha/PixArt-XL-2-1024-MS",
        "pipeline": "PixArtAlphaPipeline",
        "dtype": "float16",
        "default_steps": 20,
        "default_guidance": 4.5,
    },
    "pixart-sigma": {
        "repo_id": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "pipeline": "PixArtSigmaPipeline",
        "dtype": "float16",
        "default_steps": 20,
        "default_guidance": 4.5,
    },
    
    # SANA models
    "sana-1.6b": {
        "repo_id": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        "pipeline": "SanaPipeline",
        "dtype": "bfloat16",
        "default_steps": 20,
        "default_guidance": 5.0,
        "requires_empty_negative": True,
    },
    
    # HunyuanDiT
    "hunyuan-dit": {
        "repo_id": "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
        "pipeline": "HunyuanDiTPipeline",
        "dtype": "float16",
        "default_steps": 50,
        "default_guidance": 6.0,
    },
}


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.float16)


def get_pipeline_class(pipeline_name: str):
    """Dynamically import and return pipeline class."""
    from diffusers import (
        FluxPipeline,
        StableDiffusion3Pipeline,
        PixArtAlphaPipeline,
        PixArtSigmaPipeline,
        SanaPipeline,
        HunyuanDiTPipeline,
    )
    
    pipeline_map = {
        "FluxPipeline": FluxPipeline,
        "StableDiffusion3Pipeline": StableDiffusion3Pipeline,
        "PixArtAlphaPipeline": PixArtAlphaPipeline,
        "PixArtSigmaPipeline": PixArtSigmaPipeline,
        "SanaPipeline": SanaPipeline,
        "HunyuanDiTPipeline": HunyuanDiTPipeline,
    }
    
    return pipeline_map.get(pipeline_name)


def load_pipeline(
    model_name: str,
    model_path: Optional[str] = None,
    device: str = "cuda",
    token: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a T2I pipeline by model name.
    
    Args:
        model_name: Key from MODEL_CONFIGS
        model_path: Override path/repo_id for model (optional)
        device: Target device (cuda, cuda:0, etc.)
        token: HuggingFace token for gated models
        
    Returns:
        Tuple of (pipeline, config_dict)
    """
    if model_name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    config = MODEL_CONFIGS[model_name]
    # Use provided model_path if given, otherwise use repo_id from config
    model_source = model_path if model_path else config["repo_id"]
    pipeline_class = get_pipeline_class(config["pipeline"])
    
    if pipeline_class is None:
        raise ValueError(f"Unknown pipeline type: {config['pipeline']}")
    
    print(f"Loading {model_name} from: {model_source}")
    print(f"Pipeline: {config['pipeline']}, dtype: {config['dtype']}")
    print(f"Cache directory: {CACHE_DIR}")
    
    # Get HF token from environment if not provided
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    # Load pipeline with cache_dir
    pipe = pipeline_class.from_pretrained(
        model_source,
        torch_dtype=get_torch_dtype(config["dtype"]),
        token=hf_token if hf_token else None,
        cache_dir=CACHE_DIR,
    )
    
    pipe = pipe.to(device)
    
    return pipe, config


def generate_image(
    pipe,
    config: Dict[str, Any],
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
) -> Image.Image:
    """
    Generate a single image using the loaded pipeline.
    
    Args:
        pipe: Loaded diffusers pipeline
        config: Model config dict
        prompt: Text prompt
        width: Output width
        height: Output height
        seed: Random seed (optional)
        num_inference_steps: Override default steps
        guidance_scale: Override default guidance
        
    Returns:
        PIL Image
    """
    # Use defaults from config if not specified
    steps = num_inference_steps or config.get("default_steps", 28)
    guidance = guidance_scale if guidance_scale is not None else config.get("default_guidance", 7.0)
    
    # Set up generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # Build generation kwargs
    gen_kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
    }
    
    # Add max_sequence_length for FLUX
    if "max_sequence_length" in config:
        gen_kwargs["max_sequence_length"] = config["max_sequence_length"]
    
    # SANA requires empty negative prompt
    if config.get("requires_empty_negative"):
        gen_kwargs["negative_prompt"] = ""
    
    # Generate
    output = pipe(**gen_kwargs)
    
    return output.images[0]


def list_available_models() -> list:
    """Return list of available model names."""
    return list(MODEL_CONFIGS.keys())


# Alias for backward compatibility
generate_single_image = generate_image


if __name__ == "__main__":
    # Test listing models
    print("Available models:")
    for name in list_available_models():
        config = MODEL_CONFIGS[name]
        print(f"  {name}: {config['repo_id']}")