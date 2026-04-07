"""
GenColorBench Image Generation Module.

Provides utilities for generating images from prompts using various T2I models.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import pandas as pd


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class GenerationResult:
    """Container for benchmark generation results."""
    task: str
    color_system: str
    n_prompts: int
    filepath: Optional[str]
    dataframe: pd.DataFrame


# =============================================================================
# Task Display Names (v3)
# =============================================================================

TASK_DISPLAY_NAMES = {
    "cna": "CNA: Color Name Accuracy",
    "ncu_rgb": "NCU: Numeric Color Understanding (RGB)",
    "ncu_hex": "NCU: Numeric Color Understanding (HEX)",
    "ncu": "NCU: Numeric Color Understanding",
    "coa": "COA: Color-Object Association",
    "moc": "MOC: Multi-Object Color Composition",
    "ica": "ICA: Implicit Color Association",
    # Legacy names for backward compatibility
    "color_name_accuracy": "CNA: Color Name Accuracy",
    "numeric_rgb": "NCU: Numeric Color Understanding (RGB)",
    "numeric_hex": "NCU: Numeric Color Understanding (HEX)",
    "color_object_association": "COA: Color-Object Association",
    "multi_object_composition": "MOC: Multi-Object Color Composition",
    "implicit_color_association": "ICA: Implicit Color Association",
}


# =============================================================================
# CSV Name Mappings (v3 <-> Old)
# =============================================================================

# v3 CSV prefix -> old CSV prefix (for backward compatibility)
NEW_TO_OLD_CSV = {
    'cna_l1': 'task1_color_name_iscc_l1',
    'cna_l2': 'task1_color_name_iscc_l2',
    'cna_l3': 'task1_color_name_iscc_l3',
    'cna_css': 'task1_color_name_css',
    'ncu_rgb_l1': 'task2_numeric_rgb_iscc_l1',
    'ncu_rgb_l2': 'task2_numeric_rgb_iscc_l2',
    'ncu_rgb_l3': 'task2_numeric_rgb_iscc_l3',
    'ncu_rgb_css': 'task2_numeric_rgb_css',
    'ncu_hex_css': 'task2_numeric_hex_css',
    'coa_l1': 'task3_color_object_iscc_l1',
    'coa_l2': 'task3_color_object_iscc_l2',
    'coa_l3': 'task3_color_object_iscc_l3',
    'coa_css': 'task3_color_object_css',
    'moc_l1': 'task4_multi_object_iscc_l1',
    'moc_l2': 'task4_multi_object_iscc_l2',
    'moc_l3': 'task4_multi_object_iscc_l3',
    'moc_css': 'task4_multi_object_css',
    'ica_l1': 'task5_implicit_iscc_l1',
    'ica_l2': 'task5_implicit_iscc_l2',
    'ica_l3': 'task5_implicit_iscc_l3',
    'ica_css': 'task5_implicit_css',
}

# Reverse mapping
OLD_TO_NEW_CSV = {v: k for k, v in NEW_TO_OLD_CSV.items()}


def get_task_from_csv(csv_name: str) -> str:
    """
    Extract task name from CSV filename.
    
    Args:
        csv_name: CSV filename (e.g., 'cna_l1.csv', 'task1_color_name_iscc_l1.csv')
        
    Returns:
        Task name ('cna', 'ncu', 'coa', 'moc', 'ica')
    """
    stem = csv_name.replace('.csv', '')
    
    # v3 naming
    if stem.startswith('cna'):
        return 'cna'
    elif stem.startswith('ncu'):
        return 'ncu'
    elif stem.startswith('coa'):
        return 'coa'
    elif stem.startswith('moc'):
        return 'moc'
    elif stem.startswith('ica'):
        return 'ica'
    
    # Old naming
    if 'task1' in stem or 'color_name' in stem:
        return 'cna'
    elif 'task2' in stem or 'numeric' in stem:
        return 'ncu'
    elif 'task3' in stem or 'color_object' in stem:
        return 'coa'
    elif 'task4' in stem or 'multi_object' in stem:
        return 'moc'
    elif 'task5' in stem or 'implicit' in stem:
        return 'ica'
    
    return 'unknown'


def get_color_system_from_csv(csv_name: str) -> str:
    """
    Extract color system from CSV filename.
    
    Args:
        csv_name: CSV filename
        
    Returns:
        Color system ('l1', 'l2', 'l3', 'css')
    """
    stem = csv_name.lower().replace('.csv', '')
    
    if '_l1' in stem or 'iscc_l1' in stem:
        return 'l1'
    elif '_l2' in stem or 'iscc_l2' in stem:
        return 'l2'
    elif '_l3' in stem or 'iscc_l3' in stem:
        return 'l3'
    elif '_css' in stem or 'css' in stem:
        return 'css'
    
    return 'unknown'


# =============================================================================
# Model Registry
# =============================================================================

SUPPORTED_MODELS = {
    'flux-dev': {
        'name': 'FLUX.1-dev',
        'repo': 'black-forest-labs/FLUX.1-dev',
        'type': 'flux',
    },
    'flux-schnell': {
        'name': 'FLUX.1-schnell',
        'repo': 'black-forest-labs/FLUX.1-schnell',
        'type': 'flux',
    },
    'sd3': {
        'name': 'Stable Diffusion 3',
        'repo': 'stabilityai/stable-diffusion-3-medium-diffusers',
        'type': 'sd3',
    },
    'sd3.5': {
        'name': 'Stable Diffusion 3.5',
        'repo': 'stabilityai/stable-diffusion-3.5-large',
        'type': 'sd3',
    },
    'sana': {
        'name': 'SANA',
        'repo': 'Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers',
        'type': 'sana',
    },
    'pixart-alpha': {
        'name': 'PixArt-α',
        'repo': 'PixArt-alpha/PixArt-XL-2-1024-MS',
        'type': 'pixart',
    },
    'pixart-sigma': {
        'name': 'PixArt-Σ',
        'repo': 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
        'type': 'pixart',
    },
}


def get_generator(model_name: str, device: str = 'cuda:0'):
    """
    Get a T2I generator for the specified model.
    
    Args:
        model_name: Model name from SUPPORTED_MODELS
        device: Device to use
        
    Returns:
        Generator function
    """
    from .generation import get_model_pipeline
    
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(SUPPORTED_MODELS.keys())}")
    
    model_info = SUPPORTED_MODELS[model_name]
    pipeline = get_model_pipeline(model_info['repo'], model_info['type'], device)
    
    return pipeline


def generate(
    model_name: str,
    prompt: str,
    num_images: int = 1,
    device: str = 'cuda:0',
    **kwargs
) -> List[Any]:
    """
    Generate images from a prompt.
    
    Args:
        model_name: Model name
        prompt: Text prompt
        num_images: Number of images to generate
        device: Device to use
        **kwargs: Additional generation arguments
        
    Returns:
        List of generated images
    """
    pipeline = get_generator(model_name, device)
    
    images = []
    for _ in range(num_images):
        result = pipeline(prompt, **kwargs)
        images.append(result.images[0])
    
    return images
