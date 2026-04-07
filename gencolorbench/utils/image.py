"""
Image loading and processing utilities.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Union, Optional


def load_image(path: Union[str, Path]) -> Tuple[Image.Image, np.ndarray]:
    """
    Load image as both PIL Image and numpy array.
    
    Args:
        path: Path to image file
    
    Returns:
        Tuple of (PIL Image, numpy array in RGB format)
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    try:
        image_pil = Image.open(path).convert("RGB")
        image_np = np.array(image_pil)
        return image_pil, image_np
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}")


def find_image_path(
    images_dir: Path,
    csv_stem: str,
    prompt_id: str,
    image_idx: int
) -> Optional[Path]:
    """
    Find image path with fallback to different directory structures.
    
    Supports:
    - Flat structure: {csv_stem}/{prompt_id}_{image_idx}.png
    - Nested structure: {csv_stem}/{prompt_id}/image_{image_idx}.png
    
    Args:
        images_dir: Base images directory
        csv_stem: CSV file stem (subdirectory name)
        prompt_id: Prompt identifier
        image_idx: Image index (1-based)
    
    Returns:
        Path to image if found, None otherwise
    """
    # Try primary subdirectory
    images_subdir = images_dir / csv_stem
    
    # Fallback: remove _mini/_full suffixes
    if not images_subdir.exists():
        alt_name = csv_stem.replace('_mini', '').replace('_full', '')
        images_subdir = images_dir / alt_name
    
    if not images_subdir.exists():
        return None
    
    # Try flat structure first: {prompt_id}_{image_idx}.png
    image_path = images_subdir / f"{prompt_id}_{image_idx}.png"
    
    if image_path.exists():
        return image_path
    
    # Fallback to nested structure: {prompt_id}/image_{image_idx}.png
    prompt_dir = images_subdir / prompt_id
    image_path = prompt_dir / f"image_{image_idx}.png"
    
    if image_path.exists():
        return image_path
    
    return None
