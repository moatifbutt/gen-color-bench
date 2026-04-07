"""
Segmentation modules for GenColorBench.

Includes mask generation and post-processing utilities.
"""
from .masks import generate_object_mask, generate_object_mask_with_exclusion, postprocess_mask

__all__ = [
    "generate_object_mask",
    "generate_object_mask_with_exclusion",
    "postprocess_mask",
]
