"""
Dominant color extraction using OneHue algorithm with GT-guided selection.

Based on Witzel & Dewis (2022) "Why bananas look yellow: The dominant hue of object colours"
Reference: GenColorBench (arXiv:2510.20586) Section 3.4

Extended with GT-guided pixel selection to handle shading/specularity:
1. OneHue to get dominant hue direction
2. Filter to high-chroma pixels (≥50th percentile)
3. Select top 5% pixels closest to GT (if GT provided)
4. Mean of selected pixels

This approach answers: "If the object was generated in the correct color, 
do we have pixels matching that color?" rather than "What's the average color?"
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from .conversion import rgb_to_lab
from .metrics import ciede2000


def extract_dominant_color(
    image_np: np.ndarray,
    mask: np.ndarray,
    chroma_percentile: float = 50.0,
    target_lab: Optional[np.ndarray] = None,
    gt_selection_percentile: float = 5.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract dominant color using OneHue algorithm with optional GT-guided selection.
    
    Algorithm:
    1. Convert masked pixels to LAB
    2. Check if achromatic (median chroma < 10)
    3. PCA on a*b* chromaticity to find dominant hue direction (PC1)
    4. IQR filtering on PC2 to remove outliers
    5. Chroma percentile filtering to select saturated pixels (≥50th percentile)
    6. If target_lab provided: select top gt_selection_percentile% closest to GT
    7. Return mean of final selected pixels
    
    Args:
        image_np: RGB image as (H, W, 3) array with values in [0, 255]
        mask: Boolean mask as (H, W) array
        chroma_percentile: Percentile for chroma filtering (default 50)
        target_lab: Optional target LAB color for GT-guided selection
        gt_selection_percentile: Percentage of pixels closest to GT to keep (default 5)
    
    Returns:
        Tuple of:
        - dominant_lab: Dominant LAB color (3,)
        - info: Dictionary with extraction metadata
    """
    if not mask.any():
        return np.array([50.0, 0.0, 0.0]), {'error': 'Empty mask', 'is_achromatic': False}
    
    pixels_rgb = image_np[mask]
    if len(pixels_rgb) < 10:
        return np.array([50.0, 0.0, 0.0]), {'error': 'Too few pixels', 'is_achromatic': False}
    
    # Convert to LAB
    pixels_lab = rgb_to_lab(pixels_rgb)
    
    # Compute chroma for each pixel
    chroma = np.sqrt(pixels_lab[:, 1]**2 + pixels_lab[:, 2]**2)
    median_chroma = np.median(chroma)
    is_achromatic = median_chroma < 10.0
    
    # Handle achromatic colors (grays, blacks, whites)
    if is_achromatic:
        low_chroma_mask = chroma < 15.0
        if low_chroma_mask.sum() > 10:
            filtered_pixels_lab = pixels_lab[low_chroma_mask]
        else:
            filtered_pixels_lab = pixels_lab
        
        # GT-guided selection for achromatic
        if target_lab is not None and len(filtered_pixels_lab) > 10:
            filtered_pixels_lab = _select_closest_to_gt(
                filtered_pixels_lab, target_lab, gt_selection_percentile
            )
        
        dominant_lab = np.mean(filtered_pixels_lab, axis=0)
        
        return dominant_lab, {
            'num_pixels': len(pixels_rgb),
            'num_filtered_pixels': len(filtered_pixels_lab),
            'is_achromatic': True,
            'median_chroma': float(median_chroma),
            'gt_guided': target_lab is not None,
        }
    
    # === OneHue Algorithm ===
    
    # PCA on a*b* chromaticity
    chromaticity = pixels_lab[:, 1:3]  # (a*, b*)
    mean_ab = np.mean(chromaticity, axis=0)
    centered = chromaticity - mean_ab
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    if cov.ndim == 0 or np.isnan(cov).any():
        # Fallback for degenerate cases
        dominant_lab = np.array([np.mean(pixels_lab[:, 0]), mean_ab[0], mean_ab[1]])
        return dominant_lab, {
            'num_pixels': len(pixels_rgb),
            'is_achromatic': False,
            'error': 'Covariance computation failed',
        }
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    pc1 = eigenvectors[:, 0]  # Dominant hue direction
    pc2 = eigenvectors[:, 1]  # Orthogonal direction
    
    # IQR filtering on PC2 (remove outliers orthogonal to dominant hue)
    proj_pc2 = centered @ pc2
    q1, q3 = np.percentile(proj_pc2, [25, 75])
    iqr = q3 - q1
    on_dominant_hue = (proj_pc2 >= q1 - 1.5 * iqr) & (proj_pc2 <= q3 + 1.5 * iqr)
    
    # Chroma percentile filtering among dominant hue pixels (≥50th percentile)
    if on_dominant_hue.sum() > 10:
        chroma_threshold = np.percentile(chroma[on_dominant_hue], chroma_percentile)
        high_chroma_mask = (chroma >= chroma_threshold) & on_dominant_hue
    else:
        chroma_threshold = np.percentile(chroma, chroma_percentile)
        high_chroma_mask = chroma >= chroma_threshold
    
    # Select filtered pixels after OneHue
    if high_chroma_mask.sum() > 5:
        filtered_pixels_lab = pixels_lab[high_chroma_mask]
    elif on_dominant_hue.sum() > 5:
        filtered_pixels_lab = pixels_lab[on_dominant_hue]
    else:
        filtered_pixels_lab = pixels_lab
    
    num_after_onehue = len(filtered_pixels_lab)
    
    # === GT-guided selection ===
    # Select top gt_selection_percentile% closest to target
    # This picks the "true color" pixels if the object was generated correctly
    if target_lab is not None and len(filtered_pixels_lab) > 10:
        filtered_pixels_lab = _select_closest_to_gt(
            filtered_pixels_lab, target_lab, gt_selection_percentile
        )
    
    # Dominant color = mean of final selected pixels
    dominant_lab = np.mean(filtered_pixels_lab, axis=0)
    
    return dominant_lab, {
        'num_pixels': len(pixels_rgb),
        'num_after_onehue': num_after_onehue,
        'num_filtered_pixels': len(filtered_pixels_lab),
        'is_achromatic': False,
        'median_chroma': float(median_chroma),
        'chroma_threshold': float(chroma_threshold),
        'gt_guided': target_lab is not None,
        'gt_selection_percentile': gt_selection_percentile if target_lab is not None else None,
    }


def _select_closest_to_gt(
    pixels_lab: np.ndarray,
    target_lab: np.ndarray,
    percentile: float = 5.0
) -> np.ndarray:
    """
    Select the top percentile% of pixels closest to target_lab.
    
    This handles shading/specularity by picking the pixels that 
    best represent the "true" color if the object was generated correctly.
    
    The logic: If an image has a pink bus with shadows, the shadows are dark pink
    while the true body color is light pink. By selecting pixels closest to the 
    target (light pink), we evaluate the "best case" - did the model generate 
    ANY pixels in the correct color?
    
    Args:
        pixels_lab: LAB pixels (N, 3)
        target_lab: Target LAB color (3,)
        percentile: Percentage of closest pixels to keep (default 5%)
    
    Returns:
        Selected LAB pixels (M, 3) where M ≈ N * percentile / 100
    """
    n_pixels = len(pixels_lab)
    
    # Compute CIEDE2000 distance to target for each pixel
    distances = np.array([ciede2000(p, target_lab) for p in pixels_lab])
    
    # Number of pixels to keep
    n_keep = max(5, int(n_pixels * percentile / 100))
    n_keep = min(n_keep, n_pixels)
    
    # Get indices of closest pixels
    closest_indices = np.argsort(distances)[:n_keep]
    
    return pixels_lab[closest_indices]


def extract_dominant_color_simple(
    image_np: np.ndarray,
    mask: np.ndarray,
    chroma_percentile: float = 50.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract dominant color using OneHue algorithm WITHOUT GT-guided selection.
    
    Use this for applications where GT is not available or when you want
    the unbiased dominant color estimate.
    
    Args:
        image_np: RGB image as (H, W, 3) array with values in [0, 255]
        mask: Boolean mask as (H, W) array
        chroma_percentile: Percentile for chroma filtering (default 50)
    
    Returns:
        Tuple of (dominant_lab, info_dict)
    """
    return extract_dominant_color(
        image_np, mask, 
        chroma_percentile=chroma_percentile,
        target_lab=None  # No GT guidance
    )
