"""
Object mask generation and post-processing.

Combines GroundingDINO detection with SAM2 segmentation,
including negative filtering and morphological cleanup.
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import morphology, color as skcolor
from typing import Tuple, List

from ..models.segmentation import SegmentationModels, get_grounding_boxes, get_sam2_masks


def postprocess_mask(
    image_np: np.ndarray,
    mask: np.ndarray,
    remove_shadows: bool = True,
    remove_grays: bool = True
) -> np.ndarray:
    """
    Apply post-processing to refine mask - remove shadows, grays, noise.
    
    Args:
        image_np: RGB image as (H, W, 3) array
        mask: Boolean mask as (H, W) array
        remove_shadows: Whether to remove shadow regions
        remove_grays: Whether to remove gray/desaturated regions
    
    Returns:
        Refined boolean mask
    """
    if not mask.any():
        return mask
    
    refined_mask = mask.copy()
    original_area = mask.sum()
    
    if remove_shadows:
        image_float = image_np.astype(np.float32) / 255.0
        hsv = skcolor.rgb2hsv(image_float)
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        color_mask = (sat > 0.12) | (val > 0.20)
        refined_mask = refined_mask & color_mask & (val > 0.1)
    
    if remove_grays:
        image_float = image_np.astype(np.float32) / 255.0
        hsv = skcolor.rgb2hsv(image_float)
        refined_mask = refined_mask & (hsv[:, :, 1] > 0.10)
    
    # Morphological cleanup
    if refined_mask.any():
        mask_uint8 = refined_mask.astype(np.uint8)
        mask_uint8 = morphology.opening(mask_uint8, morphology.disk(2))
        mask_uint8 = morphology.closing(mask_uint8, morphology.disk(3))
        refined_mask = mask_uint8.astype(bool)
    
    # Keep largest component(s)
    if refined_mask.any():
        labeled, num = ndimage.label(refined_mask)
        if num > 0:
            sizes = ndimage.sum(refined_mask, labeled, range(1, num + 1))
            max_size = max(sizes)
            threshold = max_size * 0.05
            new_mask = np.zeros_like(refined_mask)
            for i, size in enumerate(sizes, 1):
                if size >= threshold:
                    new_mask |= (labeled == i)
            refined_mask = new_mask
    
    # Safety check - don't lose too much area
    if refined_mask.sum() < original_area * 0.2:
        return mask
    
    return refined_mask


def generate_object_mask(
    models: SegmentationModels,
    image_pil: Image.Image,
    image_np: np.ndarray,
    object_name: str,
    negative_labels: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate mask for an object with negative filtering.
    
    Pipeline:
    1. Detect object with GroundingDINO
    2. Segment with SAM2
    3. Detect negative labels
    4. Subtract negative regions from object mask
    5. Post-process
    
    Args:
        models: Segmentation models container
        image_pil: PIL Image
        image_np: RGB numpy array (H, W, 3)
        object_name: Target object name
        negative_labels: List of labels to exclude
    
    Returns:
        Tuple of (final_mask, original_mask, negative_mask)
    """
    H, W = image_np.shape[:2]
    
    models.sam2_predictor.set_image(image_np)
    
    # Detect object
    boxes, labels, scores = get_grounding_boxes(models, image_pil, object_name, 0.35)
    
    if len(boxes) == 0:
        empty = np.zeros((H, W), dtype=bool)
        return empty, empty, empty
    
    # Segment
    masks = get_sam2_masks(models, boxes)
    if len(masks) == 0:
        empty = np.zeros((H, W), dtype=bool)
        return empty, empty, empty
    
    # Combine masks
    original_mask = np.zeros((H, W), dtype=bool)
    for m in masks:
        original_mask |= m.astype(bool)
    
    original_area = original_mask.sum()
    if original_area == 0:
        empty = np.zeros((H, W), dtype=bool)
        return empty, empty, empty
    
    # Negative filtering
    negative_mask = np.zeros((H, W), dtype=bool)
    
    for neg_label in negative_labels:
        if not neg_label:
            continue
        
        neg_boxes, _, _ = get_grounding_boxes(models, image_pil, neg_label, 0.25)
        
        if len(neg_boxes) == 0:
            continue
        
        neg_masks = get_sam2_masks(models, neg_boxes)
        
        for neg_m in neg_masks:
            neg_m_bool = neg_m.astype(bool)
            neg_in_obj = neg_m_bool & original_mask
            
            if neg_in_obj.sum() == 0:
                continue
            
            coverage = neg_in_obj.sum() / original_area
            if coverage > 0.95:
                continue
            
            negative_mask |= neg_in_obj
    
    filtered_mask = original_mask & (~negative_mask)
    
    # Safety check
    remaining_ratio = filtered_mask.sum() / original_area if original_area > 0 else 0
    if remaining_ratio < 0.3:
        filtered_mask = original_mask
        negative_mask = np.zeros((H, W), dtype=bool)
    
    final_mask = postprocess_mask(image_np, filtered_mask)
    
    return final_mask, original_mask, negative_mask


def generate_object_mask_with_exclusion(
    models: SegmentationModels,
    image_pil: Image.Image,
    image_np: np.ndarray,
    object_name: str,
    exclusion_mask: np.ndarray,
    negative_labels: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate mask for an object, excluding regions covered by exclusion_mask.
    
    Used for Task 3 sequential masking: gray out main object before
    detecting secondary object.
    
    Args:
        models: Segmentation models container
        image_pil: PIL Image
        image_np: RGB numpy array (H, W, 3)
        object_name: Target object name
        exclusion_mask: Regions to exclude (e.g., main object mask)
        negative_labels: List of labels to exclude
    
    Returns:
        Tuple of (final_mask, original_mask, negative_mask)
    """
    H, W = image_np.shape[:2]
    
    # Create modified image with exclusion region grayed out
    modified_image_np = image_np.copy()
    modified_image_np[exclusion_mask] = 128  # Gray out excluded region
    modified_image_pil = Image.fromarray(modified_image_np)
    
    models.sam2_predictor.set_image(modified_image_np)
    
    # Detect object in modified image
    boxes, labels, scores = get_grounding_boxes(models, modified_image_pil, object_name, 0.35)
    
    if len(boxes) == 0:
        empty = np.zeros((H, W), dtype=bool)
        return empty, empty, empty
    
    # Segment
    masks = get_sam2_masks(models, boxes)
    if len(masks) == 0:
        empty = np.zeros((H, W), dtype=bool)
        return empty, empty, empty
    
    # Combine masks
    original_mask = np.zeros((H, W), dtype=bool)
    for m in masks:
        original_mask |= m.astype(bool)
    
    # Ensure no overlap with exclusion mask
    original_mask = original_mask & (~exclusion_mask)
    
    original_area = original_mask.sum()
    if original_area == 0:
        empty = np.zeros((H, W), dtype=bool)
        return empty, empty, empty
    
    # Negative filtering (same as regular)
    negative_mask = np.zeros((H, W), dtype=bool)
    
    for neg_label in negative_labels:
        if not neg_label:
            continue
        
        neg_boxes, _, _ = get_grounding_boxes(models, modified_image_pil, neg_label, 0.25)
        
        if len(neg_boxes) == 0:
            continue
        
        neg_masks = get_sam2_masks(models, neg_boxes)
        
        for neg_m in neg_masks:
            neg_m_bool = neg_m.astype(bool)
            neg_in_obj = neg_m_bool & original_mask
            
            if neg_in_obj.sum() == 0:
                continue
            
            coverage = neg_in_obj.sum() / original_area
            if coverage > 0.95:
                continue
            
            negative_mask |= neg_in_obj
    
    filtered_mask = original_mask & (~negative_mask)
    
    remaining_ratio = filtered_mask.sum() / original_area if original_area > 0 else 0
    if remaining_ratio < 0.3:
        filtered_mask = original_mask
        negative_mask = np.zeros((H, W), dtype=bool)
    
    # Post-process using original image (not modified)
    final_mask = postprocess_mask(image_np, filtered_mask)
    
    return final_mask, original_mask, negative_mask
