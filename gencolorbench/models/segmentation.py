"""
Segmentation model loading and inference.

Wraps SAM2 and GroundingDINO for object detection and segmentation.
"""

import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image
from typing import Tuple, List, Optional

# These imports require being run from within gsam2 directory
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


DEFAULT_CACHE_DIR = "/data/144-1/users/mabutt/gencolorbench/cache"


@dataclass
class SegmentationModels:
    """Container for segmentation models."""
    sam2_predictor: SAM2ImagePredictor
    grounding_processor: AutoProcessor
    grounding_model: AutoModelForZeroShotObjectDetection
    device: str


def load_segmentation_models(
    sam2_checkpoint: str,
    sam2_config: str,
    grounding_model_id: str,
    device: str,
    cache_dir: str = DEFAULT_CACHE_DIR
) -> SegmentationModels:
    """
    Load SAM2 and GroundingDINO models.
    
    Args:
        sam2_checkpoint: Path to SAM2 checkpoint
        sam2_config: SAM2 config name
        grounding_model_id: HuggingFace model ID for GroundingDINO
        device: Device string (cuda:N or cpu)
        cache_dir: Cache directory for models
    
    Returns:
        SegmentationModels container
    """
    print("=" * 60)
    print("Loading segmentation models...")
    print("=" * 60)
    
    print(f"Loading SAM2 from {sam2_checkpoint}...")
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, cache_dir=cache_dir, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model, cache_dir=cache_dir)
    print("✓ SAM2 loaded")
    
    print(f"Loading GroundingDINO: {grounding_model_id}...")
    processor = AutoProcessor.from_pretrained(grounding_model_id, cache_dir=cache_dir)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        grounding_model_id, cache_dir=cache_dir
    ).to(device)
    print("✓ GroundingDINO loaded")
    
    return SegmentationModels(
        sam2_predictor=sam2_predictor,
        grounding_processor=processor,
        grounding_model=grounding_model,
        device=device,
    )


def get_grounding_boxes(
    models: SegmentationModels,
    image: Image.Image,
    text_prompt: str,
    box_threshold: float = 0.35
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Get bounding boxes from GroundingDINO.
    
    Args:
        models: Segmentation models container
        image: PIL Image
        text_prompt: Object text prompt
        box_threshold: Detection confidence threshold
    
    Returns:
        Tuple of (boxes, labels, scores)
    """
    text = text_prompt.lower().strip()
    if not text.endswith("."):
        text = text + "."
    
    inputs = models.grounding_processor(
        images=image, text=text, return_tensors="pt"
    ).to(models.device)
    
    with torch.no_grad():
        outputs = models.grounding_model(**inputs)
    
    results = models.grounding_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_threshold,
        target_sizes=[image.size[::-1]]
    )
    
    if len(results) == 0 or len(results[0]["boxes"]) == 0:
        return np.array([]), [], np.array([])
    
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    labels = results[0]["labels"]
    
    keep = scores >= box_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    if isinstance(labels, list):
        labels = [labels[i] for i in range(len(keep)) if keep[i]]
    
    return boxes, labels, scores


def get_sam2_masks(
    models: SegmentationModels,
    boxes: np.ndarray
) -> np.ndarray:
    """
    Get masks from SAM2 given bounding boxes.
    
    Args:
        models: Segmentation models container
        boxes: Bounding boxes array (N, 4)
    
    Returns:
        Boolean mask array (N, H, W)
    """
    if len(boxes) == 0:
        return np.array([])
    
    masks, scores, logits = models.sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    return masks
