"""
Base Task Evaluator for GenColorBench.

Provides common functionality for all task evaluators:
- Image loading
- Object segmentation
- Color lookup
- Negative labels lookup
"""

import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from PIL import Image


class BaseTaskEvaluator(ABC):
    """
    Base class for task evaluators.
    
    Provides common methods for segmentation, color lookup, and image loading.
    Subclasses must implement the `evaluate` method.
    """
    
    task_name: str = "base"
    task_display_name: str = "Base Task"
    use_vlm_for_task: bool = False
    
    def __init__(
        self,
        seg_models,  # SegmentationModels
        vlm_models = None,  # Optional VLMModels
        neg_labels_dict: Dict[str, List[str]] = None,
        color_tables: Dict[str, pd.DataFrame] = None,
        neighborhoods: Dict[str, pd.DataFrame] = None,
        color_system: str = "l1",
        jnd: float = 5.0,
        use_vlm: bool = False,
        save_viz: bool = False,
    ):
        """
        Initialize evaluator.
        
        Args:
            seg_models: Segmentation models (GroundingDINO + SAM2)
            vlm_models: VLM models (optional)
            neg_labels_dict: Negative labels for object filtering
            color_tables: Color lookup tables
            neighborhoods: Color neighborhood tables
            color_system: Color system (l1, l2, l3, css)
            jnd: Just Noticeable Difference threshold
            use_vlm: Whether to use VLM for verification
            save_viz: Whether to save visualizations
        """
        self.seg_models = seg_models
        self.vlm_models = vlm_models if use_vlm else None
        self.neg_labels_dict = neg_labels_dict or {}
        self.color_tables = color_tables or {}
        self.neighborhoods = neighborhoods or {}
        self.color_system = color_system
        self.jnd = jnd
        self.use_vlm = use_vlm
        self.save_viz = save_viz
    
    @abstractmethod
    def evaluate(
        self,
        image_path: Path,
        row: pd.Series,
        output_dir: Optional[Path] = None,
        prompt_id: str = "",
        image_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Evaluate a single image.
        
        Args:
            image_path: Path to image file
            row: DataFrame row with prompt data
            output_dir: Output directory for visualizations
            prompt_id: Prompt identifier
            image_idx: Image index
        
        Returns:
            Dictionary with evaluation results
        """
        pass
    
    def load_image(self, image_path: Path) -> Tuple[Image.Image, np.ndarray]:
        """Load image as PIL and numpy array."""
        image_pil = Image.open(image_path).convert('RGB')
        image_np = np.array(image_pil)
        return image_pil, image_np
    
    def get_negative_labels(self, object_name: str) -> List[str]:
        """Get negative labels for an object."""
        obj_key = object_name.lower().strip()
        return self.neg_labels_dict.get(obj_key, [])
    
    def segment_object(
        self,
        image_pil: Image.Image,
        image_np: np.ndarray,
        object_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment an object in the image.
        
        Returns:
            Tuple of (final_mask, original_mask, negative_mask)
        """
        from ..segmentation import generate_object_mask
        
        neg_labels = self.get_negative_labels(object_name)
        
        return generate_object_mask(
            self.seg_models,
            image_pil,
            image_np,
            object_name,
            neg_labels
        )
    
    def get_target_lab(
        self,
        color_name: str,
        row: Optional[pd.Series] = None
    ) -> Optional[np.ndarray]:
        """Get target LAB color from color name or row data."""
        from ..color.matching import get_target_lab
        
        return get_target_lab(
            color_name,
            self.color_tables,
            self.color_system,
            row
        )
    
    def get_neighbors_lab(
        self,
        color_name: str,
        target_lab: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """Get neighbor colors for matching."""
        from ..color import get_color_neighbors
        
        return get_color_neighbors(
            color_name,
            self.color_tables,
            self.neighborhoods,
            self.color_system,
            target_lab
        )
