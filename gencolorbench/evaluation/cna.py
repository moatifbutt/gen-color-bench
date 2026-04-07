"""
CNA Evaluator: Color Name Accuracy

Evaluates whether T2I models can generate an object in the color specified
by its linguistic name.

Example:
- prompt: "A pink car"
- Check if the car is actually pink

CORRECT = extracted color matches target color (within JND threshold)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from PIL import Image

from .base import BaseTaskEvaluator
from ..color import extract_dominant_color, color_matches_target, get_color_neighbors, rgb_to_lab, lab_to_rgb
from ..color.matching import lookup_color_rgb, get_target_lab


class CNAEvaluator(BaseTaskEvaluator):
    """
    CNA: Color Name Accuracy
    
    Evaluates whether generated objects match their specified color names.
    Uses GT-guided pixel selection for robust color extraction.
    """
    
    task_name = "cna"
    task_display_name = "Color Name Accuracy"
    use_vlm_for_task = False
    
    def evaluate(
        self,
        image_path: Path,
        row: pd.Series,
        output_dir: Optional[Path] = None,
        prompt_id: str = "",
        image_idx: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate a single image for CNA."""
        # Load image
        image_pil, image_np = self.load_image(image_path)
        
        # Parse CSV columns
        color_name = str(row.get('color', row.get('color_name', row.get('Color', '')))).strip()
        object_name = str(row.get('object', row.get('Object', ''))).strip()
        
        if not color_name:
            return {
                'success': False,
                'correct': False,
                'error': 'Missing color name in row',
            }
        
        # Get target LAB color
        target_lab = self.get_target_lab(color_name, row)
        
        if target_lab is None:
            return {
                'success': False,
                'correct': False,
                'error': f'Color not found in tables: {color_name}',
                'color_name': color_name,
                'object': object_name,
            }
        
        # Get color neighbors for matching
        neighbors_lab = self.get_neighbors_lab(color_name, target_lab)
        
        # Segment object
        mask, orig_mask, neg_mask = self.segment_object(image_pil, image_np, object_name)
        
        if not mask.any():
            return {
                'success': True,
                'correct': False,
                'error': 'Object segmentation failed',
                'color_name': color_name,
                'object': object_name,
            }
        
        # Extract dominant color (GT-guided)
        extracted_lab, extraction_info = extract_dominant_color(
            image_np, mask,
            chroma_percentile=50.0,
            target_lab=target_lab,
            gt_selection_percentile=10.0
        )
        
        # Match color
        matches, metrics = color_matches_target(
            extracted_lab, target_lab, neighbors_lab, self.jnd
        )
        
        result = {
            'success': True,
            'correct': matches,
            'color_name': color_name,
            'object': object_name,
            'color_system': self.color_system,
            'target_lab': target_lab.tolist(),
            'extracted_lab': extracted_lab.tolist(),
            'metrics': metrics,
            'mask_pixels': int(mask.sum()),
            'extraction_info': extraction_info,
        }
        
        # Save visualization if enabled
        if self.save_viz and output_dir is not None:
            self._save_visualization(
                image_np, mask, extracted_lab, target_lab,
                matches, metrics, output_dir, prompt_id, image_idx,
                object_name, color_name
            )
        
        return result
    
    def _save_visualization(
        self,
        image_np: np.ndarray,
        mask: np.ndarray,
        extracted_lab: np.ndarray,
        target_lab: np.ndarray,
        matches: bool,
        metrics: Dict,
        output_dir: Path,
        prompt_id: str,
        image_idx: int,
        object_name: str,
        color_name: str,
    ):
        """Save visualization with black transparent mask and metrics."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Original image
            axes[0].imshow(image_np)
            axes[0].set_title(f'Original\n{object_name}')
            axes[0].axis('off')
            
            # Mask overlay - black with 50% transparency
            overlay = image_np.copy().astype(float)
            overlay[mask] = overlay[mask] * 0.5
            axes[1].imshow(overlay.astype(np.uint8))
            axes[1].set_title(f'Mask\n({mask.sum()} px)')
            axes[1].axis('off')
            
            # Target color
            target_rgb = lab_to_rgb(target_lab)
            target_patch = np.full((100, 100, 3), target_rgb, dtype=np.uint8)
            axes[2].imshow(target_patch)
            axes[2].set_title(f'Target: {color_name}')
            axes[2].axis('off')
            
            # Extracted color with metrics
            extracted_rgb = lab_to_rgb(extracted_lab)
            extracted_patch = np.full((100, 100, 3), extracted_rgb, dtype=np.uint8)
            axes[3].imshow(extracted_patch)
            status = '✓' if matches else '✗'
            de = metrics.get('ciede2000', 0)
            dc = metrics.get('delta_chroma', 0)
            dh = metrics.get('mae_hue', 0)
            axes[3].set_title(f'Extracted {status}\nΔE={de:.2f} Δc={dc:.2f} Δh={dh:.1f}°')
            axes[3].axis('off')
            
            # Overall result
            result_str = '✓ CORRECT' if matches else '✗ INCORRECT'
            fig.suptitle(f'CNA (Color Name Accuracy): {result_str}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            viz_path = output_dir / f'{prompt_id}_{image_idx}_viz.png'
            plt.savefig(viz_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not save visualization: {e}")


# Alias for backward compatibility
Task1Evaluator = CNAEvaluator
