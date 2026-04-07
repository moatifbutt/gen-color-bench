"""
ICA Evaluator: Implicit Color Association

Evaluates whether T2I models understand implicit color references like
"of the same color", "of identical color", "of matching color".

Key: ref_obj extraction does NOT use GT-guided selection - we want to know
the actual dominant color of ref_obj, not cherry-pick pixels closest to target.

Example:
- prompt: "A Pink parrot perches near an elephant of the same color."
- object: parrot (explicitly Pink)
- ref_obj: elephant (implicitly Pink via "same color")
- Both must be Pink for CORRECT

CORRECT = object matches color AND ref_obj matches color
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from PIL import Image

from .base import BaseTaskEvaluator
from ..color import extract_dominant_color, color_matches_target, get_color_neighbors, lab_to_rgb
from ..color.matching import get_target_lab
from ..color.metrics import ciede2000, delta_chroma, mae_hue


class ICAEvaluator(BaseTaskEvaluator):
    """
    ICA: Implicit Color Association
    
    Evaluates:
    1. VLM confirms both object and ref_obj exist
    2. Objects are distinct (non-overlapping masks)
    3. BOTH objects match the target color (implicit color propagation)
    """
    
    task_name = "ica"
    task_display_name = "Implicit Color Association"
    use_vlm_for_task = True
    
    def evaluate(
        self,
        image_path: Path,
        row: pd.Series,
        output_dir: Optional[Path] = None,
        prompt_id: str = "",
        image_idx: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate a single image for ICA."""
        # Load image
        image_pil, image_np = self.load_image(image_path)
        
        # Parse CSV columns
        obj_name = str(row.get('object', '')).strip()
        ref_obj_name = str(row.get('ref_object', '')).strip()
        color_name = str(row.get('color', '')).strip()
        generation_prompt = str(row.get('prompt', ''))
        
        if not obj_name or not ref_obj_name:
            return {
                'success': False,
                'correct': False,
                'error': 'Missing object or ref_object in row',
            }
        
        # Get target color (same for both objects)
        target_lab = self.get_target_lab(color_name, row)
        
        if target_lab is None:
            return {
                'success': False,
                'correct': False,
                'error': f'Target color not found: {color_name}',
                'object': obj_name,
                'ref_object': ref_obj_name,
                'color_name': color_name,
            }
        
        # Get color neighbors
        neighbors_lab = self.get_neighbors_lab(color_name, target_lab)
        
        # === Step 1: VLM Verification ===
        if self.vlm_models is not None:
            vlm_result = self._verify_both_objects_vlm(
                str(image_path), obj_name, ref_obj_name, generation_prompt
            )
            
            if not vlm_result['obj_exists']:
                return {
                    'success': True,
                    'correct': False,
                    'error': 'VLM: Main object not found',
                    'object': obj_name,
                    'ref_object': ref_obj_name,
                    'color_name': color_name,
                    'vlm_result': vlm_result,
                }
            
            if not vlm_result['ref_exists']:
                return {
                    'success': True,
                    'correct': False,
                    'error': 'VLM: Reference object not found',
                    'object': obj_name,
                    'ref_object': ref_obj_name,
                    'color_name': color_name,
                    'vlm_result': vlm_result,
                }
        else:
            vlm_result = {'obj_exists': True, 'ref_exists': True, 'skipped': True}
        
        # === Step 2: Segment main object ===
        mask_obj, _, _ = self.segment_object(image_pil, image_np, obj_name)
        
        if not mask_obj.any():
            return {
                'success': True,
                'correct': False,
                'error': 'Main object segmentation failed',
                'object': obj_name,
                'ref_object': ref_obj_name,
                'color_name': color_name,
                'vlm_result': vlm_result,
            }
        
        # === Step 3: Segment ref_obj with exclusion ===
        mask_ref, _, _ = self._segment_with_exclusion(
            image_pil, image_np, ref_obj_name, mask_obj
        )
        
        if not mask_ref.any():
            return {
                'success': True,
                'correct': False,
                'error': 'Reference object not distinct - same as main object',
                'object': obj_name,
                'ref_object': ref_obj_name,
                'color_name': color_name,
                'vlm_result': vlm_result,
                'obj_mask_pixels': int(mask_obj.sum()),
            }
        
        # === Step 4: IoU Check ===
        iou = self._compute_iou(mask_obj, mask_ref)
        
        if iou > 0.2:
            return {
                'success': True,
                'correct': False,
                'error': f'Objects not distinct (IoU={iou:.2%})',
                'object': obj_name,
                'ref_object': ref_obj_name,
                'color_name': color_name,
                'iou': float(iou),
                'vlm_result': vlm_result,
            }
        
        # === Step 5: Extract Colors ===
        # Main object: WITH GT-guided selection
        obj_color, obj_info = extract_dominant_color(
            image_np, mask_obj,
            chroma_percentile=50.0,
            target_lab=target_lab,
            gt_selection_percentile=10.0
        )
        
        # Reference object: WITHOUT GT-guided selection
        # We want to know the actual dominant color
        ref_color, ref_info = extract_dominant_color(
            image_np, mask_ref,
            chroma_percentile=50.0,
            target_lab=None,  # NO GT-guidance
            gt_selection_percentile=10.0
        )
        
        # === Step 6: Color Matching ===
        obj_matches, obj_metrics = color_matches_target(
            obj_color, target_lab, neighbors_lab, self.jnd
        )
        
        ref_matches, ref_metrics = color_matches_target(
            ref_color, target_lab, neighbors_lab, self.jnd
        )
        
        # === Step 7: Final Evaluation ===
        # CORRECT = BOTH object and ref_obj match the target color
        correct = obj_matches and ref_matches
        
        result = {
            'success': True,
            'correct': correct,
            'object': obj_name,
            'ref_object': ref_obj_name,
            'color_name': color_name,
            'color_system': self.color_system,
            'target_lab': target_lab.tolist(),
            'obj_color': obj_color.tolist(),
            'ref_color': ref_color.tolist(),
            'obj_matches': obj_matches,
            'ref_matches': ref_matches,
            'obj_mask_pixels': int(mask_obj.sum()),
            'ref_mask_pixels': int(mask_ref.sum()),
            'iou': float(iou),
            'obj_metrics': obj_metrics,
            'ref_metrics': ref_metrics,
            'vlm_result': vlm_result,
        }
        
        # Save visualization if enabled
        if self.save_viz and output_dir is not None:
            self._save_visualization(
                image_np, mask_obj, mask_ref,
                obj_color, ref_color, target_lab,
                obj_matches, ref_matches, correct,
                obj_metrics, ref_metrics,
                output_dir, prompt_id, image_idx,
                obj_name, ref_obj_name, color_name
            )
        
        return result
    
    def _verify_both_objects_vlm(
        self,
        image_path: str,
        obj_name: str,
        ref_obj_name: str,
        generation_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify both objects exist using VLM."""
        from ..models.vlm import vlm_check_single_object
        
        obj_result = vlm_check_single_object(
            self.vlm_models, image_path, obj_name, generation_prompt
        )
        obj_exists = obj_result.main_obj_present if obj_result.main_obj_present is not None else False
        
        ref_result = vlm_check_single_object(
            self.vlm_models, image_path, ref_obj_name, generation_prompt
        )
        ref_exists = ref_result.main_obj_present if ref_result.main_obj_present is not None else False
        
        return {
            'obj_exists': obj_exists,
            'ref_exists': ref_exists,
            'both_exist': obj_exists and ref_exists,
            'objects_checked': {
                obj_name: 'yes' if obj_exists else 'no',
                ref_obj_name: 'yes' if ref_exists else 'no',
            },
            'skip_segmentation': not (obj_exists and ref_exists),
        }
    
    def _segment_with_exclusion(
        self,
        image_pil: Image.Image,
        image_np: np.ndarray,
        object_name: str,
        exclusion_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment object while excluding a region."""
        from ..segmentation import generate_object_mask_with_exclusion
        neg_labels = self.get_negative_labels(object_name)
        return generate_object_mask_with_exclusion(
            self.seg_models, image_pil, image_np,
            object_name, exclusion_mask, neg_labels
        )
    
    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union between two masks."""
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        return float(intersection / union) if union > 0 else 0.0
    
    def _save_visualization(
        self,
        image_np: np.ndarray,
        mask_obj: np.ndarray,
        mask_ref: np.ndarray,
        obj_color: np.ndarray,
        ref_color: np.ndarray,
        target_lab: np.ndarray,
        obj_matches: bool,
        ref_matches: bool,
        correct: bool,
        obj_metrics: Dict,
        ref_metrics: Dict,
        output_dir: Path,
        prompt_id: str,
        image_idx: int,
        obj_name: str,
        ref_obj_name: str,
        color_name: str,
    ):
        """Save visualization with black transparent masks and metrics."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: Image and masks (black overlay)
            axes[0, 0].imshow(image_np)
            axes[0, 0].set_title(f'Original\n{obj_name} + {ref_obj_name}')
            axes[0, 0].axis('off')
            
            # Object mask - black overlay
            overlay_obj = image_np.copy().astype(float)
            overlay_obj[mask_obj] = overlay_obj[mask_obj] * 0.5
            axes[0, 1].imshow(overlay_obj.astype(np.uint8))
            axes[0, 1].set_title(f'Object: {obj_name}\n({mask_obj.sum()} px)')
            axes[0, 1].axis('off')
            
            # Reference object mask - black overlay
            overlay_ref = image_np.copy().astype(float)
            overlay_ref[mask_ref] = overlay_ref[mask_ref] * 0.5
            axes[0, 2].imshow(overlay_ref.astype(np.uint8))
            axes[0, 2].set_title(f'Ref: {ref_obj_name}\n({mask_ref.sum()} px)')
            axes[0, 2].axis('off')
            
            # Row 2: Colors with metrics
            target_rgb = lab_to_rgb(target_lab)
            target_patch = np.full((100, 100, 3), target_rgb, dtype=np.uint8)
            axes[1, 0].imshow(target_patch)
            axes[1, 0].set_title(f'Target: {color_name}')
            axes[1, 0].axis('off')
            
            # Object color with metrics
            obj_rgb = lab_to_rgb(obj_color)
            obj_patch = np.full((100, 100, 3), obj_rgb, dtype=np.uint8)
            axes[1, 1].imshow(obj_patch)
            obj_status = '✓' if obj_matches else '✗'
            obj_de = obj_metrics.get('ciede2000', 0)
            obj_dc = obj_metrics.get('delta_chroma', 0)
            obj_dh = obj_metrics.get('mae_hue', 0)
            axes[1, 1].set_title(f'{obj_name} color {obj_status}\nΔE={obj_de:.2f} Δc={obj_dc:.2f} Δh={obj_dh:.1f}°')
            axes[1, 1].axis('off')
            
            # Reference object color with metrics
            ref_rgb = lab_to_rgb(ref_color)
            ref_patch = np.full((100, 100, 3), ref_rgb, dtype=np.uint8)
            axes[1, 2].imshow(ref_patch)
            ref_status = '✓' if ref_matches else '✗'
            ref_de = ref_metrics.get('ciede2000', 0)
            ref_dc = ref_metrics.get('delta_chroma', 0)
            ref_dh = ref_metrics.get('mae_hue', 0)
            axes[1, 2].set_title(f'{ref_obj_name} color {ref_status}\nΔE={ref_de:.2f} Δc={ref_dc:.2f} Δh={ref_dh:.1f}°')
            axes[1, 2].axis('off')
            
            # Overall result
            result_str = '✓ CORRECT' if correct else '✗ INCORRECT'
            fig.suptitle(f'ICA (Implicit Color Association): {result_str}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            viz_path = output_dir / f'{prompt_id}_{image_idx}_viz.png'
            plt.savefig(viz_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not save visualization: {e}")


# Alias for backward compatibility
Task5Evaluator = ICAEvaluator
