"""
COA Evaluator: Color-Object Association

Evaluates whether T2I models can generate an object in a specific color
while a secondary object exists and does NOT have the same color (no leakage).

Key: sec_obj extraction does NOT use GT-guided selection - we want to know
the actual dominant color of sec_obj, not cherry-pick pixels closest to target.

Evaluation flow:
1. VLM verification: Both main_obj and sec_obj exist in image
2. Sequential segmentation: Segment main_obj, then sec_obj (excluding main region)
3. Distinctness check: Ensure masks don't overlap (not same object)
4. Color matching: main_obj matches target, sec_obj does NOT match target

CORRECT = main_matches AND (NOT sec_matches)
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
from ..segmentation import generate_object_mask, generate_object_mask_with_exclusion


class COAEvaluator(BaseTaskEvaluator):
    """
    COA: Color-Object Association
    
    Evaluates:
    1. VLM confirms both main_obj and sec_obj exist
    2. Objects are distinct (not merged into one)
    3. main_obj is in target color
    4. sec_obj is NOT in target color (no color leakage)
    """
    
    task_name = "coa"
    task_display_name = "Color-Object Association"
    use_vlm_for_task = True
    
    def evaluate(
        self,
        image_path: Path,
        row: pd.Series,
        output_dir: Optional[Path] = None,
        prompt_id: str = "",
        image_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Evaluate a single image for COA.
        """
        # Load image
        image_pil, image_np = self.load_image(image_path)
        
        # Parse CSV columns
        main_obj = str(row.get('main_obj', '')).strip()
        sec_obj = str(row.get('sec_obj', '')).strip()
        color_name = str(row.get('color', '')).strip()
        
        if not main_obj or not sec_obj:
            return {
                'success': False,
                'correct': False,
                'error': 'Missing main_obj or sec_obj in row',
            }
        
        # Get target color
        target_lab = self.get_target_lab(color_name, row)
        
        if target_lab is None:
            return {
                'success': False,
                'correct': False,
                'error': f'Target color not found: {color_name}',
                'main_obj': main_obj,
                'sec_obj': sec_obj,
                'color_name': color_name,
            }
        
        # Get color neighbors
        neighbors_lab = self.get_neighbors_lab(color_name, target_lab)
        
        # Get generation prompt for VLM context
        generation_prompt = str(row.get('prompt', '')) if 'prompt' in row.index else None
        
        # === Step 1: VLM Verification ===
        if self.vlm_models is not None:
            vlm_result = self._verify_objects_vlm(
                str(image_path), main_obj, sec_obj, generation_prompt
            )
            
            if not vlm_result['main_exists']:
                return {
                    'success': True,
                    'correct': False,
                    'error': 'VLM: Main object not found',
                    'main_obj': main_obj,
                    'sec_obj': sec_obj,
                    'color_name': color_name,
                    'vlm_result': vlm_result,
                }
            
            if not vlm_result['sec_exists']:
                return {
                    'success': True,
                    'correct': False,
                    'error': 'VLM: Secondary object not found',
                    'main_obj': main_obj,
                    'sec_obj': sec_obj,
                    'color_name': color_name,
                    'vlm_result': vlm_result,
                }
        else:
            vlm_result = {'main_exists': True, 'sec_exists': True, 'skipped': True}
        
        # === Step 2: Segment main_obj ===
        mask_main, orig_main, neg_main = self.segment_object(
            image_pil, image_np, main_obj
        )
        
        if not mask_main.any():
            return {
                'success': True,
                'correct': False,
                'error': 'Main object segmentation failed',
                'main_obj': main_obj,
                'sec_obj': sec_obj,
                'color_name': color_name,
                'vlm_result': vlm_result,
            }
        
        # === Step 3: Segment sec_obj with exclusion ===
        mask_sec, orig_sec, neg_sec = self._segment_with_exclusion(
            image_pil, image_np, sec_obj, mask_main
        )
        
        if not mask_sec.any():
            return {
                'success': True,
                'correct': False,
                'error': 'Secondary object not distinct - same as main object',
                'main_obj': main_obj,
                'sec_obj': sec_obj,
                'color_name': color_name,
                'vlm_result': vlm_result,
                'main_mask_pixels': int(mask_main.sum()),
            }
        
        # === Step 4: IoU Check ===
        iou = self._compute_iou(mask_main, mask_sec)
        
        if iou > 0.2:
            return {
                'success': True,
                'correct': False,
                'error': f'Objects not distinct (IoU={iou:.2%})',
                'main_obj': main_obj,
                'sec_obj': sec_obj,
                'color_name': color_name,
                'iou': float(iou),
                'vlm_result': vlm_result,
            }
        
        # === Step 5: Extract Colors ===
        # Main object: WITH GT-guided selection
        main_color, main_info = extract_dominant_color(
            image_np, mask_main,
            chroma_percentile=50.0,
            target_lab=target_lab,
            gt_selection_percentile=10.0
        )
        
        # Secondary object: WITHOUT GT-guided selection
        # We want to know the actual dominant color, not cherry-pick target-like pixels
        sec_color, sec_info = extract_dominant_color(
            image_np, mask_sec,
            chroma_percentile=50.0,
            target_lab=None,  # NO GT-guidance
            gt_selection_percentile=10.0
        )
        
        # === Step 6: Color Matching ===
        main_matches, main_metrics = color_matches_target(
            main_color, target_lab, neighbors_lab, self.jnd
        )
        
        sec_matches, sec_metrics = color_matches_target(
            sec_color, target_lab, neighbors_lab, self.jnd
        )
        
        # === Step 7: Final Evaluation ===
        # CORRECT = main matches target AND sec does NOT match target
        correct = main_matches and (not sec_matches)
        
        result = {
            'success': True,
            'correct': correct,
            'main_obj': main_obj,
            'sec_obj': sec_obj,
            'color_name': color_name,
            'color_system': self.color_system,
            'target_lab': target_lab.tolist(),
            'main_color': main_color.tolist(),
            'sec_color': sec_color.tolist(),
            'main_matches': main_matches,
            'sec_matches': sec_matches,
            'color_leakage': sec_matches,
            'main_mask_pixels': int(mask_main.sum()),
            'sec_mask_pixels': int(mask_sec.sum()),
            'iou': float(iou),
            'main_metrics': main_metrics,
            'sec_metrics': sec_metrics,
            'vlm_result': vlm_result,
        }
        
        # Save visualization if enabled
        if self.save_viz and output_dir is not None:
            self._save_visualization(
                image_np, mask_main, mask_sec,
                main_color, sec_color, target_lab,
                main_matches, sec_matches, correct,
                main_metrics, sec_metrics,
                output_dir, prompt_id, image_idx,
                main_obj, sec_obj, color_name
            )
        
        return result
    
    def _verify_objects_vlm(
        self, 
        image_path: str,
        main_obj: str, 
        sec_obj: str,
        generation_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify both objects exist using VLM.
        Uses two simple separate yes/no questions for reliability.
        """
        from ..models.vlm import vlm_check_single_object
        
        main_result = vlm_check_single_object(
            self.vlm_models, image_path, main_obj, generation_prompt
        )
        main_exists = main_result.main_obj_present if main_result.main_obj_present is not None else False
        
        sec_result = vlm_check_single_object(
            self.vlm_models, image_path, sec_obj, generation_prompt
        )
        sec_exists = sec_result.main_obj_present if sec_result.main_obj_present is not None else False
        
        return {
            'main_exists': main_exists,
            'sec_exists': sec_exists,
            'both_exist': main_exists and sec_exists,
            'objects_checked': {
                main_obj: main_result.objects_checked.get(main_obj, 'unknown'),
                sec_obj: sec_result.objects_checked.get(sec_obj, 'unknown'),
            },
            'skip_segmentation': not (main_exists and sec_exists),
        }
    
    def _segment_with_exclusion(
        self,
        image_pil: Image.Image,
        image_np: np.ndarray,
        object_name: str,
        exclusion_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment object while excluding a region."""
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
        mask_main: np.ndarray,
        mask_sec: np.ndarray,
        main_color: np.ndarray,
        sec_color: np.ndarray,
        target_lab: np.ndarray,
        main_matches: bool,
        sec_matches: bool,
        correct: bool,
        main_metrics: Dict,
        sec_metrics: Dict,
        output_dir: Path,
        prompt_id: str,
        image_idx: int,
        main_obj: str,
        sec_obj: str,
        color_name: str,
    ):
        """Save visualization with black transparent masks and metrics."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: Image and masks (black overlay with 50% transparency)
            axes[0, 0].imshow(image_np)
            axes[0, 0].set_title(f'Original\n{main_obj} + {sec_obj}')
            axes[0, 0].axis('off')
            
            # Main object mask - black overlay
            overlay_main = image_np.copy().astype(float)
            overlay_main[mask_main] = overlay_main[mask_main] * 0.5  # Darken
            axes[0, 1].imshow(overlay_main.astype(np.uint8))
            axes[0, 1].set_title(f'Main: {main_obj}\n({mask_main.sum()} px)')
            axes[0, 1].axis('off')
            
            # Secondary object mask - black overlay
            overlay_sec = image_np.copy().astype(float)
            overlay_sec[mask_sec] = overlay_sec[mask_sec] * 0.5  # Darken
            axes[0, 2].imshow(overlay_sec.astype(np.uint8))
            axes[0, 2].set_title(f'Sec: {sec_obj}\n({mask_sec.sum()} px)')
            axes[0, 2].axis('off')
            
            # Row 2: Colors with metrics
            # Target color
            target_rgb = lab_to_rgb(target_lab)
            target_patch = np.full((100, 100, 3), target_rgb, dtype=np.uint8)
            axes[1, 0].imshow(target_patch)
            axes[1, 0].set_title(f'Target: {color_name}')
            axes[1, 0].axis('off')
            
            # Main extracted color with metrics
            main_rgb = lab_to_rgb(main_color)
            main_patch = np.full((100, 100, 3), main_rgb, dtype=np.uint8)
            axes[1, 1].imshow(main_patch)
            main_status = '✓' if main_matches else '✗'
            main_de = main_metrics.get('ciede2000', 0)
            main_dc = main_metrics.get('delta_chroma', 0)
            main_dh = main_metrics.get('mae_hue', 0)
            axes[1, 1].set_title(f'Main color {main_status}\nΔE={main_de:.2f} Δc={main_dc:.2f} Δh={main_dh:.1f}°')
            axes[1, 1].axis('off')
            
            # Secondary extracted color with metrics
            sec_rgb = lab_to_rgb(sec_color)
            sec_patch = np.full((100, 100, 3), sec_rgb, dtype=np.uint8)
            axes[1, 2].imshow(sec_patch)
            sec_status = '✓ (no leakage)' if not sec_matches else '✗ (leakage!)'
            sec_de = sec_metrics.get('ciede2000', 0)
            sec_dc = sec_metrics.get('delta_chroma', 0)
            sec_dh = sec_metrics.get('mae_hue', 0)
            axes[1, 2].set_title(f'Sec color {sec_status}\nΔE={sec_de:.2f} Δc={sec_dc:.2f} Δh={sec_dh:.1f}°')
            axes[1, 2].axis('off')
            
            # Overall result
            result_str = '✓ CORRECT' if correct else '✗ INCORRECT'
            fig.suptitle(f'COA (Color-Object Association): {result_str}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            viz_path = output_dir / f'{prompt_id}_{image_idx}_viz.png'
            plt.savefig(viz_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not save visualization: {e}")


# Alias for backward compatibility
Task3Evaluator = COAEvaluator
