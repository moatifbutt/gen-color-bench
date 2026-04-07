"""
MOC Evaluator: Multi-Object Color Generation

Evaluates whether T2I models can generate multiple objects, each in a 
different specified color.

CSV columns:
- id, color_count, prompt
- object1, color1, object2, color2, object3, color3 (obj3 may be empty)

CORRECT = ALL objects match their respective colors
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
from PIL import Image

from .base import BaseTaskEvaluator
from ..color import extract_dominant_color, color_matches_target, get_color_neighbors, lab_to_rgb
from ..color.matching import get_target_lab
from ..segmentation import generate_object_mask


class MOCEvaluator(BaseTaskEvaluator):
    """
    MOC: Multi-Object Color Generation
    
    Evaluates:
    1. VLM confirms ALL objects exist
    2. Objects are distinct (non-overlapping masks)
    3. Each object matches its specified color
    """
    
    task_name = "moc"
    task_display_name = "Multi-Object Color Generation"
    use_vlm_for_task = True
    
    def evaluate(
        self,
        image_path: Path,
        row: pd.Series,
        output_dir: Optional[Path] = None,
        prompt_id: str = "",
        image_idx: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate a single image for MOC."""
        # Load image
        image_pil, image_np = self.load_image(image_path)
        
        # Parse CSV columns - get object-color pairs
        color_count = int(row.get('color_count', 2))
        generation_prompt = str(row.get('prompt', ''))
        
        # Build list of (object_name, color_name) pairs
        object_color_pairs = []
        for i in range(1, color_count + 1):
            # Handle both 'object1' and 'obj1' column naming
            obj_col = f'object{i}'
            obj_col_alt = f'obj{i}'
            color_col = f'color{i}'
            
            obj_name = str(row.get(obj_col, row.get(obj_col_alt, ''))).strip()
            color_name = str(row.get(color_col, '')).strip()
            
            if obj_name and color_name:
                object_color_pairs.append((obj_name, color_name))
        
        if len(object_color_pairs) < 2:
            return {
                'success': False,
                'correct': False,
                'error': 'Less than 2 object-color pairs found in row',
                'color_count': color_count,
            }
        
        object_names = [pair[0] for pair in object_color_pairs]
        
        # === Step 1: VLM Verification ===
        if self.vlm_models is not None:
            vlm_result = self._verify_all_objects_vlm(
                str(image_path), object_names, generation_prompt
            )
            
            if not vlm_result['all_exist']:
                missing = vlm_result.get('missing_objects', [])
                return {
                    'success': True,
                    'correct': False,
                    'error': f'VLM: Objects not found: {missing}',
                    'object_color_pairs': object_color_pairs,
                    'vlm_result': vlm_result,
                }
        else:
            vlm_result = {'all_exist': True, 'skipped': True}
        
        # === Step 2: Sequential Segmentation ===
        masks = []
        combined_exclusion_mask = np.zeros(image_np.shape[:2], dtype=bool)
        
        for i, (obj_name, _) in enumerate(object_color_pairs):
            if i == 0:
                mask, _, _ = self.segment_object(image_pil, image_np, obj_name)
            else:
                mask, _, _ = self._segment_with_exclusion(
                    image_pil, image_np, obj_name, combined_exclusion_mask
                )
            
            if not mask.any():
                return {
                    'success': True,
                    'correct': False,
                    'error': f'Segmentation failed for object: {obj_name}',
                    'object_color_pairs': object_color_pairs,
                    'failed_object_idx': i,
                    'vlm_result': vlm_result,
                }
            
            masks.append(mask)
            combined_exclusion_mask = combined_exclusion_mask | mask
        
        # === Step 3: Check Mask Overlaps ===
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                iou = self._compute_iou(masks[i], masks[j])
                if iou > 0.2:
                    return {
                        'success': True,
                        'correct': False,
                        'error': f'Objects {i+1} and {j+1} overlap (IoU={iou:.2%})',
                        'object_color_pairs': object_color_pairs,
                        'iou': float(iou),
                        'vlm_result': vlm_result,
                    }
        
        # === Step 4: Color Extraction & Matching ===
        results_per_object = []
        all_match = True
        
        for i, ((obj_name, color_name), mask) in enumerate(zip(object_color_pairs, masks)):
            target_lab = self.get_target_lab(color_name, row)
            
            if target_lab is None:
                results_per_object.append({
                    'object': obj_name,
                    'target_color': color_name,
                    'error': 'Target color not found',
                    'matches': False,
                })
                all_match = False
                continue
            
            neighbors_lab = self.get_neighbors_lab(color_name, target_lab)
            
            # Extract dominant color (GT-guided)
            extracted_color, extraction_info = extract_dominant_color(
                image_np, mask,
                chroma_percentile=50.0,
                target_lab=target_lab,
                gt_selection_percentile=10.0
            )
            
            matches, metrics = color_matches_target(
                extracted_color, target_lab, neighbors_lab, self.jnd
            )
            
            if not matches:
                all_match = False
            
            results_per_object.append({
                'object': obj_name,
                'target_color': color_name,
                'target_lab': target_lab.tolist(),
                'extracted_lab': extracted_color.tolist(),
                'matches': matches,
                'metrics': metrics,
                'mask_pixels': int(mask.sum()),
            })
        
        correct = all_match
        
        result = {
            'success': True,
            'correct': correct,
            'color_count': color_count,
            'object_color_pairs': object_color_pairs,
            'results_per_object': results_per_object,
            'all_match': all_match,
            'vlm_result': vlm_result,
        }
        
        # Save visualization if enabled
        if self.save_viz and output_dir is not None:
            self._save_visualization(
                image_np, masks, object_color_pairs, results_per_object,
                correct, output_dir, prompt_id, image_idx
            )
        
        return result
    
    def _verify_all_objects_vlm(
        self,
        image_path: str,
        object_names: List[str],
        generation_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify ALL objects exist using VLM (separate calls)."""
        from ..models.vlm import vlm_check_single_object
        
        objects_checked = {}
        missing_objects = []
        
        for obj_name in object_names:
            result = vlm_check_single_object(
                self.vlm_models, image_path, obj_name, generation_prompt
            )
            obj_exists = result.main_obj_present if result.main_obj_present is not None else False
            objects_checked[obj_name] = 'yes' if obj_exists else 'no'
            
            if not obj_exists:
                missing_objects.append(obj_name)
        
        return {
            'all_exist': len(missing_objects) == 0,
            'objects_checked': objects_checked,
            'missing_objects': missing_objects,
            'num_objects': len(object_names),
            'num_found': len(object_names) - len(missing_objects),
        }
    
    def _segment_with_exclusion(
        self,
        image_pil: Image.Image,
        image_np: np.ndarray,
        object_name: str,
        exclusion_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment object while excluding regions already segmented."""
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
        masks: List[np.ndarray],
        object_color_pairs: List[Tuple[str, str]],
        results_per_object: List[Dict],
        correct: bool,
        output_dir: Path,
        prompt_id: str,
        image_idx: int,
    ):
        """Save visualization with all objects and their colors."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            n_objects = len(object_color_pairs)
            fig, axes = plt.subplots(2, n_objects + 1, figsize=(5 * (n_objects + 1), 10))
            
            # Row 1, Col 0: Original image
            axes[0, 0].imshow(image_np)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Row 1, Cols 1-N: Object masks (black overlay)
            for i, ((obj_name, _), mask) in enumerate(zip(object_color_pairs, masks)):
                overlay = image_np.copy().astype(float)
                overlay[mask] = overlay[mask] * 0.5
                axes[0, i + 1].imshow(overlay.astype(np.uint8))
                axes[0, i + 1].set_title(f'{obj_name}\n({mask.sum()} px)')
                axes[0, i + 1].axis('off')
            
            # Row 2, Col 0: Combined masks
            combined = image_np.copy().astype(float)
            for mask in masks:
                combined[mask] = combined[mask] * 0.5
            axes[1, 0].imshow(combined.astype(np.uint8))
            axes[1, 0].set_title('All Objects')
            axes[1, 0].axis('off')
            
            # Row 2, Cols 1-N: Color comparison with metrics
            for i, result in enumerate(results_per_object):
                if 'error' in result and 'target_lab' not in result:
                    axes[1, i + 1].text(0.5, 0.5, f"Error:\n{result['error']}", 
                                        ha='center', va='center', fontsize=10)
                    axes[1, i + 1].axis('off')
                    continue
                
                target_lab = np.array(result['target_lab'])
                extracted_lab = np.array(result['extracted_lab'])
                
                # Split color patch: left=target, right=extracted
                target_rgb = lab_to_rgb(target_lab)
                extracted_rgb = lab_to_rgb(extracted_lab)
                
                patch = np.zeros((100, 100, 3), dtype=np.uint8)
                patch[:, :50] = target_rgb
                patch[:, 50:] = extracted_rgb
                
                axes[1, i + 1].imshow(patch)
                match_str = '✓' if result['matches'] else '✗'
                metrics = result.get('metrics', {})
                de = metrics.get('ciede2000', 0)
                dc = metrics.get('delta_chroma', 0)
                dh = metrics.get('mae_hue', 0)
                axes[1, i + 1].set_title(
                    f"{result['object']}: {result['target_color']} {match_str}\n"
                    f"ΔE={de:.2f} Δc={dc:.2f} Δh={dh:.1f}°"
                )
                axes[1, i + 1].axis('off')
            
            # Overall result
            result_str = '✓ CORRECT' if correct else '✗ INCORRECT'
            fig.suptitle(f'MOC (Multi-Object Color Generation): {result_str}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            viz_path = output_dir / f'{prompt_id}_{image_idx}_viz.png'
            plt.savefig(viz_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not save visualization: {e}")


# Alias for backward compatibility
Task4Evaluator = MOCEvaluator
