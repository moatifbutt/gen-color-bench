"""
NCU Evaluator: Numeric Color Understanding

Evaluates whether T2I models can generate objects in colors specified
by numerical values (hex codes or RGB triplets).

Formats:
- Hex: #FF5733
- RGB: (255, 87, 51)

CORRECT = extracted color matches target numerical color
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from PIL import Image

from .base import BaseTaskEvaluator
from ..color import extract_dominant_color, color_matches_target, get_color_neighbors, rgb_to_lab, lab_to_rgb
from ..color.matching import get_target_lab


class NCUEvaluator(BaseTaskEvaluator):
    """
    NCU: Numeric Color Understanding
    
    Evaluates whether generated objects match colors specified as hex/RGB.
    """
    
    task_name = "ncu"
    task_display_name = "Numeric Color Understanding"
    use_vlm_for_task = False
    
    def evaluate(
        self,
        image_path: Path,
        row: pd.Series,
        output_dir: Optional[Path] = None,
        prompt_id: str = "",
        image_idx: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate a single image for NCU."""
        # Load image
        image_pil, image_np = self.load_image(image_path)
        
        # Parse CSV columns - get RGB from hex or direct RGB values
        object_name = str(row.get('object', row.get('Object', ''))).strip()
        
        # Try to get target color from various possible columns
        target_lab = self._get_numeric_target(row)
        
        if target_lab is None:
            return {
                'success': False,
                'correct': False,
                'error': 'Could not parse numeric color from row',
                'object': object_name,
            }
        
        # For NCU, use L3 neighborhood (finest granularity)
        neighbors_lab = self.get_neighbors_lab("", target_lab)
        
        # Segment object
        mask, orig_mask, neg_mask = self.segment_object(image_pil, image_np, object_name)
        
        if not mask.any():
            return {
                'success': True,
                'correct': False,
                'error': 'Object segmentation failed',
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
        
        # Get color representation for output
        color_repr = self._get_color_representation(row)
        
        result = {
            'success': True,
            'correct': matches,
            'object': object_name,
            'color_repr': color_repr,
            'color_system': self.color_system,
            'target_lab': target_lab.tolist(),
            'extracted_lab': extracted_lab.tolist(),
            'metrics': metrics,
            'mask_pixels': int(mask.sum()),
        }
        
        # Save visualization if enabled
        if self.save_viz and output_dir is not None:
            self._save_visualization(
                image_np, mask, extracted_lab, target_lab,
                matches, metrics, output_dir, prompt_id, image_idx,
                object_name, color_repr
            )
        
        return result
    
    def _get_numeric_target(self, row: pd.Series) -> Optional[np.ndarray]:
        """Extract target LAB from hex or RGB columns."""
        # Try hex first
        hex_val = row.get('hex', row.get('Hex', row.get('HEX', '')))
        if hex_val and isinstance(hex_val, str) and hex_val.startswith('#'):
            try:
                hex_clean = hex_val.lstrip('#')
                r = int(hex_clean[0:2], 16)
                g = int(hex_clean[2:4], 16)
                b = int(hex_clean[4:6], 16)
                return rgb_to_lab(np.array([r, g, b]))
            except:
                pass
        
        # Try RGB columns
        try:
            r = int(row.get('r', row.get('R', 0)))
            g = int(row.get('g', row.get('G', 0)))
            b = int(row.get('b', row.get('B', 0)))
            if r > 0 or g > 0 or b > 0:
                return rgb_to_lab(np.array([r, g, b]))
        except:
            pass
        
        return None
    
    def _get_color_representation(self, row: pd.Series) -> str:
        """Get string representation of the color (hex or RGB)."""
        hex_val = row.get('hex', row.get('Hex', ''))
        if hex_val:
            return str(hex_val)
        
        try:
            r = int(row.get('r', row.get('R', 0)))
            g = int(row.get('g', row.get('G', 0)))
            b = int(row.get('b', row.get('B', 0)))
            return f"({r}, {g}, {b})"
        except:
            return "unknown"
    
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
        color_repr: str,
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
            axes[2].set_title(f'Target: {color_repr}')
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
            fig.suptitle(f'NCU (Numeric Color Understanding): {result_str}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            viz_path = output_dir / f'{prompt_id}_{image_idx}_viz.png'
            plt.savefig(viz_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not save visualization: {e}")


# Alias for backward compatibility
Task2Evaluator = NCUEvaluator
