"""
Visualization utilities for GenColorBench evaluation results.

Generates diagnostic plots showing:
- Input image and segmentation masks
- Color distribution in a*b* space
- Color swatches (target vs extracted)
- Evaluation summary
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from typing import Dict, Optional, Any

from ..color import rgb_to_lab, lab_to_rgb


def save_task3_visualization(
    image_path: Path,
    image_np: np.ndarray,
    result: Dict[str, Any],
    main_mask: Optional[np.ndarray],
    sec_mask: Optional[np.ndarray],
    target_lab: np.ndarray,
    output_dir: Path,
    prompt_id: str,
    image_idx: int
) -> str:
    """
    Save unified visualization for Task 3 evaluation.
    
    Layout (2x4 grid):
    Row 1: [Input Image] [Main Obj Mask] [Sec Obj Mask] [Combined Masks]
    Row 2: [Main Color a*b*] [Sec Color a*b*] [Color Swatches] [Evaluation Result]
    """
    fig = plt.figure(figsize=(20, 12))
    
    main_obj = result.get('main_obj', 'N/A')
    sec_obj = result.get('sec_obj', 'N/A')
    color_name = result.get('color_name', 'N/A')
    correct = result.get('correct', False)
    decision_reason = result.get('decision_reason', 'N/A')
    
    main_color_info = result.get('main_obj_color', {})
    sec_color_info = result.get('sec_obj_color', {})
    vlm_check = result.get('vlm_check', {})
    
    # Title
    status = "CORRECT" if correct else "INCORRECT"
    title_color = 'green' if correct else 'red'
    fig.suptitle(f"GenColorBench Task 3: {color_name} {main_obj} + {sec_obj} | {status}",
                 fontsize=16, fontweight='bold', color=title_color)
    
    # Row 1: Image and Masks
    _add_image_panel(fig, 1, image_np, "Input Image")
    _add_mask_panel(fig, 2, image_np, main_mask, main_obj, "Main Mask", [0, 150, 0])
    _add_mask_panel(fig, 3, image_np, sec_mask, sec_obj, "Sec Mask", [150, 0, 0], vlm_check)
    _add_combined_mask_panel(fig, 4, image_np, main_mask, sec_mask)
    
    # Row 2: Color Analysis
    _add_color_scatter(fig, 5, image_np, main_mask, main_color_info, target_lab, "Main", True)
    _add_color_scatter(fig, 6, image_np, sec_mask, sec_color_info, target_lab, "Sec", False, vlm_check)
    _add_color_swatches(fig, 7, target_lab, main_color_info, sec_color_info)
    _add_result_summary(fig, 8, result, target_lab, main_color_info, sec_color_info, vlm_check, main_obj, sec_obj)
    
    plt.tight_layout()
    
    # Save
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    status_prefix = "correct" if correct else "incorrect"
    viz_path = viz_dir / f"{status_prefix}_{prompt_id}_img{image_idx}.png"
    
    plt.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return str(viz_path)


def save_single_object_visualization(
    image_path: Path,
    image_np: np.ndarray,
    result: Dict[str, Any],
    mask: Optional[np.ndarray],
    target_lab: np.ndarray,
    output_dir: Path,
    prompt_id: str,
    image_idx: int,
    task_name: str = "Task 1"
) -> str:
    """
    Save unified visualization for Task 1/2/5 evaluation (single object).
    
    Layout (2x4 grid):
    Row 1: [Input Image] [SAM2 Mask] [Neg Filter Mask] [Final Masked Image]
    Row 2: [Color Distribution a*b*] [Post-Processed] [Color Swatches] [Evaluation Result]
    """
    fig = plt.figure(figsize=(20, 12))
    
    object_name = result.get('object', 'N/A')
    color_name = result.get('color_name', result.get('implicit_color', 'N/A'))
    correct = result.get('correct', False)
    color_info = result.get('color_info', {})
    
    # Title
    status = "CORRECT" if correct else "INCORRECT"
    title_color = 'green' if correct else 'red'
    fig.suptitle(f"GenColorBench {task_name}: {object_name} - {color_name} | {status}",
                 fontsize=16, fontweight='bold', color=title_color)
    
    # Row 1
    _add_image_panel(fig, 1, image_np, "Input Image")
    _add_mask_panel(fig, 2, image_np, mask, object_name, "SAM2 Mask", [0, 150, 0])
    
    # Negative mask placeholder
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(np.ones_like(image_np) * 245)
    ax3.set_title("Negative Mask\n(None)", fontsize=11)
    ax3.axis('off')
    
    # Final masked image
    ax4 = fig.add_subplot(2, 4, 4)
    if mask is not None and mask.any():
        final_viz = np.ones_like(image_np) * 128
        final_viz[mask] = image_np[mask]
        border = ndimage.binary_dilation(mask, iterations=3) & ~mask
        final_viz[border] = [0, 255, 0]
        ax4.imshow(final_viz)
    else:
        ax4.imshow(np.ones_like(image_np) * 128)
    ax4.set_title("Final Masked Image", fontsize=11)
    ax4.axis('off')
    
    # Row 2
    _add_color_scatter(fig, 5, image_np, mask, color_info, target_lab, "Object", True)
    
    # Post-processed placeholder
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.imshow(np.ones_like(image_np) * 245)
    ax6.set_title("Post-Processed\n(Same as SAM2)", fontsize=11)
    ax6.axis('off')
    
    # Color swatches (single object version)
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.axis('off')
    _draw_single_object_swatches(ax7, target_lab, color_info)
    
    # Result summary
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    _draw_single_result_summary(ax8, result, target_lab, color_info, object_name, color_name)
    
    plt.tight_layout()
    
    # Save
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    status_prefix = "correct" if correct else "incorrect"
    viz_path = viz_dir / f"{status_prefix}_{prompt_id}_img{image_idx}.png"
    
    plt.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return str(viz_path)


def save_multi_object_visualization(
    image_path: Path,
    image_np: np.ndarray,
    result: Dict[str, Any],
    masks: Dict[str, np.ndarray],
    target_labs: Dict[str, np.ndarray],
    output_dir: Path,
    prompt_id: str,
    image_idx: int
) -> str:
    """
    Save unified visualization for Task 4 evaluation (multi-object).
    
    Layout: Adaptive based on number of objects
    """
    n_objects = len(masks)
    correct = result.get('correct', False)
    object_results = result.get('object_results', {})
    vlm_check = result.get('vlm_check', {})
    
    n_cols = 4
    n_rows = 2 + (n_objects + 1) // 2
    
    fig = plt.figure(figsize=(20, 5 * n_rows))
    
    # Title
    status = "CORRECT" if correct else "INCORRECT"
    title_color = 'green' if correct else 'red'
    
    obj_color_pairs = [
        f"{obj}={object_results.get(obj, {}).get('color_name', '?')}" 
        for obj in masks.keys()
    ]
    title_str = " | ".join(obj_color_pairs[:3])
    if len(obj_color_pairs) > 3:
        title_str += f" | +{len(obj_color_pairs)-3} more"
    
    fig.suptitle(f"GenColorBench Task 4: {title_str} | {status}",
                 fontsize=14, fontweight='bold', color=title_color)
    
    # Row 1: Overview
    ax1 = fig.add_subplot(n_rows, n_cols, 1)
    ax1.imshow(image_np)
    ax1.set_title("Input Image", fontsize=11)
    ax1.axis('off')
    
    # Combined masks
    ax2 = fig.add_subplot(n_rows, n_cols, 2)
    combined_viz = np.ones((*image_np.shape[:2], 3), dtype=np.uint8) * 245
    colors = [[0, 150, 0], [150, 0, 0], [0, 0, 150], [150, 150, 0], [150, 0, 150], [0, 150, 150]]
    for i, (obj_name, mask) in enumerate(masks.items()):
        if mask is not None and mask.any():
            combined_viz[mask] = colors[i % len(colors)]
    ax2.imshow(combined_viz)
    ax2.set_title("Combined Masks", fontsize=11)
    ax2.axis('off')
    
    # Final masked image
    ax3 = fig.add_subplot(n_rows, n_cols, 3)
    final_viz = np.ones_like(image_np) * 128
    for obj_name, mask in masks.items():
        if mask is not None and mask.any():
            final_viz[mask] = image_np[mask]
    ax3.imshow(final_viz)
    ax3.set_title("Final Masked Image", fontsize=11)
    ax3.axis('off')
    
    # VLM summary
    ax4 = fig.add_subplot(n_rows, n_cols, 4)
    ax4.axis('off')
    vlm_text = "VLM Check:\n"
    if vlm_check:
        for obj_name, status in vlm_check.items():
            if obj_name != 'both_verified':
                vlm_text += f"  {obj_name}: {status.upper() if isinstance(status, str) else 'YES' if status else 'NO'}\n"
    else:
        vlm_text += "  (Disabled)\n"
    ax4.text(0.1, 0.9, vlm_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax4.set_title("VLM Status", fontsize=11)
    
    # Per-object details
    plot_idx = n_cols + 1
    for obj_name, mask in masks.items():
        obj_result = object_results.get(obj_name, {})
        target_lab = target_labs.get(obj_name, np.array([50, 0, 0]))
        color_name = obj_result.get('color_name', 'unknown')
        matches = obj_result.get('matches', False)
        
        # Object mask
        ax = fig.add_subplot(n_rows, n_cols, plot_idx)
        if mask is not None and mask.any():
            mask_viz = np.ones((*image_np.shape[:2], 3), dtype=np.uint8) * 245
            mask_viz[mask] = [0, 150, 0]
            ax.imshow(mask_viz)
            ax.set_title(f"{obj_name}: {color_name}\n({mask.sum():,} px)", fontsize=10,
                        color='green' if matches else 'red')
        else:
            ax.imshow(np.ones_like(image_np) * 245)
            ax.set_title(f"{obj_name}: {color_name}\n(No mask)", fontsize=10, color='red')
        ax.axis('off')
        plot_idx += 1
        
        # Color a*b* plot
        ax = fig.add_subplot(n_rows, n_cols, plot_idx)
        if mask is not None and mask.any() and obj_result:
            pixels_rgb = image_np[mask]
            if len(pixels_rgb) > 1000:
                indices = np.random.choice(len(pixels_rgb), 1000, replace=False)
                pixels_rgb = pixels_rgb[indices]
            pixels_lab = rgb_to_lab(pixels_rgb)
            
            ax.scatter(pixels_lab[:, 1], pixels_lab[:, 2], c=pixels_rgb/255.0,
                      alpha=0.5, s=2, marker='.')
            
            target_rgb = lab_to_rgb(target_lab)
            ax.scatter([target_lab[1]], [target_lab[2]], 
                      c=[target_rgb/255.0], s=200, 
                      marker='s', edgecolors='black', linewidths=2, label='Target', zorder=10)
            
            dom_lab = np.array(obj_result.get('dominant_lab', [50, 0, 0]))
            dom_rgb = lab_to_rgb(dom_lab)
            ax.scatter([dom_lab[1]], [dom_lab[2]], 
                      c=[dom_rgb/255.0], s=150,
                      marker='*', edgecolors='black', linewidths=1.5, label='Dominant', zorder=10)
            
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
            ax.legend(fontsize=7, loc='upper right')
            
            match_str = "✓" if matches else "✗"
            de = obj_result.get('metrics', {}).get('ciede2000', 0)
            ax.set_title(f"{match_str} dE={de:.1f}", fontsize=10, color='green' if matches else 'red')
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=10)
        plot_idx += 1
    
    # Summary in last cell
    while plot_idx <= n_rows * n_cols:
        ax = fig.add_subplot(n_rows, n_cols, plot_idx)
        ax.axis('off')
        
        if plot_idx == n_rows * n_cols:
            summary_text = "EVALUATION SUMMARY\n" + "─" * 25 + "\n\n"
            
            for obj_name, obj_result in object_results.items():
                matches = obj_result.get('matches', False)
                color_name = obj_result.get('color_name', '?')
                de = obj_result.get('metrics', {}).get('ciede2000', 0)
                status_icon = "✓" if matches else "✗"
                summary_text += f"{status_icon} {obj_name} ({color_name}): dE={de:.1f}\n"
            
            summary_text += "\n" + "─" * 25 + "\n"
            n_correct = sum(1 for r in object_results.values() if r.get('matches', False))
            summary_text += f"Correct: {n_correct}/{len(object_results)}\n"
            summary_text += f"Final: {'✓ CORRECT' if correct else '✗ INCORRECT'}"
            
            bbox_color = 'lightgreen' if correct else 'lightcoral'
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.7))
        
        plot_idx += 1
    
    plt.tight_layout()
    
    # Save
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    status_prefix = "correct" if correct else "incorrect"
    viz_path = viz_dir / f"{status_prefix}_{prompt_id}_img{image_idx}.png"
    
    plt.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return str(viz_path)


# =============================================================================
# Helper functions
# =============================================================================

def _add_image_panel(fig, pos, image_np, title):
    """Add image panel to figure."""
    ax = fig.add_subplot(2, 4, pos)
    ax.imshow(image_np)
    ax.set_title(title, fontsize=12)
    ax.axis('off')


def _add_mask_panel(fig, pos, image_np, mask, obj_name, title_prefix, color, vlm_check=None):
    """Add mask panel to figure."""
    ax = fig.add_subplot(2, 4, pos)
    
    if mask is not None and mask.any():
        mask_viz = np.ones((*image_np.shape[:2], 3), dtype=np.uint8) * 245
        mask_viz[mask] = color
        ax.imshow(mask_viz)
        ax.set_title(f"{title_prefix}: {obj_name}\n({mask.sum():,} px)", fontsize=11)
    else:
        ax.imshow(np.ones_like(image_np) * 245)
        if vlm_check and vlm_check.get('sec_obj_present') == False:
            status_txt = "(Not present - VLM)"
        else:
            status_txt = "(No mask)"
        ax.set_title(f"{title_prefix}: {obj_name}\n{status_txt}", fontsize=11)
    ax.axis('off')


def _add_combined_mask_panel(fig, pos, image_np, main_mask, sec_mask):
    """Add combined mask panel."""
    ax = fig.add_subplot(2, 4, pos)
    
    gray_bg = np.ones_like(image_np) * 128
    final_viz = gray_bg.copy().astype(np.float32)
    
    if main_mask is not None and main_mask.any():
        final_viz[main_mask] = image_np[main_mask]
        main_border = ndimage.binary_dilation(main_mask, iterations=3) & ~main_mask
        final_viz[main_border] = [0, 255, 0]
    
    if sec_mask is not None and sec_mask.any():
        final_viz[sec_mask] = image_np[sec_mask]
        sec_border = ndimage.binary_dilation(sec_mask, iterations=3) & ~sec_mask
        final_viz[sec_border] = [255, 0, 0]
    
    ax.imshow(final_viz.astype(np.uint8))
    ax.set_title("Final Masked Image\n(Green=Main, Red=Sec)", fontsize=11)
    ax.axis('off')


def _add_color_scatter(fig, pos, image_np, mask, color_info, target_lab, label, is_main, vlm_check=None):
    """Add color scatter plot panel."""
    ax = fig.add_subplot(2, 4, pos)
    
    if color_info and mask is not None and mask.any():
        pixels_rgb = image_np[mask]
        if len(pixels_rgb) > 2000:
            indices = np.random.choice(len(pixels_rgb), 2000, replace=False)
            pixels_rgb = pixels_rgb[indices]
        pixels_lab = rgb_to_lab(pixels_rgb)
        
        ax.scatter(pixels_lab[:, 1], pixels_lab[:, 2], c=pixels_rgb/255.0, 
                   alpha=0.5, s=2, marker='.')
        
        target_rgb = lab_to_rgb(target_lab)
        ax.scatter([target_lab[1]], [target_lab[2]], 
                   c=[target_rgb/255.0], s=250, 
                   marker='s', edgecolors='black', linewidths=2, label='Target', zorder=10)
        
        dom_lab = np.array(color_info.get('dominant_lab', [50, 0, 0]))
        dom_rgb = lab_to_rgb(dom_lab)
        ax.scatter([dom_lab[1]], [dom_lab[2]], 
                   c=[dom_rgb/255.0], s=200,
                   marker='*', edgecolors='black', linewidths=1.5, label='Dominant', zorder=10)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("a* (green-red)", fontsize=9)
        ax.set_ylabel("b* (blue-yellow)", fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        
        matches = color_info.get('matches', False)
        if is_main:
            match_str = "MATCH" if matches else "NO MATCH"
        else:
            match_str = "LEAKAGE!" if matches else "Different (OK)"
        color = 'green' if (is_main and matches) or (not is_main and not matches) else 'red'
        ax.set_title(f"{label} Color (a*b*): {match_str}", fontsize=11, color=color)
    else:
        if vlm_check and vlm_check.get('sec_obj_present') == False:
            ax.text(0.5, 0.5, "Sec obj not present\n(No leakage possible)", 
                    ha='center', va='center', fontsize=11)
        else:
            ax.text(0.5, 0.5, f"No {label.lower()} color data", ha='center', va='center', fontsize=12)
        ax.set_title(f"{label} Color (a*b*)", fontsize=11)


def _add_color_swatches(fig, pos, target_lab, main_color_info, sec_color_info):
    """Add color swatches panel for Task 3."""
    ax = fig.add_subplot(2, 4, pos)
    ax.axis('off')
    
    swatch_h = 60
    swatch_w = 80
    gap = 20
    
    target_rgb = lab_to_rgb(target_lab)
    target_swatch = np.ones((swatch_h, swatch_w, 3), dtype=np.uint8) * target_rgb
    
    if main_color_info:
        main_dom_lab = np.array(main_color_info.get('dominant_lab', [50, 0, 0]))
        main_rgb = lab_to_rgb(main_dom_lab)
    else:
        main_rgb = np.array([128, 128, 128])
    main_swatch = np.ones((swatch_h, swatch_w, 3), dtype=np.uint8) * main_rgb
    
    if sec_color_info:
        sec_dom_lab = np.array(sec_color_info.get('dominant_lab', [50, 0, 0]))
        sec_rgb = lab_to_rgb(sec_dom_lab)
    else:
        sec_rgb = np.array([128, 128, 128])
    sec_swatch = np.ones((swatch_h, swatch_w, 3), dtype=np.uint8) * sec_rgb
    
    combined_swatch = np.ones((swatch_h * 2 + gap, swatch_w * 3 + gap * 2, 3), dtype=np.uint8) * 255
    y_start = swatch_h + gap
    combined_swatch[y_start:y_start+swatch_h, 0:swatch_w] = target_swatch
    combined_swatch[y_start:y_start+swatch_h, swatch_w+gap:swatch_w*2+gap] = main_swatch
    combined_swatch[y_start:y_start+swatch_h, swatch_w*2+gap*2:swatch_w*3+gap*2] = sec_swatch
    
    ax.imshow(combined_swatch)
    ax.text(swatch_w/2, swatch_h/2, "Target", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(swatch_w*1.5+gap, swatch_h/2, "Main", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(swatch_w*2.5+gap*2, swatch_h/2, "Sec", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.set_title("Color Comparison", fontsize=11)


def _add_result_summary(fig, pos, result, target_lab, main_color_info, sec_color_info, vlm_check, main_obj, sec_obj):
    """Add result summary panel for Task 3."""
    ax = fig.add_subplot(2, 4, pos)
    ax.axis('off')
    
    correct = result.get('correct', False)
    decision_reason = result.get('decision_reason', 'N/A')
    color_name = result.get('color_name', 'N/A')
    
    result_text = f"Target Color: {color_name}\n"
    result_text += f"Target LAB: ({target_lab[0]:.1f}, {target_lab[1]:.1f}, {target_lab[2]:.1f})\n\n"
    
    if vlm_check:
        main_vlm = vlm_check.get('main_obj_present', None)
        sec_vlm = vlm_check.get('sec_obj_present', None)
        result_text += f"VLM Check:\n"
        result_text += f"  {main_obj}: {'YES' if main_vlm else 'NO'}\n"
        result_text += f"  {sec_obj}: {'YES' if sec_vlm else 'NO'}\n\n"
    else:
        result_text += f"VLM Check: DISABLED\n\n"
    
    if main_color_info:
        main_dom_lab = np.array(main_color_info.get('dominant_lab', [50, 0, 0]))
        main_metrics = main_color_info.get('metrics', {})
        main_matches = main_color_info.get('matches', False)
        result_text += f"Main Object ({main_obj}):\n"
        result_text += f"  Dominant LAB: ({main_dom_lab[0]:.1f}, {main_dom_lab[1]:.1f}, {main_dom_lab[2]:.1f})\n"
        result_text += f"  ΔChroma: {main_metrics.get('delta_chroma', 0):.2f}\n"
        result_text += f"  CIEDE2000: {main_metrics.get('ciede2000', 0):.2f}\n"
        result_text += f"  MAE Hue: {main_metrics.get('mae_hue', 0):.2f}°\n"
        result_text += f"  Result: {'✓ MATCH' if main_matches else '✗ NO MATCH'}\n\n"
    
    if sec_color_info:
        sec_dom_lab = np.array(sec_color_info.get('dominant_lab', [50, 0, 0]))
        sec_metrics = sec_color_info.get('metrics', {})
        sec_matches = sec_color_info.get('matches', False)
        result_text += f"Sec Object ({sec_obj}):\n"
        result_text += f"  Dominant LAB: ({sec_dom_lab[0]:.1f}, {sec_dom_lab[1]:.1f}, {sec_dom_lab[2]:.1f})\n"
        result_text += f"  ΔChroma: {sec_metrics.get('delta_chroma', 0):.2f}\n"
        result_text += f"  CIEDE2000: {sec_metrics.get('ciede2000', 0):.2f}\n"
        result_text += f"  Result: {'⚠️ LEAKAGE' if sec_matches else '✓ Different'}\n\n"
    elif vlm_check and vlm_check.get('sec_obj_present') == False:
        result_text += f"Sec Object ({sec_obj}):\n"
        result_text += f"  Not present (VLM)\n"
        result_text += f"  No leakage possible\n\n"
    
    result_text += f"─────────────────────\n"
    result_text += f"Decision: {decision_reason}\n"
    result_text += f"Final: {'✓ CORRECT' if correct else '✗ INCORRECT'}"
    
    bbox_color = 'lightgreen' if correct else 'lightcoral'
    ax.text(0.05, 0.95, result_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Evaluation Result", fontsize=11)


def _draw_single_object_swatches(ax, target_lab, color_info):
    """Draw color swatches for single object visualization."""
    swatch_h = 80
    swatch_w = 100
    gap = 30
    
    target_rgb = lab_to_rgb(target_lab)
    target_swatch = np.ones((swatch_h, swatch_w, 3), dtype=np.uint8) * target_rgb
    
    if color_info:
        dom_lab = np.array(color_info.get('dominant_lab', [50, 0, 0]))
        extracted_rgb = lab_to_rgb(dom_lab)
    else:
        extracted_rgb = np.array([128, 128, 128])
    extracted_swatch = np.ones((swatch_h, swatch_w, 3), dtype=np.uint8) * extracted_rgb
    
    combined_swatch = np.ones((swatch_h * 2 + gap, swatch_w * 2 + gap, 3), dtype=np.uint8) * 255
    y_start = swatch_h + gap
    combined_swatch[y_start:y_start+swatch_h, 0:swatch_w] = target_swatch
    combined_swatch[y_start:y_start+swatch_h, swatch_w+gap:swatch_w*2+gap] = extracted_swatch
    
    ax.imshow(combined_swatch)
    ax.text(swatch_w/2, swatch_h/2, "Target", ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(swatch_w*1.5+gap, swatch_h/2, "Extracted", ha='center', va='center', fontsize=11, fontweight='bold')
    ax.set_title("Color Comparison", fontsize=11)


def _draw_single_result_summary(ax, result, target_lab, color_info, object_name, color_name):
    """Draw result summary for single object visualization."""
    correct = result.get('correct', False)
    decision_reason = result.get('decision_reason', 'N/A')
    vlm_check = result.get('vlm_check', None)
    
    result_text = f"Target Color: {color_name}\n"
    result_text += f"Target LAB: ({target_lab[0]:.1f}, {target_lab[1]:.1f}, {target_lab[2]:.1f})\n\n"
    
    if vlm_check:
        obj_vlm = vlm_check.get(object_name, vlm_check.get('main_obj_present', 'N/A'))
        result_text += f"VLM Check:\n"
        if isinstance(obj_vlm, str):
            result_text += f"  {object_name}: {obj_vlm.upper()}\n\n"
        else:
            result_text += f"  {object_name}: {'YES' if obj_vlm else 'NO'}\n\n"
    
    if color_info:
        dom_lab = np.array(color_info.get('dominant_lab', [50, 0, 0]))
        metrics = color_info.get('metrics', {})
        matches = color_info.get('matches', False)
        result_text += f"Object ({object_name}):\n"
        result_text += f"  Dominant LAB: ({dom_lab[0]:.1f}, {dom_lab[1]:.1f}, {dom_lab[2]:.1f})\n"
        result_text += f"  ΔChroma: {metrics.get('delta_chroma', 0):.2f}\n"
        result_text += f"  CIEDE2000: {metrics.get('ciede2000', 0):.2f}\n"
        result_text += f"  MAE Hue: {metrics.get('mae_hue', 0):.2f}°\n"
        result_text += f"  Result: {'✓ MATCH' if matches else '✗ NO MATCH'}\n\n"
    
    result_text += f"─────────────────────\n"
    result_text += f"Decision: {decision_reason}\n"
    result_text += f"Final: {'✓ CORRECT' if correct else '✗ INCORRECT'}"
    
    bbox_color = 'lightgreen' if correct else 'lightcoral'
    ax.text(0.05, 0.95, result_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Evaluation Result", fontsize=11)
