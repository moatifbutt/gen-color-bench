"""
GenColorBench Full Benchmark: Paper-Aligned Distribution (~50K prompts) - v3 Naming.

Implements the carefully designed prompt distribution for the full benchmark:

CNA (Color Name Accuracy):
- CSS: 148 colors × 49 objects × 1 template = ~7,252
- L2: 29 colors × 100 objects = 2,900
- L3: 147 colors (5 per L2) × 47 objects = ~6,909

NCU (Numeric Color Understanding):
- HEX: 148 CSS colors × 53 objects = ~7,844
- RGB: 29 L2 colors × 101 objects = 2,929

COA (Color-Object Association):
- CSS: 148 colors × 67 templates = ~9,916
- L2: 29 colors × 67 templates = ~1,943
- L3: 260 colors × 15 prompts = ~3,900

MOC (Multi-Object Composition):
- 30 templates, each color appears 5 times
- CSS: ~753, L2: ~149, L3: ~1,351

ICA (Implicit Color Association):
- 12 templates per color
- CSS: ~1,776, L2: ~348, L3: ~3,120

Total: ~50K prompts

CSV Naming (v3):
- cna_l1.csv, cna_l2.csv, cna_l3.csv, cna_css.csv
- ncu_rgb_l1.csv, ncu_rgb_l2.csv, ncu_rgb_l3.csv, ncu_hex_css.csv
- coa_l1.csv, coa_l2.csv, coa_l3.csv, coa_css.csv
- moc_l1.csv, moc_l2.csv, moc_l3.csv, moc_css.csv
- ica_l1.csv, ica_l2.csv, ica_l3.csv, ica_css.csv
"""

import os
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from .loaders import (
    load_color_system, load_objects, load_task5_templates,
    Color, ObjectItem as Object
)
from .prompts.color_object import TEMPLATES_WITH_GT as COA_TEMPLATES
from .prompts.multi_object import TEMPLATES_WITH_GT as MOC_TEMPLATES
from .mini_benchmark import (
    GenerationResult,
    TASK_DISPLAY_NAMES,
    CNA_TEMPLATES,
    NCU_RGB_TEMPLATES,
    NCU_HEX_TEMPLATES,
    load_l3_to_l1_mapping,
    stratify_objects,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FullBenchmarkConfig:
    """Configuration for full benchmark generation.
    
    Paper-aligned distribution targeting ~50K prompts.
    """
    
    # L3 stratification: 5 shades per L2 category
    l3_shades_per_l2: int = 5
    
    # CNA config: color × object (1 random template per pair)
    cna_css_objects: int = 49      # 148 × 49 = 7,252
    cna_l2_objects: int = 100      # 29 × 100 = 2,900
    cna_l3_objects: int = 47       # ~147 × 47 ≈ 6,909
    
    # NCU config
    ncu_hex_objects: int = 53      # 148 × 53 = 7,844
    ncu_rgb_objects: int = 101     # 29 × 101 = 2,929
    
    # COA config: color × templates (67 templates for CSS/L2, 15 for L3)
    coa_templates: int = 67        # Full set for CSS/L2
    coa_l3_prompts_per_color: int = 15  # 260 × 15 = 3,900
    
    # MOC config: templates, each color appears multiple times
    moc_templates: int = 30
    moc_appearances_per_color: int = 5
    
    # ICA config: templates per color
    ica_templates_per_color: int = 12  # ~148×12=1776, 29×12=348, 260×12=3120
    
    seed: int = 42


# =============================================================================
# L3 Stratification (5 per L2 category)
# =============================================================================

def load_l3_to_l2_mapping() -> Dict[int, List[str]]:
    """Load mapping from L2 color IDs to L3 color names."""
    pkg_dir = Path(__file__).parent
    neighborhood_path = pkg_dir / 'data' / 'neighborhoods' / 'iscc_neighborhood.csv'
    
    df = pd.read_csv(neighborhood_path)
    
    mapping = {}
    for l2_id in df['Level2'].unique():
        l3_colors = df[df['Level2'] == l2_id]['ISCCNBS'].unique().tolist()
        mapping[l2_id] = l3_colors
    
    return mapping


def stratify_l3_colors_by_l2(
    l3_to_l2: Dict[int, List[str]],
    shades_per_l2: int = 5,
    seed: int = 42
) -> List[str]:
    """Stratified sampling of L3 colors: 5 per L2 category."""
    random.seed(seed)
    
    selected = []
    
    priority_prefixes = [
        ['Vivid', 'Strong', 'Brilliant'],
        ['Deep', 'Dark', 'Very dark'],
        ['Light', 'Pale', 'Brilliant'],
        ['Grayish', 'Moderate', 'Dusky'],
    ]
    
    for l2_id in sorted(l3_to_l2.keys()):
        l3_colors = l3_to_l2[l2_id]
        
        if len(l3_colors) <= shades_per_l2:
            selected.extend(l3_colors)
        else:
            l2_selected = []
            remaining = list(l3_colors)
            
            for prefixes in priority_prefixes:
                if len(l2_selected) >= shades_per_l2 - 1:
                    break
                for prefix in prefixes:
                    matches = [c for c in remaining if c.startswith(prefix)]
                    if matches:
                        choice = random.choice(matches)
                        l2_selected.append(choice)
                        remaining.remove(choice)
                        break
            
            while len(l2_selected) < shades_per_l2 and remaining:
                choice = random.choice(remaining)
                l2_selected.append(choice)
                remaining.remove(choice)
            
            selected.extend(l2_selected)
    
    return selected


# =============================================================================
# Task Generation Functions
# =============================================================================

def generate_cna_full(
    config: FullBenchmarkConfig,
    all_objects: List[Object],
    seed: int = 42
) -> List[GenerationResult]:
    """Generate CNA prompts for full benchmark."""
    random.seed(seed)
    results = []
    
    css_colors = load_color_system('css')
    l2_colors = load_color_system('iscc_l2')
    l3_colors_full = load_color_system('iscc_l3')
    
    # Stratify L3 colors (5 per L2)
    l3_to_l2 = load_l3_to_l2_mapping()
    l3_selected_names = stratify_l3_colors_by_l2(l3_to_l2, config.l3_shades_per_l2, seed)
    l3_colors = [c for c in l3_colors_full if c.name in l3_selected_names]
    
    # Stratify objects
    css_objects = stratify_objects(all_objects, sample_ratio=0.49, min_per_category=3, seed=seed)
    if len(css_objects) > config.cna_css_objects:
        css_objects = random.sample(css_objects, config.cna_css_objects)
    
    l3_objects = stratify_objects(all_objects, sample_ratio=0.73, min_per_category=5, seed=seed)
    if len(l3_objects) > config.cna_l3_objects:
        l3_objects = random.sample(l3_objects, config.cna_l3_objects)
    
    # --- CSS ---
    records = []
    idx = 1
    for color in css_colors:
        for obj in css_objects:
            template = random.choice(CNA_TEMPLATES)
            prompt = template.format(color=color.name, object=obj.name)
            records.append({
                "id": idx,
                "color": color.name,
                "object": obj.name,
                "prompt": prompt,
            })
            idx += 1
    
    df = pd.DataFrame(records)
    results.append(GenerationResult(
        task="cna",
        color_system="css",
        n_prompts=len(df),
        filepath=None,
        dataframe=df,
    ))
    
    # --- L2 ---
    records = []
    idx = 1
    for color in l2_colors:
        sampled = random.sample(all_objects, min(config.cna_l2_objects, len(all_objects)))
        for obj in sampled:
            template = random.choice(CNA_TEMPLATES)
            prompt = template.format(color=color.name, object=obj.name)
            records.append({
                "id": idx,
                "color": color.name,
                "object": obj.name,
                "prompt": prompt,
            })
            idx += 1
    
    df = pd.DataFrame(records)
    results.append(GenerationResult(
        task="cna",
        color_system="iscc_l2",
        n_prompts=len(df),
        filepath=None,
        dataframe=df,
    ))
    
    # --- L3 ---
    records = []
    idx = 1
    for color in l3_colors:
        for obj in l3_objects:
            template = random.choice(CNA_TEMPLATES)
            prompt = template.format(color=color.name, object=obj.name)
            records.append({
                "id": idx,
                "color": color.name,
                "object": obj.name,
                "prompt": prompt,
            })
            idx += 1
    
    df = pd.DataFrame(records)
    results.append(GenerationResult(
        task="cna",
        color_system="iscc_l3",
        n_prompts=len(df),
        filepath=None,
        dataframe=df,
    ))
    
    return results


def generate_ncu_full(
    config: FullBenchmarkConfig,
    all_objects: List[Object],
    seed: int = 42
) -> List[GenerationResult]:
    """Generate NCU prompts for full benchmark."""
    random.seed(seed)
    results = []
    
    css_colors = load_color_system('css')
    l2_colors = load_color_system('iscc_l2')
    
    # --- HEX (CSS only) ---
    records = []
    idx = 1
    hex_objects = random.sample(all_objects, min(config.ncu_hex_objects, len(all_objects)))
    
    for color in css_colors:
        hex_val = color.hex if color.hex else f"#{color.r:02X}{color.g:02X}{color.b:02X}"
        for obj in hex_objects:
            template = random.choice(NCU_HEX_TEMPLATES)
            prompt = template.format(object=obj.name, hex=hex_val)
            records.append({
                "id": idx,
                "css_name": color.css_name if color.css_name else color.name,
                "color_name": color.name,
                "hex": hex_val,
                "r": color.r,
                "g": color.g,
                "b": color.b,
                "object": obj.name,
                "prompt": prompt,
            })
            idx += 1
    
    df = pd.DataFrame(records)
    results.append(GenerationResult(
        task="ncu_hex",
        color_system="css",
        n_prompts=len(df),
        filepath=None,
        dataframe=df,
    ))
    
    # --- RGB (L2) ---
    records = []
    idx = 1
    rgb_objects = random.sample(all_objects, min(config.ncu_rgb_objects, len(all_objects)))
    
    for color in l2_colors:
        for obj in rgb_objects:
            template = random.choice(NCU_RGB_TEMPLATES)
            prompt = template.format(object=obj.name, r=color.r, g=color.g, b=color.b)
            records.append({
                "id": idx,
                "color_name": color.name,
                "r": color.r,
                "g": color.g,
                "b": color.b,
                "object": obj.name,
                "prompt": prompt,
            })
            idx += 1
    
    df = pd.DataFrame(records)
    results.append(GenerationResult(
        task="ncu_rgb",
        color_system="iscc_l2",
        n_prompts=len(df),
        filepath=None,
        dataframe=df,
    ))
    
    return results


def generate_coa_full(
    config: FullBenchmarkConfig,
    seed: int = 42
) -> List[GenerationResult]:
    """Generate COA prompts for full benchmark."""
    random.seed(seed)
    results = []
    
    css_colors = load_color_system('css')
    l2_colors = load_color_system('iscc_l2')
    l3_colors_full = load_color_system('iscc_l3')
    
    # Stratify L3
    l3_to_l2 = load_l3_to_l2_mapping()
    l3_selected_names = stratify_l3_colors_by_l2(l3_to_l2, config.l3_shades_per_l2, seed)
    l3_colors = [c for c in l3_colors_full if c.name in l3_selected_names]
    
    templates = COA_TEMPLATES[:config.coa_templates]
    
    def generate_for_system(colors: List[Color], system_name: str, n_templates: int) -> GenerationResult:
        records = []
        idx = 1
        
        for color in colors:
            sampled = random.sample(templates, min(n_templates, len(templates)))
            for main_obj, sec_obj, template in sampled:
                prompt = template.format(color=color.name)
                records.append({
                    "id": idx,
                    "color": color.name,
                    "main_obj": main_obj,
                    "sec_obj": sec_obj,
                    "prompt": prompt,
                })
                idx += 1
        
        df = pd.DataFrame(records)
        return GenerationResult(
            task="coa",
            color_system=system_name,
            n_prompts=len(df),
            filepath=None,
            dataframe=df,
        )
    
    results.append(generate_for_system(css_colors, "css", config.coa_templates))
    results.append(generate_for_system(l2_colors, "iscc_l2", config.coa_templates))
    results.append(generate_for_system(l3_colors, "iscc_l3", config.coa_l3_prompts_per_color))
    
    return results


def generate_moc_full(
    config: FullBenchmarkConfig,
    seed: int = 42
) -> List[GenerationResult]:
    """Generate MOC prompts for full benchmark."""
    random.seed(seed)
    results = []
    
    css_colors = load_color_system('css')
    l2_colors = load_color_system('iscc_l2')
    l3_colors_full = load_color_system('iscc_l3')
    
    # Stratify L3
    l3_to_l2 = load_l3_to_l2_mapping()
    l3_selected_names = stratify_l3_colors_by_l2(l3_to_l2, config.l3_shades_per_l2, seed)
    l3_colors = [c for c in l3_colors_full if c.name in l3_selected_names]
    
    templates = MOC_TEMPLATES[:config.moc_templates]
    
    def generate_for_system(colors: List[Color], system_name: str) -> GenerationResult:
        records = []
        color_usage = {c.name: 0 for c in colors}
        
        idx = 1
        for _ in range(config.moc_appearances_per_color):
            random.shuffle(templates)
            for obj1, obj2, obj3, template in templates:
                n_colors_needed = 3 if obj3 else 2
                
                sorted_colors = sorted(colors, key=lambda c: color_usage[c.name])
                selected = sorted_colors[:n_colors_needed]
                
                for c in selected:
                    color_usage[c.name] += 1
                
                prompt = template
                for c in selected:
                    prompt = prompt.replace("{color}", c.name, 1)
                
                records.append({
                    "id": idx,
                    "color_count": n_colors_needed,
                    "object1": obj1,
                    "color1": selected[0].name,
                    "object2": obj2,
                    "color2": selected[1].name,
                    "object3": obj3 if obj3 else "",
                    "color3": selected[2].name if n_colors_needed == 3 else "",
                    "prompt": prompt,
                })
                idx += 1
        
        df = pd.DataFrame(records)
        return GenerationResult(
            task="moc",
            color_system=system_name,
            n_prompts=len(df),
            filepath=None,
            dataframe=df,
        )
    
    results.append(generate_for_system(css_colors, "css"))
    results.append(generate_for_system(l2_colors, "iscc_l2"))
    results.append(generate_for_system(l3_colors, "iscc_l3"))
    
    return results


def generate_ica_full(
    config: FullBenchmarkConfig,
    seed: int = 42
) -> List[GenerationResult]:
    """Generate ICA prompts for full benchmark."""
    random.seed(seed)
    results = []
    
    css_colors = load_color_system('css')
    l2_colors = load_color_system('iscc_l2')
    l3_colors_full = load_color_system('iscc_l3')
    
    # Stratify L3
    l3_to_l2 = load_l3_to_l2_mapping()
    l3_selected_names = stratify_l3_colors_by_l2(l3_to_l2, config.l3_shades_per_l2, seed)
    l3_colors = [c for c in l3_colors_full if c.name in l3_selected_names]
    
    all_templates = load_task5_templates()
    
    def generate_for_system(colors: List[Color], system_name: str) -> GenerationResult:
        records = []
        idx = 1
        
        for color in colors:
            sampled = random.sample(
                all_templates,
                min(config.ica_templates_per_color, len(all_templates))
            )
            for t in sampled:
                prompt = t.prompt.replace("{color}", color.name)
                records.append({
                    "id": idx,
                    "object": t.object,
                    "color": color.name,
                    "ref_object": t.ref_object,
                    "prompt": prompt,
                })
                idx += 1
        
        df = pd.DataFrame(records)
        return GenerationResult(
            task="ica",
            color_system=system_name,
            n_prompts=len(df),
            filepath=None,
            dataframe=df,
        )
    
    results.append(generate_for_system(css_colors, "css"))
    results.append(generate_for_system(l2_colors, "iscc_l2"))
    results.append(generate_for_system(l3_colors, "iscc_l3"))
    
    return results


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_full_benchmark(
    output_dir: str = "./full_bench_prompt/",
    config: Optional[FullBenchmarkConfig] = None,
    seed: int = 42,
    verbose: bool = True,
) -> List[GenerationResult]:
    """
    Generate GenColorBench Full Benchmark with v3 naming convention.
    
    Target: ~50K prompts
    
    Args:
        output_dir: Directory to save CSV files
        config: FullBenchmarkConfig instance (uses defaults if None)
        seed: Random seed for reproducibility
        verbose: Print progress
        
    Returns:
        List of GenerationResult objects
    """
    if config is None:
        config = FullBenchmarkConfig(seed=seed)
    
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    if verbose:
        print("=" * 70)
        print("GenColorBench Full Benchmark Generation (v3 Naming)")
        print("=" * 70)
        print(f"Output directory: {output_dir}")
        print(f"Seed: {seed}")
        print("=" * 70)
        print()
    
    all_objects = load_objects()
    
    # =========================================================================
    # CNA: Color Name Accuracy
    # =========================================================================
    if verbose:
        print("CNA: Color Name Accuracy")
        print("-" * 50)
    
    cna_results = generate_cna_full(config, all_objects, seed)
    for r in cna_results:
        cs_short = r.color_system.replace('iscc_', '')
        filename = f"cna_{cs_short}.csv"
        filepath = os.path.join(output_dir, filename)
        r.dataframe.to_csv(filepath, index=False)
        r = GenerationResult(r.task, r.color_system, r.n_prompts, filepath, r.dataframe)
        all_results.append(r)
        if verbose:
            print(f"  {r.color_system}: {r.n_prompts:,} prompts")
    
    if verbose:
        print()
    
    # =========================================================================
    # NCU: Numeric Color Understanding
    # =========================================================================
    if verbose:
        print("NCU: Numeric Color Understanding")
        print("-" * 50)
    
    ncu_results = generate_ncu_full(config, all_objects, seed)
    for r in ncu_results:
        if r.task == "ncu_hex":
            filename = "ncu_hex_css.csv"
        else:
            cs_short = r.color_system.replace('iscc_', '')
            filename = f"ncu_rgb_{cs_short}.csv"
        
        filepath = os.path.join(output_dir, filename)
        r.dataframe.to_csv(filepath, index=False)
        r = GenerationResult(r.task, r.color_system, r.n_prompts, filepath, r.dataframe)
        all_results.append(r)
        if verbose:
            task_label = "HEX" if r.task == "ncu_hex" else "RGB"
            print(f"  {task_label} ({r.color_system}): {r.n_prompts:,} prompts")
    
    if verbose:
        print()
    
    # =========================================================================
    # COA: Color-Object Association
    # =========================================================================
    if verbose:
        print("COA: Color-Object Association")
        print("-" * 50)
    
    coa_results = generate_coa_full(config, seed)
    for r in coa_results:
        cs_short = r.color_system.replace('iscc_', '')
        filename = f"coa_{cs_short}.csv"
        filepath = os.path.join(output_dir, filename)
        r.dataframe.to_csv(filepath, index=False)
        r = GenerationResult(r.task, r.color_system, r.n_prompts, filepath, r.dataframe)
        all_results.append(r)
        if verbose:
            print(f"  {r.color_system}: {r.n_prompts:,} prompts")
    
    if verbose:
        print()
    
    # =========================================================================
    # MOC: Multi-Object Color Composition
    # =========================================================================
    if verbose:
        print("MOC: Multi-Object Color Composition")
        print("-" * 50)
    
    moc_results = generate_moc_full(config, seed)
    for r in moc_results:
        cs_short = r.color_system.replace('iscc_', '')
        filename = f"moc_{cs_short}.csv"
        filepath = os.path.join(output_dir, filename)
        r.dataframe.to_csv(filepath, index=False)
        r = GenerationResult(r.task, r.color_system, r.n_prompts, filepath, r.dataframe)
        all_results.append(r)
        if verbose:
            print(f"  {r.color_system}: {r.n_prompts:,} prompts")
    
    if verbose:
        print()
    
    # =========================================================================
    # ICA: Implicit Color Association
    # =========================================================================
    if verbose:
        print("ICA: Implicit Color Association")
        print("-" * 50)
    
    ica_results = generate_ica_full(config, seed)
    for r in ica_results:
        cs_short = r.color_system.replace('iscc_', '')
        filename = f"ica_{cs_short}.csv"
        filepath = os.path.join(output_dir, filename)
        r.dataframe.to_csv(filepath, index=False)
        r = GenerationResult(r.task, r.color_system, r.n_prompts, filepath, r.dataframe)
        all_results.append(r)
        if verbose:
            print(f"  {r.color_system}: {r.n_prompts:,} prompts")
    
    # =========================================================================
    # Summary
    # =========================================================================
    total_prompts = sum(r.n_prompts for r in all_results)
    
    if verbose:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        task_totals = {}
        for r in all_results:
            task_key = r.task.split('_')[0] if '_' in r.task else r.task
            if task_key not in task_totals:
                task_totals[task_key] = 0
            task_totals[task_key] += r.n_prompts
        
        for task, count in task_totals.items():
            display_name = TASK_DISPLAY_NAMES.get(task, task.upper())
            print(f"  {display_name}: {count:,} prompts")
        
        print(f"\n  TOTAL: {total_prompts:,} prompts")
        print(f"  Files: {len(all_results)}")
        print("=" * 70)
    
    # Save manifest
    manifest = {
        'version': 'v3',
        'total_prompts': total_prompts,
        'n_files': len(all_results),
        'seed': seed,
        'distribution': 'paper-aligned',
        'naming_convention': {
            'cna': 'Color Name Accuracy (was Task 1)',
            'ncu': 'Numeric Color Understanding (was Task 2)',
            'coa': 'Color-Object Association (was Task 3)',
            'moc': 'Multi-Object Color Composition (was Task 4)',
            'ica': 'Implicit Color Association (was Task 5)',
        },
        'config': {
            'l3_shades_per_l2': config.l3_shades_per_l2,
            'cna_css_objects': config.cna_css_objects,
            'cna_l3_objects': config.cna_l3_objects,
            'ncu_hex_objects': config.ncu_hex_objects,
            'coa_templates': config.coa_templates,
            'coa_l3_prompts_per_color': config.coa_l3_prompts_per_color,
            'moc_templates': config.moc_templates,
            'moc_appearances_per_color': config.moc_appearances_per_color,
            'ica_templates_per_color': config.ica_templates_per_color,
        },
        'files': [
            {
                'task': r.task,
                'color_system': r.color_system,
                'n_prompts': r.n_prompts,
                'filename': os.path.basename(r.filepath) if r.filepath else None,
            }
            for r in all_results
        ]
    }
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return all_results


if __name__ == "__main__":
    results = generate_full_benchmark(
        output_dir="./full_bench_prompt",
        seed=42,
        verbose=True,
    )
