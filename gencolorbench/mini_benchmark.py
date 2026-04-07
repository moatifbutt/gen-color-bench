"""
GenColorBench-Mini: Stratified ~10K Prompt Benchmark (v3 Naming).

Generates a representative subset of the full benchmark with intelligent
stratification across:
- Color systems (L1, L2, L3, CSS)
- Object categories (proportional representation)
- L3 color shades (sampled within each L1 parent hue)

Target: ~10,220 prompts across 5 tasks

Naming Convention (v3):
- CNA: Color Name Accuracy (was Task 1)
- NCU: Numeric Color Understanding (was Task 2)
- COA: Color-Object Association (was Task 3)
- MOC: Multi-Object Color Composition (was Task 4)
- ICA: Implicit Color Association (was Task 5)

CSV Naming:
- cna_iscc_l1.csv, cna_iscc_l2.csv, cna_iscc_l3.csv, cna_css.csv
- ncu_rgb_l1.csv, ncu_rgb_l2.csv, ncu_rgb_l3.csv, ncu_hex_css.csv
- coa_iscc_l1.csv, coa_iscc_l2.csv, coa_iscc_l3.csv, coa_css.csv
- moc_iscc_l1.csv, moc_iscc_l2.csv, moc_iscc_l3.csv, moc_css.csv
- ica_iscc_l1.csv, ica_iscc_l2.csv, ica_iscc_l3.csv, ica_css.csv
"""

import os
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

import pandas as pd
import numpy as np

from .loaders import (
    load_color_system, load_objects, load_task5_templates,
    Color, ObjectItem as Object
)
from .prompts.color_object import TEMPLATES_WITH_GT as COA_TEMPLATES
from .prompts.multi_object import TEMPLATES_WITH_GT as MOC_TEMPLATES


# =============================================================================
# Task Display Names (v3)
# =============================================================================

TASK_DISPLAY_NAMES = {
    "cna": "CNA: Color Name Accuracy",
    "ncu_rgb": "NCU: Numeric Color Understanding (RGB)",
    "ncu_hex": "NCU: Numeric Color Understanding (HEX)",
    "coa": "COA: Color-Object Association",
    "moc": "MOC: Multi-Object Color Composition",
    "ica": "ICA: Implicit Color Association",
}


# =============================================================================
# Generation Result Container
# =============================================================================

@dataclass
class GenerationResult:
    """Container for generation results."""
    task: str
    color_system: str
    n_prompts: int
    filepath: Optional[str]
    dataframe: pd.DataFrame


# =============================================================================
# Templates
# =============================================================================

CNA_TEMPLATES = [
    "A {color} {object}.",
    "The {object} is {color}.",
    "A photo of a {color} {object}.",
    "A {object} that is entirely {color}.",
    "An image of a {color} {object}.",
    "A {color} colored {object}.",
    "A single {color} {object}.",
    "A {object}, and it's {color}.",
    "A {object} in a {color} color.",
    "A {object} rendered in {color} color.",
    "A {object} with a {color} color.",
    "A realistic {object} in {color}.",
]

NCU_RGB_TEMPLATES = [
    "A {object} in rgb({r}, {g}, {b}).",
    "A {object} with the color rgb({r}, {g}, {b}).",
    "A {object} rendered in RGB color rgb({r}, {g}, {b}).",
    "A photo of a {object} in color rgb({r}, {g}, {b}).",
    "A {object} with color rgb({r}, {g}, {b}).",
    "An image of a {object} in rgb({r}, {g}, {b}).",
    "A {object} colored rgb({r}, {g}, {b}).",
    "A realistic {object} in rgb({r}, {g}, {b}).",
]

NCU_HEX_TEMPLATES = [
    "An image of a {object} in hex color {hex}.",
    "A {object} in color {hex}.",
    "A {object} with hex color {hex}.",
    "A close-up of a {object} in the color {hex}.",
    "A {object} rendered in {hex} color.",
    "A photo of a {object} in the color {hex}.",
    "A {object} rendered entirely in {hex}.",
    "A {object} designed in {hex} color.",
    "A realistic {hex}-colored {object}.",
    "A highly detailed {object} in hex {hex}.",
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MiniConfig:
    """Configuration for Mini benchmark generation."""
    
    # L3 shades to sample per L1 parent
    l3_shades_per_l1: int = 4
    
    # CSS colors to sample
    css_sample_size: int = 45
    
    # Object sampling per category (proportional with minimum)
    min_objects_per_category: int = 3
    object_sample_ratio: float = 0.45
    
    # Task-specific object counts per color
    cna_objects_per_color: Dict[str, int] = field(default_factory=lambda: {
        'iscc_l1': 40,
        'iscc_l2': 35,
        'iscc_l3': 30,
        'css': 25,
    })
    
    ncu_objects_per_color: Dict[str, int] = field(default_factory=lambda: {
        'iscc_l1': 25,
        'iscc_l2': 20,
        'iscc_l3': 15,
        'css': 15,
    })
    
    coa_templates_per_color: Dict[str, int] = field(default_factory=lambda: {
        'iscc_l1': 15,
        'iscc_l2': 12,
        'iscc_l3': 8,
        'css': 8,
    })
    
    moc_combos_per_system: Dict[str, int] = field(default_factory=lambda: {
        'iscc_l1': 200,
        'iscc_l2': 350,
        'iscc_l3': 400,
        'css': 350,
    })
    
    ica_templates_per_color: Dict[str, int] = field(default_factory=lambda: {
        'iscc_l1': 10,
        'iscc_l2': 8,
        'iscc_l3': 5,
        'css': 5,
    })
    
    seed: int = 42


# =============================================================================
# L3 Color Stratification
# =============================================================================

def load_l3_to_l1_mapping() -> Dict[int, List[str]]:
    """Load mapping from L1 color IDs to L3 color names."""
    pkg_dir = Path(__file__).parent
    neighborhood_path = pkg_dir / 'data' / 'neighborhoods' / 'iscc_neighborhood.csv'
    
    df = pd.read_csv(neighborhood_path)
    
    mapping = {}
    for l1_id in df['Level1'].unique():
        l3_colors = df[df['Level1'] == l1_id]['ISCCNBS'].unique().tolist()
        mapping[l1_id] = l3_colors
    
    return mapping


def stratify_l3_colors(
    l3_to_l1: Dict[int, List[str]],
    shades_per_l1: int = 4,
    seed: int = 42
) -> List[str]:
    """Stratified sampling of L3 colors within each L1 parent."""
    random.seed(seed)
    
    selected = []
    priority_prefixes = [
        ['Vivid', 'Strong', 'Brilliant'],
        ['Deep', 'Dark', 'Very dark'],
        ['Light', 'Pale', 'Brilliant'],
        ['Grayish', 'Moderate', 'Dusky'],
    ]
    
    for l1_id, l3_colors in sorted(l3_to_l1.items()):
        if len(l3_colors) <= shades_per_l1:
            selected.extend(l3_colors)
        else:
            l1_selected = []
            remaining = list(l3_colors)
            
            for prefixes in priority_prefixes:
                if len(l1_selected) >= shades_per_l1:
                    break
                for prefix in prefixes:
                    matches = [c for c in remaining if c.startswith(prefix)]
                    if matches:
                        choice = random.choice(matches)
                        l1_selected.append(choice)
                        remaining.remove(choice)
                        break
            
            while len(l1_selected) < shades_per_l1 and remaining:
                choice = random.choice(remaining)
                l1_selected.append(choice)
                remaining.remove(choice)
            
            selected.extend(l1_selected)
    
    return selected


def stratify_css_colors(
    colors: List[Color],
    sample_size: int = 45,
    seed: int = 42
) -> List[Color]:
    """Stratified sampling of CSS colors by hue region."""
    random.seed(seed)
    
    hue_groups = {
        'red': [], 'orange': [], 'yellow': [], 'green': [],
        'cyan': [], 'blue': [], 'purple': [], 'neutral': []
    }
    
    for color in colors:
        r, g, b = color.r, color.g, color.b
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        
        if max_c - min_c < 30:
            hue_groups['neutral'].append(color)
        elif r >= g and r >= b:
            if g > b + 50:
                hue_groups['orange'].append(color)
            elif b > g + 50:
                hue_groups['purple'].append(color)
            else:
                hue_groups['red'].append(color)
        elif g >= r and g >= b:
            if b > r + 30:
                hue_groups['cyan'].append(color)
            elif r > b + 30:
                hue_groups['yellow'].append(color)
            else:
                hue_groups['green'].append(color)
        else:
            if r > g + 30:
                hue_groups['purple'].append(color)
            elif g > r + 30:
                hue_groups['cyan'].append(color)
            else:
                hue_groups['blue'].append(color)
    
    selected = []
    total = sum(len(g) for g in hue_groups.values())
    
    for group_name, group_colors in hue_groups.items():
        if len(group_colors) == 0:
            continue
        n_sample = max(2, int(sample_size * len(group_colors) / total))
        n_sample = min(n_sample, len(group_colors))
        selected.extend(random.sample(group_colors, n_sample))
    
    if len(selected) > sample_size:
        selected = random.sample(selected, sample_size)
    elif len(selected) < sample_size:
        remaining = [c for c in colors if c not in selected]
        need = sample_size - len(selected)
        if len(remaining) >= need:
            selected.extend(random.sample(remaining, need))
    
    return selected


def stratify_objects(
    objects: List[Object],
    sample_ratio: float = 0.45,
    min_per_category: int = 3,
    seed: int = 42
) -> List[Object]:
    """Stratified sampling of objects by category."""
    random.seed(seed)
    
    by_category: Dict[str, List[Object]] = {}
    for obj in objects:
        if obj.category not in by_category:
            by_category[obj.category] = []
        by_category[obj.category].append(obj)
    
    selected = []
    for category, cat_objects in by_category.items():
        n_sample = max(min_per_category, int(len(cat_objects) * sample_ratio))
        n_sample = min(n_sample, len(cat_objects))
        selected.extend(random.sample(cat_objects, n_sample))
    
    return selected


# =============================================================================
# Task Generation Functions
# =============================================================================

def generate_cna_mini(
    colors: List[Color],
    objects: List[Object],
    objects_per_color: int,
    color_system: str,
    seed: int
) -> pd.DataFrame:
    """Generate CNA (Color Name Accuracy) prompts."""
    random.seed(seed)
    
    records = []
    idx = 1
    
    for color in colors:
        sampled_objects = random.sample(objects, min(objects_per_color, len(objects)))
        for obj in sampled_objects:
            template = random.choice(CNA_TEMPLATES)
            prompt = template.format(color=color.name, object=obj.name)
            records.append({
                "id": idx,
                "color": color.name,
                "object": obj.name,
                "prompt": prompt,
            })
            idx += 1
    
    return pd.DataFrame(records)


def generate_ncu_rgb_mini(
    colors: List[Color],
    objects: List[Object],
    objects_per_color: int,
    color_system: str,
    seed: int
) -> pd.DataFrame:
    """Generate NCU RGB prompts."""
    random.seed(seed)
    
    records = []
    idx = 1
    
    for color in colors:
        sampled_objects = random.sample(objects, min(objects_per_color, len(objects)))
        for obj in sampled_objects:
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
    
    return pd.DataFrame(records)


def generate_ncu_hex_mini(
    colors: List[Color],
    objects: List[Object],
    objects_per_color: int,
    seed: int
) -> pd.DataFrame:
    """Generate NCU HEX prompts (CSS only)."""
    random.seed(seed)
    
    records = []
    idx = 1
    
    for color in colors:
        sampled_objects = random.sample(objects, min(objects_per_color, len(objects)))
        hex_val = color.hex if color.hex else f"#{color.r:02X}{color.g:02X}{color.b:02X}"
        
        for obj in sampled_objects:
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
    
    return pd.DataFrame(records)


def generate_coa_mini(
    colors: List[Color],
    templates_per_color: int,
    color_system: str,
    seed: int
) -> pd.DataFrame:
    """Generate COA (Color-Object Association) prompts."""
    random.seed(seed)
    
    records = []
    idx = 1
    
    for color in colors:
        sampled_templates = random.sample(
            COA_TEMPLATES, 
            min(templates_per_color, len(COA_TEMPLATES))
        )
        for main_obj, sec_obj, template in sampled_templates:
            prompt = template.format(color=color.name)
            records.append({
                "id": idx,
                "color": color.name,
                "main_obj": main_obj,
                "sec_obj": sec_obj,
                "prompt": prompt,
            })
            idx += 1
    
    return pd.DataFrame(records)


def generate_moc_mini(
    colors: List[Color],
    n_prompts: int,
    color_system: str,
    seed: int
) -> pd.DataFrame:
    """Generate MOC (Multi-Object Color Composition) prompts."""
    random.seed(seed)
    
    records = []
    color_usage = {c.name: 0 for c in colors}
    
    templates = MOC_TEMPLATES.copy()
    random.shuffle(templates)
    
    # Cycle through templates
    template_idx = 0
    for idx in range(1, n_prompts + 1):
        obj1, obj2, obj3, template = templates[template_idx % len(templates)]
        template_idx += 1
        
        n_colors_needed = 3 if obj3 else 2
        
        # Sample colors prioritizing underused
        sorted_colors = sorted(colors, key=lambda c: color_usage[c.name])
        selected_colors = sorted_colors[:n_colors_needed]
        
        for c in selected_colors:
            color_usage[c.name] += 1
        
        prompt = template
        for c in selected_colors:
            prompt = prompt.replace("{color}", c.name, 1)
        
        records.append({
            "id": idx,
            "color_count": n_colors_needed,
            "object1": obj1,
            "color1": selected_colors[0].name,
            "object2": obj2,
            "color2": selected_colors[1].name,
            "object3": obj3 if obj3 else "",
            "color3": selected_colors[2].name if n_colors_needed == 3 else "",
            "prompt": prompt,
        })
    
    return pd.DataFrame(records)


def generate_ica_mini(
    colors: List[Color],
    templates_per_color: int,
    color_system: str,
    seed: int
) -> pd.DataFrame:
    """Generate ICA (Implicit Color Association) prompts."""
    random.seed(seed)
    
    all_templates = load_task5_templates()
    
    records = []
    idx = 1
    
    for color in colors:
        sampled_templates = random.sample(
            all_templates,
            min(templates_per_color, len(all_templates))
        )
        for t in sampled_templates:
            prompt = t.prompt.replace("{color}", color.name)
            records.append({
                "id": idx,
                "color": color.name,
                "object": t.object,
                "ref_object": t.ref_object,
                "prompt": prompt,
            })
            idx += 1
    
    return pd.DataFrame(records)


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_mini_benchmark(
    output_dir: str = "./mini_bench_prompt/",
    config: Optional[MiniConfig] = None,
    seed: int = 42,
    verbose: bool = True,
) -> List[GenerationResult]:
    """
    Generate GenColorBench-Mini with v3 naming convention.
    
    Args:
        output_dir: Directory to save CSV files
        config: MiniConfig instance (uses defaults if None)
        seed: Random seed for reproducibility
        verbose: Print progress
        
    Returns:
        List of GenerationResult objects
    """
    if config is None:
        config = MiniConfig(seed=seed)
    
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total_prompts = 0
    
    if verbose:
        print("=" * 60)
        print("GenColorBench-Mini Generation (v3 Naming)")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Seed: {seed}")
        print("=" * 60)
        print()
    
    # =========================================================================
    # Load and stratify colors
    # =========================================================================
    l1_colors = load_color_system('iscc_l1')
    l2_colors = load_color_system('iscc_l2')
    l3_colors_full = load_color_system('iscc_l3')
    css_colors_full = load_color_system('css')
    
    # Stratify L3 colors
    l3_to_l1 = load_l3_to_l1_mapping()
    l3_selected_names = stratify_l3_colors(l3_to_l1, config.l3_shades_per_l1, seed)
    l3_colors = [c for c in l3_colors_full if c.name in l3_selected_names]
    
    # Stratify CSS colors
    css_colors = stratify_css_colors(css_colors_full, config.css_sample_size, seed)
    
    # Load and stratify objects
    all_objects = load_objects()
    stratified_objects = stratify_objects(
        all_objects, config.object_sample_ratio, config.min_objects_per_category, seed
    )
    
    if verbose:
        print(f"Colors: L1={len(l1_colors)}, L2={len(l2_colors)}, "
              f"L3={len(l3_colors)} (from {len(l3_colors_full)}), "
              f"CSS={len(css_colors)} (from {len(css_colors_full)})")
        print(f"Objects: {len(stratified_objects)} (from {len(all_objects)})")
        print()
    
    color_systems = {
        'iscc_l1': l1_colors,
        'iscc_l2': l2_colors,
        'iscc_l3': l3_colors,
        'css': css_colors,
    }
    
    # =========================================================================
    # CNA: Color Name Accuracy
    # =========================================================================
    if verbose:
        print("CNA: Color Name Accuracy")
        print("-" * 40)
    
    for cs_name, colors in color_systems.items():
        objects_per_color = config.cna_objects_per_color[cs_name]
        df = generate_cna_mini(colors, stratified_objects, objects_per_color, cs_name, seed)
        
        # v3 naming: cna_iscc_l1.csv, cna_css.csv
        filename = f"cna_{cs_name.replace('iscc_', '')}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        results.append(GenerationResult(
            task="cna",
            color_system=cs_name,
            n_prompts=len(df),
            filepath=filepath,
            dataframe=df,
        ))
        total_prompts += len(df)
        
        if verbose:
            print(f"  {cs_name}: {len(colors)} colors × {objects_per_color} objects = {len(df)} prompts")
    
    if verbose:
        print()
    
    # =========================================================================
    # NCU: Numeric Color Understanding (RGB)
    # =========================================================================
    if verbose:
        print("NCU: Numeric Color Understanding (RGB)")
        print("-" * 40)
    
    for cs_name, colors in color_systems.items():
        objects_per_color = config.ncu_objects_per_color[cs_name]
        df = generate_ncu_rgb_mini(colors, stratified_objects, objects_per_color, cs_name, seed)
        
        # v3 naming: ncu_rgb_l1.csv, ncu_rgb_css.csv
        filename = f"ncu_rgb_{cs_name.replace('iscc_', '')}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        results.append(GenerationResult(
            task="ncu_rgb",
            color_system=cs_name,
            n_prompts=len(df),
            filepath=filepath,
            dataframe=df,
        ))
        total_prompts += len(df)
        
        if verbose:
            print(f"  {cs_name}: {len(colors)} colors × {objects_per_color} objects = {len(df)} prompts")
    
    if verbose:
        print()
    
    # =========================================================================
    # NCU: Numeric Color Understanding (HEX) - CSS only
    # =========================================================================
    if verbose:
        print("NCU: Numeric Color Understanding (HEX)")
        print("-" * 40)
    
    objects_per_color = config.ncu_objects_per_color['css']
    df = generate_ncu_hex_mini(css_colors, stratified_objects, objects_per_color, seed)
    
    filename = "ncu_hex_css.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    
    results.append(GenerationResult(
        task="ncu_hex",
        color_system="css",
        n_prompts=len(df),
        filepath=filepath,
        dataframe=df,
    ))
    total_prompts += len(df)
    
    if verbose:
        print(f"  css: {len(css_colors)} colors × {objects_per_color} objects = {len(df)} prompts")
        print()
    
    # =========================================================================
    # COA: Color-Object Association
    # =========================================================================
    if verbose:
        print("COA: Color-Object Association")
        print("-" * 40)
    
    for cs_name, colors in color_systems.items():
        templates_per_color = config.coa_templates_per_color[cs_name]
        df = generate_coa_mini(colors, templates_per_color, cs_name, seed)
        
        filename = f"coa_{cs_name.replace('iscc_', '')}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        results.append(GenerationResult(
            task="coa",
            color_system=cs_name,
            n_prompts=len(df),
            filepath=filepath,
            dataframe=df,
        ))
        total_prompts += len(df)
        
        if verbose:
            print(f"  {cs_name}: {len(colors)} colors × {templates_per_color} templates = {len(df)} prompts")
    
    if verbose:
        print()
    
    # =========================================================================
    # MOC: Multi-Object Color Composition
    # =========================================================================
    if verbose:
        print("MOC: Multi-Object Color Composition")
        print("-" * 40)
    
    for cs_name, colors in color_systems.items():
        n_combos = config.moc_combos_per_system[cs_name]
        df = generate_moc_mini(colors, n_combos, cs_name, seed)
        
        filename = f"moc_{cs_name.replace('iscc_', '')}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        results.append(GenerationResult(
            task="moc",
            color_system=cs_name,
            n_prompts=len(df),
            filepath=filepath,
            dataframe=df,
        ))
        total_prompts += len(df)
        
        if verbose:
            print(f"  {cs_name}: {len(df)} prompts (from {len(colors)} colors)")
    
    if verbose:
        print()
    
    # =========================================================================
    # ICA: Implicit Color Association
    # =========================================================================
    if verbose:
        print("ICA: Implicit Color Association")
        print("-" * 40)
    
    for cs_name, colors in color_systems.items():
        templates_per_color = config.ica_templates_per_color[cs_name]
        df = generate_ica_mini(colors, templates_per_color, cs_name, seed)
        
        filename = f"ica_{cs_name.replace('iscc_', '')}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        results.append(GenerationResult(
            task="ica",
            color_system=cs_name,
            n_prompts=len(df),
            filepath=filepath,
            dataframe=df,
        ))
        total_prompts += len(df)
        
        if verbose:
            print(f"  {cs_name}: {len(colors)} colors × {templates_per_color} templates = {len(df)} prompts")
    
    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        task_totals = {}
        for r in results:
            if r.task not in task_totals:
                task_totals[r.task] = 0
            task_totals[r.task] += r.n_prompts
        
        for task, count in task_totals.items():
            display_name = TASK_DISPLAY_NAMES.get(task, task.upper())
            print(f"  {display_name}: {count:,} prompts")
        
        print(f"\n  TOTAL: {total_prompts:,} prompts")
        print(f"  Files: {len(results)}")
        print("=" * 60)
    
    # Save manifest
    manifest = {
        'version': 'v3',
        'total_prompts': total_prompts,
        'n_files': len(results),
        'seed': seed,
        'naming_convention': {
            'cna': 'Color Name Accuracy (was Task 1)',
            'ncu': 'Numeric Color Understanding (was Task 2)',
            'coa': 'Color-Object Association (was Task 3)',
            'moc': 'Multi-Object Color Composition (was Task 4)',
            'ica': 'Implicit Color Association (was Task 5)',
        },
        'stratification': {
            'l3_shades_per_l1': config.l3_shades_per_l1,
            'css_sample_size': config.css_sample_size,
            'n_objects_stratified': len(stratified_objects),
            'n_l3_colors_stratified': len(l3_colors),
        },
        'files': [
            {
                'task': r.task,
                'color_system': r.color_system,
                'n_prompts': r.n_prompts,
                'filename': os.path.basename(r.filepath),
            }
            for r in results
        ]
    }
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = generate_mini_benchmark(
        output_dir="./mini_bench_prompt",
        seed=42,
        verbose=True,
    )
