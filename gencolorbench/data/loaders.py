"""
Data loading utilities for GenColorBench.

Handles loading of:
- Negative labels for object segmentation
- Color neighborhood mappings
- Color lookup tables (CSS, ISCC-NBS)
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union


def load_negative_labels(neg_csv_path: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Load negative labels from CSV.
    
    Negative labels help filter out parts of an object that shouldn't
    be included in color evaluation (e.g., exclude "handle" from "cup").
    
    Args:
        neg_csv_path: Path to negative labels CSV
    
    Returns:
        Dictionary mapping object_name -> list of negative labels
    """
    neg_csv_path = Path(neg_csv_path)
    
    if not neg_csv_path.exists():
        print(f"Warning: Negative labels file not found: {neg_csv_path}")
        return {}
    
    df = pd.read_csv(neg_csv_path)
    neg_labels = {}
    
    for _, row in df.iterrows():
        obj_name = str(row['Class_Name']).lower().strip()
        neg_str = str(row['Negative_Labels']) if pd.notna(row['Negative_Labels']) else ''
        labels = [l.strip() for l in neg_str.split('.') if l.strip()]
        neg_labels[obj_name] = labels
    
    return neg_labels


def load_color_neighborhoods(colors_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Load color neighborhood files.
    
    Neighborhoods define which colors are considered "similar" for
    matching purposes. Uses ISCC-NBS color naming system.
    
    Args:
        colors_dir: Directory containing neighborhood CSV files
    
    Returns:
        Dictionary with 'css' and 'iscc' DataFrames
    """
    colors_dir = Path(colors_dir)
    neighborhoods = {}
    
    css_path = colors_dir / 'css_neighborhood.csv'
    if css_path.exists():
        neighborhoods['css'] = pd.read_csv(css_path)
    
    iscc_path = colors_dir / 'iscc_neighborhood.csv'
    if iscc_path.exists():
        neighborhoods['iscc'] = pd.read_csv(iscc_path)
    
    return neighborhoods


def load_color_tables(colors_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Load color lookup tables.
    
    Tables provide RGB values for named colors in different systems:
    - CSS3/X11 named colors
    - ISCC-NBS Level 1/2/3 colors
    
    Args:
        colors_dir: Directory containing color table CSV files
    
    Returns:
        Dictionary with 'css', 'l1', 'l2', 'l3' DataFrames
    """
    colors_dir = Path(colors_dir)
    tables = {}
    
    file_mappings = {
        'l1': 'iscc_nbs_l1.csv',
        'l2': 'iscc_nbs_l2.csv',
        'l3': 'iscc_nbs_l3.csv',
        'css': 'css3_x11_colors.csv',
    }
    
    for key, filename in file_mappings.items():
        path = colors_dir / filename
        if path.exists():
            tables[key] = pd.read_csv(path)
    
    return tables
