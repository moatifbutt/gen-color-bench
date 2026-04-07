"""
Color matching utilities with neighborhood support.

Handles target color lookup and matching with JND thresholds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from .conversion import rgb_to_lab
from .metrics import ciede2000, delta_chroma, mae_hue


def lookup_color_rgb(
    color_name: str,
    color_tables: Dict[str, pd.DataFrame],
    color_system: str
) -> Optional[np.ndarray]:
    """
    Look up RGB values for a color name.
    
    Args:
        color_name: Color name to look up
        color_tables: Dictionary of color lookup tables
        color_system: Color system (css, l1, l2, l3)
    
    Returns:
        RGB array (3,) or None if not found
    """
    if color_system not in color_tables:
        return None
    
    df = color_tables[color_system]
    color_name_clean = color_name.strip().lower()
    
    if color_system == 'css':
        # CSS table may have different column names
        for col in ['css_name', 'color_name']:
            if col in df.columns:
                mask = df[col].str.lower() == color_name_clean
                if not mask.any():
                    # Try without spaces
                    mask = df[col].str.lower().str.replace(' ', '') == color_name_clean.replace(' ', '')
                if mask.any():
                    row = df[mask].iloc[0]
                    return np.array([int(row['r']), int(row['g']), int(row['b'])])
    else:
        # ISCC tables
        if 'color_name' in df.columns:
            mask = df['color_name'].str.lower() == color_name_clean
            if mask.any():
                row = df[mask].iloc[0]
                return np.array([int(row['r']), int(row['g']), int(row['b'])])
    
    return None


def get_target_lab(
    color_name: str,
    color_tables: Dict[str, pd.DataFrame],
    color_system: str,
    row: Optional[pd.Series] = None
) -> Optional[np.ndarray]:
    """
    Get target LAB color from row data or color tables.
    
    Priority:
    1. RGB columns in row (R, G, B or r, g, b)
    2. Hex column in row
    3. Color name lookup in tables
    
    Args:
        color_name: Color name
        color_tables: Color lookup tables
        color_system: Color system identifier
        row: Optional row with RGB/hex data
    
    Returns:
        LAB array (3,) or None if not found
    """
    target_lab = None
    
    # Try RGB columns first
    if row is not None:
        for r_col, g_col, b_col in [('R', 'G', 'B'), ('r', 'g', 'b')]:
            if r_col in row.index and pd.notna(row[r_col]):
                target_lab = rgb_to_lab(np.array([
                    int(row[r_col]),
                    int(row[g_col]),
                    int(row[b_col])
                ]))
                break
        
        # Try hex column
        if target_lab is None and 'hex' in row.index and pd.notna(row['hex']):
            hex_val = str(row['hex']).strip().lstrip('#')
            if len(hex_val) == 6:
                try:
                    r = int(hex_val[0:2], 16)
                    g = int(hex_val[2:4], 16)
                    b = int(hex_val[4:6], 16)
                    target_lab = rgb_to_lab(np.array([r, g, b]))
                except ValueError:
                    pass
    
    # Try color name lookup
    if target_lab is None:
        rgb = lookup_color_rgb(color_name, color_tables, color_system)
        if rgb is not None:
            target_lab = rgb_to_lab(rgb)
    
    return target_lab


def get_color_neighbors(
    color_name: str,
    color_tables: Dict[str, pd.DataFrame],
    neighborhoods: Dict[str, pd.DataFrame],
    color_system: str,
    target_lab: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Get neighbor LAB colors for a color name.
    
    For CSS colors: Map to closest ISCC L3 color, then get L3 neighbors.
    For ISCC L1/L2/L3: Get neighbors from the corresponding level.
    
    Note: No cap on number of neighbors - returns all neighbors in the
    color category for comprehensive matching.
    
    Args:
        color_name: Color name
        color_tables: Color lookup tables
        neighborhoods: Neighborhood lookup tables
        color_system: Color system (css, l1, l2, l3)
        target_lab: Target LAB for CSS → L3 mapping
    
    Returns:
        List of neighbor LAB colors
    """
    neighbors_lab = []
    
    # For CSS: map to L3 neighborhood
    if color_system == 'css':
        neighbors_lab = _get_css_neighbors(target_lab, color_tables, neighborhoods)
    # For ISCC levels
    elif color_system in ['l1', 'l2', 'l3']:
        neighbors_lab = _get_iscc_neighbors(color_name, color_system, color_tables, neighborhoods)
    
    return neighbors_lab


def _get_css_neighbors(
    target_lab: Optional[np.ndarray],
    color_tables: Dict[str, pd.DataFrame],
    neighborhoods: Dict[str, pd.DataFrame]
) -> List[np.ndarray]:
    """Get neighbors for CSS colors by mapping to closest L3."""
    neighbors_lab = []
    
    if 'iscc' not in neighborhoods or 'l3' not in color_tables:
        return neighbors_lab
    
    if target_lab is None:
        return neighbors_lab
    
    # Find closest L3 color to target CSS color
    l3_df = color_tables['l3']
    best_l3_id = None
    best_dist = float('inf')
    
    for _, row in l3_df.iterrows():
        l3_rgb = np.array([int(row['r']), int(row['g']), int(row['b'])])
        l3_lab = rgb_to_lab(l3_rgb)
        dist = ciede2000(target_lab, l3_lab)
        if dist < best_dist:
            best_dist = dist
            best_l3_id = int(row['id'])
    
    # Get all colors in that L3 neighborhood (no cap)
    if best_l3_id is not None:
        iscc_df = neighborhoods['iscc']
        if 'Level3' in iscc_df.columns:
            mask = iscc_df['Level3'] == best_l3_id
            if mask.any():
                all_rgb = iscc_df[mask][['R', 'G', 'B']].values
                for rgb in all_rgb:
                    neighbors_lab.append(rgb_to_lab(np.array(rgb)))
    
    return neighbors_lab


def _get_iscc_neighbors(
    color_name: str,
    color_system: str,
    color_tables: Dict[str, pd.DataFrame],
    neighborhoods: Dict[str, pd.DataFrame]
) -> List[np.ndarray]:
    """Get neighbors for ISCC colors (L1/L2/L3)."""
    neighbors_lab = []
    
    if 'iscc' not in neighborhoods:
        return neighbors_lab
    
    df = neighborhoods['iscc']
    color_name_clean = color_name.strip().lower()
    
    level_id = None
    level_col = f'Level{color_system[1]}'  # Level1, Level2, Level3
    
    if color_system in color_tables:
        ct = color_tables[color_system]
        mask = ct['color_name'].str.lower() == color_name_clean
        if mask.any():
            level_id = int(ct[mask].iloc[0]['id'])
    
    # Get all colors in the neighborhood (no cap)
    if level_id is not None and level_col in df.columns:
        mask = df[level_col] == level_id
        if mask.any():
            all_rgb = df[mask][['R', 'G', 'B']].values
            for rgb in all_rgb:
                neighbors_lab.append(rgb_to_lab(np.array(rgb)))
    
    return neighbors_lab


def color_matches_target(
    dominant_lab: np.ndarray,
    target_lab: np.ndarray,
    neighbors_lab: List[np.ndarray],
    jnd: float = 5.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if extracted color matches target (with neighborhood fallback).
    
    A color matches if ALL three conditions are met:
    - delta_chroma < jnd
    - ciede2000 < jnd  
    - mae_hue < jnd * 6
    
    First checks target directly, then tries neighbors until a match is found.
    
    Args:
        dominant_lab: Extracted dominant LAB color
        target_lab: Target LAB color
        neighbors_lab: List of neighbor LAB colors
        jnd: Just Noticeable Difference threshold (default 5.0)
    
    Returns:
        Tuple of (matches, metrics_dict)
    """
    # Check against target first
    dc = delta_chroma(dominant_lab, target_lab)
    de = ciede2000(dominant_lab, target_lab)
    mh = mae_hue(dominant_lab, target_lab)
    
    target_matches = (dc < jnd and de < jnd and mh < jnd * 6)
    
    metrics = {
        'delta_chroma': dc,
        'ciede2000': de,
        'mae_hue': mh,
        'matched_color': target_lab.tolist(),
        'match_type': 'target' if target_matches else 'none',
    }
    
    if target_matches:
        metrics['matches'] = True
        return True, metrics
    
    # Try neighbors - stop when first match found
    best_total = dc + de + mh
    
    for ni, neighbor in enumerate(neighbors_lab):
        n_dc = delta_chroma(dominant_lab, neighbor)
        n_de = ciede2000(dominant_lab, neighbor)
        n_mh = mae_hue(dominant_lab, neighbor)
        
        neighbor_matches = (n_dc < jnd and n_de < jnd and n_mh < jnd * 6)
        
        if neighbor_matches:
            total = n_dc + n_de + n_mh
            if total < best_total:
                best_total = total
                metrics = {
                    'delta_chroma': n_dc,
                    'ciede2000': n_de,
                    'mae_hue': n_mh,
                    'matched_color': neighbor.tolist(),
                    'match_type': 'neighbor',
                    'neighbor_idx': ni,
                    'matches': True,
                }
                
                # Early exit if very good match
                if n_dc < jnd / 2 and n_de < jnd / 2:
                    break
    
    if metrics.get('matches'):
        return True, metrics
    
    metrics['matches'] = False
    return False, metrics
