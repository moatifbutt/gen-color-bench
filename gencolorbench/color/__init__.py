"""
Color processing modules for GenColorBench.

Includes:
- Color space conversions (RGB ↔ LAB)
- Color metrics (CIEDE2000, delta chroma, MAE hue)
- Dominant color extraction (OneHue algorithm)
- Color matching with neighborhood support
"""
from .conversion import rgb_to_lab, lab_to_rgb, rgb_to_luv, luv_to_lab
from .metrics import ciede2000, delta_chroma, mae_hue
from .extraction import extract_dominant_color
from .matching import color_matches_target, get_color_neighbors, lookup_color_rgb, get_target_lab

__all__ = [
    # Conversion
    "rgb_to_lab",
    "lab_to_rgb",
    "rgb_to_luv",
    "luv_to_lab",
    # Metrics
    "ciede2000",
    "delta_chroma",
    "mae_hue",
    # Extraction
    "extract_dominant_color",
    # Matching
    "color_matches_target",
    "get_color_neighbors",
    "lookup_color_rgb",
    "get_target_lab",
]
