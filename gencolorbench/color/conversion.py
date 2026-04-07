"""
Color space conversion utilities.

Implements RGB ↔ CIELAB conversions using D65 illuminant.
"""

import numpy as np


# D65 reference white
XYZ_REF = np.array([0.95047, 1.0, 1.08883])

# sRGB to XYZ matrix
RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

# XYZ to sRGB matrix (inverse)
XYZ_TO_RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]
])

# LAB constants
LAB_EPSILON = 0.008856
LAB_KAPPA = 903.3


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB [0-255] to CIELAB.
    
    Args:
        rgb: RGB values as (3,) or (N, 3) array with values in [0, 255]
    
    Returns:
        LAB values as (3,) or (N, 3) array
    """
    single = rgb.ndim == 1
    if single:
        rgb = rgb.reshape(1, -1)
    
    # Normalize to [0, 1]
    rgb_norm = rgb.astype(np.float64) / 255.0
    
    # sRGB gamma expansion
    mask = rgb_norm > 0.04045
    rgb_linear = np.where(
        mask,
        ((rgb_norm + 0.055) / 1.055) ** 2.4,
        rgb_norm / 12.92
    )
    
    # RGB to XYZ
    xyz = rgb_linear @ RGB_TO_XYZ.T
    
    # Normalize by reference white
    xyz_norm = xyz / XYZ_REF
    
    # XYZ to LAB
    f_xyz = np.where(
        xyz_norm > LAB_EPSILON,
        np.cbrt(xyz_norm),
        (LAB_KAPPA * xyz_norm + 16) / 116
    )
    
    L = 116 * f_xyz[:, 1] - 16
    a = 500 * (f_xyz[:, 0] - f_xyz[:, 1])
    b = 200 * (f_xyz[:, 1] - f_xyz[:, 2])
    
    lab = np.stack([L, a, b], axis=1)
    
    return lab[0] if single else lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert CIELAB to RGB [0-255].
    
    Args:
        lab: LAB values as (3,) or (N, 3) array
    
    Returns:
        RGB values as (3,) or (N, 3) array with values in [0, 255]
    """
    single = lab.ndim == 1
    if single:
        lab = lab.reshape(1, -1)
    
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    
    # LAB to XYZ (f values)
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    # Inverse f function
    xr = np.where(fx ** 3 > LAB_EPSILON, fx ** 3, (116 * fx - 16) / LAB_KAPPA)
    yr = np.where(L > LAB_KAPPA * LAB_EPSILON, ((L + 16) / 116) ** 3, L / LAB_KAPPA)
    zr = np.where(fz ** 3 > LAB_EPSILON, fz ** 3, (116 * fz - 16) / LAB_KAPPA)
    
    # Denormalize by reference white
    xyz = np.stack([xr, yr, zr], axis=1) * XYZ_REF
    
    # XYZ to RGB (linear)
    rgb_linear = xyz @ XYZ_TO_RGB.T
    rgb_linear = np.clip(rgb_linear, 0, 1)
    
    # sRGB gamma compression
    rgb = np.where(
        rgb_linear > 0.0031308,
        1.055 * (rgb_linear ** (1/2.4)) - 0.055,
        12.92 * rgb_linear
    )
    
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    
    return rgb[0] if single else rgb


def rgb_to_luv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0-255] to CIELUV (D65 illuminant)."""
    from skimage import color as skcolor
    
    single = rgb.ndim == 1
    if single:
        rgb = rgb.reshape(1, 1, 3)
    elif rgb.ndim == 2:
        rgb = rgb.reshape(-1, 1, 3)
    
    rgb_norm = rgb.astype(np.float64) / 255.0
    luv = skcolor.rgb2luv(rgb_norm)
    
    if single:
        return luv.reshape(3)
    else:
        return luv.reshape(-1, 3)


def luv_to_lab(luv: np.ndarray) -> np.ndarray:
    """Convert CIELUV to CIELAB via XYZ (D65 illuminant)."""
    from skimage import color as skcolor
    
    single = luv.ndim == 1
    if single:
        luv = luv.reshape(1, 1, 3)
    elif luv.ndim == 2:
        luv = luv.reshape(-1, 1, 3)
    
    # Luv -> XYZ -> LAB
    xyz = skcolor.luv2xyz(luv)
    lab = skcolor.xyz2lab(xyz)
    
    if single:
        return lab.reshape(3)
    else:
        return lab.reshape(-1, 3)
