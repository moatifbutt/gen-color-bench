"""
Color difference metrics.

Implements CIEDE2000, delta chroma, and hue angle metrics.
"""

import numpy as np


def delta_chroma(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    Euclidean distance in a*b* plane (chroma difference).
    
    Args:
        lab1: First LAB color (3,)
        lab2: Second LAB color (3,)
    
    Returns:
        Chroma difference
    """
    return float(np.sqrt((lab1[1] - lab2[1])**2 + (lab1[2] - lab2[2])**2))


def mae_hue(lab1: np.ndarray, lab2: np.ndarray, chroma_threshold: float = 5.0) -> float:
    """
    Mean absolute error in hue angle (degrees).
    
    Returns 0 for achromatic colors (chroma below threshold).
    
    Args:
        lab1: First LAB color (3,)
        lab2: Second LAB color (3,)
        chroma_threshold: Minimum chroma for hue comparison
    
    Returns:
        Hue angle difference in degrees [0, 180]
    """
    c1 = np.sqrt(lab1[1]**2 + lab1[2]**2)
    c2 = np.sqrt(lab2[1]**2 + lab2[2]**2)
    
    # Achromatic colors have undefined hue
    if c1 < chroma_threshold or c2 < chroma_threshold:
        return 0.0
    
    h1 = np.degrees(np.arctan2(lab1[2], lab1[1]))
    h2 = np.degrees(np.arctan2(lab2[2], lab2[1]))
    
    # Normalize to [0, 360)
    h1 = h1 + 360 if h1 < 0 else h1
    h2 = h2 + 360 if h2 < 0 else h2
    
    diff = abs(h1 - h2)
    return float(diff if diff <= 180 else 360 - diff)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    CIEDE2000 color difference formula.
    
    Reference: Sharma et al. (2005), "The CIEDE2000 Color-Difference Formula"
    
    Args:
        lab1: First LAB color (3,)
        lab2: Second LAB color (3,)
    
    Returns:
        CIEDE2000 color difference (ΔE00)
    """
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])
    
    # Calculate C (chroma)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0
    
    # Calculate G factor
    C_avg_7 = C_avg**7
    G = 0.5 * (1 - np.sqrt(C_avg_7 / (C_avg_7 + 25**7)))
    
    # Adjusted a' values
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    # C' values
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    # h' values
    def calc_h_prime(a_p, b_val):
        if a_p == 0 and b_val == 0:
            return 0.0
        h = np.degrees(np.arctan2(b_val, a_p))
        return h + 360 if h < 0 else h
    
    h1_prime = calc_h_prime(a1_prime, b1)
    h2_prime = calc_h_prime(a2_prime, b2)
    
    # ΔL', ΔC', Δh'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    # Δh' calculation with wrapping
    if C1_prime * C2_prime == 0:
        delta_h_prime = 0
    else:
        diff = h2_prime - h1_prime
        if abs(diff) <= 180:
            delta_h_prime = diff
        elif diff > 180:
            delta_h_prime = diff - 360
        else:
            delta_h_prime = diff + 360
    
    # ΔH'
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2))
    
    # Average values
    L_avg_prime = (L1 + L2) / 2.0
    C_avg_prime = (C1_prime + C2_prime) / 2.0
    
    # h' average with wrapping
    if C1_prime * C2_prime == 0:
        h_avg_prime = h1_prime + h2_prime
    else:
        diff = abs(h1_prime - h2_prime)
        sum_h = h1_prime + h2_prime
        if diff <= 180:
            h_avg_prime = sum_h / 2.0
        elif sum_h < 360:
            h_avg_prime = (sum_h + 360) / 2.0
        else:
            h_avg_prime = (sum_h - 360) / 2.0
    
    # T factor
    T = (1 - 0.17 * np.cos(np.radians(h_avg_prime - 30)) +
         0.24 * np.cos(np.radians(2 * h_avg_prime)) +
         0.32 * np.cos(np.radians(3 * h_avg_prime + 6)) -
         0.20 * np.cos(np.radians(4 * h_avg_prime - 63)))
    
    # Δθ and RC
    delta_theta = 30 * np.exp(-((h_avg_prime - 275) / 25)**2)
    C_avg_prime_7 = C_avg_prime**7
    R_C = 2 * np.sqrt(C_avg_prime_7 / (C_avg_prime_7 + 25**7))
    
    # SL, SC, SH
    L_avg_prime_minus_50_sq = (L_avg_prime - 50)**2
    S_L = 1 + (0.015 * L_avg_prime_minus_50_sq) / np.sqrt(20 + L_avg_prime_minus_50_sq)
    S_C = 1 + 0.045 * C_avg_prime
    S_H = 1 + 0.015 * C_avg_prime * T
    
    # RT
    R_T = -np.sin(np.radians(2 * delta_theta)) * R_C
    
    # Final ΔE00
    delta_E = np.sqrt(
        (delta_L_prime / S_L)**2 +
        (delta_C_prime / S_C)**2 +
        (delta_H_prime / S_H)**2 +
        R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
    )
    
    return float(delta_E)
