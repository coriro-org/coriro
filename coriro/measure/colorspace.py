# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Color space conversions.

Conversion chain: sRGB → Linear RGB → OKLab → OKLCH

References:
- OKLab: https://bottosson.github.io/posts/oklab/
- OKLCH: Cylindrical form of OKLab (Lightness, Chroma, Hue)

All conversions are pure NumPy for determinism and no external dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# sRGB ↔ Linear RGB
# =============================================================================


def srgb_to_linear(srgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert sRGB values [0,1] to linear RGB.
    
    sRGB uses a piecewise gamma curve:
    - For values <= 0.04045: linear/12.92
    - For values > 0.04045: ((value + 0.055) / 1.055) ^ 2.4
    """
    srgb = np.asarray(srgb, dtype=np.float64)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4)
    )
    return linear


def linear_to_srgb(linear: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert linear RGB to sRGB values [0,1].
    
    Inverse of srgb_to_linear.
    """
    linear = np.asarray(linear, dtype=np.float64)
    # Clip negative values to avoid NaN in power function
    linear_safe = np.maximum(linear, 0.0)
    srgb = np.where(
        linear_safe <= 0.0031308,
        linear_safe * 12.92,
        1.055 * np.power(linear_safe, 1.0 / 2.4) - 0.055
    )
    return np.clip(srgb, 0.0, 1.0)


# =============================================================================
# Linear RGB ↔ OKLab
# =============================================================================

# Matrices from https://bottosson.github.io/posts/oklab/

# Linear sRGB to LMS (cone responses)
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float64)

# LMS to OKLab
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=np.float64)

# Inverse matrices
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def linear_rgb_to_oklab(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert linear RGB to OKLab.
    
    Args:
        rgb: Array of shape (..., 3) with linear RGB values
        
    Returns:
        Array of shape (..., 3) with OKLab values (L, a, b)
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    
    # RGB to LMS
    lms = np.einsum('...j,ij->...i', rgb, _M1)
    
    # Cube root (handle negative values for out-of-gamut colors)
    lms_cbrt = np.sign(lms) * np.abs(lms) ** (1.0 / 3.0)
    
    # LMS to OKLab
    lab = np.einsum('...j,ij->...i', lms_cbrt, _M2)
    
    return lab


def oklab_to_linear_rgb(lab: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert OKLab to linear RGB.
    
    Args:
        lab: Array of shape (..., 3) with OKLab values (L, a, b)
        
    Returns:
        Array of shape (..., 3) with linear RGB values
    """
    lab = np.asarray(lab, dtype=np.float64)
    
    # OKLab to LMS (cubed)
    lms_cbrt = np.einsum('...j,ij->...i', lab, _M2_INV)
    
    # Cube
    lms = lms_cbrt ** 3
    
    # LMS to RGB
    rgb = np.einsum('...j,ij->...i', lms, _M1_INV)
    
    return rgb


# =============================================================================
# OKLab ↔ OKLCH
# =============================================================================


def oklab_to_oklch(lab: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert OKLab to OKLCH (cylindrical coordinates).
    
    Args:
        lab: Array of shape (..., 3) with OKLab values (L, a, b)
        
    Returns:
        Array of shape (..., 3) with OKLCH values (L, C, H)
        H is in degrees [0, 360)
    """
    lab = np.asarray(lab, dtype=np.float64)
    
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    
    C = np.sqrt(a**2 + b**2)
    H = np.degrees(np.arctan2(b, a)) % 360.0
    
    return np.stack([L, C, H], axis=-1)


def oklch_to_oklab(lch: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert OKLCH to OKLab.
    
    Args:
        lch: Array of shape (..., 3) with OKLCH values (L, C, H)
        H is in degrees
        
    Returns:
        Array of shape (..., 3) with OKLab values (L, a, b)
    """
    lch = np.asarray(lch, dtype=np.float64)
    
    L = lch[..., 0]
    C = lch[..., 1]
    H_rad = np.radians(lch[..., 2])
    
    a = C * np.cos(H_rad)
    b = C * np.sin(H_rad)
    
    return np.stack([L, a, b], axis=-1)


# =============================================================================
# Convenience: sRGB ↔ OKLCH (full chain)
# =============================================================================


def srgb_to_oklch(srgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert sRGB [0,1] to OKLCH.
    
    Full chain: sRGB → Linear RGB → OKLab → OKLCH
    
    Args:
        srgb: Array of shape (..., 3) with sRGB values [0, 1]
        
    Returns:
        Array of shape (..., 3) with OKLCH values (L, C, H)
        - L: Lightness [0, 1]
        - C: Chroma [0, ~0.4 for sRGB gamut]
        - H: Hue in degrees [0, 360)
    """
    linear = srgb_to_linear(srgb)
    lab = linear_rgb_to_oklab(linear)
    lch = oklab_to_oklch(lab)
    return lch


def oklch_to_srgb(lch: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert OKLCH to sRGB [0,1].
    
    Full chain: OKLCH → OKLab → Linear RGB → sRGB
    
    Args:
        lch: Array of shape (..., 3) with OKLCH values (L, C, H)
        
    Returns:
        Array of shape (..., 3) with sRGB values [0, 1]
        Values are clipped to [0, 1] (gamut mapped)
    """
    lab = oklch_to_oklab(lch)
    linear = oklab_to_linear_rgb(lab)
    srgb = linear_to_srgb(linear)
    return np.clip(srgb, 0.0, 1.0)


def srgb_uint8_to_oklch(pixels: NDArray[np.uint8]) -> NDArray[np.float64]:
    """
    Convert uint8 sRGB pixels [0,255] to OKLCH.
    
    Convenience wrapper for common image format.
    
    Args:
        pixels: Array of shape (..., 3) with uint8 sRGB values [0, 255]
        
    Returns:
        Array of shape (..., 3) with OKLCH values
    """
    srgb_float = pixels.astype(np.float64) / 255.0
    return srgb_to_oklch(srgb_float)


def oklch_to_hex(L: float, C: float, H: float | None) -> str:
    """
    Convert OKLCH values to hex color string.
    
    Args:
        L: Lightness [0, 1]
        C: Chroma [0, ~0.4]
        H: Hue in degrees [0, 360), or None for achromatic
        
    Returns:
        Hex color string like "#3941C8"
    """
    # Handle achromatic (no hue)
    if H is None:
        H = 0.0
    
    lch = np.array([L, C, H], dtype=np.float64)
    srgb = oklch_to_srgb(lch)
    
    # Convert to 0-255 and format as hex
    r, g, b = (srgb * 255).round().astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_oklch(hex_color: str) -> tuple[float, float, float | None]:
    """
    Convert hex color string to OKLCH values.
    
    Args:
        hex_color: Hex string like "#3941C8" or "3941C8"
        
    Returns:
        Tuple of (L, C, H) where H may be None for achromatic colors
    """
    # Strip # if present
    hex_color = hex_color.lstrip("#")
    
    # Parse RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Convert to OKLCH
    pixels = np.array([r, g, b], dtype=np.uint8)
    lch = srgb_uint8_to_oklch(pixels)
    
    L, C, H = float(lch[0]), float(lch[1]), float(lch[2])
    
    # Mark as achromatic if chroma is very low
    if C < 0.01:
        H = None
    
    return L, C, H


# =============================================================================
# ΔE Distance (Perceptual Color Difference)
# =============================================================================


def delta_e_oklch(
    L1: float, C1: float, H1: float | None,
    L2: float, C2: float, H2: float | None,
) -> float:
    """
    Calculate perceptual color difference (ΔE) in OKLCH space.
    
    Uses Euclidean distance in OKLab (not OKLCH) because:
    - Hue is angular, direct OKLCH distance would be wrong
    - OKLab is designed for perceptual uniformity
    
    Reference thresholds (OKLab Euclidean, 0-1 scale):
    - ΔE ≈ 0.02: barely perceptible (expert eye)
    - ΔE ≈ 0.04: noticeable difference
    - ΔE ≈ 0.08+: clearly different colors
    
    Args:
        L1, C1, H1: First color in OKLCH (H in degrees, None for achromatic)
        L2, C2, H2: Second color in OKLCH
        
    Returns:
        ΔE value (lower = more similar)
    """
    # Handle achromatic (hue undefined)
    h1 = H1 if H1 is not None else 0.0
    h2 = H2 if H2 is not None else 0.0
    
    # Convert to OKLab for proper Euclidean distance
    lch1 = np.array([L1, C1, h1], dtype=np.float64)
    lch2 = np.array([L2, C2, h2], dtype=np.float64)
    
    lab1 = oklch_to_oklab(lch1)
    lab2 = oklch_to_oklab(lch2)
    
    # Euclidean distance in Lab space
    delta = lab1 - lab2
    return float(np.sqrt(np.sum(delta ** 2)))


def delta_e_oklch_batch(
    colors1: NDArray[np.float64],
    colors2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Vectorized ΔE calculation for arrays of OKLCH colors.
    
    Args:
        colors1: Array of shape (N, 3) with OKLCH values (L, C, H in degrees)
        colors2: Array of shape (N, 3) with OKLCH values
        
    Returns:
        Array of shape (N,) with ΔE values
    """
    lab1 = oklch_to_oklab(colors1)
    lab2 = oklch_to_oklab(colors2)
    
    delta = lab1 - lab2
    return np.sqrt(np.sum(delta ** 2, axis=-1))

