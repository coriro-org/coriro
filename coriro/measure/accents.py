# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Accent region detection for solid-color UI elements.

Detects small but significant solid-color regions (CTAs, icons, badges)
that may be missed by the area-dominant surface pass. Filters by absolute
pixel count rather than percentage, catching small but coherent UI elements.

Requires: scipy (pip install scipy)
"""

from __future__ import annotations

from typing import Optional
from collections import Counter
import numpy as np
from numpy.typing import NDArray

from coriro.schema.color_measurement import (
    AccentMeasurement,
    AccentRegion,
    OKLCHColor,
    WeightedColor,
)
from coriro.measure.colorspace import srgb_uint8_to_oklch, delta_e_oklch


def is_available() -> bool:
    """Check if accent region detection is available (scipy installed)."""
    try:
        from scipy import ndimage
        return True
    except ImportError:
        return False


def extract_accent_regions(
    rgb_pixels: NDArray[np.uint8],
    height: int,
    width: int,
    *,
    min_pixels: int = 2000,
    color_tolerance: int = 12,  # Reduces anti-aliasing fragmentation
    existing_palette: Optional[tuple[WeightedColor, ...]] = None,
    delta_e_exclude: float = 0.05,
) -> Optional[AccentMeasurement]:
    """
    Detect solid-color accent regions in the image.
    
    Args:
        rgb_pixels: Flattened RGB pixel array (N, 3)
        height: Image height
        width: Image width
        min_pixels: Minimum pixel count for a region to be included
        color_tolerance: RGB tolerance for color quantization (0-255)
        existing_palette: Palette colors to exclude (already captured)
        delta_e_exclude: ΔE threshold for excluding colors similar to palette
    
    Returns:
        AccentMeasurement with detected regions, or None if no accents found.
        Returns None if scipy is not installed.
    """
    # Lazy import - scipy is optional
    try:
        from scipy import ndimage
    except ImportError:
        return None
    
    # Reshape to 2D image
    img = rgb_pixels.reshape(height, width, 3)
    
    # Quantize colors aggressively to reduce unique colors
    quantized = (img // color_tolerance) * color_tolerance + color_tolerance // 2
    quantized = np.clip(quantized, 0, 255).astype(np.uint8)
    
    # Convert to single integer for fast counting
    color_ids = (
        quantized[:, :, 0].astype(np.int32) * 65536 +
        quantized[:, :, 1].astype(np.int32) * 256 +
        quantized[:, :, 2].astype(np.int32)
    )
    
    # OPTIMIZATION: First count total pixels per color to filter early
    flat_ids = color_ids.flatten()
    color_counts = Counter(flat_ids)
    
    # Only process colors that have at least min_pixels total
    # (Even if not contiguous, they won't have a region >= min_pixels if total < min_pixels)
    promising_colors = [c for c, count in color_counts.items() if count >= min_pixels]
    
    # Build set of palette colors to exclude
    palette_oklch = []
    if existing_palette:
        for wc in existing_palette:
            palette_oklch.append((wc.color.L, wc.color.C, wc.color.H))
    
    # Detect solid regions only for promising colors
    accent_regions: list[AccentRegion] = []
    
    for color_id in promising_colors:
        # Create binary mask for this color
        mask = (color_ids == color_id)
        
        # Find connected components
        labeled, num_features = ndimage.label(mask)
        
        if num_features == 0:
            continue
        
        # Get size of each component
        component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        
        for i, size in enumerate(component_sizes):
            if size < min_pixels:
                continue
            
            # Get the actual color from the center of this region
            component_mask = (labeled == i + 1)
            coords = np.argwhere(component_mask)
            center_y, center_x = coords[len(coords) // 2]
            
            # Get the ORIGINAL (non-quantized) color at this location
            r, g, b = img[center_y, center_x]
            
            # SOLIDITY CHECK: Verify this region is truly solid, not a gradient fragment
            if not _is_solid_region(img, component_mask):
                continue
            
            # Convert to OKLCH
            oklch = srgb_uint8_to_oklch(np.array([[r, g, b]], dtype=np.uint8))[0]
            L, C, H = float(oklch[0]), float(oklch[1]), float(oklch[2])
            
            # Check if this color is too similar to existing palette
            is_duplicate = False
            for p_L, p_C, p_H in palette_oklch:
                de = delta_e_oklch(L, C, H if C >= 0.02 else None,
                                   p_L, p_C, p_H if p_C >= 0.02 else None)
                if de < delta_e_exclude:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            # Determine grid location (R1C1, R1C2, etc.)
            location = _get_grid_location(center_y, center_x, height, width)
            
            # Create hex
            hex_color = f"#{r:02X}{g:02X}{b:02X}"
            
            # Create OKLCHColor (wrap hue 360 to 0)
            H_rounded = round(H, 0) % 360 if C >= 0.02 else None
            color = OKLCHColor(
                L=round(L, 2),
                C=round(C, 2),
                H=H_rounded,
                sample_hex=hex_color,
            )
            
            accent_regions.append(AccentRegion(
                color=color,
                pixels=int(size),
                location=location,
            ))
    
    if not accent_regions:
        return None
    
    # Sort by pixel count (largest first)
    accent_regions.sort(key=lambda r: r.pixels, reverse=True)
    
    # Deduplicate regions with similar colors (keep largest)
    unique_regions: list[AccentRegion] = []
    
    for region in accent_regions:
        is_duplicate = False
        for existing in unique_regions:
            de = delta_e_oklch(
                region.color.L, region.color.C, 
                region.color.H if region.color.C >= 0.02 else None,
                existing.color.L, existing.color.C,
                existing.color.H if existing.color.C >= 0.02 else None
            )
            if de < 0.08:  # Similar color threshold for accents
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_regions.append(region)
    
    if not unique_regions:
        return None
    
    return AccentMeasurement(
        regions=tuple(unique_regions[:5]),  # Cap at 5 accent regions
        min_pixels=min_pixels,
    )


def _is_solid_region(
    img: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    max_std: float = 4.0,
    percentile_pct: float = 0.85,
    percentile_tolerance: int = 20,
) -> bool:
    """
    Check if a region is truly solid (uniform color) vs a gradient fragment.

    Two-pronged approach:
    1. Relaxed std check (max_std=4.0) catches gross gradients while
       allowing anti-aliased edges that inflate std.
    2. Percentile check: verify that 85% of pixels are within 20 RGB units
       of the median. This handles anti-aliased edges (which affect only a
       few % of edge pixels) while still rejecting true gradients.

    Args:
        img: Original RGB image
        mask: Boolean mask for this region
        max_std: Maximum average std deviation to be considered solid
        percentile_pct: Fraction of pixels that must be near median
        percentile_tolerance: Max RGB distance from median for "near"

    Returns:
        True if region is solid, False if it's a gradient fragment
    """
    region_pixels = img[mask].astype(np.float32)

    # Prong 1: relaxed std — reject obvious gradients
    r_std = np.std(region_pixels[:, 0])
    g_std = np.std(region_pixels[:, 1])
    b_std = np.std(region_pixels[:, 2])
    avg_std = (r_std + g_std + b_std) / 3

    if avg_std > max_std:
        return False

    # Prong 2: percentile-based — verify core is uniform
    median = np.median(region_pixels, axis=0)  # (3,)
    distances = np.max(np.abs(region_pixels - median), axis=1)  # max channel diff
    near_median = np.sum(distances <= percentile_tolerance)
    fraction_near = near_median / len(region_pixels)

    return fraction_near >= percentile_pct


def _get_grid_location(y: int, x: int, height: int, width: int) -> str:
    """Determine which 2x2 grid cell a point falls into."""
    row = 1 if y < height // 2 else 2
    col = 1 if x < width // 2 else 2
    return f"R{row}C{col}"

