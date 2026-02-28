# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
OCR-based text color extraction.

Optional measurement pass that uses Tesseract OCR to identify text regions,
then extracts foreground colors from those pixels.

Requires: pytesseract (pip install pytesseract) and Tesseract OCR on the system.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from coriro.schema import TextColorMeasurement, WeightedColor, OKLCHColor
from coriro.measure.colorspace import srgb_uint8_to_oklch


def is_available() -> bool:
    """Check if text color extraction is available (pytesseract installed)."""
    try:
        import pytesseract
        return True
    except ImportError:
        return False


def extract_text_colors(
    pixels: NDArray[np.uint8],
    *,
    max_colors: int = 5,
    min_area_pct: float = 0.1,
    delta_e_threshold: float = 0.15,  # Collapse threshold for anti-aliased text
    exclude_colors: Optional[list[tuple[int, int, int]]] = None,
    exclude_delta_e: float = 0.05,  # Exclude colors within ΔE 0.05 of background
) -> Optional[TextColorMeasurement]:
    """
    Extract text foreground colors using OCR.
    
    Uses Tesseract OCR to detect text regions, creates a mask of text pixels,
    then extracts the most common colors from those pixels.
    
    Key insight: Within text bounding boxes, the BACKGROUND dominates (text strokes
    are thin). So we find the dominant color in each box and EXCLUDE it, keeping
    only the minority colors which are the actual text.
    
    Args:
        pixels: Image array of shape (H, W, 3) with uint8 sRGB values
        max_colors: Maximum number of text colors to return
        min_area_pct: Minimum area percentage for a color to be included
        delta_e_threshold: ΔE threshold for collapsing similar colors
        exclude_colors: RGB tuples to exclude (e.g., known background colors)
        exclude_delta_e: ΔE threshold for excluding colors similar to exclude_colors
        
    Returns:
        TextColorMeasurement if text was detected, None otherwise.
        Returns None if pytesseract is not installed.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        # pytesseract not installed, skip text extraction
        return None
    
    # Convert to PIL Image for pytesseract
    img = Image.fromarray(pixels)
    
    try:
        # Get bounding boxes for each character/word
        # Using image_to_data for detailed bounding box info
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception:
        # OCR failed (Tesseract not installed, image issue, etc.)
        return None
    
    # Create mask of text pixels
    height, width = pixels.shape[:2]
    text_mask = np.zeros((height, width), dtype=bool)
    
    # Mark pixels within detected text bounding boxes
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        # Skip empty detections
        if not data['text'][i].strip():
            continue
        
        # Get bounding box
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        
        # Clamp to image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)
        
        # Mark as text region
        text_mask[y1:y2, x1:x2] = True
    
    # Extract pixels from text regions
    text_pixels = pixels[text_mask]
    
    if len(text_pixels) == 0:
        # No text detected
        return None
    
    # Count exact pixel colors (mode-based for accuracy)
    rgb_tuples = [tuple(p) for p in text_pixels]
    counter = Counter(rgb_tuples)
    
    # Get all colors sorted by frequency
    all_colors = counter.most_common()
    
    if not all_colors:
        return None
    
    # KEY INSIGHT: We need to exclude BACKGROUND colors from text regions.
    # The background color should be provided via exclude_colors (from surface palette).
    # If not provided, we DON'T assume the local dominant is background -
    # that assumption fails when text fills most of the bounding box.
    
    from coriro.measure.colorspace import delta_e_oklch
    
    # Build exclude list from provided colors only (surface backgrounds)
    colors_to_exclude = []
    
    if exclude_colors:
        for rgb in exclude_colors:
            oklch = srgb_uint8_to_oklch(np.array(rgb, dtype=np.uint8))
            colors_to_exclude.append((oklch[0], oklch[1], oklch[2]))
    
    # Filter out colors too similar to excluded colors
    filtered_colors = []
    for (r, g, b), count in all_colors:
        oklch = srgb_uint8_to_oklch(np.array([r, g, b], dtype=np.uint8))
        L, C, H = float(oklch[0]), float(oklch[1]), float(oklch[2])
        
        # Check if this color should be excluded
        should_exclude = False
        for exc_L, exc_C, exc_H in colors_to_exclude:
            de = delta_e_oklch(L, C, H if C >= 0.02 else None, 
                              exc_L, exc_C, exc_H if exc_C >= 0.02 else None)
            if de < exclude_delta_e:
                should_exclude = True
                break
        
        if not should_exclude:
            filtered_colors.append(((r, g, b), count))
    
    # If we filtered everything, return None (no distinct text colors)
    if not filtered_colors:
        return None
    
    # Use filtered colors for the rest
    top_colors = filtered_colors[:max_colors * 2]
    
    # Calculate total for weight normalization (use filtered total)
    total_text_pixels = sum(count for _, count in filtered_colors)
    
    # Build weighted colors
    results: list[WeightedColor] = []
    
    for (r, g, b), count in top_colors:
        weight = count / total_text_pixels
        
        # Skip if below threshold
        if weight * 100 < min_area_pct:
            continue
        
        # Convert to OKLCH
        rgb_array = np.array([r, g, b], dtype=np.uint8)
        oklch = srgb_uint8_to_oklch(rgb_array)
        L, C, H = float(oklch[0]), float(oklch[1]), float(oklch[2])
        
        # Mark as achromatic if low chroma
        h_value = H if C >= 0.02 else None
        
        # Store exact hex as sample_hex
        hex_val = f"#{r:02X}{g:02X}{b:02X}"
        
        color = OKLCHColor(
            L=L,
            C=C,
            H=h_value,
            sample_hex=hex_val,
        )
        results.append(WeightedColor(color=color, weight=weight))
    
    if not results:
        return None
    
    # Collapse similar colors (simple version - just take top N after sorting)
    results = _collapse_similar_text_colors(results, delta_e_threshold)
    
    # Limit to max_colors
    results = results[:max_colors]
    
    # Renormalize weights
    total_weight = sum(wc.weight for wc in results)
    if total_weight > 0:
        results = [
            WeightedColor(color=wc.color, weight=wc.weight / total_weight)
            for wc in results
        ]
    
    return TextColorMeasurement(
        colors=tuple(results),
        scope="glyph_foreground",
        coverage="complete",
        min_area_pct=min_area_pct,
        ocr_engine="tesseract",
    )


def _collapse_similar_text_colors(
    colors: list[WeightedColor],
    delta_e_threshold: float,
) -> list[WeightedColor]:
    """
    Collapse similar text colors based on ΔE threshold.
    
    Simple greedy approach: iterate through colors, merge any that are
    within threshold of the current representative.
    """
    from coriro.measure.colorspace import delta_e_oklch
    
    if not colors:
        return colors
    
    # Sort by weight descending
    colors = sorted(colors, key=lambda wc: wc.weight, reverse=True)
    
    result: list[WeightedColor] = []
    used = [False] * len(colors)
    
    for i, wc in enumerate(colors):
        if used[i]:
            continue
        
        # Start with this color
        total_weight = wc.weight
        used[i] = True
        
        # Find similar colors
        for j in range(i + 1, len(colors)):
            if used[j]:
                continue
            
            de = delta_e_oklch(
                wc.color.L, wc.color.C, wc.color.H,
                colors[j].color.L, colors[j].color.C, colors[j].color.H,
            )
            
            if de < delta_e_threshold:
                total_weight += colors[j].weight
                used[j] = True
        
        result.append(WeightedColor(color=wc.color, weight=total_weight))
    
    return result

