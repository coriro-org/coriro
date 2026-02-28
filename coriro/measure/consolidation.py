# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Color consolidation layer.

Post-measurement processing that collapses near-identical colors,
normalizes black/white families, and produces design-friendly palettes.

This is applied AFTER raw measurement, BEFORE serialization.
It does not change what we measure—it changes how we summarize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from coriro.schema import OKLCHColor, WeightedColor
from coriro.measure.colorspace import delta_e_oklch


@dataclass(frozen=True)
class ConsolidationConfig:
    """Configuration for color consolidation."""
    
    # ΔE threshold for collapsing similar colors (OKLab scale)
    # OKLab ΔE is 0-1 scale, NOT 0-100 like CIEDE2000!
    # 0.02 = imperceptible
    # 0.05 = just noticeable
    # 0.10 = clearly different
    # 0.30+ = very different colors
    delta_e_threshold: float = 0.03  # Conservative: only merge near-identical
    
    # Maximum colors in final palette (after collapse)
    max_palette_size: int = 5
    
    # Minimum weight (as fraction) to keep in final palette
    # Colors below this are filtered out as noise
    min_weight: float = 0.01  # 1%
    
    # Black normalization: L < threshold and C < chroma_threshold
    black_l_threshold: float = 0.25
    black_c_threshold: float = 0.02
    
    # White normalization: L > threshold and C < chroma_threshold
    white_l_threshold: float = 0.95
    white_c_threshold: float = 0.02


def _is_achromatic(color: OKLCHColor, threshold: float = 0.02) -> bool:
    """Check if a color is achromatic (near-gray/black/white)."""
    return color.C < threshold


def collapse_similar_colors(
    colors: tuple[WeightedColor, ...],
    config: Optional[ConsolidationConfig] = None,
) -> tuple[WeightedColor, ...]:
    """
    Collapse similar colors based on ΔE threshold.
    
    Colors within delta_e_threshold of each other are merged.
    The representative color is chosen as the one with highest weight
    in the merged group (preserving its sample_hex for accuracy).
    
    Important: Achromatic colors (C < 0.02) are NEVER merged with
    chromatic colors, even if ΔE is small. This prevents dark reds
    from being merged with blacks, etc.
    
    Args:
        colors: Input palette (ordered by weight descending)
        config: Consolidation settings (uses defaults if None)
        
    Returns:
        Collapsed palette, still ordered by weight descending.
        Weights are re-normalized to sum to 1.0.
    """
    if not colors:
        return colors
    
    cfg = config or ConsolidationConfig()
    
    # Track which colors have been merged
    merged: list[bool] = [False] * len(colors)
    result_groups: list[list[WeightedColor]] = []
    
    for i, wc in enumerate(colors):
        if merged[i]:
            continue
        
        # Start a new group with this color
        group = [wc]
        merged[i] = True
        
        # Determine if this color is achromatic
        i_achromatic = _is_achromatic(wc.color)
        
        # Find all similar colors that haven't been merged yet
        for j in range(i + 1, len(colors)):
            if merged[j]:
                continue
            
            # CRITICAL: Don't merge achromatic with chromatic
            j_achromatic = _is_achromatic(colors[j].color)
            if i_achromatic != j_achromatic:
                continue  # Skip - different color family
            
            # Calculate ΔE between representative color and candidate
            de = delta_e_oklch(
                wc.color.L, wc.color.C, wc.color.H,
                colors[j].color.L, colors[j].color.C, colors[j].color.H,
            )
            
            if de < cfg.delta_e_threshold:
                group.append(colors[j])
                merged[j] = True
        
        result_groups.append(group)
    
    # Build collapsed colors
    collapsed: list[WeightedColor] = []
    
    for group in result_groups:
        # Sum weights
        total_weight = sum(wc.weight for wc in group)
        
        # Pick representative: highest weight in group (first, since sorted)
        # This preserves the best sample_hex
        representative = group[0]
        
        collapsed.append(WeightedColor(
            color=representative.color,
            weight=total_weight,
        ))
    
    # Sort by weight descending
    collapsed.sort(key=lambda wc: wc.weight, reverse=True)
    
    # Normalize weights
    total = sum(wc.weight for wc in collapsed)
    if total > 0:
        collapsed = [
            WeightedColor(color=wc.color, weight=wc.weight / total)
            for wc in collapsed
        ]
    
    return tuple(collapsed)


def normalize_black_white(
    colors: tuple[WeightedColor, ...],
    config: Optional[ConsolidationConfig] = None,
) -> tuple[WeightedColor, ...]:
    """
    Normalize near-black and near-white colors into single representatives.
    
    Groups all colors matching black/white criteria and collapses them
    into single entries (most common black, most common white).
    
    Args:
        colors: Input palette
        config: Consolidation settings
        
    Returns:
        Palette with normalized black/white entries
    """
    if not colors:
        return colors
    
    cfg = config or ConsolidationConfig()
    
    blacks: list[WeightedColor] = []
    whites: list[WeightedColor] = []
    others: list[WeightedColor] = []
    
    for wc in colors:
        c = wc.color
        
        # Check for black family
        if c.L < cfg.black_l_threshold and c.C < cfg.black_c_threshold:
            blacks.append(wc)
        # Check for white family
        elif c.L > cfg.white_l_threshold and c.C < cfg.white_c_threshold:
            whites.append(wc)
        else:
            others.append(wc)
    
    result = list(others)
    
    # Collapse blacks into single entry (pick MOST COMMON, not darkest)
    # For dark mode UIs, the most frequent dark shade is the intentional choice
    if blacks:
        total_weight = sum(wc.weight for wc in blacks)
        # Find most common (highest weight) - preserves design intent
        most_common = max(blacks, key=lambda wc: wc.weight)
        result.append(WeightedColor(color=most_common.color, weight=total_weight))
    
    # Collapse whites into single entry (pick MOST COMMON, not lightest)
    if whites:
        total_weight = sum(wc.weight for wc in whites)
        # Find most common (highest weight) - preserves design intent
        most_common = max(whites, key=lambda wc: wc.weight)
        result.append(WeightedColor(color=most_common.color, weight=total_weight))
    
    # Sort by weight descending
    result.sort(key=lambda wc: wc.weight, reverse=True)
    
    # Normalize weights
    total = sum(wc.weight for wc in result)
    if total > 0:
        result = [
            WeightedColor(color=wc.color, weight=wc.weight / total)
            for wc in result
        ]
    
    return tuple(result)


def consolidate_palette(
    colors: tuple[WeightedColor, ...],
    config: Optional[ConsolidationConfig] = None,
) -> tuple[WeightedColor, ...]:
    """
    Full consolidation pipeline for a color palette.
    
    Applies:
    1. Black/white normalization
    2. ΔE-based collapse
    3. Size limiting
    
    Args:
        colors: Raw extracted palette
        config: Consolidation settings
        
    Returns:
        Consolidated, design-friendly palette
    """
    cfg = config or ConsolidationConfig()
    
    # Step 1: Normalize black/white first
    # (This reduces noise before collapse)
    colors = normalize_black_white(colors, cfg)
    
    # Step 2: Collapse similar colors
    colors = collapse_similar_colors(colors, cfg)
    
    # Step 3: Filter out noise (below min_weight)
    colors = tuple(wc for wc in colors if wc.weight >= cfg.min_weight)
    
    # Renormalize weights after filtering
    if colors:
        total = sum(wc.weight for wc in colors)
        if total > 0:
            colors = tuple(
                WeightedColor(color=wc.color, weight=wc.weight / total)
                for wc in colors
            )
    
    # Step 4: Limit size
    if len(colors) > cfg.max_palette_size:
        # Keep top N, renormalize weights
        colors = colors[:cfg.max_palette_size]
        total = sum(wc.weight for wc in colors)
        if total > 0:
            colors = tuple(
                WeightedColor(color=wc.color, weight=wc.weight / total)
                for wc in colors
            )
    
    return colors

