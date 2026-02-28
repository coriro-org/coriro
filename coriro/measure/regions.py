# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Spatial region extraction.

Partitions an image into fixed grid regions and extracts color
measurements from each region.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from coriro.schema import (
    GridSize,
    RegionColor,
    SpatialBins,
    REGION_IDS,
)
from coriro.measure.palette import extract_palette


def _grid_dimensions(grid: GridSize) -> int:
    """Return the number of rows/cols for a grid size."""
    return {
        GridSize.GRID_2X2: 2,
        GridSize.GRID_3X3: 3,
        GridSize.GRID_4X4: 4,
    }[grid]


def extract_spatial_bins(
    oklch_pixels: NDArray[np.float64],
    height: int,
    width: int,
    grid: GridSize = GridSize.GRID_2X2,
    colors_per_region: int = 3,
    rgb_flat: NDArray[np.uint8] | None = None,
) -> SpatialBins:
    """
    Partition image into grid regions and extract color palettes.
    
    Args:
        oklch_pixels: Array of shape (H*W, 3) with OKLCH values, flattened
        height: Image height in pixels
        width: Image width in pixels
        grid: Grid size (2x2, 3x3, or 4x4)
        colors_per_region: Number of colors in each region's mini-palette
        rgb_flat: Optional (H*W, 3) array of RGB values for sample_hex computation
        
    Returns:
        SpatialBins with region palettes
    """
    # Reshape to 2D image
    oklch_image = oklch_pixels.reshape(height, width, 3)
    rgb_image = rgb_flat.reshape(height, width, 3) if rgb_flat is not None else None
    
    dim = _grid_dimensions(grid)
    region_ids = REGION_IDS[grid]
    
    # Calculate region boundaries
    row_edges = np.linspace(0, height, dim + 1, dtype=int)
    col_edges = np.linspace(0, width, dim + 1, dtype=int)
    
    regions = []
    region_idx = 0
    
    for row in range(dim):
        for col in range(dim):
            # Extract region pixels
            r_start, r_end = row_edges[row], row_edges[row + 1]
            c_start, c_end = col_edges[col], col_edges[col + 1]
            
            region_pixels = oklch_image[r_start:r_end, c_start:c_end]
            region_pixels_flat = region_pixels.reshape(-1, 3)
            
            # Get corresponding RGB pixels for sample_hex
            region_rgb_flat = None
            if rgb_image is not None:
                region_rgb = rgb_image[r_start:r_end, c_start:c_end]
                region_rgb_flat = region_rgb.reshape(-1, 3)
            
            if len(region_pixels_flat) == 0:
                raise ValueError(
                    f"Empty region {region_ids[region_idx]}: image {width}x{height} "
                    f"is too small for a {grid.value} grid (minimum {dim}x{dim} pixels)"
                )
            
            # Extract mini-palette for this region (with sample_hex if RGB available)
            palette = extract_palette(
                region_pixels_flat,
                n_colors=colors_per_region,
                rgb_pixels=region_rgb_flat,
            )
            
            regions.append(RegionColor(
                region_id=region_ids[region_idx],
                palette=palette,
            ))
            
            region_idx += 1
    
    return SpatialBins(grid=grid, regions=tuple(regions))

