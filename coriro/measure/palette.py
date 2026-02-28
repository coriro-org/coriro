# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Palette extraction using k-means clustering or mode-based counting.

Two approaches:
1. K-means: Finds statistical clusters (averaged colors)
2. Mode: Finds most common exact pixel values (exact colors)

For simple screenshots, mode is more accurate.
For photos/gradients, k-means is more robust.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from coriro.schema import OKLCHColor, WeightedColor
from coriro.measure.colorspace import srgb_uint8_to_oklch


def extract_palette(
    oklch_pixels: NDArray[np.float64],
    n_colors: int = 5,
    max_iter: int = 100,
    seed: Optional[int] = 42,
    rgb_pixels: Optional[NDArray[np.uint8]] = None,
) -> tuple[WeightedColor, ...]:
    """
    Extract a palette of dominant colors from OKLCH pixels.
    
    Uses k-means clustering in OKLCH space. Colors are returned
    ordered by weight (most dominant first).
    
    For each cluster, we record:
    - OKLCH centroid (average) for perceptual reasoning
    - sample_hex (nearest real pixel to centroid) for implementation accuracy
    
    Args:
        oklch_pixels: Array of shape (N, 3) with OKLCH values
        n_colors: Number of colors to extract (may return fewer if image has fewer unique colors)
        max_iter: Maximum k-means iterations
        seed: Random seed for reproducibility (None for random)
        rgb_pixels: Optional array of shape (N, 3) with original RGB [0-255] values.
            If provided, sample_hex will be set to the nearest real pixel.
        
    Returns:
        Tuple of WeightedColor, ordered by weight descending.
        Weights sum to 1.0.
    """
    if len(oklch_pixels) == 0:
        raise ValueError("Cannot extract palette from empty pixel array")
    
    n_pixels = len(oklch_pixels)
    
    # K-means clustering (will adjust k internally based on unique colors)
    centroids, labels = _kmeans(
        oklch_pixels, 
        k=n_colors, 
        max_iter=max_iter, 
        seed=seed
    )
    
    # Actual number of clusters found
    actual_k = len(centroids)
    
    # Count pixels per cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    # Build weighted colors
    weighted = []
    total = n_pixels
    
    for i in range(actual_k):
        L, C, H = centroids[i]
        count = cluster_sizes.get(i, 0)
        weight = count / total
        
        # Skip clusters with no pixels
        if weight == 0:
            continue
        
        # Handle achromatic colors (low chroma)
        h_value = float(H) if C >= 0.02 else None
        
        # Find nearest real pixel to centroid (for sample_hex)
        sample_hex = None
        if rgb_pixels is not None:
            sample_hex = _find_nearest_pixel_hex(
                centroid=centroids[i],
                oklch_pixels=oklch_pixels,
                rgb_pixels=rgb_pixels,
                labels=labels,
                cluster_id=i,
            )
        
        color = OKLCHColor(
            L=float(np.clip(L, 0.0, 1.0)),
            C=float(max(0.0, C)),
            H=h_value,
            sample_hex=sample_hex,
        )
        weighted.append((color, weight))
    
    # Sort by weight descending
    weighted.sort(key=lambda x: x[1], reverse=True)
    
    # Normalize weights to sum to 1.0 exactly
    total_weight = sum(w for _, w in weighted)
    if total_weight > 0:
        weighted = [(c, w / total_weight) for c, w in weighted]
    
    return tuple(WeightedColor(color=c, weight=w) for c, w in weighted)


def _find_nearest_pixel_hex(
    centroid: NDArray[np.float64],
    oklch_pixels: NDArray[np.float64],
    rgb_pixels: NDArray[np.uint8],
    labels: NDArray[np.int64],
    cluster_id: int,
) -> str:
    """
    Find the nearest real pixel to the centroid within a cluster.
    
    Returns the hex value of that pixel. This is the "representative pixel"
    that provides implementation-accurate hex values.
    
    Args:
        centroid: (3,) array with OKLCH centroid values
        oklch_pixels: (N, 3) array of all OKLCH pixels
        rgb_pixels: (N, 3) array of corresponding RGB pixels [0-255]
        labels: (N,) array of cluster assignments
        cluster_id: Which cluster to search
        
    Returns:
        Hex string like "#F6C767"
    """
    # Get pixels in this cluster
    mask = labels == cluster_id
    cluster_oklch = oklch_pixels[mask]
    cluster_rgb = rgb_pixels[mask]
    
    if len(cluster_oklch) == 0:
        # Fallback: compute from centroid
        from coriro.measure.colorspace import oklch_to_hex
        return oklch_to_hex(centroid[0], centroid[1], centroid[2])
    
    # Find nearest pixel to centroid (Euclidean distance in OKLCH)
    distances = np.sum((cluster_oklch - centroid) ** 2, axis=1)
    nearest_idx = np.argmin(distances)
    
    # Get RGB of nearest pixel
    r, g, b = cluster_rgb[nearest_idx]
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def _kmeans(
    data: NDArray[np.float64],
    k: int,
    max_iter: int = 100,
    seed: Optional[int] = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Vectorized k-means implementation.
    
    Uses k-means++ initialization for better convergence.
    Fully vectorized for performance on large pixel arrays.
    
    Args:
        data: Array of shape (N, D)
        k: Number of clusters
        max_iter: Maximum iterations
        seed: Random seed
        
    Returns:
        (centroids, labels) where:
        - centroids: (k, D) array of cluster centers
        - labels: (N,) array of cluster assignments
    """
    rng = np.random.default_rng(seed)
    n, d = data.shape
    
    # Find unique points to avoid issues with duplicates
    unique_data = np.unique(data, axis=0)
    n_unique = len(unique_data)
    
    # Adjust k if we have fewer unique points
    k = min(k, n_unique)
    
    if k == 0:
        raise ValueError("No valid data points for clustering")
    
    # k-means++ initialization
    centroids = np.empty((k, d), dtype=np.float64)
    
    # First centroid: random unique point
    idx = rng.integers(n_unique)
    centroids[0] = unique_data[idx]
    
    # Remaining centroids: weighted by distance squared
    for i in range(1, k):
        # Distance to nearest existing centroid (vectorized)
        dists_to_centroids = np.sum(
            (unique_data[:, np.newaxis, :] - centroids[np.newaxis, :i, :]) ** 2,
            axis=2
        )
        dists = np.min(dists_to_centroids, axis=1)
        
        # Handle case where all distances are zero
        total = dists.sum()
        if total == 0:
            remaining_idx = rng.integers(n_unique)
            centroids[i] = unique_data[remaining_idx]
        else:
            probs = dists / total
            idx = rng.choice(n_unique, p=probs)
            centroids[i] = unique_data[idx]
    
    # Iterate on full data (vectorized)
    labels = np.zeros(n, dtype=np.int64)
    
    for _ in range(max_iter):
        old_labels = labels.copy()
        
        # Vectorized distance computation: (N, k)
        # Using broadcasting: data is (N, D), centroids is (k, D)
        dists = np.sum(
            (data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2
        )
        labels = np.argmin(dists, axis=1)
        
        # Check convergence
        if np.array_equal(labels, old_labels):
            break
        
        # Update centroids
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centroids[j] = data[mask].mean(axis=0)
    
    return centroids, labels


def extract_dominant_color(
    oklch_pixels: NDArray[np.float64],
) -> OKLCHColor:
    """
    Extract the single most dominant color from OKLCH pixels.

    Delegates to extract_palette with n_colors=1 for consistency.

    Args:
        oklch_pixels: Array of shape (N, 3) with OKLCH values

    Returns:
        Single OKLCHColor representing the dominant color
    """
    if len(oklch_pixels) == 0:
        raise ValueError("Cannot extract dominant color from empty pixel array")
    
    # Use palette extraction with k=1 for consistency
    palette = extract_palette(oklch_pixels, n_colors=1)
    return palette[0].color


def extract_palette_mode(
    rgb_pixels: NDArray[np.uint8],
    n_colors: int = 10,
    min_count: int = 10,
) -> tuple[WeightedColor, ...]:
    """
    Extract palette using MODE (most common exact pixel values).
    
    This is more accurate than k-means for simple screenshots because
    it finds the actual pixel values rather than averaged centroids.
    
    Args:
        rgb_pixels: Array of shape (N, 3) with RGB values [0-255]
        n_colors: Maximum number of colors to return
        min_count: Minimum pixel count to include a color
        
    Returns:
        Tuple of WeightedColor with exact pixel values, ordered by frequency.
        Weights are normalized to sum to 1.0.
    """
    if len(rgb_pixels) == 0:
        raise ValueError("Cannot extract palette from empty pixel array")
    
    # Count exact pixel values
    rgb_tuples = [tuple(p) for p in rgb_pixels]
    counter = Counter(rgb_tuples)
    
    # Get top colors
    top_colors = counter.most_common(n_colors)
    
    # Filter by min_count and collect
    filtered = [(rgb, count) for rgb, count in top_colors if count >= min_count]
    
    if not filtered:
        # Fallback: just take top colors regardless of min_count
        filtered = top_colors[:n_colors]
    
    # Calculate total for normalization
    total_count = sum(count for _, count in filtered)
    
    results = []
    for (r, g, b), count in filtered:
        weight = count / total_count  # Normalized to sum to 1.0
        
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
    
    return tuple(results)

