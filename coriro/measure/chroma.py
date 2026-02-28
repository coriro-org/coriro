# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Chroma-aware palette supplementation.

Finds perceptually significant colors missed by the area-dominant pass:
1. Chroma outliers: high-saturation colors (e.g. yellow CTA on blue page)
2. Uncovered colors: pixel clusters poorly represented by the palette

Both functions return small tuples of WeightedColor to be appended to
the palette. Weights are computed relative to the full image.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from coriro.schema.color_measurement import WeightedColor, OKLCHColor
from coriro.measure.colorspace import oklch_to_oklab, srgb_uint8_to_oklch
from coriro.measure.palette import extract_palette_mode


def _delta_e_lab(lab1: NDArray[np.float64], lab2: NDArray[np.float64]) -> float:
    """Euclidean distance between two OKLab vectors."""
    delta = lab1 - lab2
    return float(np.sqrt(np.sum(delta ** 2)))


def _wc_to_lab(wc: WeightedColor) -> NDArray[np.float64]:
    """Convert a single WeightedColor to OKLab (3,) vector."""
    h = wc.color.H if wc.color.H is not None else 0.0
    lch = np.array([wc.color.L, wc.color.C, h], dtype=np.float64)
    return oklch_to_oklab(lch)


def _palette_to_lab(palette: tuple[WeightedColor, ...]) -> NDArray[np.float64]:
    """Convert palette entries to OKLab array (K, 3)."""
    lch = np.array([
        [wc.color.L, wc.color.C, wc.color.H if wc.color.H is not None else 0.0]
        for wc in palette
    ], dtype=np.float64)
    return oklch_to_oklab(lch)


def _dedup_novel(
    candidates: list[WeightedColor],
    threshold: float,
) -> list[WeightedColor]:
    """Deduplicate a list of WeightedColor against each other.

    Keeps the first (highest-weight) entry and drops any subsequent entry
    that is within ΔE < threshold of an already-accepted entry.
    """
    accepted: list[WeightedColor] = []
    accepted_labs: list[NDArray[np.float64]] = []

    for wc in candidates:
        wc_lab = _wc_to_lab(wc)
        is_dup = any(
            _delta_e_lab(wc_lab, a_lab) < threshold
            for a_lab in accepted_labs
        )
        if not is_dup:
            accepted.append(wc)
            accepted_labs.append(wc_lab)

    return accepted


def find_chroma_outliers(
    oklch_flat: NDArray[np.float64],
    rgb_flat: NDArray[np.uint8],
    existing_palette: tuple[WeightedColor, ...],
    *,
    z_threshold: float = 2.0,
    min_outlier_pixels: int = 100,
    novelty_threshold: float = 0.08,
    max_supplements: int = 3,
) -> tuple[WeightedColor, ...]:
    """
    Find colors with unusually high chroma relative to the image mean.

    A yellow CTA (C~0.17) on a blue page (mean C~0.05, std~0.03) sits at
    z-score ~4.0, well above the default threshold of 2.0.

    Args:
        oklch_flat: (N, 3) OKLCH pixel array
        rgb_flat: (N, 3) uint8 RGB pixel array
        existing_palette: Current palette to check novelty against
        z_threshold: Std deviations above mean chroma to qualify as outlier
        min_outlier_pixels: Minimum outlier pixels to proceed
        novelty_threshold: Min ΔE from all palette entries to keep
        max_supplements: Cap on returned supplements

    Returns:
        Tuple of WeightedColor (may be empty), weights relative to full image.
    """
    n_total = len(oklch_flat)

    # 1. Compute chroma statistics
    chroma = oklch_flat[:, 1]
    c_mean = float(np.mean(chroma))
    c_std = float(np.std(chroma))

    if c_std < 1e-6:
        return ()  # Uniform chroma — nothing to find

    threshold = c_mean + z_threshold * c_std

    # 2. Mask high-chroma pixels
    outlier_mask = chroma > threshold
    n_outliers = int(np.sum(outlier_mask))

    if n_outliers < min_outlier_pixels:
        return ()

    # 3. Cluster the outlier pixels
    outlier_rgb = rgb_flat[outlier_mask]
    outlier_palette = extract_palette_mode(outlier_rgb, n_colors=max_supplements + 2, min_count=1)

    # 4. Filter by novelty (ΔE from existing palette)
    if not existing_palette:
        novel = list(outlier_palette)
    else:
        palette_lab = _palette_to_lab(existing_palette)
        novel = []
        for wc in outlier_palette:
            wc_lab = _wc_to_lab(wc)
            min_de = min(_delta_e_lab(wc_lab, palette_lab[i]) for i in range(len(palette_lab)))
            if min_de > novelty_threshold:
                novel.append(wc)

    if not novel:
        return ()

    # 5. Internal dedup — prevent near-duplicate supplements (e.g. two yellows)
    novel = _dedup_novel(novel, novelty_threshold)

    # 6. Recompute weights relative to full image
    results = []
    for wc in novel[:max_supplements]:
        # Count how many full-image pixels match this color exactly
        hex_val = wc.color.sample_hex
        if hex_val:
            r, g, b = int(hex_val[1:3], 16), int(hex_val[3:5], 16), int(hex_val[5:7], 16)
            match_mask = (
                (rgb_flat[:, 0] == r) &
                (rgb_flat[:, 1] == g) &
                (rgb_flat[:, 2] == b)
            )
            pixel_count = int(np.sum(match_mask))
        else:
            pixel_count = n_outliers // len(novel)

        weight = max(pixel_count / n_total, 1e-4)  # Floor to avoid zero
        results.append(WeightedColor(color=wc.color, weight=weight))

    return tuple(results)


def find_uncovered_colors(
    oklch_flat: NDArray[np.float64],
    rgb_flat: NDArray[np.uint8],
    palette: tuple[WeightedColor, ...],
    *,
    coverage_threshold: float = 0.15,
    min_uncovered_pct: float = 0.005,
    novelty_threshold: float = 0.10,
    max_supplements: int = 2,
    chunk_size: int = 500_000,
) -> tuple[WeightedColor, ...]:
    """
    Safety net — find pixel clusters poorly covered by the palette.

    Computes per-pixel min ΔE to nearest palette color. Pixels with
    distance > coverage_threshold are "uncovered". If enough exist,
    clusters them and returns novel supplements.

    Args:
        oklch_flat: (N, 3) OKLCH pixel array
        rgb_flat: (N, 3) uint8 RGB pixel array
        palette: Current palette (including any chroma supplements)
        coverage_threshold: ΔE above which a pixel is uncovered
        min_uncovered_pct: Min fraction of pixels uncovered to proceed
        novelty_threshold: Min ΔE from palette to keep a supplement
        max_supplements: Cap on returned supplements
        chunk_size: Pixels per batch for memory efficiency

    Returns:
        Tuple of WeightedColor (may be empty), weights relative to full image.
    """
    if not palette:
        return ()

    n_total = len(oklch_flat)
    palette_lab = _palette_to_lab(palette)  # (K, 3)
    k = len(palette_lab)

    # Process in chunks to limit memory (N×K distance matrix)
    uncovered_mask = np.zeros(n_total, dtype=np.bool_)

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk_lch = oklch_flat[start:end]

        # Convert chunk to OKLab
        chunk_lab = oklch_to_oklab(chunk_lch)  # (chunk, 3)

        # Distance to each palette color: (chunk, K)
        dists = np.sqrt(np.sum(
            (chunk_lab[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]) ** 2,
            axis=2,
        ))

        # Min distance to nearest palette color
        min_dist = np.min(dists, axis=1)  # (chunk,)
        uncovered_mask[start:end] = min_dist > coverage_threshold

    n_uncovered = int(np.sum(uncovered_mask))

    if n_uncovered / n_total < min_uncovered_pct:
        return ()  # Palette covers the image well

    # Cluster uncovered pixels
    uncovered_rgb = rgb_flat[uncovered_mask]
    uncovered_palette = extract_palette_mode(uncovered_rgb, n_colors=max_supplements + 2, min_count=1)

    # Filter by novelty
    novel = []
    for wc in uncovered_palette:
        wc_lab = _wc_to_lab(wc)
        min_de = min(_delta_e_lab(wc_lab, palette_lab[i]) for i in range(k))
        if min_de > novelty_threshold:
            novel.append(wc)

    if not novel:
        return ()

    # Internal dedup
    novel = _dedup_novel(novel, novelty_threshold)

    # Recompute weights relative to full image
    results = []
    for wc in novel[:max_supplements]:
        hex_val = wc.color.sample_hex
        if hex_val:
            r, g, b = int(hex_val[1:3], 16), int(hex_val[3:5], 16), int(hex_val[5:7], 16)
            match_mask = (
                (rgb_flat[:, 0] == r) &
                (rgb_flat[:, 1] == g) &
                (rgb_flat[:, 2] == b)
            )
            pixel_count = int(np.sum(match_mask))
        else:
            pixel_count = n_uncovered // len(novel)

        weight = max(pixel_count / n_total, 1e-4)
        results.append(WeightedColor(color=wc.color, weight=weight))

    return tuple(results)
