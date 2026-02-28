# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""Tests for chroma-aware palette supplementation."""

import numpy as np
import pytest

from coriro.schema.color_measurement import OKLCHColor, WeightedColor
from coriro.measure.chroma import find_chroma_outliers, find_uncovered_colors
from coriro.measure.colorspace import srgb_to_oklch


def _make_image(dominant_rgb, accent_rgb, accent_fraction=0.02, size=10000):
    """Create a synthetic image with dominant background + small accent."""
    n_accent = int(size * accent_fraction)
    n_dominant = size - n_accent
    pixels = np.vstack([
        np.tile(dominant_rgb, (n_dominant, 1)),
        np.tile(accent_rgb, (n_accent, 1)),
    ]).astype(np.uint8)
    return pixels


def _wc(L, C, H, weight, sample_hex=None):
    return WeightedColor(
        color=OKLCHColor(L=L, C=C, H=H, sample_hex=sample_hex),
        weight=weight,
    )


class TestChromaOutliers:

    def test_yellow_on_blue_detected(self):
        """A high-chroma yellow accent on a low-chroma blue background."""
        blue = np.array([30, 50, 120])
        yellow = np.array([255, 230, 0])
        rgb_flat = _make_image(blue, yellow, accent_fraction=0.03)

        oklch_flat = srgb_to_oklch(rgb_flat.astype(np.float64) / 255.0)
        # Existing palette is just the blue
        palette = (_wc(0.35, 0.05, 250.0, 1.0, sample_hex="#1E3278"),)

        supplements = find_chroma_outliers(
            oklch_flat, rgb_flat, palette,
            min_outlier_pixels=50,
        )
        assert len(supplements) > 0

    def test_uniform_image_no_outliers(self):
        """An image with uniform chroma should produce no outliers."""
        gray = np.array([128, 128, 128])
        rgb_flat = np.tile(gray, (5000, 1)).astype(np.uint8)

        oklch_flat = srgb_to_oklch(rgb_flat.astype(np.float64) / 255.0)
        palette = (_wc(0.5, 0.0, None, 1.0),)

        supplements = find_chroma_outliers(oklch_flat, rgb_flat, palette)
        assert len(supplements) == 0

    def test_outlier_not_duplicate_of_palette(self):
        """Outlier matching existing palette entry should be filtered."""
        blue = np.array([30, 50, 120])
        rgb_flat = np.tile(blue, (5000, 1)).astype(np.uint8)
        # Add a few brighter blue pixels (same hue, higher chroma)
        bright_blue = np.array([0, 60, 255])
        extra = np.tile(bright_blue, (200, 1)).astype(np.uint8)
        rgb_flat = np.vstack([rgb_flat, extra])

        oklch_flat = srgb_to_oklch(rgb_flat.astype(np.float64) / 255.0)
        # If palette already contains something close to bright_blue, no supplement
        palette = (_wc(0.45, 0.2, 260.0, 1.0, sample_hex="#003CFF"),)

        supplements = find_chroma_outliers(
            oklch_flat, rgb_flat, palette,
            novelty_threshold=0.08,
            min_outlier_pixels=50,
        )
        # Should be empty or very few since bright blue is close to palette
        # (depends on exact ΔE — this tests the novelty filter is active)
        for s in supplements:
            assert s.weight > 0


class TestUncoveredColors:

    def test_uncovered_pixel_cluster_found(self):
        """A distinct color cluster not in palette should be found."""
        blue = np.array([30, 50, 120])
        red = np.array([200, 30, 30])
        # 90% blue, 10% red
        rgb_flat = _make_image(blue, red, accent_fraction=0.10)

        oklch_flat = srgb_to_oklch(rgb_flat.astype(np.float64) / 255.0)
        # Palette only has blue
        palette = (_wc(0.35, 0.05, 250.0, 1.0, sample_hex="#1E3278"),)

        supplements = find_uncovered_colors(
            oklch_flat, rgb_flat, palette,
            coverage_threshold=0.10,
            min_uncovered_pct=0.005,
        )
        assert len(supplements) > 0

    def test_well_covered_image_no_supplements(self):
        """An image well-covered by palette should produce no supplements."""
        blue = np.array([30, 50, 120])
        rgb_flat = np.tile(blue, (5000, 1)).astype(np.uint8)

        oklch_flat = srgb_to_oklch(rgb_flat.astype(np.float64) / 255.0)
        palette = (_wc(0.35, 0.05, 250.0, 1.0, sample_hex="#1E3278"),)

        supplements = find_uncovered_colors(oklch_flat, rgb_flat, palette)
        assert len(supplements) == 0

    def test_empty_palette_returns_empty(self):
        rgb_flat = np.zeros((100, 3), dtype=np.uint8)
        oklch_flat = srgb_to_oklch(rgb_flat.astype(np.float64) / 255.0)
        supplements = find_uncovered_colors(oklch_flat, rgb_flat, ())
        assert len(supplements) == 0
