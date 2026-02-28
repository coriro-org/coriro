# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""Tests for color consolidation (ΔE collapse, black/white normalization)."""

import pytest

from coriro.schema.color_measurement import OKLCHColor, WeightedColor
from coriro.measure.consolidation import (
    collapse_similar_colors,
    normalize_black_white,
    consolidate_palette,
    ConsolidationConfig,
)


def _wc(L, C, H, weight):
    return WeightedColor(color=OKLCHColor(L=L, C=C, H=H), weight=weight)


def _achromatic_wc(L, weight):
    return WeightedColor(color=OKLCHColor(L=L, C=0.0), weight=weight)


class TestCollapseSimilar:

    def test_identical_colors_merge(self):
        colors = (
            _wc(0.5, 0.1, 200.0, 0.4),
            _wc(0.5, 0.1, 200.0, 0.3),
            _wc(0.8, 0.05, 30.0, 0.3),
        )
        result = collapse_similar_colors(colors)
        # Two identical blues should merge, leaving 2 groups
        assert len(result) == 2
        # Weights should be renormalized to sum to 1.0
        total = sum(wc.weight for wc in result)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_distinct_colors_preserved(self):
        colors = (
            _wc(0.5, 0.1, 200.0, 0.5),  # Blue
            _wc(0.6, 0.15, 30.0, 0.3),   # Orange
            _wc(0.7, 0.2, 120.0, 0.2),   # Green
        )
        result = collapse_similar_colors(colors)
        assert len(result) == 3

    def test_achromatic_not_merged_with_chromatic(self):
        """Dark gray and dark red should not merge even if ΔE is small."""
        colors = (
            _achromatic_wc(0.2, 0.5),     # Dark gray
            _wc(0.2, 0.03, 30.0, 0.5),    # Dark, slightly red
        )
        config = ConsolidationConfig(delta_e_threshold=0.10)
        result = collapse_similar_colors(colors, config)
        assert len(result) == 2

    def test_empty_palette(self):
        assert collapse_similar_colors(()) == ()


class TestBlackWhiteNormalization:

    def test_multiple_blacks_collapse(self):
        colors = (
            _achromatic_wc(0.1, 0.3),   # Near-black 1
            _achromatic_wc(0.15, 0.2),   # Near-black 2
            _wc(0.5, 0.1, 200.0, 0.5),  # Blue
        )
        result = normalize_black_white(colors)
        blacks = [wc for wc in result if wc.color.L < 0.25]
        assert len(blacks) == 1  # Collapsed to one

    def test_multiple_whites_collapse(self):
        colors = (
            _achromatic_wc(0.96, 0.3),
            _achromatic_wc(0.98, 0.2),
            _wc(0.5, 0.1, 200.0, 0.5),
        )
        result = normalize_black_white(colors)
        whites = [wc for wc in result if wc.color.L > 0.95]
        assert len(whites) == 1

    def test_weight_sum_preserved(self):
        colors = (
            _achromatic_wc(0.1, 0.3),
            _achromatic_wc(0.15, 0.2),
            _wc(0.5, 0.1, 200.0, 0.5),
        )
        result = normalize_black_white(colors)
        total = sum(wc.weight for wc in result)
        assert total == pytest.approx(1.0, abs=0.01)


class TestConsolidatePipeline:

    def test_full_pipeline(self):
        colors = (
            _wc(0.5, 0.1, 200.0, 0.25),
            _wc(0.5, 0.1, 201.0, 0.20),  # Near-identical to first
            _achromatic_wc(0.1, 0.15),
            _achromatic_wc(0.12, 0.10),   # Near-identical black
            _wc(0.7, 0.2, 120.0, 0.15),
            _wc(0.6, 0.15, 30.0, 0.15),
        )
        config = ConsolidationConfig(max_palette_size=4, delta_e_threshold=0.05)
        result = consolidate_palette(colors, config)

        assert len(result) <= 4
        total = sum(wc.weight for wc in result)
        assert total == pytest.approx(1.0, abs=0.01)
        # Sorted by weight descending
        weights = [wc.weight for wc in result]
        assert weights == sorted(weights, reverse=True)

    def test_size_limiting(self):
        colors = tuple(
            _wc(0.3 + i * 0.1, 0.1, 30.0 * i, 1.0 / 8)
            for i in range(8)
        )
        config = ConsolidationConfig(max_palette_size=3, delta_e_threshold=0.001)
        result = consolidate_palette(colors, config)
        assert len(result) <= 3
