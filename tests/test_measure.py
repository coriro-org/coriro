# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""Integration tests for the measure() entry point."""

import numpy as np
import pytest
import importlib

from coriro import measure, ColorMeasurement, GridSize


def _solid_image(r, g, b, height=100, width=100):
    """Create a solid-color image."""
    return np.full((height, width, 3), [r, g, b], dtype=np.uint8)


def _two_tone_image(rgb1, rgb2, height=100, width=200):
    """Create an image that is half one color, half another."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :width // 2] = rgb1
    img[:, width // 2:] = rgb2
    return img


class TestMeasureBasic:

    def test_solid_red(self):
        pixels = _solid_image(255, 0, 0)
        m = measure(pixels, include_hash=False)
        assert isinstance(m, ColorMeasurement)
        assert m.dominant.L > 0  # Red has non-zero lightness
        assert m.dominant.C > 0  # Red is chromatic

    def test_solid_white(self):
        pixels = _solid_image(255, 255, 255)
        m = measure(pixels, include_hash=False)
        assert m.dominant.L > 0.9
        assert m.dominant.is_achromatic

    def test_solid_black(self):
        pixels = _solid_image(0, 0, 0)
        m = measure(pixels, include_hash=False)
        assert m.dominant.L < 0.1
        assert m.dominant.is_achromatic

    def test_palette_weights_sum_to_one(self):
        pixels = _two_tone_image([255, 0, 0], [0, 0, 255])
        m = measure(pixels, include_hash=False)
        total = sum(wc.weight for wc in m.palette)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_hash_included(self):
        pixels = _solid_image(128, 128, 128)
        m = measure(pixels, include_hash=True)
        assert m.image_hash is not None
        assert m.image_hash.startswith("sha256:")

    def test_hash_excluded(self):
        pixels = _solid_image(128, 128, 128)
        m = measure(pixels, include_hash=False)
        assert m.image_hash is None


class TestMeasureDeterminism:

    def test_same_input_same_output(self):
        pixels = _two_tone_image([200, 50, 50], [50, 50, 200])
        m1 = measure(pixels, include_hash=False)
        m2 = measure(pixels, include_hash=False)

        assert m1.dominant.L == m2.dominant.L
        assert m1.dominant.C == m2.dominant.C
        assert m1.dominant.H == m2.dominant.H
        assert len(m1.palette) == len(m2.palette)
        for wc1, wc2 in zip(m1.palette, m2.palette):
            assert wc1.weight == wc2.weight


class TestSpatialGrid:

    def test_2x2_produces_4_regions(self):
        pixels = _solid_image(128, 128, 128)
        m = measure(pixels, grid=GridSize.GRID_2X2, include_hash=False)
        assert len(m.spatial.regions) == 4

    def test_3x3_produces_9_regions(self):
        pixels = _solid_image(128, 128, 128)
        m = measure(pixels, grid=GridSize.GRID_3X3, include_hash=False)
        assert len(m.spatial.regions) == 9

    def test_4x4_produces_16_regions(self):
        pixels = _solid_image(128, 128, 128)
        m = measure(pixels, grid=GridSize.GRID_4X4, include_hash=False)
        assert len(m.spatial.regions) == 16

    def test_region_palette_weights_sum_to_one(self):
        pixels = _solid_image(128, 128, 128)
        m = measure(pixels, include_hash=False)
        for region in m.spatial.regions:
            total = sum(wc.weight for wc in region.palette)
            assert total == pytest.approx(1.0, abs=0.01)


class TestMeasurementMeta:

    def test_meta_present_when_consolidated(self):
        pixels = _solid_image(128, 128, 128)
        m = measure(pixels, consolidate=True, include_hash=False)
        assert m.measurement is not None
        assert m.measurement.scope == "area_dominant_surfaces"
        assert m.measurement.coverage == "complete"

    def test_meta_absent_when_not_consolidated(self):
        pixels = _solid_image(128, 128, 128)
        m = measure(pixels, consolidate=False, include_hash=False)
        assert m.measurement is None


class TestOptionalPasses:

    def test_text_colors_off_by_default(self):
        pixels = _solid_image(128, 128, 128)
        m = measure(pixels, include_hash=False)
        assert m.text_colors is None

    def test_accent_regions_off_by_default(self):
        pixels = _solid_image(128, 128, 128)
        m = measure(pixels, include_hash=False)
        assert m.accent_regions is None

    def test_gradient_always_none(self):
        """Gradient detection is not implemented — always None."""
        pixels = _solid_image(128, 128, 128)
        m = measure(pixels, include_hash=False)
        assert m.gradient is None

    def test_accents_use_original_dimensions_when_downsampled(self, monkeypatch):
        """Accent pass should receive original dimensions even if surfaces are downsampled."""
        pixels = _solid_image(128, 128, 128, height=200, width=300)
        captured = {}

        def fake_extract_accent_regions(rgb_pixels, height, width, **kwargs):
            captured["shape"] = rgb_pixels.shape
            captured["height"] = height
            captured["width"] = width
            return None

        extract_module = importlib.import_module("coriro.measure.extract")
        monkeypatch.setattr(extract_module, "extract_accent_regions", fake_extract_accent_regions)

        measure(
            pixels,
            include_hash=False,
            include_accents=True,
            max_pixels=1000,  # Force downsampling of surface pipeline
        )

        assert captured["shape"] == (200 * 300, 3)
        assert captured["height"] == 200
        assert captured["width"] == 300


class TestInputValidation:

    def test_invalid_shape_raises(self):
        pixels = np.zeros((100, 100), dtype=np.uint8)  # 2D, no channels
        with pytest.raises(ValueError, match="Expected.*H, W, 3"):
            measure(pixels)

    def test_invalid_dtype_raises(self):
        pixels = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected uint8"):
            measure(pixels)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected file path or numpy"):
            measure(42)
