# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""Tests for color space conversions (sRGB ↔ OKLab ↔ OKLCH)."""

import numpy as np
import pytest

from coriro.measure.colorspace import (
    srgb_to_linear,
    linear_to_srgb,
    linear_rgb_to_oklab,
    oklab_to_linear_rgb,
    oklab_to_oklch,
    oklch_to_oklab,
    srgb_to_oklch,
    oklch_to_srgb,
    srgb_uint8_to_oklch,
    oklch_to_hex,
    hex_to_oklch,
    delta_e_oklch,
    delta_e_oklch_batch,
)


class TestSRGBLinearRoundtrip:
    """sRGB ↔ Linear RGB conversions must roundtrip accurately."""

    def test_roundtrip_mid_gray(self):
        srgb = np.array([0.5, 0.5, 0.5])
        linear = srgb_to_linear(srgb)
        recovered = linear_to_srgb(linear)
        np.testing.assert_allclose(recovered, srgb, atol=1e-10)

    def test_roundtrip_black(self):
        srgb = np.array([0.0, 0.0, 0.0])
        recovered = linear_to_srgb(srgb_to_linear(srgb))
        np.testing.assert_allclose(recovered, srgb, atol=1e-10)

    def test_roundtrip_white(self):
        srgb = np.array([1.0, 1.0, 1.0])
        recovered = linear_to_srgb(srgb_to_linear(srgb))
        np.testing.assert_allclose(recovered, srgb, atol=1e-10)

    def test_roundtrip_primary_red(self):
        srgb = np.array([1.0, 0.0, 0.0])
        recovered = linear_to_srgb(srgb_to_linear(srgb))
        np.testing.assert_allclose(recovered, srgb, atol=1e-10)

    def test_gamma_threshold(self):
        """Values below 0.04045 use linear segment."""
        val = 0.03
        linear = srgb_to_linear(np.array([val]))
        assert float(linear[0]) == pytest.approx(val / 12.92, abs=1e-10)

    def test_batch_roundtrip(self):
        srgb = np.random.RandomState(42).random((100, 3))
        recovered = linear_to_srgb(srgb_to_linear(srgb))
        np.testing.assert_allclose(recovered, srgb, atol=1e-10)


class TestOKLabRoundtrip:
    """Linear RGB ↔ OKLab conversions must roundtrip accurately."""

    def test_roundtrip_white(self):
        rgb = np.array([1.0, 1.0, 1.0])
        lab = linear_rgb_to_oklab(rgb)
        recovered = oklab_to_linear_rgb(lab)
        np.testing.assert_allclose(recovered, rgb, atol=1e-8)

    def test_roundtrip_black(self):
        rgb = np.array([0.0, 0.0, 0.0])
        lab = linear_rgb_to_oklab(rgb)
        recovered = oklab_to_linear_rgb(lab)
        np.testing.assert_allclose(recovered, rgb, atol=1e-8)

    def test_white_lightness_is_one(self):
        rgb = np.array([1.0, 1.0, 1.0])
        lab = linear_rgb_to_oklab(rgb)
        assert lab[0] == pytest.approx(1.0, abs=1e-6)

    def test_black_lightness_is_zero(self):
        rgb = np.array([0.0, 0.0, 0.0])
        lab = linear_rgb_to_oklab(rgb)
        assert lab[0] == pytest.approx(0.0, abs=1e-6)

    def test_batch_roundtrip(self):
        rgb = np.random.RandomState(42).random((50, 3))
        recovered = oklab_to_linear_rgb(linear_rgb_to_oklab(rgb))
        np.testing.assert_allclose(recovered, rgb, atol=1e-8)


class TestOKLCHRoundtrip:
    """OKLab ↔ OKLCH conversions must roundtrip accurately."""

    def test_roundtrip_chromatic(self):
        lab = np.array([0.7, 0.1, -0.05])
        lch = oklab_to_oklch(lab)
        recovered = oklch_to_oklab(lch)
        np.testing.assert_allclose(recovered, lab, atol=1e-10)

    def test_chroma_calculation(self):
        lab = np.array([0.5, 0.3, 0.4])
        lch = oklab_to_oklch(lab)
        expected_c = np.sqrt(0.3**2 + 0.4**2)
        assert lch[1] == pytest.approx(expected_c, abs=1e-10)

    def test_achromatic_zero_chroma(self):
        lab = np.array([0.5, 0.0, 0.0])
        lch = oklab_to_oklch(lab)
        assert lch[1] == pytest.approx(0.0, abs=1e-10)

    def test_hue_range(self):
        """Hue must be in [0, 360)."""
        lab = np.array([0.5, -0.1, 0.1])
        lch = oklab_to_oklch(lab)
        assert 0.0 <= lch[2] < 360.0


class TestFullChainRoundtrip:
    """sRGB → OKLCH → sRGB must roundtrip for in-gamut colors."""

    def test_roundtrip_red(self):
        srgb = np.array([1.0, 0.0, 0.0])
        lch = srgb_to_oklch(srgb)
        recovered = oklch_to_srgb(lch)
        np.testing.assert_allclose(recovered, srgb, atol=1e-4)

    def test_roundtrip_green(self):
        srgb = np.array([0.0, 1.0, 0.0])
        recovered = oklch_to_srgb(srgb_to_oklch(srgb))
        np.testing.assert_allclose(recovered, srgb, atol=1e-4)

    def test_roundtrip_blue(self):
        srgb = np.array([0.0, 0.0, 1.0])
        recovered = oklch_to_srgb(srgb_to_oklch(srgb))
        np.testing.assert_allclose(recovered, srgb, atol=1e-4)

    def test_uint8_conversion(self):
        pixels = np.array([[128, 64, 200]], dtype=np.uint8)
        lch = srgb_uint8_to_oklch(pixels)
        assert lch.shape == (1, 3)
        assert 0.0 <= lch[0, 0] <= 1.0  # L in range
        assert lch[0, 1] >= 0.0  # C non-negative


class TestHexConversion:
    """Hex ↔ OKLCH conversion tests."""

    def test_hex_roundtrip_red(self):
        hex_val = oklch_to_hex(0.6, 0.2, 30.0)
        assert hex_val.startswith("#")
        assert len(hex_val) == 7

    def test_hex_to_oklch_black(self):
        L, C, H = hex_to_oklch("#000000")
        assert L == pytest.approx(0.0, abs=0.01)

    def test_hex_to_oklch_white(self):
        L, C, H = hex_to_oklch("#FFFFFF")
        assert L == pytest.approx(1.0, abs=0.01)
        assert H is None  # Achromatic

    def test_hex_to_oklch_achromatic_detection(self):
        L, C, H = hex_to_oklch("#808080")
        assert H is None  # Gray is achromatic


class TestDeltaE:
    """Perceptual color difference tests."""

    def test_identical_colors_zero(self):
        de = delta_e_oklch(0.5, 0.1, 200.0, 0.5, 0.1, 200.0)
        assert de == pytest.approx(0.0, abs=1e-10)

    def test_black_white_large_distance(self):
        de = delta_e_oklch(0.0, 0.0, None, 1.0, 0.0, None)
        assert de > 0.5  # Black to white is a large distance

    def test_similar_colors_small_distance(self):
        de = delta_e_oklch(0.5, 0.1, 200.0, 0.51, 0.1, 200.0)
        assert de < 0.05  # Barely different

    def test_achromatic_handling(self):
        """None hue should not cause errors."""
        de = delta_e_oklch(0.5, 0.0, None, 0.5, 0.0, None)
        assert de == pytest.approx(0.0, abs=1e-10)

    def test_batch_matches_scalar(self):
        colors1 = np.array([[0.5, 0.1, 200.0], [0.8, 0.05, 30.0]])
        colors2 = np.array([[0.6, 0.15, 210.0], [0.7, 0.1, 40.0]])
        batch = delta_e_oklch_batch(colors1, colors2)

        scalar0 = delta_e_oklch(0.5, 0.1, 200.0, 0.6, 0.15, 210.0)
        scalar1 = delta_e_oklch(0.8, 0.05, 30.0, 0.7, 0.1, 40.0)

        assert batch[0] == pytest.approx(scalar0, abs=1e-10)
        assert batch[1] == pytest.approx(scalar1, abs=1e-10)
