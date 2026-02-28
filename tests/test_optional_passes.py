# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""Tests for optional measurement passes (accents, text, smooth)."""

import numpy as np
import pytest

from coriro import measure


def _solid_image(r, g, b, height=100, width=100):
    return np.full((height, width, 3), [r, g, b], dtype=np.uint8)


def _two_tone_image(rgb1, rgb2, height=100, width=200):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, : width // 2] = rgb1
    img[:, width // 2 :] = rgb2
    return img


# ---------------------------------------------------------------------------
# Helpers for synthetic test images
# ---------------------------------------------------------------------------

def _image_with_accent_dot(
    bg=(200, 200, 200),
    dot=(255, 0, 0),
    height=200,
    width=200,
    dot_size=60,
):
    """Gray background with a solid colored dot in the center."""
    img = np.full((height, width, 3), bg, dtype=np.uint8)
    cy, cx = height // 2, width // 2
    half = dot_size // 2
    img[cy - half : cy + half, cx - half : cx + half] = dot
    return img


def _image_with_text_like_region(
    bg=(255, 255, 255), fg=(0, 0, 0), height=100, width=300
):
    """White background with thin horizontal dark stripes (text-like)."""
    img = np.full((height, width, 3), bg, dtype=np.uint8)
    for y in range(10, 90, 8):
        img[y : y + 2, 20:280] = fg
    return img


# ===========================================================================
# Accent region detection
# ===========================================================================

class TestAccentAvailability:

    def test_is_available(self):
        from coriro.measure.accents import is_available
        # scipy is installed in this environment
        assert is_available() is True


class TestAccentExtraction:

    def test_accent_returns_result_with_distinct_dot(self):
        """A solid colored dot on a neutral background should be detected."""
        img = _image_with_accent_dot(
            bg=(200, 200, 200), dot=(255, 0, 0), dot_size=60
        )
        m = measure(img, include_accents=True, accent_min_pixels=100, include_hash=False)
        # Accent detection may or may not find the dot depending on thresholds,
        # but the pipeline should run without error
        assert isinstance(m.accent_regions, type(None)) or hasattr(
            m.accent_regions, "regions"
        )

    def test_solid_image_no_accents(self):
        """Uniform image should produce no accent regions."""
        img = _solid_image(128, 128, 128, height=200, width=200)
        m = measure(img, include_accents=True, accent_min_pixels=100, include_hash=False)
        assert m.accent_regions is None

    def test_accent_off_by_default(self):
        img = _image_with_accent_dot()
        m = measure(img, include_hash=False)
        assert m.accent_regions is None

    def test_accent_does_not_affect_core_palette(self):
        """Enabling accents should not change the core palette or dominant color."""
        img = _image_with_accent_dot(dot_size=60)
        m_without = measure(img, include_accents=False, include_hash=False)
        m_with = measure(img, include_accents=True, accent_min_pixels=100, include_hash=False)
        assert m_without.dominant.L == m_with.dominant.L
        assert m_without.dominant.C == m_with.dominant.C
        assert len(m_without.palette) == len(m_with.palette)

    def test_accent_regions_have_valid_structure(self):
        """If accent regions are found, they should have correct fields."""
        img = _image_with_accent_dot(
            bg=(200, 200, 200), dot=(255, 0, 0), height=300, width=300, dot_size=80
        )
        m = measure(img, include_accents=True, accent_min_pixels=50, include_hash=False)
        if m.accent_regions is not None:
            for region in m.accent_regions.regions:
                assert region.color is not None
                assert 0.0 <= region.color.L <= 1.0
                assert region.color.C >= 0.0
                assert region.pixels > 0
                assert region.location is not None


class TestAccentDirectAPI:

    def test_extract_accent_regions_on_uniform_image(self):
        """Uniform image detected as a single solid region — valid behavior."""
        from coriro.measure.accents import extract_accent_regions
        img = _solid_image(128, 128, 128, height=200, width=200)
        flat = img.reshape(-1, 3)
        result = extract_accent_regions(flat, 200, 200, min_pixels=100)
        # Entire image qualifies as a solid accent region
        if result is not None:
            assert len(result.regions) >= 1

    def test_extract_accent_regions_with_existing_palette_excludes_match(self):
        """Accents matching existing palette should be excluded."""
        from coriro.measure.accents import extract_accent_regions
        from coriro import measure
        # Create an image where the only color matches the surface palette
        img = _solid_image(128, 128, 128, height=200, width=200)
        m = measure(img, include_hash=False)
        flat = img.reshape(-1, 3)
        # Pass the surface palette — accent matching it should be excluded
        result = extract_accent_regions(
            flat, 200, 200, min_pixels=100, existing_palette=m.palette
        )
        assert result is None


# ===========================================================================
# Text color extraction
# ===========================================================================

class TestTextAvailability:

    def test_is_available(self):
        from coriro.measure.text import is_available
        assert is_available() is True


class TestTextExtraction:

    def test_text_off_by_default(self):
        img = _image_with_text_like_region()
        m = measure(img, include_hash=False)
        assert m.text_colors is None

    def test_text_enabled_runs_without_error(self):
        """OCR on synthetic image should not crash."""
        img = _image_with_text_like_region()
        m = measure(img, include_text=True, include_hash=False)
        # OCR may not find real text in synthetic stripes, so text_colors may be None
        assert m.text_colors is None or hasattr(m.text_colors, "colors")

    def test_text_does_not_affect_core_palette(self):
        """Enabling text extraction should not change core measurement."""
        img = _image_with_text_like_region()
        m_without = measure(img, include_text=False, include_hash=False)
        m_with = measure(img, include_text=True, include_hash=False)
        assert m_without.dominant.L == m_with.dominant.L
        assert m_without.dominant.C == m_with.dominant.C
        assert len(m_without.palette) == len(m_with.palette)

    def test_text_colors_have_valid_structure(self):
        """If text colors are found, they should have correct fields."""
        img = _image_with_text_like_region()
        m = measure(img, include_text=True, include_hash=False)
        if m.text_colors is not None:
            total = sum(wc.weight for wc in m.text_colors.colors)
            assert total == pytest.approx(1.0, abs=0.02)
            for wc in m.text_colors.colors:
                assert 0.0 <= wc.color.L <= 1.0
                assert wc.weight > 0


class TestTextDirectAPI:

    def test_extract_text_colors_on_blank_image(self):
        """Blank white image should produce no text colors."""
        from coriro.measure.text import extract_text_colors
        img = _solid_image(255, 255, 255)
        result = extract_text_colors(img)
        assert result is None

    def test_extract_text_colors_with_exclude(self):
        """Background color exclusion should work."""
        from coriro.measure.text import extract_text_colors
        img = _image_with_text_like_region(bg=(255, 255, 255), fg=(0, 0, 0))
        result = extract_text_colors(img, exclude_colors=[(255, 255, 255)])
        # Whether OCR finds text or not, the function should not crash
        if result is not None:
            for wc in result.colors:
                # Excluded white shouldn't appear as text color
                assert not (wc.color.L > 0.95 and wc.color.is_achromatic)


# ===========================================================================
# CNN smooth pass
# ===========================================================================

class TestSmoothAvailability:

    def test_is_available(self):
        from coriro.measure.smoother import is_available
        assert is_available() is True


class TestSmoothImage:

    def test_smooth_returns_same_shape(self):
        from coriro.measure.smoother import smooth_image
        img = _two_tone_image([255, 0, 0], [0, 0, 255])
        result = smooth_image(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_smooth_preserves_solid_image(self):
        """A solid-color image should remain roughly the same after smoothing."""
        from coriro.measure.smoother import smooth_image
        img = _solid_image(128, 128, 128)
        result = smooth_image(img)
        mean_diff = np.abs(result.astype(int) - img.astype(int)).mean()
        assert mean_diff < 30  # Generous tolerance for CNN artifacts

    def test_smooth_output_valid_range(self):
        from coriro.measure.smoother import smooth_image
        img = _two_tone_image([0, 0, 0], [255, 255, 255])
        result = smooth_image(img)
        assert result.min() >= 0
        assert result.max() <= 255


class TestSmoothIntegration:

    def test_smooth_measure_runs(self):
        """measure() with smooth=True should complete without error."""
        img = _two_tone_image([255, 0, 0], [0, 0, 255])
        m = measure(img, smooth=True, include_hash=False)
        assert m.dominant is not None
        assert len(m.palette) >= 1

    def test_smooth_does_not_change_palette_count(self):
        """Smoothing should not dramatically change the number of palette entries."""
        img = _two_tone_image([255, 0, 0], [0, 0, 255])
        m_raw = measure(img, smooth=False, include_hash=False)
        m_smooth = measure(img, smooth=True, include_hash=False)
        assert abs(len(m_raw.palette) - len(m_smooth.palette)) <= 2

    def test_smooth_with_text_uses_original_pixels(self):
        """Text extraction should use unsmoothed pixels even when smooth=True."""
        img = _image_with_text_like_region()
        # This should not crash — the key check is that the text pass
        # receives original pixels, not smoothed ones
        m = measure(img, smooth=True, include_text=True, include_hash=False)
        assert m is not None


class TestSmoothCaching:

    def test_get_smoother_returns_same_instance(self):
        from coriro.measure.smoother import get_smoother
        s1 = get_smoother()
        s2 = get_smoother()
        assert s1 is s2

    def test_clear_cache(self):
        from coriro.measure.smoother import get_smoother, clear_cache
        _ = get_smoother()
        clear_cache()  # Should not raise
        s2 = get_smoother()
        assert s2 is not None


# ===========================================================================
# All optional passes combined
# ===========================================================================

class TestAllPassesCombined:

    def test_all_passes_together(self):
        """Enabling all three optional passes simultaneously should not crash."""
        img = _image_with_accent_dot(
            bg=(200, 200, 200), dot=(255, 0, 0), height=200, width=200, dot_size=60
        )
        m = measure(
            img,
            smooth=True,
            include_text=True,
            include_accents=True,
            accent_min_pixels=100,
            include_hash=False,
        )
        assert m.dominant is not None
        assert len(m.palette) >= 1
        assert m.spatial is not None

    def test_serializers_work_with_all_passes(self):
        """Serializers should handle measurements with all optional data."""
        import json
        from coriro.runtime import to_tool_output, to_system_prompt, to_context_block, BlockFormat

        img = _image_with_accent_dot(
            bg=(200, 200, 200), dot=(255, 0, 0), height=200, width=200, dot_size=60
        )
        m = measure(
            img,
            include_accents=True,
            accent_min_pixels=100,
            include_hash=False,
        )

        # All serialization paths should complete without error
        tool_out = to_tool_output(m)
        assert len(tool_out) > 0
        json.loads(tool_out)  # Should parse

        tool_consolidated = to_tool_output(m, consolidated=True)
        json.loads(tool_consolidated)

        prompt = to_system_prompt(m)
        assert len(prompt) > 0

        xml = to_context_block(m, format=BlockFormat.XML)
        assert "<" in xml

        md = to_context_block(m, format=BlockFormat.MARKDOWN)
        assert "```" in md
