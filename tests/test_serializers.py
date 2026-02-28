# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""Tests for the runtime serializers (tool, system, block)."""

import json

import numpy as np
import pytest

from coriro import measure
from coriro.runtime import (
    BlockFormat,
    to_context_block,
    to_system_prompt,
    to_tool_output,
)


def _solid_image(r, g, b, height=100, width=100):
    return np.full((height, width, 3), [r, g, b], dtype=np.uint8)


def _two_tone_image(rgb1, rgb2, height=100, width=200):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, : width // 2] = rgb1
    img[:, width // 2 :] = rgb2
    return img


def _three_color_image(height=100, width=300):
    """Red / green / blue vertical stripes."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    third = width // 3
    img[:, :third] = [255, 0, 0]
    img[:, third : 2 * third] = [0, 255, 0]
    img[:, 2 * third :] = [0, 0, 255]
    return img


@pytest.fixture
def solid_measurement():
    return measure(_solid_image(100, 100, 100), include_hash=False)


@pytest.fixture
def two_tone_measurement():
    return measure(_two_tone_image([255, 0, 0], [0, 0, 255]), include_hash=False)


@pytest.fixture
def three_color_measurement():
    return measure(_three_color_image(), include_hash=False)


# ---------------------------------------------------------------------------
# to_tool_output — standard mode
# ---------------------------------------------------------------------------

class TestToolOutputStandard:

    def test_returns_string(self, two_tone_measurement):
        result = to_tool_output(two_tone_measurement)
        assert isinstance(result, str)

    def test_parses_as_json(self, two_tone_measurement):
        result = to_tool_output(two_tone_measurement)
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_contains_tool_key(self, two_tone_measurement):
        data = json.loads(to_tool_output(two_tone_measurement))
        assert data["tool"] == "coriro_color_measurement"

    def test_contains_dominant(self, two_tone_measurement):
        data = json.loads(to_tool_output(two_tone_measurement))
        assert "dominant" in data

    def test_contains_palette(self, two_tone_measurement):
        data = json.loads(to_tool_output(two_tone_measurement))
        assert "palette" in data
        assert len(data["palette"]) > 0

    def test_palette_has_weight(self, two_tone_measurement):
        data = json.loads(to_tool_output(two_tone_measurement))
        for entry in data["palette"]:
            assert "weight" in entry

    def test_palette_weights_sum_near_one(self, two_tone_measurement):
        data = json.loads(to_tool_output(two_tone_measurement))
        total = sum(e["weight"] for e in data["palette"])
        assert total == pytest.approx(1.0, abs=0.02)

    def test_spatial_excluded_by_default(self, two_tone_measurement):
        data = json.loads(to_tool_output(two_tone_measurement))
        assert "spatial" not in data

    def test_spatial_included_when_requested(self, two_tone_measurement):
        data = json.loads(to_tool_output(two_tone_measurement, include_spatial=True))
        assert "spatial" in data


# ---------------------------------------------------------------------------
# to_tool_output — consolidated mode
# ---------------------------------------------------------------------------

class TestToolOutputConsolidated:

    def test_parses_as_json(self, two_tone_measurement):
        result = to_tool_output(two_tone_measurement, consolidated=True)
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_contains_measurement_block(self, two_tone_measurement):
        data = json.loads(to_tool_output(two_tone_measurement, consolidated=True))
        assert "measurement" in data

    def test_measurement_block_has_scope(self, two_tone_measurement):
        data = json.loads(to_tool_output(two_tone_measurement, consolidated=True))
        assert "scope" in data["measurement"]
        assert data["measurement"]["scope"] == "area_dominant_surfaces"

    def test_dominant_has_hex_and_oklch(self, two_tone_measurement):
        data = json.loads(to_tool_output(two_tone_measurement, consolidated=True))
        dom = data["dominant"]
        assert "hex" in dom
        assert "oklch" in dom

    def test_palette_entries_have_hex_and_oklch(self, two_tone_measurement):
        data = json.loads(to_tool_output(two_tone_measurement, consolidated=True))
        for entry in data["palette"]:
            assert "hex" in entry
            assert "oklch" in entry
            assert "weight" in entry

    def test_hex_format_valid(self, two_tone_measurement):
        """Hex values should be #RRGGBB format."""
        data = json.loads(to_tool_output(two_tone_measurement, consolidated=True))
        import re
        hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
        assert hex_re.match(data["dominant"]["hex"])
        for entry in data["palette"]:
            assert hex_re.match(entry["hex"])


# ---------------------------------------------------------------------------
# to_tool_output — compact mode
# ---------------------------------------------------------------------------

class TestToolOutputCompact:

    def test_parses_as_json(self, two_tone_measurement):
        result = to_tool_output(two_tone_measurement, compact=True)
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_shorter_than_standard(self, two_tone_measurement):
        standard = to_tool_output(two_tone_measurement)
        compact = to_tool_output(two_tone_measurement, compact=True)
        assert len(compact) < len(standard)


# ---------------------------------------------------------------------------
# to_tool_output — hex_only mode
# ---------------------------------------------------------------------------

class TestToolOutputHexOnly:

    def test_parses_as_json(self, two_tone_measurement):
        result = to_tool_output(two_tone_measurement, hex_only=True)
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_shorter_than_compact(self, two_tone_measurement):
        compact = to_tool_output(two_tone_measurement, compact=True)
        hex_only = to_tool_output(two_tone_measurement, hex_only=True)
        assert len(hex_only) < len(compact)

    def test_palette_entries_contain_hex(self, two_tone_measurement):
        data = json.loads(to_tool_output(two_tone_measurement, hex_only=True))
        import re
        hex_re = re.compile(r"#[0-9A-Fa-f]{6}")
        for entry in data["palette"]:
            # hex_only format uses strings like "#FF0000 (50%)"
            if isinstance(entry, str):
                assert hex_re.search(entry)
            elif isinstance(entry, dict):
                assert any(hex_re.search(str(v)) for v in entry.values())


# ---------------------------------------------------------------------------
# to_tool_output — solid image (single-color edge case)
# ---------------------------------------------------------------------------

class TestToolOutputSolidImage:

    def test_single_color_palette(self, solid_measurement):
        data = json.loads(to_tool_output(solid_measurement))
        assert len(data["palette"]) >= 1

    def test_dominant_matches_palette_top(self, solid_measurement):
        data = json.loads(to_tool_output(solid_measurement, consolidated=True))
        assert data["dominant"]["hex"] == data["palette"][0]["hex"]

    def test_consolidated_single_color(self, solid_measurement):
        data = json.loads(to_tool_output(solid_measurement, consolidated=True))
        assert data["palette"][0]["weight"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# to_system_prompt — natural language
# ---------------------------------------------------------------------------

class TestSystemPromptNatural:

    def test_returns_string(self, two_tone_measurement):
        result = to_system_prompt(two_tone_measurement)
        assert isinstance(result, str)

    def test_not_json(self, two_tone_measurement):
        result = to_system_prompt(two_tone_measurement)
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    def test_mentions_dominant(self, two_tone_measurement):
        result = to_system_prompt(two_tone_measurement)
        assert "dominant" in result.lower() or "primary" in result.lower()

    def test_contains_color_descriptions(self, two_tone_measurement):
        result = to_system_prompt(two_tone_measurement)
        # Natural format uses color names and OKLCH values, not hex
        assert "L=" in result
        assert "C=" in result

    def test_spatial_excluded_by_default(self, two_tone_measurement):
        result = to_system_prompt(two_tone_measurement)
        assert "R1C1" not in result

    def test_spatial_included_when_requested(self, two_tone_measurement):
        result = to_system_prompt(two_tone_measurement, include_spatial=True)
        assert "R1C1" in result or "top" in result.lower()


# ---------------------------------------------------------------------------
# to_system_prompt — JSON format
# ---------------------------------------------------------------------------

class TestSystemPromptJSON:

    def test_returns_string(self, two_tone_measurement):
        from coriro.runtime.serializers.base import SerializerFormat
        result = to_system_prompt(two_tone_measurement, format=SerializerFormat.JSON)
        assert isinstance(result, str)

    def test_contains_json(self, two_tone_measurement):
        from coriro.runtime.serializers.base import SerializerFormat
        result = to_system_prompt(two_tone_measurement, format=SerializerFormat.JSON)
        # Should contain parseable JSON somewhere (possibly wrapped in text)
        assert "{" in result


# ---------------------------------------------------------------------------
# to_context_block — XML
# ---------------------------------------------------------------------------

class TestContextBlockXML:

    def test_returns_string(self, two_tone_measurement):
        result = to_context_block(two_tone_measurement, format=BlockFormat.XML)
        assert isinstance(result, str)

    def test_has_xml_tags(self, two_tone_measurement):
        result = to_context_block(two_tone_measurement, format=BlockFormat.XML)
        assert result.startswith("<")
        assert result.endswith(">")

    def test_default_tag_name(self, two_tone_measurement):
        result = to_context_block(two_tone_measurement, format=BlockFormat.XML)
        assert "<color_measurement" in result
        assert "</color_measurement>" in result

    def test_custom_tag_name(self, two_tone_measurement):
        result = to_context_block(
            two_tone_measurement, format=BlockFormat.XML, tag_name="color_data"
        )
        assert "<color_data" in result
        assert "</color_data>" in result

    def test_contains_dominant(self, two_tone_measurement):
        result = to_context_block(two_tone_measurement, format=BlockFormat.XML)
        assert "<dominant" in result or "dominant" in result

    def test_contains_palette(self, two_tone_measurement):
        result = to_context_block(two_tone_measurement, format=BlockFormat.XML)
        assert "<palette" in result or "palette" in result

    def test_spatial_excluded_by_default(self, two_tone_measurement):
        result = to_context_block(two_tone_measurement, format=BlockFormat.XML)
        assert "<spatial" not in result

    def test_spatial_included_when_requested(self, two_tone_measurement):
        result = to_context_block(
            two_tone_measurement, format=BlockFormat.XML, include_spatial=True
        )
        assert "spatial" in result or "region" in result.lower()


# ---------------------------------------------------------------------------
# to_context_block — JSON
# ---------------------------------------------------------------------------

class TestContextBlockJSON:

    def test_returns_string(self, two_tone_measurement):
        result = to_context_block(two_tone_measurement, format=BlockFormat.JSON)
        assert isinstance(result, str)

    def test_parses_as_json(self, two_tone_measurement):
        result = to_context_block(two_tone_measurement, format=BlockFormat.JSON)
        data = json.loads(result)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# to_context_block — Markdown
# ---------------------------------------------------------------------------

class TestContextBlockMarkdown:

    def test_returns_string(self, two_tone_measurement):
        result = to_context_block(two_tone_measurement, format=BlockFormat.MARKDOWN)
        assert isinstance(result, str)

    def test_has_code_fence(self, two_tone_measurement):
        result = to_context_block(two_tone_measurement, format=BlockFormat.MARKDOWN)
        assert "```" in result

    def test_contains_json_inside_fence(self, two_tone_measurement):
        result = to_context_block(two_tone_measurement, format=BlockFormat.MARKDOWN)
        # Extract content between fences
        lines = result.split("\n")
        json_lines = []
        inside = False
        for line in lines:
            if line.startswith("```") and not inside:
                inside = True
                continue
            if line.startswith("```") and inside:
                break
            if inside:
                json_lines.append(line)
        json_str = "\n".join(json_lines)
        if json_str.strip():
            data = json.loads(json_str)
            assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Cross-format consistency
# ---------------------------------------------------------------------------

class TestCrossFormatConsistency:

    def test_all_formats_produce_output(self, two_tone_measurement):
        """Every serialization path should return a non-empty string."""
        assert len(to_tool_output(two_tone_measurement)) > 0
        assert len(to_tool_output(two_tone_measurement, consolidated=True)) > 0
        assert len(to_tool_output(two_tone_measurement, compact=True)) > 0
        assert len(to_tool_output(two_tone_measurement, hex_only=True)) > 0
        assert len(to_system_prompt(two_tone_measurement)) > 0
        assert len(to_context_block(two_tone_measurement, format=BlockFormat.XML)) > 0
        assert len(to_context_block(two_tone_measurement, format=BlockFormat.JSON)) > 0
        assert len(to_context_block(two_tone_measurement, format=BlockFormat.MARKDOWN)) > 0

    def test_dominant_consistent_across_formats(self, two_tone_measurement):
        """Dominant color OKLCH values should be consistent across formats."""
        import re

        tool_data = json.loads(
            to_tool_output(two_tone_measurement, consolidated=True)
        )
        tool_oklch = tool_data["dominant"]["oklch"]

        xml = to_context_block(two_tone_measurement, format=BlockFormat.XML)
        # consolidated oklch is like "L0.63/C0.26/H29"
        # XML dominant is like L="0.628" C="0.258" H="29.2"
        # Both derive from the same measurement — verify L values are close
        tool_l = float(re.search(r"L([\d.]+)", tool_oklch).group(1))
        xml_l = float(re.search(r'L="([\d.]+)"', xml).group(1))
        assert abs(tool_l - xml_l) < 0.01

    def test_three_color_image_all_colors_present(self, three_color_measurement):
        """All distinct colors should appear across serialization formats."""
        data = json.loads(
            to_tool_output(three_color_measurement, consolidated=True)
        )
        assert len(data["palette"]) >= 3
