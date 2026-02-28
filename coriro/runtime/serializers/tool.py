# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Tool output serializer for function-calling VLMs.

Formats ColorMeasurement as a tool/function result that can be returned
to models supporting tool use (OpenAI, Claude, Qwen, etc.).

The output is structured JSON that models can parse and reason over.
"""

from __future__ import annotations

import json

from coriro.runtime.serializers.base import SerializerFormat
from coriro.schema import ColorMeasurement


def to_tool_output(
    measurement: ColorMeasurement,
    *,
    format: SerializerFormat = SerializerFormat.JSON,
    include_spatial: bool = False,
    include_gradient: bool = True,
    compact_colors: bool = False,
    compact: bool = False,
    include_hex: bool = False,
    hex_only: bool = False,
    image_id: str | None = None,
    consolidated: bool = False,
) -> str:
    """Serialize a ColorMeasurement as tool output JSON.

    This format is designed for VLMs with tool/function calling support.
    The measurement is returned as if it were the result of a
    ``get_image_colors`` tool call.

    Args:
        measurement: The ColorMeasurement to serialize.
        format: Output format (JSON or JSON_PRETTY).
        include_spatial: Include spatial region data.
        include_gradient: Include gradient data if present.
        compact_colors: Use compact color representation (``L0.85/C0.15/H220``).
        compact: Use minimal representation (~100 tokens vs ~1000).
            Implies compact_colors. Only regional dominant (no regional
            palettes). Rounded values.
        include_hex: Include hex color values alongside OKLCH.
        hex_only: Ultra-compact hex-only format (~60 tokens).
            Only hex values, no OKLCH. Implies compact. Best for
            implementation tasks.
        image_id: Optional image identifier for multi-image contexts.
        consolidated: Design-friendly consolidated format.
            Dominant as ``{ "hex": "#...", "oklch": "L.../C.../H..." }``,
            palette with hex + oklch + weight, spatial bins as hex.
            Best for screenshot-to-code workflows.

    Returns:
        JSON string suitable for tool output.

    Example (consolidated=True)::

        {
          "tool": "coriro_color_measurement",
          "image_id": "hero_banner.png",
          "dominant": { "hex": "#5C0D07", "oklch": "L0.31/C0.11/H30" },
          "palette": [
            { "hex": "#5C0D07", "oklch": "L0.31/C0.11/H30", "weight": 0.70 },
            { "hex": "#121212", "oklch": "L0.18/C0.00", "weight": 0.21 }
          ],
          "spatial": {
            "R1C1": "#5C0D07",
            "R1C2": "#5C0D07"
          }
        }
    """
    if consolidated:
        data = _build_consolidated_data(measurement, image_id=image_id, include_spatial=include_spatial)
    else:
        # hex_only implies compact
        if hex_only:
            compact = True
            include_hex = True

        # compact mode implies compact_colors
        if compact:
            compact_colors = True

        data = _build_tool_data(
            measurement,
            include_spatial=include_spatial,
            include_gradient=include_gradient,
            compact_colors=compact_colors,
            compact=compact,
            include_hex=include_hex,
            hex_only=hex_only,
            image_id=image_id,
        )

    if format == SerializerFormat.JSON_PRETTY:
        return json.dumps(data, indent=2)
    else:
        return json.dumps(data, separators=(",", ":"))


def _get_hex(color) -> str:
    """Get hex value from OKLCHColor, preferring sample_hex if available."""
    if color.sample_hex:
        return color.sample_hex
    return color.centroid_hex


def _format_oklch(color) -> str:
    """Format as compact OKLCH string."""
    h_str = f"/H{color.H:.0f}" if color.H is not None else ""
    return f"L{color.L:.2f}/C{color.C:.2f}{h_str}"


def _build_consolidated_data(
    measurement: ColorMeasurement,
    image_id: str | None = None,
    include_spatial: bool = False,
) -> dict:
    """Build design-friendly consolidated output.

    Format optimized for screenshot-to-code workflows:
    - Measurement metadata (closes the world)
    - Hex values primary (for implementation)
    - OKLCH secondary (for perceptual intelligence)
    - Spatial bins as hex (opt-in, off by default)
    """
    result: dict = {
        "tool": "coriro_color_measurement",
    }

    if image_id:
        result["image_id"] = image_id

    # Measurement metadata tells the LLM that the palette is complete
    # above the configured thresholds.
    if measurement.measurement is not None:
        result["measurement"] = measurement.measurement.to_dict()

    # Dominant: paired hex + oklch
    result["dominant"] = {
        "hex": _get_hex(measurement.dominant),
        "oklch": _format_oklch(measurement.dominant),
    }

    # Palette: hex + oklch + weight for each color
    result["palette"] = [
        {
            "hex": _get_hex(wc.color),
            "oklch": _format_oklch(wc.color),
            "weight": round(wc.weight, 2),
        }
        for wc in measurement.palette
    ]

    # Spatial: hex-first (opt-in — low signal for most web screenshots)
    if include_spatial:
        result["spatial"] = {
            region.region_id: _get_hex(region.dominant)
            for region in measurement.spatial.regions
        }

    # Text colors (if OCR pass was enabled)
    if measurement.text_colors is not None:
        result["text_colors"] = measurement.text_colors.to_dict()

    # Accent regions (if solid region detection was enabled)
    if measurement.accent_regions is not None:
        result["accent_regions"] = measurement.accent_regions.to_dict()

    return result


def _build_tool_data(
    measurement: ColorMeasurement,
    include_spatial: bool,
    include_gradient: bool,
    compact_colors: bool,
    compact: bool = False,
    include_hex: bool = False,
    hex_only: bool = False,
    image_id: str | None = None,
) -> dict:
    """Build the tool output data structure."""

    def format_oklch(color) -> str:
        """Format as pure OKLCH string."""
        h_str = f"/H{color.H:.0f}" if color.H is not None else ""
        return f"L{color.L:.2f}/C{color.C:.2f}{h_str}"

    def format_color(color) -> dict | str:
        if hex_only:
            return color.centroid_hex
        if compact_colors:
            return format_oklch(color)
        return color.to_dict(include_hex=include_hex)

    def format_weighted(wc) -> dict | str:
        if compact:
            pct = int(wc.weight * 100)
            hex_val = wc.color.sample_hex or wc.color.centroid_hex
            if hex_only:
                return f"{hex_val} ({pct}%)"
            return f"{hex_val} {format_oklch(wc.color)} ({pct}%)"
        return {
            "color": format_color(wc.color),
            "weight": round(wc.weight, 3),
        }

    result: dict = {
        "tool": "coriro_color_measurement",
    }

    if image_id:
        result["image_id"] = image_id

    if not compact:
        result["version"] = measurement.version

    result["dominant"] = format_color(measurement.dominant)
    result["palette"] = [format_weighted(wc) for wc in measurement.palette]

    if include_spatial:
        if compact:
            result["spatial"] = {
                region.region_id: format_color(region.dominant)
                for region in measurement.spatial.regions
            }
        else:
            result["spatial"] = {
                "grid": measurement.spatial.grid.value,
                "regions": {
                    region.region_id: {
                        "dominant": format_color(region.dominant),
                        "palette": [format_weighted(wc) for wc in region.palette],
                    }
                    for region in measurement.spatial.regions
                },
            }

    if include_gradient and measurement.gradient is not None:
        result["gradient"] = {
            "direction": measurement.gradient.direction.value,
            "stops": [
                {"position": s.position, "color": format_color(s.color)}
                for s in measurement.gradient.stops
            ],
            "confidence": measurement.gradient.confidence,
        }

    if not compact and measurement.image_hash:
        result["image_hash"] = measurement.image_hash

    return result
