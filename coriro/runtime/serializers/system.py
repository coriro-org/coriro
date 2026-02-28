# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
System prompt serializer for VLM system instructions.

Formats ColorMeasurement for injection into the system prompt, providing
the model with authoritative color facts before processing the user's
request.
"""

from __future__ import annotations

import json

from coriro.runtime.serializers.base import SerializerFormat
from coriro.schema import ColorMeasurement


def to_system_prompt(
    measurement: ColorMeasurement,
    *,
    format: SerializerFormat = SerializerFormat.NATURAL,
    include_spatial: bool = False,
    preamble: bool = True,
) -> str:
    """Serialize a ColorMeasurement for system prompt injection.

    This format is designed to be prepended or appended to a system
    prompt, giving the model authoritative color information.

    Args:
        measurement: The ColorMeasurement to serialize.
        format: NATURAL (human-readable) or JSON.
        include_spatial: Include spatial region breakdown.
        preamble: Include explanatory preamble.

    Returns:
        String suitable for system prompt injection.

    Example (NATURAL)::

        ## Coriro Color Measurement

        **Dominant Color:** Light blue (L=0.85, C=0.15, H=220)

        **Palette:**
        1. Light blue (60%) -- L=0.85, C=0.15, H=220
        2. White (40%) -- L=0.98, C=0.01

        **Spatial Distribution (2x2 grid):**
        - R1C1 (top-left): Light blue
        - R1C2 (top-right): White
        ...
    """
    if format == SerializerFormat.NATURAL:
        return _to_natural(measurement, include_spatial, preamble)
    else:
        return _to_json_block(measurement, include_spatial, preamble)


def _to_natural(
    measurement: ColorMeasurement,
    include_spatial: bool,
    preamble: bool,
) -> str:
    """Generate natural language representation."""
    lines: list[str] = []

    if preamble:
        lines.extend([
            "## Coriro Color Measurement",
            "",
        ])

    # Dominant color
    dom = measurement.dominant
    dom_desc = _describe_color(dom)
    lines.append(f"**Dominant Color:** {dom_desc}")
    lines.append("")

    # Palette
    lines.append("**Palette:**")
    for i, wc in enumerate(measurement.palette, 1):
        pct = wc.weight * 100
        desc = _describe_color(wc.color)
        if pct >= 1.0:
            lines.append(f"{i}. {desc} ({pct:.0f}%)")
        elif wc.weight > 0:
            lines.append(f"{i}. {desc} (<1%)")
        else:
            lines.append(f"{i}. {desc} (<1%)")
    lines.append("")

    # Spatial
    if include_spatial:
        grid = measurement.spatial.grid.value
        lines.append(f"**Spatial Distribution ({grid} grid):**")

        region_names = {
            "R1C1": "top-left",
            "R1C2": "top-right",
            "R1C3": "top-center-right",
            "R1C4": "top-far-right",
            "R2C1": "upper-left",
            "R2C2": "upper-right",
            "R2C3": "upper-center-right",
            "R2C4": "upper-far-right",
            "R3C1": "lower-left",
            "R3C2": "lower-right",
            "R3C3": "lower-center-right",
            "R3C4": "lower-far-right",
            "R4C1": "bottom-left",
            "R4C2": "bottom-right",
            "R4C3": "bottom-center-right",
            "R4C4": "bottom-far-right",
        }

        for region in measurement.spatial.regions:
            name = region_names.get(region.region_id, region.region_id)
            desc = _describe_color(region.dominant, short=True)
            lines.append(f"- {region.region_id} ({name}): {desc}")
        lines.append("")

    return "\n".join(lines)


def _to_json_block(
    measurement: ColorMeasurement,
    include_spatial: bool,
    preamble: bool,
) -> str:
    """Generate JSON block representation."""
    lines: list[str] = []

    if preamble:
        lines.extend([
            "## Coriro Color Measurement",
            "",
        ])

    data = measurement.to_dict()
    if not include_spatial:
        data.pop("spatial", None)

    lines.append("```json")
    lines.append(json.dumps(data, indent=2))
    lines.append("```")

    return "\n".join(lines)


def _describe_color(color, short: bool = False) -> str:
    """Generate a human-readable color description."""
    if color.is_achromatic:
        if color.L > 0.9:
            name = "White"
        elif color.L < 0.1:
            name = "Black"
        elif color.L < 0.35:
            name = "Dark gray"
        elif color.L > 0.75:
            name = "Light gray"
        else:
            name = "Gray"
    else:
        name = _hue_to_name(color.H)

        if color.L > 0.75:
            name = f"Light {name.lower()}"
        elif color.L < 0.35:
            name = f"Dark {name.lower()}"

    if short:
        return name

    if color.H is not None:
        return f"{name} (L={color.L:.2f}, C={color.C:.2f}, H={color.H:.0f}\u00b0)"
    return f"{name} (L={color.L:.2f}, C={color.C:.2f})"


def _hue_to_name(hue: float) -> str:
    """Convert OKLCH hue angle to an approximate color name.

    OKLCH hue wheel (approximate ranges used here):
      0-29, 340-359: Red
      30-59: Orange
      60-109: Yellow
      110-159: Green
      160-199: Cyan
      200-259: Blue
      260-309: Purple
      310-339: Pink
    """
    if hue < 30 or hue >= 340:
        return "Red"
    elif hue < 60:
        return "Orange"
    elif hue < 110:
        return "Yellow"
    elif hue < 160:
        return "Green"
    elif hue < 200:
        return "Cyan"
    elif hue < 260:
        return "Blue"
    elif hue < 310:
        return "Purple"
    else:
        return "Pink"
