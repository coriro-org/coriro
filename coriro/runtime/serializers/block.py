# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Context block serializer for inline injection.

Formats ColorMeasurement as a structured block (XML, JSON, or Markdown) that
can be inserted directly into the conversation context, typically alongside
or before an image.
"""

from __future__ import annotations

import json
from enum import Enum

from coriro.schema import ColorMeasurement


class BlockFormat(Enum):
    """Block format options."""

    XML = "xml"
    JSON = "json"
    MARKDOWN = "markdown"


def to_context_block(
    measurement: ColorMeasurement,
    *,
    format: BlockFormat = BlockFormat.XML,
    include_spatial: bool = False,
    tag_name: str = "color_measurement",
) -> str:
    """Serialize a ColorMeasurement as a context block.

    This format is designed for inline injection into the conversation,
    typically placed immediately before or after an image reference.

    Args:
        measurement: The ColorMeasurement to serialize.
        format: Block format (XML, JSON, or MARKDOWN).
        include_spatial: Include spatial region data.
        tag_name: XML/markdown tag name for the block.

    Returns:
        Formatted block string.

    Example (XML)::

        <color_measurement version="1.0">
          <dominant L="0.85" C="0.15" H="220"/>
          <palette>
            <color L="0.85" C="0.15" H="220" weight="0.60"/>
            <color L="0.98" C="0.01" weight="0.40"/>
          </palette>
          <spatial grid="2x2">
            <region id="R1C1" L="0.85" C="0.15" H="220"/>
            <region id="R1C2" L="0.85" C="0.15" H="220"/>
            <region id="R2C1" L="0.98" C="0.01"/>
            <region id="R2C2" L="0.98" C="0.01"/>
          </spatial>
        </color_measurement>
    """
    if format == BlockFormat.XML:
        return _to_xml(measurement, include_spatial, tag_name)
    elif format == BlockFormat.JSON:
        return _to_json(measurement, include_spatial, tag_name)
    else:
        return _to_markdown(measurement, include_spatial, tag_name)


def _to_xml(
    measurement: ColorMeasurement,
    include_spatial: bool,
    tag_name: str,
) -> str:
    """Generate XML block."""
    lines = [f'<{tag_name} version="{measurement.version}" source="coriro">']

    # Dominant
    dom = measurement.dominant
    h_attr = f' H="{dom.H:.1f}"' if dom.H is not None else ""
    lines.append(f'  <dominant L="{dom.L:.3f}" C="{dom.C:.3f}"{h_attr}/>')

    # Palette
    lines.append("  <palette>")
    for wc in measurement.palette:
        c = wc.color
        h_attr = f' H="{c.H:.1f}"' if c.H is not None else ""
        lines.append(
            f'    <color L="{c.L:.3f}" C="{c.C:.3f}"{h_attr} '
            f'weight="{wc.weight:.3f}"/>'
        )
    lines.append("  </palette>")

    # Spatial
    if include_spatial:
        grid = measurement.spatial.grid.value
        lines.append(f'  <spatial grid="{grid}">')
        for region in measurement.spatial.regions:
            dom = region.dominant
            h_attr = f' H="{dom.H:.1f}"' if dom.H is not None else ""
            lines.append(
                f'    <region id="{region.region_id}" '
                f'L="{dom.L:.3f}" C="{dom.C:.3f}"{h_attr}/>'
            )
        lines.append("  </spatial>")

    # Gradient (if present)
    if measurement.gradient is not None:
        g = measurement.gradient
        lines.append(
            f'  <gradient direction="{g.direction.value}" '
            f'confidence="{g.confidence:.2f}">'
        )
        for stop in g.stops:
            c = stop.color
            h_attr = f' H="{c.H:.1f}"' if c.H is not None else ""
            lines.append(
                f'    <stop position="{stop.position:.2f}" '
                f'L="{c.L:.3f}" C="{c.C:.3f}"{h_attr}/>'
            )
        lines.append("  </gradient>")

    lines.append(f"</{tag_name}>")
    return "\n".join(lines)


def _to_json(
    measurement: ColorMeasurement,
    include_spatial: bool,
    tag_name: str,
) -> str:
    """Generate JSON block with wrapper."""
    data = measurement.to_dict()
    if not include_spatial:
        data.pop("spatial", None)

    wrapped = {tag_name: data}
    return json.dumps(wrapped, indent=2)


def _to_markdown(
    measurement: ColorMeasurement,
    include_spatial: bool,
    tag_name: str,
) -> str:
    """Generate markdown block with code fence."""
    data = measurement.to_dict()
    if not include_spatial:
        data.pop("spatial", None)

    lines = [
        f"<!-- {tag_name} -->",
        "```json",
        json.dumps(data, indent=2),
        "```",
        f"<!-- /{tag_name} -->",
    ]
    return "\n".join(lines)
