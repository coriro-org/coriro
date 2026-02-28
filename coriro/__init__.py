# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Coriro -- Color measurement instrument for VLM pipelines.

Extracts explicit, structured color facts from image pixels and delivers
them to the language model as sidecar data.

Quick start::

    from coriro import measure

    m = measure("image.png")
    m.to_prompt()   # Human-readable for VLM prompts
    m.to_xml()      # Structured XML block
    m.to_json()     # Compact JSON
"""

from __future__ import annotations

__version__ = "1.0.0"

from coriro.measure import measure
from coriro.schema import (
    ColorMeasurement,
    GridSize,
    OKLCHColor,
    RegionColor,
    SpatialBins,
    WeightedColor,
)

__all__ = [
    # Core API
    "measure",
    "ColorMeasurement",
    # Types (commonly needed)
    "OKLCHColor",
    "GridSize",
    "WeightedColor",
    "SpatialBins",
    "RegionColor",
    # Version
    "__version__",
]

