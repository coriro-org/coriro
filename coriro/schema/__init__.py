# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Schema definitions for color measurements.

All types in this module are immutable (frozen dataclasses).
Once a measurement is produced, it is a fact and cannot be altered.

Spatial color distribution is always computed; serialization is opt-in.
"""

from coriro.schema.color_measurement import (
    REGION_IDS,
    SCHEMA_VERSION,
    AccentMeasurement,
    AccentRegion,
    ColorMeasurement,
    GradientDirection,
    GradientHint,
    GradientStop,
    GridSize,
    MeasurementMeta,
    OKLCHColor,
    RegionColor,
    SpatialBins,
    TextColorMeasurement,
    WeightedColor,
)

# Alias for consistency with naming convention
AccentRegionMeasurement = AccentMeasurement

__all__ = [
    # Version
    "SCHEMA_VERSION",
    # Core types
    "OKLCHColor",
    "WeightedColor",
    # Measurement metadata (closes the world)
    "MeasurementMeta",
    # Text color measurement (OCR-based, optional)
    "TextColorMeasurement",
    # Accent region measurement (solid regions, optional)
    "AccentRegion",
    "AccentMeasurement",
    "AccentRegionMeasurement",  # Alias for AccentMeasurement
    # Spatial types (always computed, serialization opt-in)
    "GridSize",
    "RegionColor",
    "SpatialBins",
    "REGION_IDS",
    # Gradient types (optional)
    "GradientDirection",
    "GradientStop",
    "GradientHint",
    # Top-level container
    "ColorMeasurement",
]

