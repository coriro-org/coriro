# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
ColorMeasurement v1.0 — Canonical schema for color extraction.

Design principles:
- Immutable: All types are frozen dataclasses
- Deterministic: Same input → same measurement
- Authoritative: Measurements are facts, not inferences
- Serializable: JSON-ready for sidecar injection

Spatial Color Distribution:
    Spatial color distribution is always computed and included in the measurement.
    Serializers control whether it appears in VLM output (off by default).

    Vision encoders irreversibly destroy spatial color relationships through patching
    and downsampling. Coriro partitions the image into fixed spatial bins and reports
    color measurements per region, preserving where colors appear — not just which
    colors exist.

OKLCH Color Space:
- L (Lightness): 0.0 = black, 1.0 = white
- C (Chroma): 0.0 = gray, ~0.32 = max saturation in sRGB
- H (Hue): 0-360 degrees (≈30=orange, ≈90=yellow, ≈145=green, ≈250=blue, ≈330=pink/red)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# =============================================================================
# Schema Version
# =============================================================================

SCHEMA_VERSION = "1.0"


# =============================================================================
# Parsing Helpers
# =============================================================================

_OKLCH_RE = re.compile(r"L([\d.]+)/C([\d.]+)(?:/H([\d.]+))?")


def _parse_oklch_string(s: str) -> tuple[float, float, Optional[float]]:
    """Parse a compact OKLCH string like 'L0.85/C0.15/H220' into (L, C, H).

    Returns (0.5, 0.0, None) as fallback for unparseable input.
    """
    m = _OKLCH_RE.match(s)
    if not m:
        return 0.5, 0.0, None
    L = float(m.group(1))
    C = float(m.group(2))
    H = float(m.group(3)) if m.group(3) is not None else None
    return L, C, H


# =============================================================================
# Core Color Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class OKLCHColor:
    """
    A single color in OKLCH color space.
    
    OKLCH is perceptually uniform and provides human-readable hue values.
    This is the canonical representation for all colors in Coriro.
    
    Attributes:
        L: Lightness (0.0 = black, 1.0 = white)
        C: Chroma (0.0 = neutral gray, typical max ~0.32 for sRGB)
        H: Hue in degrees (0-360, where 0≈red, 120≈green, 240≈blue)
           None for achromatic colors (C ≈ 0)
        sample_hex: Optional hex of a representative real pixel from the cluster.
           The centroid is an average (not a real pixel). For implementation-level
           accuracy (CSS, tokens), use sample_hex. For reasoning, use L/C/H.
    """
    L: float
    C: float
    H: Optional[float] = None
    sample_hex: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate color values are within expected ranges."""
        if not 0.0 <= self.L <= 1.0:
            raise ValueError(f"Lightness must be 0-1, got {self.L}")
        if self.C < 0.0:
            raise ValueError(f"Chroma must be >= 0, got {self.C}")
        if self.H is not None and not 0.0 <= self.H < 360.0:
            raise ValueError(f"Hue must be 0-360, got {self.H}")
    
    @property
    def is_achromatic(self) -> bool:
        """True if the color has no perceptible hue (gray/white/black)."""
        return self.C < 0.02 or self.H is None
    
    @property
    def hex(self) -> str:
        """
        Get hex color string.
        
        Returns sample_hex if available (exact pixel), otherwise computes
        from OKLCH centroid values.
        
        Returns:
            Hex string like "#3941C8"
        """
        if self.sample_hex is not None:
            return self.sample_hex
        from coriro.measure.colorspace import oklch_to_hex
        return oklch_to_hex(self.L, self.C, self.H)
    
    @property
    def centroid_hex(self) -> str:
        """
        Hex computed from OKLCH centroid values (always, ignoring sample_hex).
        
        Use this when you specifically want the cluster average, not the
        representative pixel.
        """
        from coriro.measure.colorspace import oklch_to_hex
        return oklch_to_hex(self.L, self.C, self.H)
    
    def to_dict(self, include_hex: bool = False) -> dict:
        """
        Serialize to dictionary.
        
        Args:
            include_hex: If True, include hex value (sample_hex if available)
        """
        d = {"L": self.L, "C": self.C, "H": self.H}
        if self.sample_hex is not None:
            d["sample_hex"] = self.sample_hex
        if include_hex:
            d["hex"] = self.hex
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> OKLCHColor:
        """Deserialize from dictionary."""
        return cls(
            L=data["L"], 
            C=data["C"], 
            H=data.get("H"),
            sample_hex=data.get("sample_hex"),
        )


@dataclass(frozen=True, slots=True)
class WeightedColor:
    """
    A color with an associated weight indicating relative dominance.
    
    Used in palettes where colors are ranked by visual prominence.
    
    Attributes:
        color: The OKLCH color value
        weight: Relative dominance (0.0-1.0), where weights in a palette sum to 1.0
    """
    color: OKLCHColor
    weight: float
    
    def __post_init__(self) -> None:
        """Validate weight is in valid range."""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be 0-1, got {self.weight}")
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"color": self.color.to_dict(), "weight": self.weight}
    
    @classmethod
    def from_dict(cls, data: dict) -> WeightedColor:
        """Deserialize from dictionary."""
        return cls(
            color=OKLCHColor.from_dict(data["color"]),
            weight=data["weight"],
        )


# =============================================================================
# Spatial Types
# =============================================================================


class GridSize(Enum):
    """
    Supported spatial bin grid sizes.
    
    These are fixed, deterministic partitions — not adaptive or learned.
    """
    GRID_2X2 = "2x2"  # 4 regions  — minimum, default
    GRID_3X3 = "3x3"  # 9 regions  — intermediate
    GRID_4X4 = "4x4"  # 16 regions — high fidelity


def _grid_dimensions(grid: GridSize) -> int:
    """Return the dimension (rows/cols) for a grid size."""
    return {
        GridSize.GRID_2X2: 2,
        GridSize.GRID_3X3: 3,
        GridSize.GRID_4X4: 4,
    }[grid]


def _generate_region_ids(grid: GridSize) -> tuple[str, ...]:
    """
    Generate region IDs for a grid in reading order (left→right, top→bottom).
    
    Naming convention: R{row}C{col} where row and col are 1-indexed.
    
    Example for 2x2:
        ┌───────┬───────┐
        │ R1C1  │ R1C2  │
        ├───────┼───────┤
        │ R2C1  │ R2C2  │
        └───────┴───────┘
    """
    dim = _grid_dimensions(grid)
    return tuple(
        f"R{row}C{col}"
        for row in range(1, dim + 1)
        for col in range(1, dim + 1)
    )


# Pre-computed region IDs for each grid size
REGION_IDS = {
    GridSize.GRID_2X2: _generate_region_ids(GridSize.GRID_2X2),
    GridSize.GRID_3X3: _generate_region_ids(GridSize.GRID_3X3),
    GridSize.GRID_4X4: _generate_region_ids(GridSize.GRID_4X4),
}


@dataclass(frozen=True, slots=True)
class RegionColor:
    """
    Color measurement for a single spatial region.
    
    Each region contains a mini-palette of the most prominent colors
    within that spatial partition. This provides sufficient accuracy
    for layout-sensitive tasks.
    
    Attributes:
        region_id: Stable identifier (e.g., "R1C1", "R2C3")
        palette: Weighted colors in this region, ordered by dominance
                 Weights sum to 1.0 within the region
    """
    region_id: str
    palette: tuple[WeightedColor, ...]
    
    def __post_init__(self) -> None:
        """Validate region structure."""
        if not self.region_id:
            raise ValueError("region_id cannot be empty")
        if not self.palette:
            raise ValueError("Region palette cannot be empty")
        # Verify weights sum to approximately 1.0
        total_weight = sum(wc.weight for wc in self.palette)
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(
                f"Region palette weights must sum to 1.0, got {total_weight:.3f}"
            )
    
    @property
    def dominant(self) -> OKLCHColor:
        """The most prominent color in this region (convenience accessor)."""
        return self.palette[0].color
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "region_id": self.region_id,
            "palette": [wc.to_dict() for wc in self.palette],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> RegionColor:
        """Deserialize from dictionary."""
        return cls(
            region_id=data["region_id"],
            palette=tuple(WeightedColor.from_dict(wc) for wc in data["palette"]),
        )


@dataclass(frozen=True, slots=True)
class SpatialBins:
    """
    Deterministic spatial partitioning of color measurements.
    
    Coriro does not infer layout, objects, or meaning. Instead, it partitions
    the image into fixed, deterministic regions and reports color measurements
    per region. This preserves spatial color facts that vision encoders
    irreversibly destroy during patching and downsampling.
    
    Spatial bins are:
        - Fixed grid partitions (2×2, 3×3, 4×4)
        - Identified by stable IDs (R1C1, R1C2, R2C1, etc.)
        - Computed directly from pixel coordinates
        - Invariant across models and tasks
    
    Spatial bins are NOT:
        - Bounding boxes
        - Object regions
        - Adaptive segments
        - Attention-derived areas
        - Learned masks
    
    Region layout (reading order, left→right, top→bottom):
    
        2×2:                    4×4:
        ┌───────┬───────┐       ┌────┬────┬────┬────┐
        │ R1C1  │ R1C2  │       │R1C1│R1C2│R1C3│R1C4│
        ├───────┼───────┤       ├────┼────┼────┼────┤
        │ R2C1  │ R2C2  │       │R2C1│R2C2│R2C3│R2C4│
        └───────┴───────┘       ├────┼────┼────┼────┤
                                │R3C1│R3C2│R3C3│R3C4│
                                ├────┼────┼────┼────┤
                                │R4C1│R4C2│R4C3│R4C4│
                                └────┴────┴────┴────┘
    
    Attributes:
        grid: Grid size (2x2, 3x3, or 4x4)
        regions: Tuple of RegionColor, one per grid cell, in reading order
    """
    grid: GridSize
    regions: tuple[RegionColor, ...]
    
    def __post_init__(self) -> None:
        """Validate spatial bin structure."""
        expected_ids = REGION_IDS[self.grid]
        expected_count = len(expected_ids)
        
        if len(self.regions) != expected_count:
            raise ValueError(
                f"Grid {self.grid.value} requires {expected_count} regions, "
                f"got {len(self.regions)}"
            )
        
        # Verify region IDs match expected IDs in order
        actual_ids = tuple(r.region_id for r in self.regions)
        if actual_ids != expected_ids:
            raise ValueError(
                f"Region IDs must be {expected_ids}, got {actual_ids}"
            )
    
    def get_region(self, region_id: str) -> RegionColor:
        """Get a specific region by ID."""
        for region in self.regions:
            if region.region_id == region_id:
                return region
        raise KeyError(f"No region with ID '{region_id}'")
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "grid": self.grid.value,
            "regions": [r.to_dict() for r in self.regions],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> SpatialBins:
        """Deserialize from dictionary."""
        return cls(
            grid=GridSize(data["grid"]),
            regions=tuple(RegionColor.from_dict(r) for r in data["regions"]),
        )


# =============================================================================
# Gradient Types
# =============================================================================


class GradientDirection(Enum):
    """Detected gradient direction in the image."""
    HORIZONTAL = "horizontal"        # left → right
    VERTICAL = "vertical"            # top → bottom
    DIAGONAL_DOWN = "diagonal_down"  # top-left → bottom-right
    DIAGONAL_UP = "diagonal_up"      # bottom-left → top-right
    RADIAL = "radial"                # center → edges


@dataclass(frozen=True, slots=True)
class GradientStop:
    """
    A single stop in a gradient.
    
    Attributes:
        position: Normalized position along gradient axis (0.0-1.0)
        color: Color at this position
    """
    position: float
    color: OKLCHColor
    
    def __post_init__(self) -> None:
        """Validate position is in range."""
        if not 0.0 <= self.position <= 1.0:
            raise ValueError(f"Position must be 0-1, got {self.position}")
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"position": self.position, "color": self.color.to_dict()}
    
    @classmethod
    def from_dict(cls, data: dict) -> GradientStop:
        """Deserialize from dictionary."""
        return cls(
            position=data["position"],
            color=OKLCHColor.from_dict(data["color"]),
        )


@dataclass(frozen=True, slots=True)
class GradientHint:
    """
    Detected gradient information in the image.
    
    This is a hint, not a precise reconstruction. It indicates that
    a noticeable color gradient exists and provides approximate stops.
    
    Attributes:
        direction: Primary direction of the gradient
        stops: Tuple of gradient stops (at least 2)
        confidence: Detection confidence (0.0-1.0)
    """
    direction: GradientDirection
    stops: tuple[GradientStop, ...]
    confidence: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate gradient structure."""
        if len(self.stops) < 2:
            raise ValueError("Gradient must have at least 2 stops")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        # Verify stops are ordered by position
        positions = [s.position for s in self.stops]
        if positions != sorted(positions):
            raise ValueError("Gradient stops must be ordered by position")
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "direction": self.direction.value,
            "stops": [s.to_dict() for s in self.stops],
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> GradientHint:
        """Deserialize from dictionary."""
        return cls(
            direction=GradientDirection(data["direction"]),
            stops=tuple(GradientStop.from_dict(s) for s in data["stops"]),
            confidence=data.get("confidence", 1.0),
        )


# =============================================================================
# Measurement Metadata (Closes the World)
# =============================================================================


@dataclass(frozen=True, slots=True)
class MeasurementMeta:
    """
    Metadata about how a measurement was performed.
    
    This block communicates measurement scope and completeness, turning
    the output from "here are some colors" to "here are ALL colors that
    meet these criteria."
    
    Without this metadata, LLMs treat the palette as advisory and may
    fill gaps with vision estimates. With it, omission is meaningful:
    any color not listed is below the measurement thresholds.
    
    Attributes:
        scope: What was measured
            - "area_dominant_surfaces": large contiguous surface-defining colors
        coverage: Whether result is complete ("complete") or sampled ("partial")
        min_area_pct: Minimum area percentage for inclusion (e.g., 1.0 = 1%)
        delta_e_collapse: ΔE threshold for merging similar colors
        palette_cap: Maximum colors in output palette
        spatial_role: Role of spatial bins
            - "diagnostic": layout nuance, may include below-threshold colors
        perceptual_supplements: Number of colors added by chroma-aware
            supplementation (beyond the area-dominant set). Tells VLMs the
            palette includes perceptually significant colors that cover
            less than min_area_pct of pixels.
        version: Schema version for forward compatibility
    """
    version: str = "1.0"
    scope: str = "area_dominant_surfaces"
    coverage: str = "complete"
    min_area_pct: float = 1.0
    delta_e_collapse: float = 1.5
    palette_cap: int = 5
    spatial_role: str = "diagnostic"
    perceptual_supplements: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        d = {
            "version": self.version,
            "scope": self.scope,
            "coverage": self.coverage,
            "thresholds": {
                "min_area_pct": self.min_area_pct,
                "delta_e_collapse": self.delta_e_collapse,
            },
            "palette_cap": self.palette_cap,
            "spatial_role": self.spatial_role,
        }
        if self.perceptual_supplements > 0:
            d["perceptual_supplements"] = self.perceptual_supplements
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "MeasurementMeta":
        """Deserialize from dictionary."""
        thresholds = data.get("thresholds", {})
        return cls(
            scope=data.get("scope", "area_dominant_surfaces"),
            coverage=data.get("coverage", "complete"),
            min_area_pct=thresholds.get("min_area_pct", 1.0),
            delta_e_collapse=thresholds.get("delta_e_collapse", 1.5),
            palette_cap=data.get("palette_cap", 5),
            spatial_role=data.get("spatial_role", "diagnostic"),
            perceptual_supplements=data.get("perceptual_supplements", 0),
        )


# =============================================================================
# Text Color Measurement (OCR-based, orthogonal pass)
# =============================================================================


@dataclass(frozen=True, slots=True)
class TextColorMeasurement:
    """
    Text color measurement from OCR-based glyph detection.
    
    This is an orthogonal measurement pass that extracts foreground text colors
    using OCR to identify glyph regions. It does NOT contaminate the surface
    palette — it's a separate, bounded measurement.
    
    Attributes:
        scope: What was measured ("glyph_foreground")
        coverage: Whether result is complete ("complete") or partial ("partial")
        min_area_pct: Minimum area percentage for inclusion (text is sparse, so lower)
        colors: Extracted text colors with weights
        ocr_engine: Which OCR engine was used (e.g., "tesseract")
    """
    colors: tuple[WeightedColor, ...]
    scope: str = "glyph_foreground"
    coverage: str = "complete"
    min_area_pct: float = 0.1  # Text is sparse, lower threshold
    ocr_engine: str = "tesseract"
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "scope": self.scope,
            "coverage": self.coverage,
            "thresholds": {
                "min_area_pct": self.min_area_pct,
            },
            "ocr_engine": self.ocr_engine,
            "colors": [
                {
                    "hex": wc.color.sample_hex or wc.color.centroid_hex,
                    "oklch": f"L{wc.color.L:.2f}/C{wc.color.C:.2f}" + 
                             (f"/H{wc.color.H:.0f}" if wc.color.H is not None else ""),
                    "weight": round(wc.weight, 2),
                }
                for wc in self.colors
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TextColorMeasurement":
        """Deserialize from dictionary."""
        thresholds = data.get("thresholds", {})
        colors_data = data.get("colors", [])

        colors = []
        for c in colors_data:
            L, C, H = _parse_oklch_string(c.get("oklch", ""))
            color = OKLCHColor(L=L, C=C, H=H, sample_hex=c.get("hex"))
            colors.append(WeightedColor(color=color, weight=c.get("weight", 0.0)))

        return cls(
            colors=tuple(colors),
            scope=data.get("scope", "glyph_foreground"),
            coverage=data.get("coverage", "complete"),
            min_area_pct=thresholds.get("min_area_pct", 0.1),
            ocr_engine=data.get("ocr_engine", "tesseract"),
        )


# =============================================================================
# Accent Region Measurement (Solid UI elements, orthogonal pass)
# =============================================================================


@dataclass(frozen=True, slots=True)
class AccentRegion:
    """
    A single solid-color accent region detected in the image.
    
    Accent regions are contiguous areas of solid color that may be small
    in percentage terms but are significant UI elements (CTAs, icons, badges).
    
    Attributes:
        color: The color of the region
        pixels: Number of pixels in the region
        location: Approximate spatial location (e.g., "R1C1" grid cell)
    """
    color: OKLCHColor
    pixels: int
    location: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        result = {
            "hex": self.color.sample_hex or self.color.centroid_hex,
            "oklch": f"L{self.color.L:.2f}/C{self.color.C:.2f}" + 
                     (f"/H{self.color.H:.0f}" if self.color.H is not None else ""),
            "pixels": self.pixels,
        }
        if self.location:
            result["location"] = self.location
        return result


@dataclass(frozen=True, slots=True)
class AccentMeasurement:
    """
    Accent region measurement from solid-color detection.
    
    This is an orthogonal measurement pass that detects small but significant
    solid-color UI elements (CTAs, icons, badges) that may be missed by the
    area-dominant surface pass due to their small percentage of total pixels.
    
    Unlike the surface pass (which filters by percentage), this pass filters
    by absolute pixel count, catching small but coherent UI elements.
    
    Attributes:
        scope: What was measured ("solid_accent_regions")
        coverage: Whether result is complete ("complete") or partial ("partial")
        min_pixels: Minimum pixel count for inclusion
        regions: Detected accent regions
    """
    regions: tuple[AccentRegion, ...]
    scope: str = "solid_accent_regions"
    coverage: str = "complete"
    min_pixels: int = 5000
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "scope": self.scope,
            "coverage": self.coverage,
            "thresholds": {
                "min_pixels": self.min_pixels,
            },
            "regions": [r.to_dict() for r in self.regions],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AccentMeasurement":
        """Deserialize from dictionary."""
        thresholds = data.get("thresholds", {})
        regions_data = data.get("regions", [])

        regions = []
        for r in regions_data:
            L, C, H = _parse_oklch_string(r.get("oklch", ""))
            color = OKLCHColor(L=L, C=C, H=H, sample_hex=r.get("hex"))
            regions.append(AccentRegion(
                color=color,
                pixels=r.get("pixels", 0),
                location=r.get("location"),
            ))

        return cls(
            regions=tuple(regions),
            scope=data.get("scope", "solid_accent_regions"),
            coverage=data.get("coverage", "complete"),
            min_pixels=thresholds.get("min_pixels", 5000),
        )


# =============================================================================
# Top-Level Measurement Container
# =============================================================================


@dataclass(frozen=True, slots=True)
class ColorMeasurement:
    """
    Complete color measurement for an image.
    
    This is the top-level container produced by Coriro.
    It contains all extracted color information as immutable facts.
    
    Required fields:
        - dominant: Global dominant color
        - palette: Global palette (weighted colors)
        - spatial: Spatial color distribution (always computed)

    Optional fields:
        - measurement: Metadata about how measurement was performed (closes the world)
        - text_colors: OCR-based text foreground colors
        - accent_regions: Solid accent region detection
        - gradient: Detected gradient information
        - image_hash: Hash of source image for verification
    
    Attributes:
        version: Schema version (e.g., "1.0")
        dominant: The single most prominent color in the image
        palette: Ordered tuple of weighted colors (most to least dominant)
        spatial: Grid-based spatial color distribution (REQUIRED)
        measurement: Optional metadata that declares scope/coverage (enables closed-world reasoning)
        gradient: Optional detected gradient information
        image_hash: Optional hash of the source image for verification
    
    Usage:
        measurement = ColorMeasurement(
            dominant=OKLCHColor(L=0.7, C=0.15, H=250.0),
            palette=(
                WeightedColor(OKLCHColor(L=0.7, C=0.15, H=250.0), 0.6),
                WeightedColor(OKLCHColor(L=0.9, C=0.02, H=None), 0.4),
            ),
            spatial=SpatialBins(
                grid=GridSize.GRID_2X2,
                regions=(...),
            ),
        )
    """
    dominant: OKLCHColor
    palette: tuple[WeightedColor, ...]
    spatial: SpatialBins  # REQUIRED — spatial distribution is core, not optional
    version: str = field(default=SCHEMA_VERSION)
    measurement: Optional[MeasurementMeta] = None  # Closes the world when present
    text_colors: Optional[TextColorMeasurement] = None  # OCR-based text pass (optional)
    accent_regions: Optional[AccentMeasurement] = None  # Solid region detection (optional)
    gradient: Optional[GradientHint] = None
    image_hash: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate measurement structure."""
        if not self.palette:
            raise ValueError("Palette cannot be empty")
        # Verify weights sum to approximately 1.0
        total_weight = sum(wc.weight for wc in self.palette)
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(
                f"Palette weights must sum to 1.0, got {total_weight:.3f}"
            )
    
    def to_dict(self) -> dict:
        """
        Serialize to dictionary for JSON output.
        
        This is the primary serialization method for sidecar injection.
        """
        result = {
            "version": self.version,
            "dominant": self.dominant.to_dict(),
            "palette": [wc.to_dict() for wc in self.palette],
            "spatial": self.spatial.to_dict(),
        }
        if self.measurement is not None:
            result["measurement"] = self.measurement.to_dict()
        if self.text_colors is not None:
            result["text_colors"] = self.text_colors.to_dict()
        if self.accent_regions is not None:
            result["accent_regions"] = self.accent_regions.to_dict()
        if self.gradient is not None:
            result["gradient"] = self.gradient.to_dict()
        if self.image_hash is not None:
            result["image_hash"] = self.image_hash
        return result
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_prompt(self, include_spatial: bool = True) -> str:
        """
        Serialize to human-readable format for VLM context.
        
        Authority comes from channel (system payload), not from
        explicit instructions. The output is clean structured data.
        
        Example output:
            ## Coriro Color Measurement
            
            **Dominant Color:** Light blue (L=0.85, C=0.15, H=220°)
            
            **Spatial Distribution (2×2 grid):**
            - R1C1 (top-left): Light blue
            ...
        """
        # Import here to avoid circular imports
        from coriro.runtime.serializers.system import to_system_prompt
        from coriro.runtime.serializers.base import SerializerFormat
        return to_system_prompt(
            self,
            format=SerializerFormat.NATURAL,
            include_spatial=include_spatial,
            preamble=True,
        )
    
    def to_xml(self, include_spatial: bool = True) -> str:
        """
        Serialize to XML block format.
        
        Best for: Claude, Qwen, structured parsing.
        Authority comes from structure, not instructions.
        
        Example output:
            <color_measurement version="1.0" source="coriro">
              <dominant L="0.850" C="0.150" H="220.0"/>
              ...
            </color_measurement>
        """
        # Import here to avoid circular imports
        from coriro.runtime.serializers.block import to_context_block, BlockFormat
        return to_context_block(
            self,
            format=BlockFormat.XML,
            include_spatial=include_spatial,
        )
    
    @classmethod
    def from_dict(cls, data: dict) -> ColorMeasurement:
        """Deserialize from dictionary."""
        return cls(
            version=data.get("version", SCHEMA_VERSION),
            dominant=OKLCHColor.from_dict(data["dominant"]),
            palette=tuple(WeightedColor.from_dict(wc) for wc in data["palette"]),
            spatial=SpatialBins.from_dict(data["spatial"]),
            measurement=(
                MeasurementMeta.from_dict(data["measurement"])
                if data.get("measurement") else None
            ),
            text_colors=(
                TextColorMeasurement.from_dict(data["text_colors"])
                if data.get("text_colors") else None
            ),
            accent_regions=(
                AccentMeasurement.from_dict(data["accent_regions"])
                if data.get("accent_regions") else None
            ),
            gradient=(
                GradientHint.from_dict(data["gradient"])
                if data.get("gradient") else None
            ),
            image_hash=data.get("image_hash"),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> ColorMeasurement:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
