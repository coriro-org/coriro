# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""Tests for schema types and serialization roundtrips."""

import pytest

from coriro.schema.color_measurement import (
    OKLCHColor,
    WeightedColor,
    GridSize,
    RegionColor,
    SpatialBins,
    MeasurementMeta,
    TextColorMeasurement,
    AccentMeasurement,
    AccentRegion,
    GradientHint,
    GradientStop,
    GradientDirection,
    ColorMeasurement,
    REGION_IDS,
)


class TestOKLCHColor:

    def test_valid_color(self):
        c = OKLCHColor(L=0.5, C=0.1, H=200.0)
        assert c.L == 0.5
        assert c.C == 0.1
        assert c.H == 200.0

    def test_achromatic_none_hue(self):
        c = OKLCHColor(L=0.5, C=0.0)
        assert c.H is None
        assert c.is_achromatic

    def test_achromatic_low_chroma(self):
        c = OKLCHColor(L=0.5, C=0.01, H=200.0)
        assert c.is_achromatic  # C < 0.02

    def test_invalid_lightness(self):
        with pytest.raises(ValueError, match="Lightness"):
            OKLCHColor(L=1.5, C=0.1, H=200.0)

    def test_invalid_chroma(self):
        with pytest.raises(ValueError, match="Chroma"):
            OKLCHColor(L=0.5, C=-0.1)

    def test_invalid_hue(self):
        with pytest.raises(ValueError, match="Hue"):
            OKLCHColor(L=0.5, C=0.1, H=400.0)

    def test_to_dict_roundtrip(self):
        c = OKLCHColor(L=0.5, C=0.1, H=200.0, sample_hex="#3366CC")
        d = c.to_dict()
        recovered = OKLCHColor.from_dict(d)
        assert recovered.L == c.L
        assert recovered.C == c.C
        assert recovered.H == c.H
        assert recovered.sample_hex == c.sample_hex

    def test_sample_hex_preferred(self):
        c = OKLCHColor(L=0.5, C=0.1, H=200.0, sample_hex="#3366CC")
        assert c.hex == "#3366CC"

    def test_frozen(self):
        c = OKLCHColor(L=0.5, C=0.1, H=200.0)
        with pytest.raises(AttributeError):
            c.L = 0.6


class TestWeightedColor:

    def test_valid(self):
        wc = WeightedColor(
            color=OKLCHColor(L=0.5, C=0.1, H=200.0),
            weight=0.6,
        )
        assert wc.weight == 0.6

    def test_invalid_weight(self):
        with pytest.raises(ValueError, match="Weight"):
            WeightedColor(
                color=OKLCHColor(L=0.5, C=0.1, H=200.0),
                weight=1.5,
            )

    def test_to_dict_roundtrip(self):
        wc = WeightedColor(
            color=OKLCHColor(L=0.5, C=0.1, H=200.0),
            weight=0.6,
        )
        recovered = WeightedColor.from_dict(wc.to_dict())
        assert recovered.weight == wc.weight
        assert recovered.color.L == wc.color.L


class TestSpatialBins:

    def _make_region(self, region_id: str) -> RegionColor:
        return RegionColor(
            region_id=region_id,
            palette=(
                WeightedColor(OKLCHColor(L=0.5, C=0.1, H=200.0), 0.6),
                WeightedColor(OKLCHColor(L=0.8, C=0.02), 0.4),
            ),
        )

    def test_2x2_valid(self):
        regions = tuple(self._make_region(rid) for rid in REGION_IDS[GridSize.GRID_2X2])
        spatial = SpatialBins(grid=GridSize.GRID_2X2, regions=regions)
        assert len(spatial.regions) == 4

    def test_3x3_valid(self):
        regions = tuple(self._make_region(rid) for rid in REGION_IDS[GridSize.GRID_3X3])
        spatial = SpatialBins(grid=GridSize.GRID_3X3, regions=regions)
        assert len(spatial.regions) == 9

    def test_wrong_count_raises(self):
        regions = tuple(self._make_region(rid) for rid in REGION_IDS[GridSize.GRID_2X2][:3])
        with pytest.raises(ValueError, match="requires 4 regions"):
            SpatialBins(grid=GridSize.GRID_2X2, regions=regions)

    def test_get_region(self):
        regions = tuple(self._make_region(rid) for rid in REGION_IDS[GridSize.GRID_2X2])
        spatial = SpatialBins(grid=GridSize.GRID_2X2, regions=regions)
        r = spatial.get_region("R1C1")
        assert r.region_id == "R1C1"

    def test_to_dict_roundtrip(self):
        regions = tuple(self._make_region(rid) for rid in REGION_IDS[GridSize.GRID_2X2])
        spatial = SpatialBins(grid=GridSize.GRID_2X2, regions=regions)
        recovered = SpatialBins.from_dict(spatial.to_dict())
        assert recovered.grid == spatial.grid
        assert len(recovered.regions) == len(spatial.regions)


class TestMeasurementMeta:

    def test_defaults(self):
        meta = MeasurementMeta()
        assert meta.scope == "area_dominant_surfaces"
        assert meta.coverage == "complete"
        assert meta.min_area_pct == 1.0

    def test_to_dict_roundtrip(self):
        meta = MeasurementMeta(
            min_area_pct=2.0,
            delta_e_collapse=0.05,
            palette_cap=8,
            perceptual_supplements=3,
        )
        recovered = MeasurementMeta.from_dict(meta.to_dict())
        assert recovered.min_area_pct == meta.min_area_pct
        assert recovered.delta_e_collapse == meta.delta_e_collapse
        assert recovered.palette_cap == meta.palette_cap
        assert recovered.perceptual_supplements == meta.perceptual_supplements

    def test_supplements_omitted_when_zero(self):
        meta = MeasurementMeta()
        d = meta.to_dict()
        assert "perceptual_supplements" not in d


class TestColorMeasurement:

    def _make_minimal(self, **overrides) -> ColorMeasurement:
        dom = OKLCHColor(L=0.5, C=0.1, H=200.0)
        palette = (
            WeightedColor(dom, 0.6),
            WeightedColor(OKLCHColor(L=0.9, C=0.01), 0.4),
        )
        region = lambda rid: RegionColor(
            region_id=rid,
            palette=(
                WeightedColor(dom, 0.7),
                WeightedColor(OKLCHColor(L=0.9, C=0.01), 0.3),
            ),
        )
        spatial = SpatialBins(
            grid=GridSize.GRID_2X2,
            regions=tuple(region(rid) for rid in REGION_IDS[GridSize.GRID_2X2]),
        )
        kwargs = dict(dominant=dom, palette=palette, spatial=spatial)
        kwargs.update(overrides)
        return ColorMeasurement(**kwargs)

    def test_minimal_valid(self):
        m = self._make_minimal()
        assert m.dominant.L == 0.5
        assert len(m.palette) == 2
        assert m.text_colors is None
        assert m.accent_regions is None

    def test_empty_palette_raises(self):
        with pytest.raises(ValueError, match="Palette cannot be empty"):
            self._make_minimal(palette=())

    def test_to_dict_roundtrip(self):
        m = self._make_minimal(
            measurement=MeasurementMeta(),
            image_hash="sha256:abc123",
        )
        d = m.to_dict()
        recovered = ColorMeasurement.from_dict(d)
        assert recovered.dominant.L == m.dominant.L
        assert recovered.measurement is not None
        assert recovered.image_hash == "sha256:abc123"

    def test_to_json_parses(self):
        import json
        m = self._make_minimal()
        j = m.to_json()
        data = json.loads(j)
        assert data["version"] == "1.0"
        assert "dominant" in data

    def test_optional_fields_absent_in_dict(self):
        m = self._make_minimal()
        d = m.to_dict()
        assert "text_colors" not in d
        assert "accent_regions" not in d
        assert "gradient" not in d
