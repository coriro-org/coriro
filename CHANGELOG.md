# Changelog

## 2.0.0

### Text extraction

- Context ring background identification (exterior 3px sampling outside bounding box)
- Palette-informed background validation (ΔE 0.10 snap to nearest surface color)
- Pixel-ratio inversion guard
- Hue-protected text color collapse (30° threshold)
- Tesseract box padding
- OnnxTR fallback hardening
- OnnxTR model: INT8 → FP32

### Spatial

- Mode-based per-region extraction (replaces k-means centroids)
- Consolidated spatial format includes `coverage` weight per region
- NATURAL prompt spatial shows top-3 palette colors with area percentages

### Performance

- Vectorized palette extraction via NumPy bit-packing and `np.unique`
- Smart downsampling (400px max dimension, NEAREST resampling)
- Singleton OnnxTR model cache
- Vectorized text pixel counting

### Fixed

- Spatial region labels incorrect for 2×2 and 3×3 grids in `to_prompt()`
- Chroma supplement z-score included achromatic pixels in mean/std

### Changed

- Chroma z-score threshold: 2.0σ → 1.5σ
- Text color `min_area_pct` default: 0.1% → 0.5%
- Text color `max_colors` default: 5 → 0 (noise floor filtering)
- Achromatic-aware text color collapse (wider ΔE 0.20 for achromatic pairs)
- Hue-aware deduplication for chroma supplements

### Breaking changes

- Consolidated spatial: `"R1C1": "#hex"` → `"R1C1": {"hex": "#hex", "coverage": 0.72}`
- `extract_text_colors()` gains optional `palette` parameter

### Dependencies

- New optional: `onnxtr[cpu]` >=0.8.0 (Tesseract fallback preserved)

## 1.0.0

Initial release.

- Core measurement pipeline: dominant color, weighted palette, spatial distribution
- OKLCH color space with perceptual ΔE consolidation
- Chroma-aware supplementation (outlier detection, uncovered color recovery)
- ICC profile conversion (Display P3, Adobe RGB → sRGB)
- Closed-world measurement metadata
- Serialization: consolidated JSON, compact JSON, hex-only, XML, Markdown, natural language
- Optional passes: text colors (OCR), accent regions (scipy), CNN smoothing (torch/timm)
- Frozen dataclass schema throughout
- Python 3.9+, pure Python core, no GPU required
