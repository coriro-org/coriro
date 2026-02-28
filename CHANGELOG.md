# Changelog

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
