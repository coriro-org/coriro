# Coriro

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

Color measurement runtime for vision-language and multimodal inference pipelines.

Coriro extracts color data from image pixels — dominant colors, weighted palettes, spatial distribution — and passes them alongside the image into any vision-language or multimodal model pipeline.

Core runtime is pure Python. No GPU required. Zero lock-in.

```python
from coriro import measure

m = measure("image.png")

m.to_json()    # Raw measurement JSON (schema)
m.to_xml()     # XML block for context injection
m.to_prompt()  # Natural language for system prompts
```

---

## Why Coriro exists

It reads actual image pixels, extracts palettes in perceptually uniform OKLCH, consolidates near-identical colors, catches high-saturation outliers, and maps spatial distribution. The results are output as structured data (JSON, XML, natural language) that any pipeline can consume.

The output includes measurement criteria, coverage thresholds, collapse distances, and palette caps, so the receiving model knows exactly what was measured and what was excluded.

The most immediate reason to use it: multimodal and vision-language models lose color precision during vision encoding. Patch-based encoders downsample and average pixels, so the language model generates color values that are sometimes close, often wrong. Coriro runs pixel-level colorimetry outside the model and passes the results as plain text, the channel where language models are already precise.

But the value extends beyond compensating for encoder limitations. Even a model with perfect color perception does not produce a consolidated, weighted palette in a perceptually uniform color space with spatial distribution and measurement metadata. That is a measurement task. Coriro's output is useful as pipeline data, whether the model's vision is approximate or exact.

```
┌──────────────────────────────────────────────────────┐
│                     VLM Pipeline                     │
│                                                      │
│  ┌──────────┐    ┌────────────────────────────────┐  │
│  │  Image   │───▶│  Vision Encoder                │  │
│  └──────────┘    │  Sees layout, structure, UX    │  │
│       │          │  Approximate color             │  │
│       │          └────────────────┬───────────────┘  │
│       │                           │                  │
│       ▼                           ▼                  │
│  ┌──────────┐    ┌────────────────────────────────┐  │
│  │  Coriro  │───▶│  Language Model Context        │  │
│  │ measure()│    │  Image + Color sidecar data    │  │
│  └──────────┘    │  = Complete information        │  │
│  Pixel-level     └────────────────────────────────┘  │
│  colorimetry                                         │
│  (sidecar)                                           │
└──────────────────────────────────────────────────────┘
```

**The sidecar principle:** Coriro output runs *alongside* the image, not as a replacement. The model gets the image for layout, hierarchy, and semantics, plus measured color data for precise implementation. Remove Coriro from the pipeline and the model works exactly as before. No weights are modified. No configuration is changed.

---

## Use cases

- **VLM color grounding** — Provide measured palettes as structured data alongside the image, compensating for vision-encoder color loss documented in published benchmarks across many models.
- **Screenshot-to-code** — Provide exact hex values and spatial color distribution so code generation uses measured colors instead of vision-inferred estimates.
- **Product color metadata** — Attach verifiable weighted palettes to product listings for perceptual color search, cross-SKU consistency checks, and return-reduction workflows.
- **Design system compliance** — Measure rendered screenshots against design token definitions using perceptual ΔE distance to catch cross-platform drift.
- **Accessibility contrast auditing** — Compute contrast from rendered pixels in perceptually uniform OKLCH, including text over gradients and background images that DOM-only scanners can miss.
- **Color-aware asset search** — Index images by weighted palette and spatial distribution for retrieval by color proportion, placement, and perceptual similarity.

---

## Install

```bash
pip install coriro
```

Core installation requires only `numpy` and `Pillow`. Optional passes have separate dependencies:

| Feature | Install | Requires |
|---------|---------|----------|
| Text colors | `pip install coriro[text]` | `pytesseract` + system Tesseract OCR |
| Accent regions | `pip install coriro[accents]` | `scipy` |
| CNN smoothing | `pip install coriro[cnn]` | `torch`, `timm` |
| All features | `pip install coriro[all]` | All of the above |

---

## Quick start

### Measure an image

```python
from coriro import measure

m = measure("image.png")

# Dominant color (OKLCH)
print(m.dominant.hex)            # hex from a real pixel in the cluster
print(m.dominant.L)              # Lightness (0.0–1.0)
print(m.dominant.C)              # Chroma (0.0–~0.32)
print(m.dominant.is_achromatic)  # True when C < 0.02

# Palette (weighted, most dominant first)
for wc in m.palette:
    print(f"{wc.color.hex} — {wc.weight:.0%}")

# Spatial distribution
region = m.spatial.get_region("R1C1")  # Top-left
print(region.dominant.hex)
```

### Optional passes

Three additional passes are available. Each is off by default and independently toggleable:

```python
m = measure(
    "image.png",
    smooth=True,            # CNN pixel stabilization (requires torch + timm)
    include_text=True,      # Text foreground colors via OCR (requires pytesseract)
    include_accents=True,   # Solid accent region detection (requires scipy)
)
```

> **CNN smoothing** stabilizes color surfaces before extraction — reducing noise from compression artifacts, gradient banding, and anti-aliasing. It is off by default because it requires `torch` + `timm` (`pip install coriro[cnn]`). If your environment already includes `torch` + `timm`, consider enabling `smooth=True` for improved measurement quality.

> **Latency control:** Coriro processes images at full resolution by default (`max_pixels=0`). For latency-sensitive pipelines, set `max_pixels` to cap the pixel count before processing. Because color measurement is statistical, equivalent palettes are usually preserved at reduced resolution.

### Pipeline integration

```python
from coriro import measure
from coriro.runtime import to_tool_output

m = measure("image.png", include_text=True)

# Structured measurement — pass to your pipeline
coriro_data = to_tool_output(m, consolidated=True)
```

The consolidated format produces:

```json
{
  "tool": "coriro_color_measurement",
  "measurement": {
    "version": "1.0",
    "scope": "area_dominant_surfaces",
    "coverage": "complete",
    "thresholds": { "min_area_pct": 1.0, "delta_e_collapse": 0.03 },
    "palette_cap": 5,
    "spatial_role": "diagnostic"
  },
  "dominant": { "hex": "#1A1A2E", "oklch": "L0.23/C0.04/H283" },
  "palette": [
    { "hex": "#1A1A2E", "oklch": "L0.23/C0.04/H283", "weight": 0.62 },
    { "hex": "#E8453C", "oklch": "L0.63/C0.20/H28", "weight": 0.23 },
    { "hex": "#F5F5F5", "oklch": "L0.97/C0.00", "weight": 0.15 }
  ]
}
```

The `measurement` block tells the receiving system: *this palette is complete above 1% area with ΔE > 0.03 between entries.* Any color not listed is below those thresholds. Omission becomes signal.

### Advanced integration

The simplest use is appending Coriro output as text alongside the image in a VLM prompt when building your system prompts. But the schema supports integration at any level of your architecture.

Measurements are normalized numerical values (L, C, H in perceptually uniform ranges, weights summing to 1.0) with explicit thresholds. Depending on your pipeline, you can:

- Convert palette values into auxiliary feature vectors for multimodal fusion
- Map measurements through a projection layer into your model's embedding space
- Inject color features as auxiliary tokens alongside vision embeddings
- Concatenate structured color vectors with image embeddings before downstream processing
- Use measurements as conditioning signals for adapters, LoRA modules, or FiLM-style modulation layers
- Feed data into custom embedding or feature store pipelines

Because Coriro exposes explicit perceptual thresholds (ΔE collapse distance, coverage floor), these measurements function as deterministic, reproducible conditioning signals. Not heuristic approximations that vary with prompt phrasing.

Coriro does not prescribe how the data is consumed. It provides structured, perceptually grounded color measurements. Whether you integrate them as prompt metadata, auxiliary embeddings, model conditioning, or pipeline features is implementation-specific.

For serialization formats and code examples, see the [documentation](https://coriro.org/docs).

---

## Deployment

Coriro is a Python library, not an HTTP server. It integrates in two common ways:

- **Local Python pipeline (notebooks, scripts, backend workers)** — Import `measure()` in the same Python process that builds model requests, tool payloads, or feature pipelines. Coriro output is available as JSON, XML, or natural language. Pass it however your architecture consumes structured data.
- **Hosted sidecar API (for JS/Next.js/Vercel frontends)** — Run Coriro behind a small Python service that you host separately. Your frontend sends the image, receives the color measurement payload, and integrates it into your pipeline.

This package ships the measurement and serialization library only. If you need an HTTP endpoint, wrap it in any Python web framework. A minimal FastAPI example:

```python
from fastapi import FastAPI, UploadFile
from coriro import measure
import tempfile, os

app = FastAPI()

@app.post("/measure")
async def measure_image(file: UploadFile):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(await file.read())
    tmp.close()
    m = measure(tmp.name)
    os.unlink(tmp.name)
    return m.to_dict()
```

That's the entire service — `pip install coriro fastapi uvicorn python-multipart`, run with `uvicorn app:app`, and your frontend has a color measurement endpoint.

---

## What it measures

### Core (always on)

- **Dominant color** — The single most prominent color, extracted from the weighted palette
- **Area-ranked global palette** — Colors ordered by pixel area, represented in perceptually uniform OKLCH with implementation-accurate `sample_hex` values from real pixels
- **ICC profile conversion** — Automatically converts Display P3, Adobe RGB, and other profiled images to sRGB before measurement via `PIL.ImageCms`, ensuring colors match what color pickers show
- **Color consolidation** — Collapses near-identical colors using perceptual ΔE thresholds in OKLab space, merges black/white families into single representatives, and limits output to a design-friendly palette size
- **Chroma-aware supplementation** — Two safety-net passes catch perceptually significant colors missed by area-dominant extraction:
  - *Chroma outliers:* High-saturation pixels via z-score (>2.0 std deviations above mean chroma). Catches a yellow CTA button on a low-chroma blue page.
  - *Uncovered colors:* Clusters pixels with ΔE > 0.15 from the nearest palette entry. Catches distinct color groups below the area threshold.
- **Spatial color distribution** — Fixed grid partitioning (2×2, 3×3, or 4×4) with per-region palettes. Preserves *where* colors appear — not just *which* colors exist
- **Closed-world measurement metadata** — States the palette's measurement criteria (coverage floor, collapse distance, palette cap). If a color isn't listed, it's below the threshold — omission is signal, not oversight

### Optional passes (independently toggleable, off by default)

- **Text foreground colors** — OCR-assisted glyph region detection via Tesseract, extracting foreground colors with background exclusion. Uses original (unsmoothed) pixels for accuracy
- **Solid accent regions** — Connected-component detection for small but significant solid-color UI elements (CTAs, icons, badges) that fall below area-dominant thresholds. Filters by absolute pixel count, not percentage
- **CNN-guided pixel stabilization** — Shallow ConvNeXt stem (stem + stage 1 only) for reducing compression artifacts, gradient banding, and anti-aliasing. A stabilizer, not an interpreter — measurement logic remains the authority

---

## How it works

Calling `measure()` runs a seven-stage pipeline:

1. **Load & convert** — Reads the image, converts ICC-profiled pixels (Display P3, Adobe RGB) to sRGB via `PIL.ImageCms`. NumPy arrays are accepted directly but must already be sRGB.

2. **Smooth** *(optional)* — CNN pixel stabilization (`smooth=True`). Runs before extraction to reduce compression artifacts, gradient banding, and anti-aliasing before any color analysis.

3. **Color extraction** — Extracts a raw palette using mode-based counting (exact pixel values, default) or k-means clustering (better for photos and gradients). Colors are represented in OKLCH.

4. **Consolidation** — Collapses near-identical colors, merges black/white families, and filters noise. Produces a design-friendly palette at the requested size.

5. **Chroma supplementation** — Two passes recover perceptually significant colors missed by area-dominant extraction (high-saturation outliers, uncovered color clusters). Both filter by novelty against the existing palette to avoid duplicates.

6. **Spatial binning** — Partitions the image into a fixed grid and extracts per-region palettes. Region IDs follow reading order: `R1C1` (top-left) through `R2C2` (bottom-right) for a 2×2 grid.

7. **Optional passes** — Text colors (OCR) and accent regions (connected components). Each is independently toggleable and does not affect the core palette.

---

## Serialization

Coriro separates measurement from delivery. The same `ColorMeasurement` can be serialized for different pipeline targets:

| Format | Function | Use case |
|--------|----------|----------|
| Consolidated JSON | `to_tool_output(m, consolidated=True)` | **Recommended.** Hex + OKLCH + weights in one object |
| Compact JSON | `to_tool_output(m, compact=True)` | Token-constrained pipelines |
| Hex-only | `to_tool_output(m, hex_only=True)` | Minimal output. Implementation tasks only |
| Full JSON | `to_tool_output(m)` | Complete OKLCH data with all metadata |
| XML block | `to_context_block(m, format=BlockFormat.XML)` | Inline context injection (Claude, Qwen) |
| Natural language | `to_system_prompt(m)` | Human-readable for system prompts |
| Markdown | `to_context_block(m, format=BlockFormat.MARKDOWN)` | Markdown-fenced JSON |

See the [documentation](https://coriro.org/docs) for more details.

---

## Contributing

Coriro is open source under the MIT license. Contributions are welcome.

---

## License

MIT License. Copyright © 2026 Saad Irfan. See [LICENSE](LICENSE).

---

[coriro.org](https://coriro.org) · [Documentation](https://coriro.org/docs) · [GitHub](https://github.com/coriro-org/coriro)