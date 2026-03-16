# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
OCR-based text color extraction.

Optional measurement pass that detects text regions and extracts foreground
colors from those pixels.

Primary engine: OnnxTR (DBNet + MobileNetV3, FP32, ~5MB model).
Fallback engine: Tesseract OCR (pytesseract + system Tesseract).

Extraction method: Per-box context-ring background exclusion + global ΔE
collapse. Background identified via exterior context ring (3px outside box),
with palette-informed validation and pixel-ratio inversion safety net.
Falls back to interior border ring when exterior ring is empty.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from coriro.schema import TextColorMeasurement, WeightedColor, OKLCHColor
from coriro.measure.colorspace import srgb_uint8_to_oklch

logger = logging.getLogger(__name__)


def _pack_rgb(pixels: NDArray[np.uint8]) -> NDArray[np.uint32]:
    """Pack (N, 3) uint8 RGB array into (N,) uint32 for fast np.unique."""
    return (pixels[:, 0].astype(np.uint32) << 16 |
            pixels[:, 1].astype(np.uint32) << 8 |
            pixels[:, 2].astype(np.uint32))


def _unpack_rgb(packed: int) -> tuple[int, int, int]:
    """Unpack a single uint32 back to (R, G, B) tuple."""
    return ((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF)


# Singleton cache for OnnxTR predictor — avoids reloading the ONNX model
# and recreating the InferenceSession on every call (~1.5s saved per call).
_onnxtr_predictor_cache: dict[str, object] = {}


def _get_onnxtr_predictor(
    arch: str = "db_mobilenet_v3_large",
    load_in_8_bit: bool = False,
):
    """Get or create a cached OnnxTR detection predictor.

    FP32 by default. INT8 quantization is available but not recommended
    as it significantly reduces detection coverage.
    """
    key = f"{arch}:int8={load_in_8_bit}"
    if key not in _onnxtr_predictor_cache:
        from onnxtr.models import detection_predictor
        predictor = detection_predictor(
            arch=arch,
            load_in_8_bit=load_in_8_bit,
        )
        _onnxtr_predictor_cache[key] = predictor

        # Log active execution provider (e.g., CoreML on Apple Silicon).
        try:
            session = predictor.model.session
            providers = session.get_providers()
            logger.info("OnnxTR EP: %s (arch=%s, int8=%s)", providers, arch, load_in_8_bit)
        except Exception:
            pass

    return _onnxtr_predictor_cache[key]


def is_available() -> bool:
    """Check if text color extraction is available (onnxtr or pytesseract)."""
    try:
        import onnxtr  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_text_boxes_onnxtr(
    pixels: NDArray[np.uint8],
) -> tuple[list[tuple[int, int, int, int]], str] | None:
    """Detect text bounding boxes using OnnxTR's DBNet detector.

    Returns list of (x1, y1, x2, y2) pixel-coordinate boxes and engine name,
    or None if onnxtr is not available.
    """
    try:
        predictor = _get_onnxtr_predictor()
    except ImportError:
        return None
    except Exception as e:
        logger.warning("OnnxTR failed (%s: %s), falling back to Tesseract", type(e).__name__, e)
        return None

    height, width = pixels.shape[:2]
    # Result is list[ndarray] — one array per input image, shape (N, 5)
    # Columns: [x_min, y_min, x_max, y_max, confidence] in [0,1] coords
    result = predictor([pixels])

    if len(result) == 0 or len(result[0]) == 0:
        return None

    boxes: list[tuple[int, int, int, int]] = []
    for xmin, ymin, xmax, ymax, _conf in result[0]:
        x1 = max(0, int(xmin * width))
        y1 = max(0, int(ymin * height))
        x2 = min(width, int(xmax * width))
        y2 = min(height, int(ymax * height))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return (boxes, "onnxtr") if boxes else None


def _detect_text_boxes_tesseract(
    pixels: NDArray[np.uint8],
) -> tuple[list[tuple[int, int, int, int]], str] | None:
    """Detect text bounding boxes using Tesseract OCR.

    Returns list of (x1, y1, x2, y2) pixel-coordinate boxes and engine name,
    or None if pytesseract is not available or OCR fails.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None

    img = Image.fromarray(pixels)
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception:
        return None

    height, width = pixels.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        if not data["text"][i].strip():
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        # Pad Tesseract boxes: Tesseract produces tight bounding boxes that
        # place the border ring on text pixels instead of background.
        # Expand by max(2, 15% of box height) on all sides, clamped to image.
        pad = max(2, int(h * 0.15))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return (boxes, "tesseract") if boxes else None


def _compute_context_ring_bg(
    pixels: NDArray[np.uint8],
    x1: int, y1: int, x2: int, y2: int,
    all_boxes: list[tuple[int, int, int, int]],
    box_index: int,
    pad: int = 3,
) -> tuple[int, int, int] | None:
    """Sample background from a ring OUTSIDE the bounding box.

    Text cannot exist outside its bounding box, so exterior pixels are
    guaranteed background. This avoids the border-pixel inversion defect
    where tight boxes (e.g. Tesseract) place the interior ring on text
    pixels instead of background.

    Masks out other detected boxes to prevent text contamination from
    neighboring boxes.

    Returns the mode RGB tuple, or None if the exterior ring is empty
    (e.g. box fills the entire image edge).
    """
    img_h, img_w = pixels.shape[:2]

    # Outer rectangle (expanded by pad, clamped to image)
    ox1 = max(0, x1 - pad)
    oy1 = max(0, y1 - pad)
    ox2 = min(img_w, x2 + pad)
    oy2 = min(img_h, y2 + pad)

    # Build mask: outer rectangle minus inner box
    ring_h = oy2 - oy1
    ring_w = ox2 - ox1
    if ring_h < 1 or ring_w < 1:
        return None

    mask = np.ones((ring_h, ring_w), dtype=bool)

    # Carve out the original box
    inner_y1 = y1 - oy1
    inner_y2 = y2 - oy1
    inner_x1 = x1 - ox1
    inner_x2 = x2 - ox1
    mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

    # Carve out other detected boxes (prevent neighbor text contamination)
    for j, (bx1, by1, bx2, by2) in enumerate(all_boxes):
        if j == box_index:
            continue
        # Overlap with the outer rectangle
        cx1 = max(bx1, ox1) - ox1
        cy1 = max(by1, oy1) - oy1
        cx2 = min(bx2, ox2) - ox1
        cy2 = min(by2, oy2) - oy1
        if cx2 > cx1 and cy2 > cy1:
            mask[cy1:cy2, cx1:cx2] = False

    ring_region = pixels[oy1:oy2, ox1:ox2]
    ring_pixels = ring_region[mask]

    if len(ring_pixels) == 0:
        return None

    packed = _pack_rgb(ring_pixels)
    unique, counts = np.unique(packed, return_counts=True)
    mode_packed = unique[np.argmax(counts)]
    return _unpack_rgb(int(mode_packed))


def _interior_border_bg(
    box: NDArray[np.uint8],
) -> tuple[int, int, int] | None:
    """Fallback: interior 2px border ring mode when context ring is empty."""
    bh, bw = box.shape[:2]
    if bh < 4 or bw < 4:
        return None
    border_mask = np.zeros((bh, bw), dtype=bool)
    border_mask[:2, :] = True
    border_mask[-2:, :] = True
    border_mask[:, :2] = True
    border_mask[:, -2:] = True
    border_pixels = box[border_mask]
    packed = _pack_rgb(border_pixels)
    unique, counts = np.unique(packed, return_counts=True)
    return _unpack_rgb(int(unique[np.argmax(counts)]))


def _snap_to_palette(
    bg_rgb: tuple[int, int, int],
    palette: list[tuple[int, int, int]],
    max_delta_e: float = 0.10,
) -> tuple[int, int, int]:
    """Snap a background color to the nearest palette color within ΔE threshold.

    Uses the pre-computed surface palette as a high-confidence prior for
    background identification. If the context-ring mode is within ΔE of a
    palette color, use the palette color instead (more stable on gradients
    and multi-surface boundaries).
    """
    from coriro.measure.colorspace import delta_e_oklch

    bg_oklch = srgb_uint8_to_oklch(np.array(bg_rgb, dtype=np.uint8))
    bg_L, bg_C, bg_H = float(bg_oklch[0]), float(bg_oklch[1]), float(bg_oklch[2])
    bg_H_eff = bg_H if bg_C >= 0.02 else None

    best_de = max_delta_e
    best_rgb = bg_rgb

    for p_rgb in palette:
        p_oklch = srgb_uint8_to_oklch(np.array(p_rgb, dtype=np.uint8))
        p_L, p_C, p_H = float(p_oklch[0]), float(p_oklch[1]), float(p_oklch[2])
        p_H_eff = p_H if p_C >= 0.02 else None
        de = delta_e_oklch(bg_L, bg_C, bg_H_eff, p_L, p_C, p_H_eff)
        if de < best_de:
            best_de = de
            best_rgb = p_rgb

    return best_rgb


def _extract_foreground(
    box: NDArray[np.uint8],
    bg_rgb: tuple[int, int, int],
    bg_exclude_delta_e: float,
) -> tuple[Counter, int]:
    """Extract foreground pixels from a box given a background color.

    Returns (foreground_counter, total_box_pixels).
    """
    from coriro.measure.colorspace import delta_e_oklch

    bg_oklch = srgb_uint8_to_oklch(np.array(bg_rgb, dtype=np.uint8))
    bg_L, bg_C, bg_H = float(bg_oklch[0]), float(bg_oklch[1]), float(bg_oklch[2])
    bg_H_eff = bg_H if bg_C >= 0.02 else None

    box_flat = box.reshape(-1, 3)
    box_packed = _pack_rgb(box_flat)
    unique_pixels, pixel_counts = np.unique(box_packed, return_counts=True)

    fg_counter: Counter = Counter()
    for idx in range(len(unique_pixels)):
        rgb = _unpack_rgb(int(unique_pixels[idx]))
        count = int(pixel_counts[idx])
        oklch = srgb_uint8_to_oklch(np.array(rgb, dtype=np.uint8))
        L, C, H = float(oklch[0]), float(oklch[1]), float(oklch[2])
        H_eff = H if C >= 0.02 else None
        de = delta_e_oklch(L, C, H_eff, bg_L, bg_C, bg_H_eff)
        if de >= bg_exclude_delta_e:
            fg_counter[rgb] += count

    return fg_counter, len(box_flat)


# Pixel-ratio inversion threshold: text glyphs occupy 20-40% of a bounding
# box. If foreground exceeds this ratio, background identification is likely
# inverted (text color was identified as background).
_INVERSION_FG_RATIO = 0.60


def extract_text_colors(
    pixels: NDArray[np.uint8],
    *,
    max_colors: int = 0,
    min_area_pct: float = 0.5,
    delta_e_threshold: float = 0.15,  # Collapse AA variants globally
    bg_exclude_delta_e: float = 0.20,  # Per-box local background exclusion radius
    palette: list[tuple[int, int, int]] | None = None,
) -> Optional[TextColorMeasurement]:
    """
    Extract text foreground colors via text detection + per-box background exclusion.

    Three-layer background identification per box:
    1. Context ring — sample pixels OUTSIDE the box (text cannot exist outside
       its bounding box, so exterior pixels are guaranteed background).
    2. Palette-informed validation — snap context-ring mode to nearest surface
       palette color within ΔE 0.10 for stability on gradients.
    3. Pixel-ratio safety net — if foreground > 60% of box area, the extraction
       is inverted (text is always minority area). Re-extract with corrected bg.

    Falls back to interior border ring when the context ring is empty
    (box at image edge with no exterior pixels).

    Then globally collapse anti-aliased variants (ΔE threshold) and filter
    by min_area_pct.

    Args:
        pixels: Image array of shape (H, W, 3) with uint8 sRGB values
        max_colors: Maximum text colors to return (0 = no cap)
        min_area_pct: Minimum weight percentage for inclusion
        delta_e_threshold: ΔE threshold for collapsing similar text colors
        bg_exclude_delta_e: ΔE radius for per-box local background exclusion
        palette: Optional surface palette as list of (R, G, B) tuples.
            When provided, context-ring background is snapped to the nearest
            palette color within ΔE 0.10 for more stable bg identification.

    Returns:
        TextColorMeasurement if text was detected, None otherwise.
    """
    # Try OnnxTR first, fall back to Tesseract
    detection = _detect_text_boxes_onnxtr(pixels)
    if detection is None:
        detection = _detect_text_boxes_tesseract(pixels)
        if detection is not None:
            logger.warning("OnnxTR unavailable, using Tesseract fallback for text detection")
    if detection is None:
        return None

    detected_boxes, engine_name = detection

    foreground_counter: Counter = Counter()

    for i, (x1, y1, x2, y2) in enumerate(detected_boxes):
        box = pixels[y1:y2, x1:x2]
        bh, bw = box.shape[:2]
        if bh < 4 or bw < 4:
            continue

        # Phase 2: Context ring (exterior sampling) — primary bg identification
        bg_rgb = _compute_context_ring_bg(pixels, x1, y1, x2, y2, detected_boxes, i)

        # Fallback to interior border ring if context ring is empty
        if bg_rgb is None:
            bg_rgb = _interior_border_bg(box)
        if bg_rgb is None:
            continue

        # Phase 3: Palette-informed validation — snap to nearest palette color
        if palette:
            bg_rgb = _snap_to_palette(bg_rgb, palette)

        # Extract foreground
        fg_counter, total_box_px = _extract_foreground(box, bg_rgb, bg_exclude_delta_e)

        # Phase 4: Pixel-ratio safety net — detect inverted extraction
        fg_px = sum(fg_counter.values())
        if total_box_px > 0 and fg_px / total_box_px > _INVERSION_FG_RATIO:
            # Foreground > 60% → likely inverted. The "background" we identified
            # is actually the text color, and inter-glyph gaps are "foreground".
            # Use the second-most-common context ring color as background.
            # If that also fails, use the current fg mode as the new background.
            fg_mode_rgb = fg_counter.most_common(1)[0][0] if fg_counter else None
            if fg_mode_rgb is not None:
                logger.debug(
                    "Box %d (%d,%d)-(%d,%d): fg_ratio=%.1f%% > 60%%, re-extracting with swapped bg",
                    i, x1, y1, x2, y2, fg_px / total_box_px * 100,
                )
                fg_counter, _ = _extract_foreground(box, fg_mode_rgb, bg_exclude_delta_e)

        foreground_counter += fg_counter

    if not foreground_counter:
        return None

    # Global counting on collected foreground pixels
    all_colors = foreground_counter.most_common()
    total_fg_pixels = sum(count for _, count in all_colors)

    # Build weighted colors, filtering by min_area_pct
    results: list[WeightedColor] = []

    for (r, g, b), count in all_colors:
        weight = count / total_fg_pixels

        if weight * 100 < min_area_pct:
            continue

        rgb_array = np.array([r, g, b], dtype=np.uint8)
        oklch = srgb_uint8_to_oklch(rgb_array)
        L, C, H = float(oklch[0]), float(oklch[1]), float(oklch[2])
        h_value = H if C >= 0.02 else None
        hex_val = f"#{r:02X}{g:02X}{b:02X}"

        color = OKLCHColor(L=L, C=C, H=h_value, sample_hex=hex_val)
        results.append(WeightedColor(color=color, weight=weight))

    if not results:
        return None

    # Collapse similar colors (merges anti-aliased variants).
    # Achromatic pairs use wider threshold to merge opacity composites.
    results = _collapse_similar_text_colors(results, delta_e_threshold)

    # Optional cap (0 = no limit)
    if max_colors > 0:
        results = results[:max_colors]

    # Renormalize weights
    total_weight = sum(wc.weight for wc in results)
    if total_weight > 0:
        results = [
            WeightedColor(color=wc.color, weight=wc.weight / total_weight)
            for wc in results
        ]

    return TextColorMeasurement(
        colors=tuple(results),
        scope="glyph_foreground",
        coverage="complete",
        min_area_pct=min_area_pct,
        ocr_engine=engine_name,
    )


# OKLab achromatic boundary (Ottosson): colors below C~0.02-0.03 carry
# no perceptible hue. Below this threshold, hue is undefined and lightness
# variation typically reflects opacity compositing rather than distinct colors.
_ACHROMATIC_CHROMA_THRESHOLD = 0.03


def _hue_angle_diff(h1: float | None, h2: float | None) -> float:
    """Circular hue angle difference in degrees [0, 180]."""
    if h1 is None or h2 is None:
        return 0.0  # Achromatic — no hue to compare
    diff = abs(h1 - h2) % 360
    return min(diff, 360 - diff)


# Hue protection threshold: chromatic colors with hue angle difference
# greater than this are never collapsed, regardless of ΔE. Prevents
# merging distinct hue families (e.g. gold H=87° into green H=134°)
# that happen to be close in OKLab due to similar L and moderate C.
# AA variants of the same color never exceed ~10° hue difference.
_HUE_PROTECTION_DEGREES = 30.0


def _collapse_similar_text_colors(
    colors: list[WeightedColor],
    delta_e_threshold: float,
    achromatic_delta_e: float = 0.20,
) -> list[WeightedColor]:
    """
    Collapse similar text colors based on ΔE threshold with hue protection.

    Simple greedy approach: iterate through colors, merge any that are
    within threshold of the current representative.

    Three collapse regimes:
    - Achromatic pairs (both C < 0.03): wider ΔE (0.20) — opacity composites.
    - Chromatic pairs with hue diff > 30°: never collapse — different families.
    - Chromatic pairs with hue diff ≤ 30°: standard ΔE threshold — AA variants.
    """
    from coriro.measure.colorspace import delta_e_oklch

    if not colors:
        return colors

    # Sort by weight descending
    colors = sorted(colors, key=lambda wc: wc.weight, reverse=True)

    result: list[WeightedColor] = []
    used = [False] * len(colors)

    for i, wc in enumerate(colors):
        if used[i]:
            continue

        # Start with this color
        total_weight = wc.weight
        used[i] = True
        i_achromatic = wc.color.C < _ACHROMATIC_CHROMA_THRESHOLD

        # Find similar colors
        for j in range(i + 1, len(colors)):
            if used[j]:
                continue

            j_achromatic = colors[j].color.C < _ACHROMATIC_CHROMA_THRESHOLD
            both_chromatic = not i_achromatic and not j_achromatic

            # Hue protection: never collapse chromatic colors from different
            # hue families. OKLab ΔE underweights hue difference at moderate
            # chroma, causing gold/green or blue/purple to merge incorrectly.
            if both_chromatic:
                hue_diff = _hue_angle_diff(wc.color.H, colors[j].color.H)
                if hue_diff > _HUE_PROTECTION_DEGREES:
                    continue

            # Use wider threshold when both colors are achromatic
            threshold = achromatic_delta_e if (i_achromatic and j_achromatic) else delta_e_threshold

            de = delta_e_oklch(
                wc.color.L, wc.color.C, wc.color.H,
                colors[j].color.L, colors[j].color.C, colors[j].color.H,
            )

            if de < threshold:
                total_weight += colors[j].weight
                used[j] = True

        result.append(WeightedColor(color=wc.color, weight=total_weight))

    return result
