# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Main measurement extraction API.

This is the primary entry point for Coriro's measurement core.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from coriro.schema import (
    ColorMeasurement,
    GridSize,
    MeasurementMeta,
    TextColorMeasurement,
)
from coriro.schema.color_measurement import AccentMeasurement, WeightedColor
from coriro.measure.colorspace import srgb_uint8_to_oklch
from coriro.measure.palette import extract_palette, extract_dominant_color, extract_palette_mode
from coriro.measure.regions import extract_spatial_bins
from coriro.measure.consolidation import (
    consolidate_palette,
    ConsolidationConfig,
)
from coriro.measure.text import extract_text_colors
from coriro.measure.accents import extract_accent_regions
from coriro.measure.chroma import find_chroma_outliers, find_uncovered_colors


def measure(
    image: Union[str, Path, NDArray[np.uint8]],
    *,
    palette_size: int = 10,  # Extract more, consolidate down
    grid: GridSize = GridSize.GRID_2X2,
    colors_per_region: int = 3,
    include_hash: bool = True,
    max_pixels: int = 0,  # 0 = no downsampling (accuracy priority)
    use_mode: bool = True,  # Use mode (exact pixels) instead of k-means
    consolidate: bool = True,  # Apply color consolidation
    delta_e_threshold: float = 0.03,  # ΔE threshold for collapsing (OKLab scale)
    max_output_colors: int = 5,  # Final palette size after consolidation
    include_text: bool = False,  # Run OCR-based text color pass (requires pytesseract)
    include_accents: bool = False,  # Run solid region detection for accent colors
    accent_min_pixels: int = 2000,  # Minimum pixel count for accent regions
    smooth: bool = False,  # Apply CNN pixel stabilization (requires torch + timm)
) -> ColorMeasurement:
    """
    Extract a complete color measurement from an image.
    
    This is the primary API for Coriro. It produces a ColorMeasurement
    containing:
    - Global dominant color
    - Global palette (weighted colors, consolidated)
    - Spatial color distribution (required)
    
    Args:
        image: One of:
            - Path to image file (str or Path) — recommended, handles ICC
              profile conversion to sRGB automatically.
            - NumPy array of shape (H, W, 3) with uint8 sRGB values.
              Caller must ensure pixels are in sRGB. PIL's convert('RGB')
              does NOT remap from embedded profiles (Display P3, Adobe RGB),
              so raw PIL arrays will produce shifted colors. Pass a file
              path instead, or use PIL.ImageCms to convert before passing.
        palette_size: Number of colors to extract initially (default: 10)
            More colors are extracted to enable better consolidation.
        grid: Spatial grid size (default: 2x2)
        colors_per_region: Colors per spatial region (default: 3)
        include_hash: Include SHA256 hash of image data (default: True)
        max_pixels: Max pixels to process. Set to 0 to disable downsampling.
        use_mode: If True (default), use mode-based extraction (exact pixel values).
            If False, use k-means clustering (averaged centroids).
            Mode is more accurate for simple screenshots.
        consolidate: If True (default), apply color consolidation:
            - Merge near-identical colors (ΔE < threshold)
            - Normalize black/white families
            - Limit to max_output_colors
        delta_e_threshold: ΔE threshold for collapsing similar colors in OKLab scale
            (default: 0.03 = conservative, only near-identical colors)
        max_output_colors: Final palette size after consolidation (default: 5)
        include_text: If True, run OCR-based text color extraction (default: False).
            Requires pytesseract and Tesseract OCR to be installed.
            This is an orthogonal pass that does not affect the surface palette.
        include_accents: If True, run solid region detection for accent colors
            (default: False). Detects small but significant UI elements.
        accent_min_pixels: Minimum pixel count for accent region inclusion
            (default: 2000).
        smooth: If True, apply CNN pixel stabilization before measurement (default: False).
            Requires torch and timm to be installed.
            This smooths gradients and reduces anti-aliasing artifacts without
            adding semantic interpretation. The CNN is a stabilizer, not a source
            of truth - measurement logic remains the authority.
        
    Returns:
        ColorMeasurement with all extracted data
        
    Example:
        >>> from coriro.measure import measure
        >>> m = measure("image.png")
        >>> print(m.dominant)
        OKLCHColor(L=0.85, C=0.02, H=None)
        >>> print(m.spatial.get_region("R1C1").dominant)
        OKLCHColor(L=0.9, C=0.15, H=220.0)
    """
    # Load image
    pixels, height, width = _load_image(image)

    if height < 2 or width < 2:
        raise ValueError(
            f"Image too small ({width}x{height}): "
            f"minimum dimensions are 2x2 pixels"
        )

    original_height, original_width = height, width

    # Keep original pixels for text extraction (smoothing hurts text accuracy)
    original_pixels = pixels
    
    # Optional: Apply CNN pixel stabilization for surface colors only
    if smooth:
        from coriro.measure.smoother import smooth_image, is_available
        if not is_available():
            raise ImportError(
                "CNN smoothing requires torch and timm. "
                "Install with: pip install coriro[cnn]"
            )
        pixels = smooth_image(pixels)
    
    # Downsample if too large (for performance)
    if max_pixels > 0:
        total_pixels = height * width
        if total_pixels > max_pixels:
            scale = (max_pixels / total_pixels) ** 0.5
            new_height = max(2, int(height * scale))
            new_width = max(2, int(width * scale))
            pixels = _downsample(pixels, new_height, new_width)
            height, width = new_height, new_width
    
    # Convert to OKLCH (keep RGB for sample_hex computation)
    rgb_flat = pixels.reshape(-1, 3)
    oklch_pixels = srgb_uint8_to_oklch(pixels)
    oklch_flat = oklch_pixels.reshape(-1, 3)
    
    # Extract global palette
    if use_mode:
        # Mode-based: exact pixel values (more accurate for screenshots)
        palette = extract_palette_mode(rgb_flat, n_colors=palette_size)
    else:
        # K-means: averaged centroids (better for photos/gradients)
        palette = extract_palette(
            oklch_flat, 
            n_colors=palette_size,
            rgb_pixels=rgb_flat,
        )
    
    # Apply consolidation: collapse similar colors, normalize black/white
    if consolidate:
        config = ConsolidationConfig(
            delta_e_threshold=delta_e_threshold,
            max_palette_size=max_output_colors,
        )
        palette = consolidate_palette(palette, config)

    # Chroma-aware supplementation: catch high-saturation colors missed by
    # area-dominant extraction (e.g. yellow CTA on blue page)
    n_supplements = 0
    chroma_supplements = find_chroma_outliers(oklch_flat, rgb_flat, palette)
    uncovered = find_uncovered_colors(oklch_flat, rgb_flat, palette + chroma_supplements)
    n_supplements = len(chroma_supplements) + len(uncovered)

    if n_supplements > 0:
        palette = palette + chroma_supplements + uncovered
        total = sum(wc.weight for wc in palette)
        palette = tuple(WeightedColor(wc.color, wc.weight / total) for wc in palette)

    # Final cap: ensure palette never exceeds max_output_colors.
    # Supplements may have pushed it over; truncate lowest-weight entries
    # and renormalize.
    if len(palette) > max_output_colors:
        palette = palette[:max_output_colors]
        total = sum(wc.weight for wc in palette)
        if total > 0:
            palette = tuple(WeightedColor(wc.color, wc.weight / total) for wc in palette)

    # Extract global dominant color (first in palette)
    dominant = palette[0].color if palette else extract_dominant_color(oklch_flat)
    
    effective_grid = _fit_grid(grid, height, width)

    # Extract spatial bins (still uses k-means for regional palettes)
    spatial = extract_spatial_bins(
        oklch_flat,
        height=height,
        width=width,
        grid=effective_grid,
        colors_per_region=colors_per_region,
        rgb_flat=rgb_flat,
    )
    
    # Compute hash if requested
    image_hash: Optional[str] = None
    if include_hash:
        image_hash = f"sha256:{hashlib.sha256(pixels.tobytes()).hexdigest()[:16]}"
    
    # Build measurement metadata (closes the world)
    # This tells LLMs that the palette is complete above the thresholds
    measurement_meta: Optional[MeasurementMeta] = None
    if consolidate:
        measurement_meta = MeasurementMeta(
            scope="area_dominant_surfaces",
            coverage="complete",
            min_area_pct=1.0,  # Colors below 1% area are not measured
            delta_e_collapse=delta_e_threshold,
            palette_cap=max_output_colors,
            spatial_role="diagnostic",
            perceptual_supplements=n_supplements,
        )
    
    # Text colors use different thresholds than surfaces (thin glyphs vs
    # large areas). Always use original (unsmoothed) pixels for text.
    text_colors: Optional[TextColorMeasurement] = None
    if include_text:
        # Convert dominant surface color to RGB for exclusion
        # This prevents background from appearing in text colors
        from coriro.measure.colorspace import oklch_to_srgb
        
        lch = np.array([dominant.L, dominant.C, dominant.H if dominant.H is not None else 0.0])
        srgb = oklch_to_srgb(lch.reshape(1, 3)).flatten()
        # Clamp to [0, 1] and convert to uint8
        srgb = np.clip(srgb, 0.0, 1.0)
        dominant_rgb = tuple((srgb * 255).round().astype(np.uint8))
        
        text_colors = extract_text_colors(
            original_pixels,  # Always use unsmoothed for text
            max_colors=3,  # Cap at 3 text colors (main + accent + one extra)
            min_area_pct=1.0,  # Filter out <1% entries (AA noise)
            # Use tighter threshold for text (0.15 in OKLab scale, not 1.5)
            delta_e_threshold=0.15,
            # Exclude the surface dominant (background) from text colors
            exclude_colors=[dominant_rgb],
        )
    
    # Accent region detection: small solid-color UI elements (CTAs, icons)
    # that may fall below the area-dominant threshold.
    accent_regions: Optional[AccentMeasurement] = None
    if include_accents:
        accent_regions = extract_accent_regions(
            original_pixels.reshape(-1, 3),  # Use original pixels
            height=original_height,
            width=original_width,
            min_pixels=accent_min_pixels,
            existing_palette=palette,  # Exclude colors already in palette
        )

    return ColorMeasurement(
        dominant=dominant,
        palette=palette,
        spatial=spatial,
        measurement=measurement_meta,
        text_colors=text_colors,
        accent_regions=accent_regions,
        image_hash=image_hash,
    )


def _load_image(
    image: Union[str, Path, NDArray[np.uint8]],
) -> tuple[NDArray[np.uint8], int, int]:
    """
    Load image from file or validate array.
    
    Applies ICC profile conversion to sRGB if the image has an embedded
    color profile. This ensures colors match what color pickers show.
    
    Returns:
        (pixels, height, width) where pixels has shape (H, W, 3)
    """
    if isinstance(image, (str, Path)):
        # Load from file using PIL
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "Pillow is required for image loading. "
                "Install with: pip install Pillow"
            ) from e
        
        img = Image.open(image)
        
        # Apply ICC profile conversion to sRGB if profile exists
        if 'icc_profile' in img.info:
            try:
                from PIL import ImageCms
                import io
                
                embedded_profile = ImageCms.ImageCmsProfile(
                    io.BytesIO(img.info['icc_profile'])
                )
                srgb_profile = ImageCms.createProfile('sRGB')
                
                # Convert to RGB first if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Apply color profile conversion
                img = ImageCms.profileToProfile(
                    img, embedded_profile, srgb_profile
                )
            except Exception:
                # If ICC conversion fails, fall back to simple RGB conversion
                if img.mode != "RGB":
                    img = img.convert("RGB")
        else:
            # No ICC profile - simple RGB conversion
            if img.mode != "RGB":
                img = img.convert("RGB")
        
        pixels = np.array(img, dtype=np.uint8)
        height, width = pixels.shape[:2]
        
    elif isinstance(image, np.ndarray):
        pixels = image
        
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) array, got shape {pixels.shape}"
            )
        
        if pixels.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 array, got {pixels.dtype}"
            )
        
        height, width = pixels.shape[:2]
        
    else:
        raise TypeError(
            f"Expected file path or numpy array, got {type(image)}"
        )
    
    return pixels, height, width


def _fit_grid(grid: GridSize, height: int, width: int) -> GridSize:
    """Step down the grid if the image has fewer pixels than grid cells."""
    _ordered = [GridSize.GRID_4X4, GridSize.GRID_3X3, GridSize.GRID_2X2]
    _dim = {GridSize.GRID_4X4: 4, GridSize.GRID_3X3: 3, GridSize.GRID_2X2: 2}

    start = _ordered.index(grid) if grid in _ordered else 0
    for g in _ordered[start:]:
        d = _dim[g]
        if height >= d and width >= d:
            return g
    return GridSize.GRID_2X2


def _downsample(
    pixels: NDArray[np.uint8],
    new_height: int,
    new_width: int,
) -> NDArray[np.uint8]:
    """Downsample image using PIL (Lanczos) with a slicing fallback."""
    try:
        from PIL import Image
    except ImportError:
        # Fallback: simple slicing (fast but lower quality)
        h, w = pixels.shape[:2]
        step_h = max(1, h // new_height)
        step_w = max(1, w // new_width)
        return pixels[::step_h, ::step_w]
    
    img = Image.fromarray(pixels)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)
