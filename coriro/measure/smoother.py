# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Pixel stabilizer for gradient smoothing and anti-aliasing reduction.

Optional pre-conditioning that smooths gradients and reduces anti-aliasing
artifacts via controlled downscale/upscale through early CNN layers.

The CNN stem determines the feature resolution; the original image is
downscaled to that resolution and bilinearly upscaled back, achieving
smoothing without complex feature-space reconstruction.

Requires: torch (pip install torch) and timm (pip install timm)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from numpy.typing import NDArray

# Lazy imports for optional dependencies
_smoother_cache: dict = {}


def _check_dependencies() -> bool:
    """Check if torch and timm are available."""
    try:
        import torch
        import timm
        return True
    except ImportError:
        return False


def create_smoother(
    model_name: str = "convnext_tiny",
    device: Optional[str] = None,
) -> "torch.nn.Module":
    """
    Create a frozen pixel stabilizer from early CNN layers.
    
    Uses ConvNeXt-Tiny stem + stage1 by default. This provides:
    - Strong gradient/color separation
    - Good anti-aliasing handling
    - Pure convolution (no attention)
    - Local receptive fields
    
    Args:
        model_name: timm model name (default: convnext_tiny)
        device: torch device (default: auto-detect)
        
    Returns:
        Frozen nn.Module that outputs smoothed feature maps
        
    Raises:
        ImportError: If torch or timm not installed
    """
    import torch
    import torch.nn as nn
    import timm
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load full model with pretrained weights
    full_model = timm.create_model(model_name, pretrained=True)
    
    # Extract stem + first stage only
    # ConvNeXt structure: stem -> stages[0] -> stages[1] -> ...
    # We only want stem + stages[0] for shallow, local features
    
    if hasattr(full_model, 'stem') and hasattr(full_model, 'stages'):
        # ConvNeXt-style architecture
        smoother = nn.Sequential(
            full_model.stem,
            full_model.stages[0],
        )
    elif hasattr(full_model, 'conv_stem') and hasattr(full_model, 'blocks'):
        # EfficientNet-style architecture
        smoother = nn.Sequential(
            full_model.conv_stem,
            full_model.bn1,
            full_model.blocks[0],
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    
    # Freeze all parameters
    for param in smoother.parameters():
        param.requires_grad = False
    
    smoother.eval()
    smoother.to(device)
    
    return smoother


def get_smoother(
    model_name: str = "convnext_tiny",
    device: Optional[str] = None,
) -> "torch.nn.Module":
    """
    Get or create a cached smoother instance.
    
    Smoother instances are cached by model name to avoid repeated loading.
    """
    cache_key = f"{model_name}:{device}"
    
    if cache_key not in _smoother_cache:
        _smoother_cache[cache_key] = create_smoother(model_name, device)
    
    return _smoother_cache[cache_key]


def smooth_image(
    pixels: NDArray[np.uint8],
    smoother: Optional["torch.nn.Module"] = None,
    model_name: str = "convnext_tiny",
) -> NDArray[np.uint8]:
    """
    Apply pixel stabilization to an image.
    
    The CNN processes the image through shallow convolutional layers,
    then reconstructs a smoothed version at the original resolution.
    This reduces anti-aliasing artifacts and consolidates gradients
    without adding semantic interpretation.
    
    Args:
        pixels: Input image (H, W, 3) uint8 sRGB
        smoother: Pre-created smoother module (optional)
        model_name: Model to use if smoother not provided
        
    Returns:
        Smoothed image (H, W, 3) uint8 sRGB, same shape as input
        
    Raises:
        ImportError: If torch or timm not installed
    """
    import torch
    import torch.nn.functional as F
    
    if smoother is None:
        smoother = get_smoother(model_name)
    
    device = next(smoother.parameters()).device
    original_shape = pixels.shape[:2]  # (H, W)
    
    # Convert to tensor: (H, W, 3) -> (1, 3, H, W)
    tensor = torch.from_numpy(pixels).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    tensor = tensor / 255.0  # Normalize to [0, 1]
    tensor = tensor.to(device)
    
    # Apply ImageNet normalization (required for pretrained models)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    tensor = (tensor - mean) / std
    
    # Forward pass to determine the feature-map resolution
    with torch.no_grad():
        features = smoother(tensor)  # (1, C, H', W')

    # Use the CNN's spatial downsampling factor: downscale the original
    # image to feature resolution, then bilinearly upscale back.
    feature_h, feature_w = features.shape[2:]
    
    # Downscale original to feature resolution
    downscaled = F.interpolate(
        tensor * std + mean,  # Undo normalization
        size=(feature_h, feature_w),
        mode='bilinear',
        align_corners=False,
    )
    
    # Upscale back to original resolution with bilinear smoothing
    smoothed = F.interpolate(
        downscaled,
        size=original_shape,
        mode='bilinear',
        align_corners=False,
    )
    
    # Convert back to numpy uint8
    smoothed = smoothed.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
    smoothed = smoothed.clamp(0, 1) * 255
    smoothed = smoothed.cpu().numpy().astype(np.uint8)
    
    return smoothed


def is_available() -> bool:
    """Check if CNN smoothing is available (torch + timm installed)."""
    return _check_dependencies()


def clear_cache() -> None:
    """
    Release cached smoother instances and free GPU memory.
    
    Call this when you're done with smoothing to release resources.
    """
    _smoother_cache.clear()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

