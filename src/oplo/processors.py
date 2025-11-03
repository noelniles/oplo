"""Image processing operations organized by category.

All processors follow the signature:
    fn(tile: np.ndarray, coords: Tuple[int, int, int, int], **kwargs) -> np.ndarray

Where coords = (y0, y1, x0, x1) tile boundaries in the full image.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
from oplo.processor_registry import processor_registry, ProcessorParam


# ========================= FILTERS =========================

@processor_registry.register(
    name="gaussian_blur",
    category="Filters",
    description="Smooth image using Gaussian blur",
    params=[ProcessorParam("sigma", "float", 1.0, "Blur radius (sigma)", min=0.1, max=20.0)],
    tags=["blur", "smooth", "denoise", "filter"]
)
def gaussian_blur(tile: np.ndarray, coords: Tuple[int, int, int, int], sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur. Tries scipy, cv2, then numpy fallback."""
    try:
        from scipy.ndimage import gaussian_filter
        if tile.ndim == 2:
            return gaussian_filter(tile, sigma=sigma)
        out = np.empty_like(tile)
        for c in range(tile.shape[2]):
            out[..., c] = gaussian_filter(tile[..., c], sigma=sigma)
        return out
    except ImportError:
        pass
    
    try:
        import cv2
        k = int(max(1, round(sigma * 4)) // 2 * 2 + 1)
        return cv2.GaussianBlur(tile, (k, k), sigmaX=sigma)
    except ImportError:
        pass
    
    # Numpy fallback: simple box filter
    kernel_size = max(1, int(round(sigma * 3)))
    k = 2 * kernel_size + 1
    kern = np.ones(k, dtype=np.float32) / float(k)
    
    def sep_filter(img2d):
        tmp = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), axis=1, arr=img2d)
        return np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), axis=0, arr=tmp)
    
    if tile.ndim == 2:
        return sep_filter(tile)
    out = np.empty_like(tile)
    for c in range(tile.shape[2]):
        out[..., c] = sep_filter(tile[..., c])
    return out


@processor_registry.register(
    name="median_filter",
    category="Filters",
    description="Remove salt-and-pepper noise using median filter",
    params=[ProcessorParam("size", "int", 3, "Kernel size (odd number)", min=3, max=21)],
    tags=["median", "denoise", "filter", "noise"]
)
def median_filter(tile: np.ndarray, coords: Tuple[int, int, int, int], size: int = 3) -> np.ndarray:
    """Apply median filter for noise removal."""
    size = int(size)
    if size % 2 == 0:
        size += 1
    try:
        from scipy.ndimage import median_filter as scipy_median
        return scipy_median(tile, size=size)
    except ImportError:
        pass
    
    try:
        import cv2
        if tile.ndim == 2:
            return cv2.medianBlur(tile.astype(np.float32), size).astype(tile.dtype)
        out = np.empty_like(tile)
        for c in range(tile.shape[2]):
            out[..., c] = cv2.medianBlur(tile[..., c].astype(np.float32), size).astype(tile.dtype)
        return out
    except ImportError:
        # Numpy fallback: simple median using percentile
        from scipy.ndimage import generic_filter
        if tile.ndim == 2:
            return generic_filter(tile, np.median, size=size)
        out = np.empty_like(tile)
        for c in range(tile.shape[2]):
            out[..., c] = generic_filter(tile[..., c], np.median, size=size)
        return out


@processor_registry.register(
    name="sharpen",
    category="Filters",
    description="Enhance edges and details",
    params=[ProcessorParam("strength", "float", 1.0, "Sharpening strength", min=0.0, max=5.0)],
    tags=["sharpen", "enhance", "edge", "filter"]
)
def sharpen(tile: np.ndarray, coords: Tuple[int, int, int, int], strength: float = 1.0) -> np.ndarray:
    """Sharpen using unsharp mask."""
    blurred = gaussian_blur(tile, coords, sigma=1.0)
    sharpened = tile + strength * (tile - blurred)
    return np.clip(sharpened, tile.min(), tile.max()).astype(tile.dtype)


# ========================= EDGE DETECTION =========================

@processor_registry.register(
    name="sobel",
    category="Edge Detection",
    description="Detect edges using Sobel operator",
    params=[],
    tags=["edge", "sobel", "gradient", "detect"]
)
def sobel(tile: np.ndarray, coords: Tuple[int, int, int, int]) -> np.ndarray:
    """Apply Sobel edge detection."""
    try:
        from scipy.ndimage import sobel as scipy_sobel
        if tile.ndim == 2:
            sx = scipy_sobel(tile, axis=0)
            sy = scipy_sobel(tile, axis=1)
            return np.hypot(sx, sy).astype(np.float32)
        # For RGB, convert to grayscale first
        gray = 0.2989 * tile[..., 0] + 0.5870 * tile[..., 1] + 0.1140 * tile[..., 2]
        sx = scipy_sobel(gray, axis=0)
        sy = scipy_sobel(gray, axis=1)
        return np.hypot(sx, sy).astype(np.float32)
    except ImportError:
        # Simple numpy gradient fallback
        if tile.ndim == 3:
            gray = 0.2989 * tile[..., 0] + 0.5870 * tile[..., 1] + 0.1140 * tile[..., 2]
        else:
            gray = tile
        gx = np.gradient(gray, axis=1)
        gy = np.gradient(gray, axis=0)
        return np.hypot(gx, gy).astype(np.float32)


@processor_registry.register(
    name="laplacian",
    category="Edge Detection",
    description="Detect edges using Laplacian operator",
    params=[],
    tags=["edge", "laplacian", "detect"]
)
def laplacian(tile: np.ndarray, coords: Tuple[int, int, int, int]) -> np.ndarray:
    """Apply Laplacian edge detection."""
    try:
        from scipy.ndimage import laplace
        if tile.ndim == 2:
            return laplace(tile).astype(np.float32)
        gray = 0.2989 * tile[..., 0] + 0.5870 * tile[..., 1] + 0.1140 * tile[..., 2]
        return laplace(gray).astype(np.float32)
    except ImportError:
        # Numpy gradient fallback
        if tile.ndim == 3:
            gray = 0.2989 * tile[..., 0] + 0.5870 * tile[..., 1] + 0.1140 * tile[..., 2]
        else:
            gray = tile
        return (np.gradient(np.gradient(gray, axis=0), axis=0) + 
                np.gradient(np.gradient(gray, axis=1), axis=1)).astype(np.float32)


# ========================= MORPHOLOGY =========================

@processor_registry.register(
    name="erode",
    category="Morphology",
    description="Erode image (shrink bright regions)",
    params=[ProcessorParam("size", "int", 3, "Kernel size", min=3, max=21)],
    tags=["morphology", "erode", "shrink"]
)
def erode(tile: np.ndarray, coords: Tuple[int, int, int, int], size: int = 3) -> np.ndarray:
    """Morphological erosion."""
    try:
        from scipy.ndimage import grey_erosion
        return grey_erosion(tile, size=size)
    except ImportError:
        try:
            import cv2
            kernel = np.ones((size, size), np.uint8)
            if tile.ndim == 2:
                return cv2.erode(tile, kernel)
            out = np.empty_like(tile)
            for c in range(tile.shape[2]):
                out[..., c] = cv2.erode(tile[..., c], kernel)
            return out
        except ImportError:
            return tile  # No op if no libs available


@processor_registry.register(
    name="dilate",
    category="Morphology",
    description="Dilate image (expand bright regions)",
    params=[ProcessorParam("size", "int", 3, "Kernel size", min=3, max=21)],
    tags=["morphology", "dilate", "expand"]
)
def dilate(tile: np.ndarray, coords: Tuple[int, int, int, int], size: int = 3) -> np.ndarray:
    """Morphological dilation."""
    try:
        from scipy.ndimage import grey_dilation
        return grey_dilation(tile, size=size)
    except ImportError:
        try:
            import cv2
            kernel = np.ones((size, size), np.uint8)
            if tile.ndim == 2:
                return cv2.dilate(tile, kernel)
            out = np.empty_like(tile)
            for c in range(tile.shape[2]):
                out[..., c] = cv2.dilate(tile[..., c], kernel)
            return out
        except ImportError:
            return tile


# ========================= ADJUSTMENTS =========================

@processor_registry.register(
    name="brightness",
    category="Adjustments",
    description="Adjust image brightness",
    params=[ProcessorParam("factor", "float", 1.0, "Brightness factor (1.0 = no change)", min=0.0, max=3.0)],
    tags=["brightness", "adjust", "intensity"]
)
def brightness(tile: np.ndarray, coords: Tuple[int, int, int, int], factor: float = 1.0) -> np.ndarray:
    """Adjust brightness by multiplying pixel values."""
    return np.clip(tile * factor, tile.min(), tile.max()).astype(tile.dtype)


@processor_registry.register(
    name="contrast",
    category="Adjustments",
    description="Adjust image contrast",
    params=[ProcessorParam("factor", "float", 1.0, "Contrast factor (1.0 = no change)", min=0.0, max=3.0)],
    tags=["contrast", "adjust"]
)
def contrast(tile: np.ndarray, coords: Tuple[int, int, int, int], factor: float = 1.0) -> np.ndarray:
    """Adjust contrast around the mean."""
    mean = tile.mean()
    adjusted = mean + factor * (tile - mean)
    return np.clip(adjusted, tile.min(), tile.max()).astype(tile.dtype)


@processor_registry.register(
    name="gamma_correct",
    category="Adjustments",
    description="Apply gamma correction",
    params=[ProcessorParam("gamma", "float", 1.0, "Gamma value (>1 brightens, <1 darkens)", min=0.1, max=5.0)],
    tags=["gamma", "adjust", "tone"]
)
def gamma_correct(tile: np.ndarray, coords: Tuple[int, int, int, int], gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction."""
    # Normalize to [0, 1], apply gamma, denormalize
    tile_min, tile_max = tile.min(), tile.max()
    if tile_max == tile_min:
        return tile
    normalized = (tile - tile_min) / (tile_max - tile_min)
    corrected = np.power(normalized, gamma)
    return (corrected * (tile_max - tile_min) + tile_min).astype(tile.dtype)


@processor_registry.register(
    name="invert",
    category="Adjustments",
    description="Invert image colors",
    params=[],
    tags=["invert", "negative", "adjust"]
)
def invert(tile: np.ndarray, coords: Tuple[int, int, int, int]) -> np.ndarray:
    """Invert pixel values."""
    tile_min, tile_max = tile.min(), tile.max()
    return (tile_max + tile_min - tile).astype(tile.dtype)


# ========================= TRANSFORMS =========================

@processor_registry.register(
    name="flip_horizontal",
    category="Transforms",
    description="Flip image horizontally",
    params=[],
    tags=["flip", "mirror", "transform"]
)
def flip_horizontal(tile: np.ndarray, coords: Tuple[int, int, int, int]) -> np.ndarray:
    """Flip tile horizontally."""
    return np.fliplr(tile)


@processor_registry.register(
    name="flip_vertical",
    category="Transforms",
    description="Flip image vertically",
    params=[],
    tags=["flip", "mirror", "transform"]
)
def flip_vertical(tile: np.ndarray, coords: Tuple[int, int, int, int]) -> np.ndarray:
    """Flip tile vertically."""
    return np.flipud(tile)


# ========================= NOISE =========================

@processor_registry.register(
    name="add_gaussian_noise",
    category="Noise",
    description="Add Gaussian noise to image",
    params=[
        ProcessorParam("mean", "float", 0.0, "Noise mean", min=-1.0, max=1.0),
        ProcessorParam("std", "float", 0.1, "Noise standard deviation", min=0.0, max=1.0)
    ],
    tags=["noise", "gaussian", "add"]
)
def add_gaussian_noise(tile: np.ndarray, coords: Tuple[int, int, int, int], mean: float = 0.0, std: float = 0.1) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.normal(mean, std, tile.shape).astype(tile.dtype)
    return np.clip(tile + noise, tile.min(), tile.max()).astype(tile.dtype)


@processor_registry.register(
    name="threshold",
    category="Segmentation",
    description="Binary threshold",
    params=[ProcessorParam("threshold", "float", 0.5, "Threshold value", min=0.0, max=1.0)],
    tags=["threshold", "binary", "segment"]
)
def threshold(tile: np.ndarray, coords: Tuple[int, int, int, int], threshold: float = 0.5) -> np.ndarray:
    """Apply binary threshold."""
    tile_min, tile_max = tile.min(), tile.max()
    thresh_val = tile_min + threshold * (tile_max - tile_min)
    return np.where(tile >= thresh_val, tile_max, tile_min).astype(tile.dtype)
