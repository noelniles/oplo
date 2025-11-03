"""Sample image processing functions (tile-compatible).

Each processor is a callable: fn(tile: np.ndarray, coords: (y0,y1,x0,x1), **kwargs) -> processed_tile
"""
from __future__ import annotations

from typing import Tuple
import numpy as np


def gaussian_blur(tile: np.ndarray, coords: Tuple[int, int, int, int], sigma: float = 1.0) -> np.ndarray:
    """Apply a Gaussian blur to the tile. Tries scipy.ndimage first, falls back to
    a simple separable filter using numpy if SciPy/OpenCV aren't available.
    """
    try:
        from scipy.ndimage import gaussian_filter

        if tile.ndim == 2:
            return gaussian_filter(tile, sigma=sigma)
        # apply per-channel
        out = np.empty_like(tile)
        for c in range(tile.shape[2]):
            out[..., c] = gaussian_filter(tile[..., c], sigma=sigma)
        return out
    except Exception:
        pass

    try:
        import cv2

        k = int(max(1, round(sigma * 4)) // 2 * 2 + 1)
        if tile.ndim == 2:
            return cv2.GaussianBlur(tile, (k, k), sigmaX=sigma)
        out = cv2.GaussianBlur(tile, (k, k), sigmaX=sigma)
        return out
    except Exception:
        pass

    # Fallback: naive box filter implemented via convolution with uniform kernel
    # This is slower and lower-quality but does not require extra deps.
    kernel_size = max(1, int(round(sigma * 3)))
    k = 2 * kernel_size + 1
    # separable 1D kernel
    kern = np.ones(k, dtype=np.float32) / float(k)

    def sep_filter(img2d):
        # horizontal then vertical via convolution using numpy pad
        tmp = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), axis=1, arr=img2d)
        out2 = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), axis=0, arr=tmp)
        return out2

    if tile.ndim == 2:
        return sep_filter(tile)
    out = np.empty_like(tile)
    for c in range(tile.shape[2]):
        out[..., c] = sep_filter(tile[..., c])
    return out
