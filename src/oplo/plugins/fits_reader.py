from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np

from oplo.io.image_io import ImageMeta
from oplo.registry import registry


class _FitsReader:
    name = "fits"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".fits", ".fit"}

    @staticmethod
    def load(p: Path) -> Tuple[np.ndarray, ImageMeta]:
        try:
            from astropy.io import fits
        except Exception as e:
            raise ImportError("astropy is required to read FITS files; install with `pip install astropy`") from e

        with fits.open(str(p), memmap=True) as hdul:
            # find the first image HDU with data
            data = None
            for h in hdul:
                if h.data is not None:
                    data = h.data
                    break
            if data is None:
                raise ValueError("No image data found in FITS file")
            arr = np.asarray(data)

        # bit depth best-effort
        try:
            bit_depth = int(arr.dtype.itemsize * 8) if arr.dtype.kind in "ui" else 32
        except Exception:
            bit_depth = None

        meta = ImageMeta(
            path=str(p),
            orig_dtype=str(arr.dtype),
            bit_depth=bit_depth,
            colorspace="mono" if arr.ndim == 2 else "multi",
            shape=tuple(arr.shape),
        )
        return arr, meta


# Register
try:
    registry.register_reader(_FitsReader, priority=220)
except Exception:
    pass
