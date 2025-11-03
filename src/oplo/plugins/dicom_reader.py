from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np

from oplo.io.image_io import ImageMeta
from oplo.registry import registry


class _PydicomReader:
    name = "pydicom"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".dcm"}

    @staticmethod
    def load(p: Path) -> Tuple[np.ndarray, ImageMeta]:
        try:
            import pydicom
        except Exception as e:
            raise ImportError("pydicom is required to read DICOM files; install with `pip install pydicom`") from e

        ds = pydicom.dcmread(str(p))

        # Pixel data access - pydicom will attempt to decode pixel data using
        # optional plugins (gdcm or pylibjpeg). If those are not installed
        # you'll get a RuntimeError here; catch it and raise an informative
        # ImportError so the user knows how to enable DICOM decompression.
        try:
            if not hasattr(ds, "pixel_array"):
                raise ValueError("DICOM file does not contain pixel data")
            arr = ds.pixel_array
        except RuntimeError as e:
            # Provide actionable installation hints
            hint = (
                "DICOM pixel data decompression requires additional libraries. "
                "Install pylibjpeg and a codec backend (recommended):\n"
                "  pip install pylibjpeg pylibjpeg-libjpeg\n"
                "or via conda:\n"
                "  conda install -c conda-forge pylibjpeg pylibjpeg-libjpeg\n"
                "Alternatively install GDCM (via conda):\n"
                "  conda install -c conda-forge gdcm\n"
            )
            raise ImportError(f"Unable to decode DICOM pixel data: {e}\n{hint}") from e
        arr = np.asarray(arr)

        # Derive bit depth
        try:
            bit_depth = int(arr.dtype.itemsize * 8) if arr.dtype.kind in "ui" else 32
        except Exception:
            bit_depth = None

        meta = ImageMeta(
            path=str(p),
            orig_dtype=str(arr.dtype),
            bit_depth=bit_depth,
            colorspace="mono" if arr.ndim == 2 else "rgb",
            shape=tuple(arr.shape),
        )
        return arr, meta


# Register with the global registry so this reader becomes available
try:
    registry.register_reader(_PydicomReader, priority=150)
except Exception:
    pass
