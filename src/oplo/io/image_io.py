from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Callable, List

import numpy as np

# Optional imports are done inside reader functions to avoid hard deps

# --------------------------- Public API ------------------------------------

@dataclass
class ImageMeta:
    path: str
    orig_dtype: str
    bit_depth: Optional[int]
    colorspace: str = "unknown"
    shape: Optional[Tuple[int, ...]] = None
    reader: Optional[str] = None


def load_image(path: str | Path) -> tuple[np.ndarray, ImageMeta]:
    """Load an image as float32 [0,1] for processing/Plotly display.

    Tries specialized readers in order. Keeps full precision internally and
    returns a view-scaled array only at the UI layer.
    """
    p = Path(path)

    for reader in _READERS:
        if reader.accepts(p):
            arr, meta = reader.load(p)
            meta.reader = reader.name
            return to_float01(arr), meta

    # Fallback to Pillow
    arr, meta = _PillowReader.load(p)
    meta.reader = _PillowReader.name
    return to_float01(arr), meta


# --------------------------- TIFF (primary) --------------------------------

class _TiffReader:
    name = "tifffile"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".tif", ".tiff"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        import tifffile as tiff
        # If extremely large, consider tiff.memmap or aszarr for pyramids
        try:
            arr = tiff.imread(str(p))  # preserves dtype and pages
        except Exception as e:
            # Try memmap as a fallback for big images
            arr = tiff.memmap(str(p))  # lazy mapping
        meta = ImageMeta(
            path=str(p),
            orig_dtype=str(arr.dtype),
            bit_depth=int(arr.dtype.itemsize * 8) if arr.dtype.kind in "ui" else 32,
            colorspace="linear",
            shape=tuple(arr.shape),
        )
        # If multi-page or multi-sample, prefer first page for now
        if arr.ndim == 3 and arr.shape[-1] not in (1, 2, 3, 4):
            # Likely a stack: (pages, H, W) or (H, W, pages)
            # Use first page; future: expose a timeline/slider
            arr = np.asarray(arr[0])
        return np.asarray(arr), meta


# --------------------------- PNG/JPEG (friendly) ---------------------------

class _PillowReader:
    name = "pillow"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        from PIL import Image
        im = Image.open(p)
        # Convert paletted/LA images to RGB or L consistently
        if im.mode in {"P", "LA"}:
            im = im.convert("RGBA" if "A" in im.mode else "RGB")
        arr = np.array(im)
        meta = ImageMeta(
            path=str(p),
            orig_dtype=str(arr.dtype),
            bit_depth=int(arr.dtype.itemsize * 8) if arr.dtype.kind in "ui" else 32,
            colorspace=getattr(im, "mode", "unknown"),
            shape=tuple(arr.shape),
        )
        return arr, meta


# --------------------------- Stubs (scaffold) ------------------------------

class _DicomReader:
    name = "dicom"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".dcm"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        # Future: pydicom.dcmread(...).pixel_array
        raise NotImplementedError("DICOM reader not implemented yet")


class _RawReader:
    name = "rawpy"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".dng", ".nef", ".cr2", ".arw"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        # Future: rawpy.imread(...).postprocess(..., output_bps=16, gamma=(1,1))
        raise NotImplementedError("RAW reader not implemented yet")


class _FitsReader:
    name = "fits"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".fits", ".fit"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        # Future: astropy.io.fits.open(..., memmap=True)
        raise NotImplementedError("FITS reader not implemented yet")


class _ExrReader:
    name = "exr"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".exr"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        # Future: imagecodecs via tifffile.imread or OpenEXR (if available)
        raise NotImplementedError("EXR reader not implemented yet")


class _NpyReader:
    name = "npy"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".npy", ".npz"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        # Future: np.load(..., mmap_mode='r')
        raise NotImplementedError("NPY/NPZ reader not implemented yet")


class _Hdf5Reader:
    name = "hdf5"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".h5", ".hdf5"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        # Future: h5py.File(...)[dataset]; expose dataset picker in UI
        raise NotImplementedError("HDF5 reader not implemented yet")


class _EventReader:
    name = "events"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".aedat", ".dat"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        # Future: parse event stream and produce an accumulation/time-surface
        raise NotImplementedError("Event reader not implemented yet")


# Reader registration order: most specific/robust first
_READERS: List[object] = [
    _TiffReader,
    _DicomReader,
    _RawReader,
    _FitsReader,
    _ExrReader,
    _NpyReader,
    _Hdf5Reader,
]


# --------------------------- Utilities -------------------------------------

def to_float01(arr: np.ndarray) -> np.ndarray:
    """Convert to float32 in [0,1] without destroying dynamic range semantics.

    Integers are scaled by their dtype max. Floats are clamped to [0,1].
    """
    a = np.asarray(arr)
    if a.dtype.kind == "f":
        out = np.clip(a, 0.0, 1.0)
    elif a.dtype == np.uint8:
        out = a.astype(np.float32) / 255.0
    elif a.dtype == np.uint16:
        out = a.astype(np.float32) / 65535.0
    elif a.dtype.kind in "ui":
        maxv = float(np.iinfo(a.dtype).max)
        out = a.astype(np.float32) / maxv
    else:
        # Fallback: scale by finite max
        finite_max = float(np.nanmax(a)) if np.isfinite(a).any() else 1.0
        out = a.astype(np.float32) / (finite_max if finite_max else 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
