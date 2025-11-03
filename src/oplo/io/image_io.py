from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# ImageBundle: preserve raw data + metadata + calibration. Visualization-only
# scaling happens via .view(...), keeping scientific data intact for analysis.
# ---------------------------------------------------------------------------

@dataclass
class ImageMeta:
    path: str
    orig_dtype: str
    bit_depth: Optional[int]
    colorspace: str = "unknown"
    shape: Optional[Tuple[int, ...]] = None
    reader: Optional[str] = None


@dataclass
class ImageBundle:
    data: np.ndarray            # raw array (original dtype when feasible)
    meta: Dict                  # metadata dict (derived from ImageMeta)
    calib: Dict                 # calibration / scaling hints (per modality)

    def view(
        self,
        policy: str = "dtype_range",
        lo: float = 0.5,
        hi: float = 99.5,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Return a float32 image in [0,1] for visualization only.
        Supports multiple policies; default is dtype-range scaling.
        """
        return to_unit_view(
            self.data,
            self.meta,
            self.calib,
            policy=policy,
            lo=lo,
            hi=hi,
            window_center=window_center,
            window_width=window_width,
            gamma=gamma,
        )


# --------------------------- Public API ------------------------------------

def load_image(path: str | Path) -> ImageBundle:
    """Load an image and return an ImageBundle (raw + meta + calib).

    Current working readers: TIFF (tifffile) and PNG/JPEG/BMP (Pillow).
    Other formats are scaffolded for future implementation.
    """
    p = Path(path)

    for reader in _READERS:
        if reader.accepts(p):
            try:
                arr, meta = reader.load(p)
            except NotImplementedError as exc:
                raise ValueError(f"Support for {p.suffix} is not implemented yet (reader: {reader.name}).") from exc
            meta.reader = reader.name
            return ImageBundle(data=arr, meta=meta.__dict__, calib={})

    raise ValueError(f"Unsupported image format: {p.suffix or 'unknown'}")


# --------------------------- TIFF (primary) --------------------------------

class _TiffReader:
    name = "tifffile"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".tif", ".tiff"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        import tifffile as tiff
        # Try normal imread first (fast for typical images). If it fails or is
        # huge, fall back to memmap. Users with gigantic TIFFs can still open.
        arr = None
        try:
            arr = tiff.imread(str(p))
        except Exception:
            pass
        if arr is None:
            try:
                arr = tiff.memmap(str(p))  # lazy mapping
            except Exception as e:
                raise
        meta = ImageMeta(
            path=str(p),
            orig_dtype=str(arr.dtype),
            bit_depth=int(arr.dtype.itemsize * 8) if arr.dtype.kind in "ui" else 32,
            colorspace="linear",
            shape=tuple(arr.shape),
        )
        # If this is a stack, just expose first page for now (viewer roadmap will add stacks)
        if arr.ndim == 3 and arr.shape[-1] not in (1, 2, 3, 4):
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


# --------------------------- Scaffolds (future) ----------------------------

class _DicomReader:
    name = "dicom"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".dcm"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        raise NotImplementedError("DICOM reader not implemented yet")


class _RawReader:
    name = "rawpy"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".dng", ".nef", ".cr2", ".arw"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        raise NotImplementedError("RAW reader not implemented yet")


class _FitsReader:
    name = "fits"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".fits", ".fit"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        raise NotImplementedError("FITS reader not implemented yet")


class _ExrReader:
    name = "exr"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".exr"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        raise NotImplementedError("EXR reader not implemented yet")


class _NpyReader:
    name = "npy"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".npy", ".npz"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        raise NotImplementedError("NPY/NPZ reader not implemented yet")


class _Hdf5Reader:
    name = "hdf5"

    @staticmethod
    def accepts(p: Path) -> bool:
        return p.suffix.lower() in {".h5", ".hdf5"}

    @staticmethod
    def load(p: Path) -> tuple[np.ndarray, ImageMeta]:
        raise NotImplementedError("HDF5 reader not implemented yet")


# Reader registration order: working + specific first
_READERS: List[object] = [
    _TiffReader,
    _PillowReader,   # allow PNG/JPEG/BMP when TIFF not matched
    _DicomReader,
    _RawReader,
    _FitsReader,
    _ExrReader,
    _NpyReader,
    _Hdf5Reader,
]


# --------------------------- View scaling ----------------------------------

def to_unit_view(
    arr: np.ndarray,
    meta: Dict,
    calib: Dict,
    policy: str = "dtype_range",
    lo: float = 0.5,
    hi: float = 99.5,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Convert to float32 in [0,1] for visualization only.

    policies:
      - "dtype_range": scale uint types by dtype max; floats min-max clamp
      - "percentile": robust scaling between given percentiles
      - "window": (future) DICOM-style window/level using calib or args
    """
    x = arr

    if policy == "percentile":
        x32 = x.astype(np.float32, copy=False)
        if np.all(np.isnan(x32)):
            y = np.zeros_like(x32, dtype=np.float32)
        else:
            lo_v = float(np.nanpercentile(x32, lo))
            hi_v = float(np.nanpercentile(x32, hi))
            denom = max(hi_v - lo_v, 1e-12)
            y = (x32 - lo_v) / denom

    elif policy == "window":
        wc = window_center if window_center is not None else calib.get("window_center")
        ww = window_width  if window_width  is not None else calib.get("window_width")
        if wc is None or ww is None:
            raise ValueError("window policy requires window_center/width")
        lo_v = wc - ww/2.0
        hi_v = wc + ww/2.0
        x32 = x.astype(np.float32, copy=False)
        y = (x32 - lo_v) / max(hi_v - lo_v, 1e-12)

    else:  # dtype_range
        if x.dtype == np.uint8:
            y = x.astype(np.float32) / 255.0
        elif x.dtype == np.uint16:
            y = x.astype(np.float32) / 65535.0
        elif x.dtype.kind in "ui":
            y = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
        else:
            x32 = x.astype(np.float32, copy=False)
            if np.all(np.isnan(x32)):
                y = np.zeros_like(x32, dtype=np.float32)
            else:
                x_min = float(np.nanmin(x32))
                x_max = float(np.nanmax(x32))
                y = (x32 - x_min) / max(x_max - x_min, 1e-12)

    y = np.clip(y, 0, 1)
    if gamma != 1.0:
        y = np.power(y, 1.0/gamma, dtype=np.float32)
    return y.astype(np.float32, copy=False)
