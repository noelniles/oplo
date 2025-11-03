from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import logging

logger = logging.getLogger("oplo.io.image_io")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[oplo.io.image_io] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

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

    def is_stack(self) -> bool:
        """Return True if this bundle represents a stack (multiple slices).

        Heuristic:
        - If data.ndim == 3 and the last axis is not a small channel axis (1..4),
          treat axis 0 as Z (Z,H,W).
        - If data.ndim == 4 and the last axis is channels (3 or 4), treat axis 0
          as Z (Z,H,W,C).
        """
        d = self.data
        if d is None:
            return False
        if d.ndim == 3:
            # could be H,W,C or Z,H,W
            if d.shape[2] in (1, 2, 3, 4):
                return False
            return True
        if d.ndim == 4:
            # assume Z,H,W,C
            return True
        return False

    def num_slices(self) -> int:
        if not self.is_stack():
            return 1
        return int(self.data.shape[0])

    def get_slice(self, idx: int) -> np.ndarray:
        """Return the raw slice at index `idx` as a numpy array.
        The returned slice will be 2D (H,W) or 3D (H,W,C).
        """
        if not self.is_stack():
            return self.data
        idx = int(idx)
        d = self.data
        if d.ndim == 3:
            return d[idx]
        if d.ndim == 4:
            return d[idx]
        raise IndexError("Unsupported data shape for slicing")

    def view_slice(self, idx: int, **kwargs) -> np.ndarray:
        """Return a visualization-ready float32 [0,1] array for a given slice.
        Calls through to to_unit_view with the bundle meta/calib.
        """
        sl = self.get_slice(idx)
        return to_unit_view(sl, self.meta, self.calib, **kwargs)

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

    # Prefer readers registered via the app registry when available. This
    # allows plugins to register new readers without editing this module.
    try:
        from oplo.registry import registry

        # First try a direct finder for this path (fast and deterministic).
        try:
            reader = registry.find_reader_for_path(p)
            if reader is not None:
                try:
                    arr, meta = reader.load(p)
                    meta.reader = getattr(reader, "name", getattr(reader, "__name__", "unknown"))
                    return ImageBundle(data=arr, meta=meta.__dict__, calib={})
                except Exception:
                    # If the chosen reader failed, fall through to trying others
                    # below so we can still attempt to read the file.
                    pass
        except Exception:
            # ignore finder failures and continue to scanning readers
            pass

        # Ensure plugins package is imported (best-effort) so third-party
        # modules have a chance to register themselves with the registry.
        try:
            import importlib

            importlib.import_module("oplo.plugins")
        except Exception:
            pass

        for reader in registry.get_readers():
            try:
                name = getattr(reader, "name", getattr(reader, "__name__", str(reader)))
                if reader.accepts(p):
                    logger.info(f"reader {name} accepts path {p}; attempting load")
                    arr, meta = reader.load(p)
                    meta.reader = name
                    return ImageBundle(data=arr, meta=meta.__dict__, calib={})
            except ImportError as e:
                # Helpful message when optional deps missing
                logger.warning(f"reader {name} could not be used (missing dependency): {e}")
                continue
            except Exception as e:
                logger.exception(f"reader {name} failed while loading {p}: {e}")
                continue
    except Exception:
        # If registry import/lookup fails entirely, fall back to local list below.
        pass

    # Fallback behavior: try the built-in reader ordering (legacy).
    for reader in _READERS:
        if reader.accepts(p):
            arr, meta = reader.load(p)
            meta.reader = getattr(reader, "name", getattr(reader, "__name__", "unknown"))
            return ImageBundle(data=arr, meta=meta.__dict__, calib={})

    # As last resort, defer to Pillow loader which is generally available.
    arr, meta = _PillowReader.load(p)
    meta.reader = _PillowReader.name
    return ImageBundle(data=arr, meta=meta.__dict__, calib={})


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
        # Preserve stacks: do not collapse 3D stacks here. Viewer logic will
        # inspect ImageBundle.is_stack() and render slices as needed.
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
        # Try to use pydicom if it is available in the environment. This
        # provides a basic fallback even if an external plugin didn't register
        # properly. For full-featured DICOM support, install pydicom or use the
        # dedicated plugin.
        try:
            import pydicom
        except Exception:
            raise NotImplementedError("DICOM reader not implemented yet; install pydicom or enable a plugin")

        ds = pydicom.dcmread(str(p))
        if not hasattr(ds, "pixel_array"):
            raise ValueError("DICOM file does not contain pixel data")
        arr = np.asarray(ds.pixel_array)
        meta = ImageMeta(
            path=str(p),
            orig_dtype=str(arr.dtype),
            bit_depth=int(arr.dtype.itemsize * 8) if arr.dtype.kind in "ui" else 32,
            colorspace="mono" if arr.ndim == 2 else "rgb",
            shape=tuple(arr.shape),
        )
        return arr, meta


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


# Register built-in readers with the global registry so third-party plugins
# can extend or override behaviour. If registry isn't available at import
# time we preserve a local _READERS fallback for robustness.
# Always expose a builtin readers list for fallback usage.
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
try:
    from oplo.registry import registry

    registry.register_reader(_TiffReader, priority=50)
    registry.register_reader(_PillowReader, priority=60)
    registry.register_reader(_DicomReader, priority=200)
    registry.register_reader(_RawReader, priority=210)
    registry.register_reader(_FitsReader, priority=220)
    registry.register_reader(_ExrReader, priority=230)
    registry.register_reader(_NpyReader, priority=240)
    registry.register_reader(_Hdf5Reader, priority=250)
except Exception:
    # registry unavailable; builtin _READERS already defined above
    pass


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
        lo_v = float(np.percentile(x32, lo))
        hi_v = float(np.percentile(x32, hi))
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
            x_min = float(np.nanmin(x32))
            x_max = float(np.nanmax(x32))
            y = (x32 - x_min) / max(x_max - x_min, 1e-12)

    y = np.clip(y, 0, 1)
    if gamma != 1.0:
        y = np.power(y, 1.0/gamma, dtype=np.float32)
    return y.astype(np.float32, copy=False)
