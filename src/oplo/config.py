from __future__ import annotations
import os
from pathlib import Path

ALLOWED_EXTS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp",
    ".fits", ".fit", ".exr", ".npy", ".npz", ".dcm",
}

def get_base_dir() -> Path:
    env = os.getenv("OPLO_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
    else:
        p = (Path.home() / "oplo-data").resolve()
        p.mkdir(parents=True, exist_ok=True)
    return p

def safe_join(base: Path, *parts: str) -> Path:
    out = base.joinpath(*parts).resolve()
    if not str(out).startswith(str(base)):
        raise ValueError("Path escapes base directory")
    return out
