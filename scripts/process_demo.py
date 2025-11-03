#!/usr/bin/env python3
"""Demo: process an image file using the pipeline and a sample processor.

Usage: python scripts/process_demo.py input.tif output.tif

This script demonstrates loading an image via `oplo.io.image_io.load_image`,
processing it with `oplo.processors.gaussian_blur` via `oplo.pipeline.process_bundle`,
and writing the result (best-effort) using tifffile if available.
"""
import sys
from pathlib import Path

from oplo.io.image_io import load_image
from oplo.pipeline import process_bundle
from oplo.processors import gaussian_blur


def main(argv):
    if len(argv) < 3:
        print("usage: process_demo.py input output")
        return 2
    inp = Path(argv[1])
    outp = Path(argv[2])
    bundle = load_image(inp)
    print(f"Loaded {inp}: shape={bundle.data.shape}")
    processed = process_bundle(bundle, gaussian_blur, tile_size=(512, 512), overlap=(16, 16), workers=4)
    # try to write TIFF if tifffile available
    try:
        import tifffile as tiff

        tiff.imwrite(str(outp), processed.data.astype(processed.data.dtype))
        print(f"Wrote {outp}")
    except Exception:
        # fallback: numpy save
        import numpy as np

        np.save(str(outp) + ".npy", processed.data)
        print(f"Wrote {outp}.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
