from __future__ import annotations

import numpy as np
from PIL import Image
from pathlib import Path

from oplo.io.image_io import to_unit_view, load_image, ImageBundle


def test_to_unit_view_dtype_range_uint8_uint16_float():
    # uint8
    arr8 = np.array([[0, 128, 255]], dtype=np.uint8)
    v8 = to_unit_view(arr8, {}, {}, policy="dtype_range")
    assert v8.dtype == np.float32
    assert np.isclose(v8[0, 0], 0.0, atol=1e-6)
    assert np.isclose(v8[0, 1], 128 / 255.0, atol=1e-6)
    assert np.isclose(v8[0, 2], 1.0, atol=1e-6)

    # uint16
    arr16 = np.array([[0, 32768, 65535]], dtype=np.uint16)
    v16 = to_unit_view(arr16, {}, {}, policy="dtype_range")
    assert np.isclose(v16[0, 1], 32768 / 65535.0, atol=1e-6)
    assert np.isclose(v16[0, 2], 1.0, atol=1e-6)

    # float (min-max scaling)
    arrf = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    vf = to_unit_view(arrf, {}, {}, policy="dtype_range")
    assert np.allclose(vf, arrf, atol=1e-6)


def test_to_unit_view_percentile_scaling():
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    v = to_unit_view(arr, {}, {}, policy="percentile", lo=1.0, hi=99.0)
    # ensure values are clipped into [0,1] and scaled
    assert v.min() >= 0.0
    assert v.max() <= 1.0
    assert v.max() > 0.99
    assert v.min() < 0.01


def test_load_image_uses_pillow(tmp_path: Path):
    # Create a small RGB PNG and ensure load_image returns a bundle
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    p = tmp_path / "test.png"
    img.save(p)

    bundle = load_image(str(p))
    assert isinstance(bundle, ImageBundle)
    # Pillow reader should be used for PNG files
    assert bundle.meta.get("reader") == "pillow"
    assert bundle.data.shape == (10, 10, 3)
