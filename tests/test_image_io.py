from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from oplo.io.image_io import to_unit_view, load_image


def test_to_unit_view_handles_all_nan_percentile():
    arr = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32)
    meta = {"orig_dtype": str(arr.dtype)}
    calib = {}

    result = to_unit_view(arr, meta, calib, policy="percentile")
    assert np.all(result == 0.0)
    assert result.dtype == np.float32


def test_to_unit_view_handles_all_nan_dtype_range():
    arr = np.array([[np.nan, np.nan]], dtype=np.float32)
    meta = {"orig_dtype": str(arr.dtype)}
    calib = {}

    result = to_unit_view(arr, meta, calib, policy="dtype_range")
    assert np.all(result == 0.0)


def test_load_image_raises_for_unimplemented_reader(tmp_path):
    fake = tmp_path / "image.dcm"
    fake.write_bytes(b"")

    with pytest.raises(ValueError) as excinfo:
        load_image(fake)

    assert "not implemented" in str(excinfo.value)


def test_load_image_rejects_unknown_format(tmp_path):
    fake = tmp_path / "image.xyz"
    fake.write_bytes(b"123")

    with pytest.raises(ValueError) as excinfo:
        load_image(fake)

    assert "Unsupported" in str(excinfo.value)
