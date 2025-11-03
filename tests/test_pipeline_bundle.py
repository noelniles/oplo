import numpy as np
from concurrent.futures import ThreadPoolExecutor
from oplo.io.image_io import ImageBundle
from oplo.pipeline import process_bundle


def _double(tile, coords):
    return tile * 2


def test_process_bundle_single_and_stack():
    # single image
    arr = np.random.RandomState(1).rand(40, 50).astype(np.float32)
    b = ImageBundle(data=arr, meta={"path": "<mem>"}, calib={})
    out = process_bundle(b, _double, tile_size=(20, 20), overlap=(5, 5), executor_class=ThreadPoolExecutor, workers=2)
    assert np.allclose(out.data, arr * 2)

    # stack
    stack = np.stack([arr, arr * 3], axis=0)
    b2 = ImageBundle(data=stack, meta={"path": "<mem>"}, calib={})
    out2 = process_bundle(b2, _double, tile_size=(20, 20), overlap=(5, 5), executor_class=ThreadPoolExecutor, workers=2)
    assert np.allclose(out2.data[0], arr * 2)
    assert np.allclose(out2.data[1], arr * 6)
