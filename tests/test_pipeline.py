import numpy as np
from oplo.pipeline import tile_coords, split_into_tiles, stitch_tiles, process_tiles


def test_tile_coords_and_roundtrip():
    H, W = 123, 205
    coords = tile_coords((H, W), (50, 50), (10, 10))
    assert len(coords) > 0
    arr = np.arange(H * W, dtype=np.float32).reshape((H, W))
    tiles = split_into_tiles(arr, (50, 50), (10, 10))
    stitched = stitch_tiles(tiles, (H, W))
    # after split+stitch with averaging we should recover original values exactly
    # because we didn't modify tile contents
    assert np.allclose(stitched, arr.astype(np.float32))


def _double(tile, coords):
    return tile * 2


def test_process_tiles_double():
    H, W = 80, 90
    arr = np.random.RandomState(0).rand(H, W).astype(np.float32)
    out = process_tiles(arr, _double, tile_size=(30, 30), overlap=(5, 5), workers=2)
    assert np.allclose(out, arr * 2, atol=1e-6)
