from __future__ import annotations

from typing import Tuple, List, Callable, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from oplo.io.image_io import ImageBundle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import math


def _normalize_tile_args(tile_size, overlap):
    th, tw = tile_size
    oh, ow = overlap
    if th <= 0 or tw <= 0:
        raise ValueError("tile_size must be positive ints")
    if oh < 0 or ow < 0:
        raise ValueError("overlap must be non-negative")
    if oh >= th or ow >= tw:
        raise ValueError("overlap must be smaller than tile dimensions")
    return int(th), int(tw), int(oh), int(ow)


def tile_coords(shape: Tuple[int, int], tile_size: Tuple[int, int], overlap: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    """Return a list of (y0,y1,x0,x1) tile coordinates covering an HxW image.

    Guarantees full coverage. Tiles at edges may be smaller than tile_size.
    """
    H, W = shape
    th, tw, oh, ow = _normalize_tile_args(tile_size, overlap)
    stride_y = th - oh
    stride_x = tw - ow
    coords = []
    if H <= 0 or W <= 0:
        return coords

    y = 0
    while y < H:
        y0 = y
        y1 = min(y0 + th, H)
        x = 0
        while x < W:
            x0 = x
            x1 = min(x0 + tw, W)
            coords.append((y0, y1, x0, x1))
            x += stride_x
        y += stride_y

    # Ensure last row/col reach the image end (sometimes step math skips tail)
    if coords:
        last_by = coords[-1][1]
        if last_by < H:
            # add a final row of tiles anchored to bottom
            new_coords = []
            y0 = max(0, H - th)
            y1 = H
            x = 0
            while x < W:
                x0 = x
                x1 = min(x0 + tw, W)
                c = (y0, y1, x0, x1)
                if c not in coords:
                    new_coords.append(c)
                x += stride_x
            coords.extend(new_coords)
        # ensure last column reaches right edge
        # note: above may have already added columns
        # deduplicate while preserving order is fine
    # dedupe while preserving order
    seen = set()
    out = []
    for c in coords:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def split_into_tiles(arr: np.ndarray, tile_size: Tuple[int, int], overlap: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Split an array into tiles. Returns list of dicts: {coords:(y0,y1,x0,x1), tile:np.ndarray}

    Works with 2D arrays (H,W) or 3D arrays (H,W,C). Channels are preserved.
    """
    if arr.ndim < 2:
        raise ValueError("array must have at least 2 dimensions")
    H, W = arr.shape[:2]
    coords = tile_coords((H, W), tile_size, overlap)
    out = []
    for (y0, y1, x0, x1) in coords:
        tile = arr[y0:y1, x0:x1].copy()
        out.append({"coords": (y0, y1, x0, x1), "tile": tile})
    return out


def stitch_tiles(tiles: List[Dict[str, Any]], out_shape: Tuple[int, int, Optional[int]]) -> np.ndarray:
    """Stitch tiles back into an array of shape (H,W) or (H,W,C).

    Overlap regions are averaged (simple weighting).
    tiles: list of dicts with keys 'coords' and 'tile'
    out_shape: (H, W) or (H, W, C)
    """
    if len(out_shape) < 2:
        raise ValueError("out_shape must be at least 2D")
    H = out_shape[0]
    W = out_shape[1]
    C = None
    if len(out_shape) > 2:
        C = out_shape[2]
    # use float accumulator for safety
    sample_tile = tiles[0]["tile"] if tiles else None
    if sample_tile is None:
        # empty image
        if C is None:
            return np.zeros((H, W), dtype=np.float32)
        else:
            return np.zeros((H, W, C), dtype=np.float32)
    acc_shape = (H, W) + (() if sample_tile.ndim == 2 else (sample_tile.shape[2],))
    acc = np.zeros(acc_shape, dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)

    for t in tiles:
        y0, y1, x0, x1 = t["coords"]
        tile = t["tile"].astype(np.float64, copy=False)
        # expand weight for channels if needed
        acc[y0:y1, x0:x1] += tile
        weight[y0:y1, x0:x1] += 1.0

    # avoid division by zero
    w = weight.copy()
    w[w == 0] = 1.0
    if sample_tile.ndim == 2:
        out = (acc / w).astype(np.float32)
    else:
        # broadcast weight to channels
        out = (acc / w[..., None]).astype(np.float32)
    return out


def _worker_apply(args):
    # worker helper so we can submit a top-level callable to ProcessPool
    func, tile, coords, kwargs = args
    return {"coords": coords, "tile": func(tile, coords, **(kwargs or {}))}


def process_tiles(
    arr: np.ndarray,
    func: Callable[[np.ndarray, Tuple[int, int, int, int]], np.ndarray],
    tile_size: Tuple[int, int] = (512, 512),
    overlap: Tuple[int, int] = (0, 0),
    workers: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    executor_class=ProcessPoolExecutor,
    func_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Apply `func` to each tile in `arr` in parallel and stitch results.

    func signature: func(tile: np.ndarray, coords: (y0,y1,x0,x1), **kwargs) -> processed_tile

    Returns a stitched array of dtype float32 (for simplicity). Use small
    tiles and a reasonable number of workers for large images.

    progress_cb: optional callable(completed, total) invoked in main process as tiles finish.
    executor_class: injectable for tests (e.g., use ThreadPoolExecutor).
    """
    tiles = split_into_tiles(arr, tile_size, overlap)
    total = len(tiles)
    if total == 0:
        return arr.copy()

    # pack work items; pass func by reference (must be picklable for ProcessPool)
    work = [(func, t["tile"], t["coords"], func_kwargs) for t in tiles]

    results = []
    completed = 0
    # choose workers
    if workers is None:
        workers = max(1, min(32, (os_cpu_count() or 1)))

    # Use ProcessPoolExecutor as default if executor_class is None
    if executor_class is None:
        executor_class = ProcessPoolExecutor

    with executor_class(max_workers=workers) as exe:
        futures = [exe.submit(_worker_apply, w) for w in work]
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            completed += 1
            if progress_cb:
                try:
                    progress_cb(completed, total)
                except Exception:
                    pass

    # stitch
    stitched = stitch_tiles(results, arr.shape[:2] + (() if arr.ndim == 2 else (arr.shape[2],)))
    return stitched


def os_cpu_count():
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


def process_bundle(
    bundle: ImageBundle,
    func: Callable[[np.ndarray, Tuple[int, int, int, int]], np.ndarray],
    tile_size: Tuple[int, int] = (512, 512),
    overlap: Tuple[int, int] = (0, 0),
    workers: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    executor_class=ProcessPoolExecutor,
    func_kwargs: Optional[Dict[str, Any]] = None,
) -> ImageBundle:
    """Process an ImageBundle by applying `func` to each tile of each slice.

    Returns a new ImageBundle with processed data and copied metadata/calib.

    If the bundle is a stack, each slice is processed independently and the
    resulting stack is returned. `executor_class` can be overridden for tests
    (e.g., ThreadPoolExecutor) to avoid pickling requirements.
    """
    if bundle is None:
        raise ValueError("bundle must be an ImageBundle")

    data = bundle.data
    # single image
    if not bundle.is_stack():
        out = process_tiles(
            data,
            func,
            tile_size=tile_size,
            overlap=overlap,
            workers=workers,
            progress_cb=progress_cb,
            executor_class=executor_class,
            func_kwargs=func_kwargs,
        )
        return ImageBundle(data=out, meta=dict(bundle.meta), calib=dict(bundle.calib))

    # stack: process each slice
    n = bundle.num_slices()
    outs = []
    for i in range(n):
        slice_img = bundle.get_slice(i)
        out_slice = process_tiles(
            slice_img,
            func,
            tile_size=tile_size,
            overlap=overlap,
            workers=workers,
            progress_cb=progress_cb,
            executor_class=executor_class,
            func_kwargs=func_kwargs,
        )
        outs.append(out_slice)
    # stack back; preserve channel dims if present
    stacked = np.stack(outs, axis=0)
    return ImageBundle(data=stacked, meta=dict(bundle.meta), calib=dict(bundle.calib))
