# Tiled Viewer (design)

Goal: interactive, high-performance viewer for very large images (multi-gigapixel)
that supports smooth pan/zoom and viewport-level processing. The viewer should
work with both on-disk multi-resolution sources (pyramid/OME-Zarr/OME-TIFF) and
with on-demand server-side tiling for single large images.

Key ideas
- Tile API: server endpoint `/tile?path=...&level=...&x=...&y=...&size=256` returns
  a PNG/JPEG tile for immediate display.
- Pyramid levels: level 0 = highest resolution. Each level halves longest side.
- Lazy generation: if multi-resolution file exists (OME-Zarr, DeepZoom, or tiled
  OME-TIFF), read tiles directly; otherwise generate tiles on-demand from base
  image and optionally cache to disk.

Server-side contract
- get_tile(path: str, level: int, x: int, y: int, tile_size: int=256) -> bytes
  - Reads minimal data required (memmap or partial read), applies transforms
    (gamma/contrast/colormap) if requested, and returns encoded PNG bytes.
  - Should accept URL parameters for transient preview transforms.

ImageBundle additions
- add method: get_tile(level:int, x:int, y:int, tile_size:int=256, transforms:dict=None)
  - Returns a tile as a numpy array (uint8 or float32 depending on downstream)
- store optional pyramid metadata in `bundle.meta['pyramid']` when available.

Caching
- Use a small on-disk cache (e.g., under `~/.cache/oplo/tiles/<hash>/...`) keyed by
  (path, level, x, y, transforms) to avoid recomputing expensive tiles.
- Provide a cache size limit and LRU eviction.

Client-side
- Integrate a lightweight JS tile layer (e.g., Leaflet or a minimal canvas-based
  layer) into the Dash app; the tile layer pulls from server `/tile` endpoint.
- Keep viewport transforms to server to allow quick response (server applies
  display gamma/colormap and returns ready-to-show PNG.)

Performance notes
- Prefer reading with ``tifffile`` memmap for huge tiffs or zarr for chunked access.
- For CPU-heavy transforms, consider a worker pool or async handlers.
- For large deployments, optional CDN or precomputed tiles (offline generation)
  for common datasets.

Security
- Reuse existing `safe_join` for file path resolution; ensure tile endpoints
  validate and authorize access to requested paths.

Next steps for implementation
1. Add server tile endpoint in `oplo.server` (or a small blueprint) returning PNG bytes.
2. Extend `ImageBundle` with `get_tile(...)` and a thin adapter that uses memmap or
   zarr reads.
3. Add a minimal on-disk LRU cache and tests for tile correctness.
4. Integrate a simple JS tile layer in the Dash viewer to display tiles.

This doc is a starting point; I can expand any section into code sketches and
implementation PRs next.
