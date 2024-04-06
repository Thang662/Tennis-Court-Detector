from typing import Tuple
import numpy as np

def gen_binary_map(wh: Tuple[int, int],
                   cxy: Tuple[float, float],
                   r: float,
                   data_type: np.dtype = np.float32,
):
    w, h   = wh
    cx, cy = cxy
    if cx < 0 or cy < 0:
        return np.zeros((h,w), dtype=data_type)
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    distmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    bmap    = np.zeros_like(distmap)
    bmap[distmap <= r**2] = 1
    return bmap.astype(data_type)

def faster_gen_binary_map(wh: Tuple[int, int],
                   cxy: Tuple[float, float],
                   r: float,
                   data_type: np.dtype = np.float32,
                   ) -> np.ndarray:
    w, h = wh
    cx, cy = cxy

    # Early exit for invalid coordinates
    if cx < 0 or cy < 0:
        return np.zeros((h, w), dtype=data_type)

    # Pre-calculate values for efficiency
    y_range = np.arange(1, h + 1)
    x_range = np.arange(1, w + 1)
    cy1 = cy + 1
    cx1 = cx + 1
    r2 = r**2

    # Vectorized distance calculation using broadcasting
    distmap = ((y_range[:, np.newaxis] - cy1)**2) + ((x_range - cx1)**2)

    # Boolean indexing for faster assignment
    bmap = np.zeros_like(distmap, dtype=bool)  # Use boolean array for thresholding
    bmap[distmap <= r2] = True

    # Convert to desired data type only at the end
    return bmap.astype(data_type)