"""Shared lpdiff math: median subtraction, x-axis median smooth, residual, and min/max plot scaling."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from nsm.statistics import temporal_median_background


def subtract_background(arr: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Broadcast subtract per-column background: ``arr - background``."""
    if background.shape != (arr.shape[1],):
        raise ValueError(
            f"background length {background.shape} != width {arr.shape[1]}"
        )
    return arr - background


def _median_footprint_x(size_x: int) -> int:
    k = max(1, int(size_x))
    if k % 2 == 0:
        k += 1
    return k


def median_smooth_along_x(arr: np.ndarray, *, size_x: int) -> np.ndarray:
    """Median filter along **x** only (shape ``(1, k)`` footprint), per time row.

    ``size_x`` is adjusted to an odd integer ≥ 1. Boundaries use ``mode='reflect'``.
    """
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D (time, x) array, got shape {x.shape}")
    k = _median_footprint_x(size_x)
    out = median_filter(x, size=(1, k), mode="reflect")
    return out.astype(np.float32)


def equidistant_time_indices(n_time: int, *, k: int = 5) -> np.ndarray:
    """``min(k, n_time)`` distinct row indices from ``0 … n_time−1``, spread across time (axis 0)."""
    if n_time <= 0:
        raise ValueError(f"need positive time extent, got {n_time}")
    k_eff = min(k, n_time)
    if k_eff == 1:
        return np.zeros(1, dtype=np.int64)
    denom = k_eff - 1
    return (np.arange(k_eff, dtype=np.int64) * (n_time - 1) // denom).astype(np.int64)


def y_axis_minmax(a: np.ndarray) -> tuple[float, float]:
    """Strict min/max for a matplotlib **y**-axis; expands by ``1e-6`` if flat."""
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def kymograph_clip_for_imshow(arr_td: np.ndarray) -> tuple[np.ndarray, float, float]:
    """``(T, X)`` kymograph → transposed display array and ``vmin``/``vmax`` (min/max clip)."""
    vmin, vmax = y_axis_minmax(arr_td)
    disp = np.clip(arr_td.T, vmin, vmax)
    return disp, vmin, vmax


def spatial_median_caption(requested_size_x: int) -> str:
    k = _median_footprint_x(requested_size_x)
    return f"median filter along x, footprint {k}"


def median_subtracted(arr_tx: np.ndarray) -> np.ndarray:
    """``(T, X)`` raw kymograph → ``raw − temporal_median`` per column (float32)."""
    median = temporal_median_background(arr_tx)
    return subtract_background(arr_tx, median)


def lpdiff_residual(
    arr_tx: np.ndarray,
    *,
    median_kernel_x: int = 15,
) -> tuple[np.ndarray, str]:
    """Median subtract → spatial median along x → ``preproc − smooth`` and caption."""
    corrected = median_subtracted(arr_tx)
    lp = median_smooth_along_x(corrected, size_x=median_kernel_x)
    res = corrected.astype(np.float32, copy=False) - lp
    return res, spatial_median_caption(median_kernel_x)
