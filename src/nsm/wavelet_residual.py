"""Kymograph wavelet pipeline: temporal median correction, LP along *x*, detail residual, smoothed squared detail."""

from __future__ import annotations

import numpy as np
import pywt
from skimage.filters import gaussian

RESIDUAL_SQ_GAUSSIAN_SIGMA = 10.0
"""Gaussian σ in pixels along **x** on squared wavelet detail (applied after squaring)."""


def temporal_median_background(arr: np.ndarray) -> np.ndarray:
    """For each position x, median intensity over time. ``arr`` is (T, X)."""
    if arr.ndim != 2:
        raise ValueError(f"expected 2D (T, X), got shape {arr.shape}")
    return np.median(arr.astype(np.float32, copy=False), axis=0).astype(np.float32)


def subtract_background(arr: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Broadcast subtract per-column background: ``arr - background``."""
    if background.shape != (arr.shape[1],):
        raise ValueError(
            f"background length {background.shape} != width {arr.shape[1]}"
        )
    return arr - background


def wavelet_lowpass_along_x(
    arr: np.ndarray,
    *,
    wavelet: str = "db4",
    level: int = 4,
) -> tuple[np.ndarray, int]:
    """1-D approximation-only low-pass along axis **x** for each time row (detail coeffs zeroed).

    Assumes ``arr`` is shaped ``(time, x)``. No smoothing along time.

    Returns the filtered array and the decomposition depth used (0 if the input
    was returned unchanged).
    """
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D (time, x) array, got shape {x.shape}")
    n_time, width = x.shape
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(int(width), w)
    if max_level < 1:
        return arr.astype(np.float32, copy=False), 0
    lvl = max(1, min(int(level), max_level))
    out = np.empty_like(x)
    for i in range(n_time):
        coeffs = pywt.wavedec(x[i], wavelet, level=lvl, mode="symmetric")
        coeffs_lp = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        rec = pywt.waverec(coeffs_lp, wavelet, mode="symmetric")
        out[i] = rec[:width]
    return out.astype(np.float32), lvl


def equidistant_time_indices(n_time: int, *, k: int = 5) -> np.ndarray:
    """``min(k, n_time)`` distinct row indices from ``0 … n_time−1``, spread across time (axis 0)."""
    if n_time <= 0:
        raise ValueError(f"need positive time extent, got {n_time}")
    k_eff = min(k, n_time)
    if k_eff == 1:
        return np.zeros(1, dtype=np.int64)
    denom = k_eff - 1
    return (np.arange(k_eff, dtype=np.int64) * (n_time - 1) // denom).astype(np.int64)


def gaussian_smooth_along_x(arr_tx: np.ndarray, *, sigma: float) -> np.ndarray:
    """Smooth each time row along **x** only. ``σ≤0`` returns ``arr_tx`` as float32."""
    z = np.asarray(arr_tx, dtype=np.float32)
    if sigma <= 0:
        return z
    out = gaussian(
        np.asarray(z, dtype=np.float64),
        sigma=(0.0, float(sigma)),
        mode="reflect",
        preserve_range=True,
        truncate=4.0,
    )
    return out.astype(np.float32, copy=False)


def squared_residual_gaussian(
    wavelet_detail_tx: np.ndarray,
    *,
    gaussian_sigma: float = RESIDUAL_SQ_GAUSSIAN_SIGMA,
) -> np.ndarray:
    """Squared wavelet-detail kymograph, then Gaussian along **x** (HDF5 panels, movies, peaks)."""
    sq = np.square(np.asarray(wavelet_detail_tx, dtype=np.float32))
    return gaussian_smooth_along_x(sq, sigma=float(gaussian_sigma))


def y_axis_minmax(a: np.ndarray) -> tuple[float, float]:
    """Strict min/max for a matplotlib **y**-axis; expands by ``1e-6`` if flat."""
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def wavelet_lp_caption(wavelet: str, lvl: int) -> str:
    if lvl:
        return f"{wavelet}, level {lvl}, axis=x"
    return f"{wavelet}, axis=x (no decomposition; LP equals preproc)"


def median_subtracted(arr_tx: np.ndarray) -> np.ndarray:
    """``(T, X)`` raw kymograph → ``raw − temporal_median`` per column (float32)."""
    median = temporal_median_background(arr_tx)
    return subtract_background(arr_tx, median)


def wavelet_detail_residual(
    arr_tx: np.ndarray,
    *,
    wavelet: str = "db4",
    wavelet_level: int = 4,
) -> tuple[np.ndarray, str]:
    """Temporal median correction → wavelet low-pass along *x* → detail = corrected − LP, plus caption."""
    corrected = median_subtracted(arr_tx)
    lp, lvl = wavelet_lowpass_along_x(
        corrected, wavelet=wavelet, level=wavelet_level
    )
    detail = corrected.astype(np.float32, copy=False) - lp
    return detail, wavelet_lp_caption(wavelet, lvl)
