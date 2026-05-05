"""Wavelet residual on kymographs: median background, LP along *x*, illumination-rescaled I², plotting scales."""

from __future__ import annotations

import numpy as np
import pywt
from skimage.filters import gaussian

I2_GAUSSIAN_SIGMA = 10.0
"""Gaussian σ in pixels along **x** on the rescaled **I²** kymograph."""

ILLUMINATION_GAUSSIAN_SIGMA = 10.0
"""Gaussian σ along **x** applied to the one-dimensional temporal-median profile."""


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


def gaussian_smooth_profile_x(profile_x: np.ndarray, *, sigma: float) -> np.ndarray:
    """Gaussian along **x** for a 1-D column profile ``(width,)``."""
    row = np.asarray(profile_x, dtype=np.float32)[np.newaxis, :]
    out = gaussian_smooth_along_x(row, sigma=sigma)
    return out[0].astype(np.float32, copy=False)


def rescale_lpdiff_sq_by_illum2(
    lpdiff_tx: np.ndarray,
    illumination_x: np.ndarray,
) -> np.ndarray:
    """``(lpdiff)² / illumin²`` with a floor on the denominator (illumination is 1-D)."""
    sq = np.square(np.asarray(lpdiff_tx, dtype=np.float32))
    ill = np.maximum(illumination_x.astype(np.float64), 1e-12)
    ill2 = np.square(ill)
    ref = float(np.max(ill2))
    eps = max(ref * 1e-6, 1e-12) if ref > 0 else 1e-12
    denom = np.maximum(ill2.astype(np.float32), float(eps))
    return sq / denom


def i2_from_lpdiff(
    lpdiff_tx: np.ndarray,
    illumination_smoothed_x: np.ndarray,
    *,
    gaussian_sigma: float = I2_GAUSSIAN_SIGMA,
) -> np.ndarray:
    """Kymograph ``(lpdiff)² / illumin²`` then Gaussian along **x** (panels / movie)."""
    if illumination_smoothed_x.shape != (lpdiff_tx.shape[1],):
        raise ValueError(
            f"illumination length {illumination_smoothed_x.shape} != width {lpdiff_tx.shape[1]}"
        )
    z = rescale_lpdiff_sq_by_illum2(lpdiff_tx, illumination_smoothed_x)
    return gaussian_smooth_along_x(z, sigma=float(gaussian_sigma))


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


def lpdiff_residual(
    arr_tx: np.ndarray,
    *,
    wavelet: str = "db4",
    wavelet_level: int = 4,
) -> tuple[np.ndarray, str]:
    """Median subtract → wavelet LP along x → ``preproc − LP`` and caption."""
    corrected = median_subtracted(arr_tx)
    lp, lvl = wavelet_lowpass_along_x(
        corrected, wavelet=wavelet, level=wavelet_level
    )
    res = corrected.astype(np.float32, copy=False) - lp
    return res, wavelet_lp_caption(wavelet, lvl)
