"""HDF5 kymograph I/O helpers and filesystem paths for PNG outputs."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


DEFAULT_KYMOGRAPH_DATASET = "kymograph"
"""Default HDF5 dataset key for `(time, position)` intensity arrays."""

RESIDUAL_SQ_GAUSSIAN_DATASET = "residual_sq_gaussian"
"""Gaussian smoothing along **x** of squared wavelet-detail kymograph."""

DEFAULT_LEADING_TIME_ROWS = 1024
"""Default leading time rows kept by ``nsm-crop`` when trimming long kymographs."""

IMAGE_CMAP = "hot"
"""Default ``matplotlib`` colormap for kymograph-style ``imshow`` panels."""

FIGSIZE_INCHES: tuple[float, float] = (10.0, 10.0)
"""Matplotlib figure (width, height) in inches — shared square canvas for all NSM plots."""

DEFAULT_DATA_ROOT = Path.home() / "data" / "nsm"
"""Default folder that contains ``*.h5`` (CLI ``directory`` default)."""

DEFAULT_PLOTS_DIR = DEFAULT_DATA_ROOT / "plots"
"""Default output directory for PNG and derived HDF5 artifacts."""


def load_kymograph(
    path: Path,
    *,
    dataset_name: str = DEFAULT_KYMOGRAPH_DATASET,
    max_time: int | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load kymograph as float32 ``(T, X)``. Returns ``(array, on-disk dataset shape)``.

    With ``max_time`` set, reads the **leading** ``min(file_T, max_time)`` rows (no striding).
    With ``max_time=None`` (default), loads the full time axis.
    """
    with h5py.File(path, "r") as f:
        if dataset_name not in f:
            avail = ", ".join(sorted(f.keys()))
            raise KeyError(
                f"{path.name}: missing '{dataset_name}'. Available: {avail or '(empty)'}"
            )
        ds = f[dataset_name]
        raw_shape = (int(ds.shape[0]), int(ds.shape[1]))
        t_end = raw_shape[0]
        if max_time is not None:
            t_end = min(t_end, max_time)
        arr = ds[:t_end, :].astype(np.float32, copy=False)
    return arr, raw_shape


def find_h5_files(data_dir: Path) -> list[Path]:
    """Return `.h5` paths under ``data_dir`` sorted by name; raise if none."""
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files under {data_dir}")
    return files


def disambiguated_output_path(template: Path, h5_path: Path, n_files: int) -> Path:
    if n_files == 1:
        return template
    return template.with_name(f"{template.stem}_{h5_path.stem}{template.suffix}")


def resolve_png_destination(out: Path, *, source_stem: str, filename: str) -> Path:
    """Directory → ``out / filename``; path ending in ``.png`` → that file (parent created)."""
    out = out.expanduser().resolve()
    if out.suffix.lower() == ".png":
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    return (out / filename).resolve()


def resolve_output_directory(out: Path) -> Path:
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def bin_kymograph_spatiotemporal_sum(
    kymo: np.ndarray,
    *,
    binning_x: int,
    binned_time_ms: float,
    raw_frame_rate_hz: float,
) -> tuple[np.ndarray, float]:
    """Sum-bin ``kymo`` along time and position to match collaborator notebook math.

    ``binning_t = max(1, round(binned_time_ms * raw_frame_rate_hz / 1000))``;
    the array is cropped to shapes divisible by ``(binning_t, binning_x)`` before
    reshaping and summing. Returns ``(binned_array, binned_frame_rate_hz)``.
    """
    if binning_x < 1:
        raise ValueError(f"binning_x must be >= 1, got {binning_x}")
    if not np.isfinite(raw_frame_rate_hz) or raw_frame_rate_hz <= 0:
        raise ValueError(f"raw_frame_rate_hz must be finite and > 0, got {raw_frame_rate_hz}")
    binning_t = max(1, int(round(binned_time_ms * raw_frame_rate_hz / 1000.0)))
    t, x = int(kymo.shape[0]), int(kymo.shape[1])
    t_binned = (t // binning_t) * binning_t
    x_binned = (x // binning_x) * binning_x
    if t_binned < 1 or x_binned < 1:
        raise ValueError(
            f"kymograph shape {(t, x)} too small for binning_t={binning_t}, binning_x={binning_x}"
        )
    block = kymo[:t_binned, :x_binned]
    out = block.reshape(
        t_binned // binning_t,
        binning_t,
        x_binned // binning_x,
        binning_x,
    ).sum(axis=(1, 3))
    binned_fps = float(raw_frame_rate_hz) / float(binning_t)
    return out.astype(np.float64, copy=False), binned_fps


def load_kymograph_binned(
    path: Path,
    *,
    dataset_name: str = DEFAULT_KYMOGRAPH_DATASET,
    binning_x: int = 2,
    binned_time_ms: float = 2.857,
    default_fps: float = 2800.0,
    trim_trailing_rows: int = 0,
    max_time: int | None = None,
) -> tuple[np.ndarray, float]:
    """Load ``dataset_name`` from HDF5 and return sum-binned ``(T', X')`` plus binned fps.

    Use the same **file** (for example an ``nsm-crop`` output) for both the main
    ``nsm`` preprocessing pipeline and :mod:`nsm.collaborator` by passing identical
    ``binning_x``, ``binned_time_ms``, and ``max_time`` (or rely on the cropped file
    length). ``trim_trailing_rows`` drops that many rows from the **end** of the raw
    dataset before binning (the legacy collaborator default was 5000).
    """
    path = path.expanduser().resolve()
    with h5py.File(path, "r") as f:
        if dataset_name not in f:
            avail = ", ".join(sorted(f.keys()))
            raise KeyError(
                f"{path.name}: missing {dataset_name!r}. Available: {avail or '(empty)'}"
            )
        ds = f[dataset_name]
        raw_shape = (int(ds.shape[0]), int(ds.shape[1]))
        raw_fps = float(f.attrs.get("fps", default_fps))
        t_end = raw_shape[0]
        if trim_trailing_rows > 0:
            t_end = max(0, t_end - trim_trailing_rows)
        if max_time is not None:
            t_end = min(t_end, max_time)
        if t_end < 1:
            raise ValueError(
                f"no time rows to read after trim/max_time (t_end={t_end}, raw={raw_shape})"
            )
        arr = ds[:t_end, :].astype(np.float32, copy=False)
    return bin_kymograph_spatiotemporal_sum(
        arr,
        binning_x=binning_x,
        binned_time_ms=binned_time_ms,
        raw_frame_rate_hz=raw_fps,
    )
