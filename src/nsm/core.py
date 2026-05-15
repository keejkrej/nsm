"""Shared CLI constants and shared helper functions.

This module centralizes reusable IO/plotting primitives so command modules can stay
small and explicit.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


HELP = "Workflow: prepare cropped input, extract trajectories, then fit drift/diffusion"
PROG_NAME = "nsm"


def normalize_argv(argv: Sequence[str] | None) -> list[str] | None:
    return list(argv) if argv is not None else None


# ---------------------------------------------------------------------------
# Kymograph filesystem and I/O primitives

DEFAULT_KYMOGRAPH_DATASET = "kymograph"
"""Default HDF5 dataset key for `(time, position)` intensity arrays."""

RESIDUAL_SQ_GAUSSIAN_DATASET = "residual_sq_gaussian"
"""Gaussian-smoothed squared wavelet-detail dataset key."""

DEFAULT_LEADING_TIME_ROWS = 1024
"""Default leading time rows retained by `crop`."""

IMAGE_CMAP = "hot"
"""Default colormap for kymograph visualizations."""

FIGSIZE_INCHES: tuple[float, float] = (10.0, 10.0)
"""Shared figure size for trajectory and intensity outputs."""

DEFAULT_DATA_ROOT = Path.home() / "data" / "nsm"
"""Default base folder for data and output products."""

DEFAULT_PLOTS_DIR = DEFAULT_DATA_ROOT / "plots"
"""Default output directory for PNGs and derived HDF5 artifacts."""


def load_kymograph(
    path: Path,
    *,
    dataset_name: str = DEFAULT_KYMOGRAPH_DATASET,
    max_time: int | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load a kymograph dataset as float32 ``(T, X)`` and return array + disk shape."""
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
    """Return all ``.h5`` files in ``data_dir`` sorted by name."""
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files under {data_dir}")
    return files


def disambiguated_output_path(template: Path, h5_path: Path, n_files: int) -> Path:
    if n_files == 1:
        return template
    return template.with_name(f"{template.stem}_{h5_path.stem}{template.suffix}")


def resolve_png_destination(out: Path, *, source_stem: str, filename: str) -> Path:
    """Resolve PNG output for a source stem."""
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
    """Sum-bin a kymograph in time and position."""
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
    return out.astype(np.float64, copy=False), float(raw_frame_rate_hz) / float(binning_t)


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
    """Load and sum-bin an HDF5 kymograph."""
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


# ---------------------------------------------------------------------------
# Plot helpers shared across workflows


def _safe_minmax(arr: np.ndarray) -> tuple[float, float]:
    z = np.asarray(arr, dtype=np.float64)
    if z.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(z, (1.0, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(z))
        hi = float(np.nanmax(z))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi


def write_kymograph_png(
    arr: np.ndarray,
    out_png: Path,
    *,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    data = np.asarray(arr, dtype=np.float64)
    if data.size == 0:
        raise ValueError("cannot plot empty kymograph")
    if vmin is None or vmax is None:
        vmin, vmax = _safe_minmax(data)
    disp = data.T.astype(np.float32, copy=False)

    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES)
    im = ax.imshow(
        disp,
        aspect="auto",
        origin="upper",
        cmap=IMAGE_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("time (axis 0)")
    ax.set_ylabel("position (pixels)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_trajectory_overlay_png(
    kymo: np.ndarray,
    stacked: np.ndarray,
    out_png: Path,
    *,
    title: str,
) -> None:
    arr = np.asarray(kymo, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot plot empty trajectory map")
    disp = arr.T.astype(np.float32, copy=False)
    vmin, vmax = _safe_minmax(arr)

    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES)
    im = ax.imshow(
        disp,
        aspect="auto",
        origin="upper",
        cmap=IMAGE_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    rows = np.asarray(stacked, dtype=np.float64)
    if rows.size:
        ids = rows[:, 0]
        x = rows[:, 1]
        t = rows[:, 2]
        vmax_id = float(np.max(ids)) if ids.size else 1.0
        ax.scatter(
            t,
            x,
            c=ids,
            cmap="gist_ncar",
            vmin=-0.5,
            vmax=max(vmax_id + 0.5, 1.0),
            s=8.0,
            alpha=0.92,
            edgecolors="black",
            linewidths=0.2,
            zorder=5,
        )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("time (axis 0)")
    ax.set_ylabel("position (pixels)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_raw_preview(
    arr: np.ndarray,
    out_png: Path,
    *,
    title: str,
) -> None:
    data = np.asarray(arr, dtype=np.float64)
    if data.size == 0:
        raise ValueError("cannot plot empty raw kymograph")
    lo, hi = np.percentile(data, (2.0, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = _safe_minmax(data)
    write_kymograph_png(
        np.clip(data, lo, hi),
        out_png,
        title=title,
        vmin=lo,
        vmax=hi,
    )
