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
