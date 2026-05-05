"""Shared HDF5 kymograph loading and plot path helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


DATASET_DEFAULT = "kymograph"
"""Default HDF5 dataset name for kymograph arrays."""

MAX_TIME_SAMPLES = 1024
"""Read at most this many rows along time (axis 0); ``None`` = full time axis."""

IMAGE_CMAP = "hot"
"""Default ``matplotlib`` colormap for kymograph-style ``imshow`` panels."""

FIGSIZE_INCHES: tuple[float, float] = (10.0, 10.0)
"""Matplotlib figure (width, height) in inches — shared square canvas for all NSM plots."""


def load_kymograph(
    path: Path,
    *,
    dataset_name: str = DATASET_DEFAULT,
    max_time: int | None = MAX_TIME_SAMPLES,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load kymograph as float32 ``(T, X)``. Returns ``(array, on-disk dataset shape)``.

    Reads the **leading** ``min(file_T, max_time)`` rows along time (no striding).
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


def discover_h5_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files under {data_dir}")
    return files


def output_path_for_file(template: Path, h5_path: Path, n_files: int) -> Path:
    if n_files == 1:
        return template
    return template.with_name(f"{template.stem}_{h5_path.stem}{template.suffix}")
