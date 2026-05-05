"""Shared HDF5 kymograph loading and plot path helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


DATASET_DEFAULT = "kymograph"
"""Default HDF5 dataset name for kymograph arrays."""

MAX_TIME_SAMPLES = 1024
"""Default leading time rows for **cropped** exports (``nsm-crop``) and movie frame caps."""

IMAGE_CMAP = "hot"
"""Default ``matplotlib`` colormap for kymograph-style ``imshow`` panels."""

FIGSIZE_INCHES: tuple[float, float] = (10.0, 10.0)
"""Matplotlib figure (width, height) in inches — shared square canvas for all NSM plots."""

DEFAULT_DATA_ROOT = Path.home() / "data" / "nsm"
"""Default folder that contains ``*.h5`` (CLI ``directory`` default)."""

DEFAULT_PLOTS_DIR = DEFAULT_DATA_ROOT / "plots"
"""Default output directory for PNG/MP4 artifacts."""


def load_kymograph(
    path: Path,
    *,
    dataset_name: str = DATASET_DEFAULT,
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


def discover_h5_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files under {data_dir}")
    return files


def output_path_for_file(template: Path, h5_path: Path, n_files: int) -> Path:
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


def resolve_video_destination(out: Path, *, source_stem: str, suffix: str) -> Path:
    """``.mp4`` / ``.gif`` → that path; otherwise treat ``out`` as a directory."""
    out = out.expanduser().resolve()
    if out.suffix.lower() in (".mp4", ".gif"):
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    return (out / f"{source_stem}{suffix}").resolve()
