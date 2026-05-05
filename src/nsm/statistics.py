"""Per-position summaries over time: median plus 1–99% and 10–90% bands."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nsm.data import (
    DATASET_DEFAULT,
    FIGSIZE_INCHES,
    discover_h5_files,
    load_kymograph,
    output_path_for_file,
)


def temporal_percentile_band_across_time(
    arr: np.ndarray, pct_low: float, pct_high: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per **x**, ``pct_low`` / ``pct_high`` percentiles over time. ``arr`` is (T, X)."""
    if arr.ndim != 2:
        raise ValueError(f"expected 2D (T, X), got shape {arr.shape}")
    if not (0.0 <= pct_low < pct_high <= 100.0):
        raise ValueError(f"need 0 <= pct_low < pct_high <= 100, got {pct_low}, {pct_high}")
    pl, ph = np.percentile(
        arr.astype(np.float32, copy=False), [pct_low, pct_high], axis=0
    )
    return pl.astype(np.float32, copy=False), ph.astype(np.float32, copy=False)


def temporal_median_background(arr: np.ndarray) -> np.ndarray:
    """For each position x, median intensity over time. ``arr`` is (T, X)."""
    if arr.ndim != 2:
        raise ValueError(f"expected 2D (T, X), got shape {arr.shape}")
    return np.median(arr.astype(np.float32, copy=False), axis=0).astype(np.float32)


def _plot_statistics_panel(
    path: Path,
    *,
    out_path: Path | None,
    show: bool,
    dataset_name: str,
) -> None:
    arr, full_shape = load_kymograph(path, dataset_name=dataset_name)
    med = temporal_median_background(arr)
    p1, p99 = temporal_percentile_band_across_time(arr, 1.0, 99.0)
    p10, p90 = temporal_percentile_band_across_time(arr, 10.0, 90.0)

    x = np.arange(med.shape[0], dtype=np.float32)
    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES)
    ax.fill_between(
        x,
        p1,
        p99,
        alpha=0.45,
        color="steelblue",
        linewidth=0,
        label="1–99% (across time)",
        zorder=1,
    )
    ax.fill_between(
        x,
        p10,
        p90,
        alpha=0.55,
        color="seagreen",
        linewidth=0,
        label="10–90% (across time)",
        zorder=2,
    )
    ax.plot(x, med, color="navy", lw=1.15, label="Median", zorder=3)

    ax.set_title(
        f"{path.name} — median, 10–90%, and 1–99% vs position\n"
        f"loaded array {tuple(arr.shape)} • on-disk {full_shape}"
    )
    ax.set_xlabel("position x (pixel index)")
    ax.set_ylabel("intensity")
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path.resolve()}")
    if not show:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Per x: median over t and percentile bands vs position "
            "(1–99%, 10–90% across time). Uses leading time slice from HDF5 "
            "(default up to MAX_TIME_SAMPLES; see nsm.data)."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=os.path.expanduser("~/data/nsm"),
        type=Path,
        help="Directory containing .h5 files (default: ~/data/nsm)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(os.path.expanduser("~/data/nsm/plots/nsm_statistics.png")),
        help=(
            "PNG path (default: ~/data/nsm/plots/nsm_statistics.png). "
            "Several .h5: stem_<file>.png in that directory."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_DEFAULT,
        help=f"HDF5 dataset name (default: {DATASET_DEFAULT})",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save PNG (use with --show)",
    )

    args = parser.parse_args()
    data_dir = args.directory.expanduser().resolve()
    paths = discover_h5_files(data_dir)
    n_files = len(paths)
    template = None if args.no_save else args.output.expanduser()

    for path in paths:
        dest = (
            None
            if template is None
            else output_path_for_file(template, path, n_files)
        )
        _plot_statistics_panel(
            path,
            out_path=dest,
            show=args.show or args.no_save,
            dataset_name=args.dataset,
        )

    if args.show or args.no_save:
        plt.show()
        plt.close("all")


if __name__ == "__main__":
    main()
