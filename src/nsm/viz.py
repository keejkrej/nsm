"""Kymograph viewer: full loaded window (no subsampling)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nsm.data import (
    DATASET_DEFAULT,
    FIGSIZE_INCHES,
    IMAGE_CMAP,
    MAX_TIME_SAMPLES,
    discover_h5_files,
    load_kymograph,
    output_path_for_file,
)


def _load_kymograph_stretched(
    path: Path,
    *,
    dataset_name: str,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """2–98% contrast stretch; returns ``(array, disk shape, loaded shape)``."""
    arr, disk_shape = load_kymograph(path, dataset_name=dataset_name)
    vmin, vmax = np.percentile(arr, (2, 98))
    stretched = np.clip(arr.astype(np.float32, copy=False), vmin, vmax)
    return stretched, disk_shape, tuple(arr.shape)


def _plot(paths: list[Path], out_path: Path | None, show: bool, dataset_name: str) -> None:
    n_files = len(paths)
    for path in paths:
        arr, disk_shape, loaded_shape = _load_kymograph_stretched(
            path, dataset_name=dataset_name
        )
        display = arr.T
        fig, ax = plt.subplots(figsize=FIGSIZE_INCHES)
        im = ax.imshow(
            display,
            aspect="equal",
            origin="upper",
            cmap=IMAGE_CMAP,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(
            f"{path.name}\nkymograph {loaded_shape} (pos×time; image transposed)\n"
            f"on-disk {disk_shape}"
        )
        ax.set_xlabel("time (axis 0)")
        ax.set_ylabel("position (pixels)")
        fig.tight_layout()
        if out_path is not None:
            dest = output_path_for_file(out_path, path, n_files)
            dest.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(dest, dpi=150)
            print(f"Wrote {dest.resolve()}")
        if not show:
            plt.close(fig)
    if show:
        plt.show()
        plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot kymographs from HDF5. Loads the leading time window only "
            f"(default {MAX_TIME_SAMPLES} rows); no subsampling."
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
        default=Path(os.path.expanduser("~/data/nsm/plots/nsm_kymographs.png")),
        help=(
            "PNG output path (default: ~/data/nsm/plots/nsm_kymographs.png). "
            "With multiple .h5 files, writes stem_<file>.png in the same directory."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_DEFAULT,
        help=f"HDF5 dataset name (default: {DATASET_DEFAULT})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after saving",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save PNG (use with --show)",
    )

    args = parser.parse_args()
    data_dir = args.directory.expanduser().resolve()

    paths = discover_h5_files(data_dir)
    out_path = None if args.no_save else args.output.expanduser()
    _plot(
        paths,
        out_path=out_path,
        show=args.show or args.no_save,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
