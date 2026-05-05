"""Kymograph viewer: full loaded window (no subsampling)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nsm.data import (
    DATASET_DEFAULT,
    DEFAULT_PLOTS_DIR,
    FIGSIZE_INCHES,
    IMAGE_CMAP,
    load_kymograph,
    resolve_png_destination,
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


def _plot(path: Path, out_path: Path | None, show: bool, dataset_name: str) -> None:
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
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path.resolve()}")
    if not show:
        plt.close(fig)
    if show:
        plt.show()
        plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one kymograph from an HDF5 file (full time axis of that file; no subsampling)."
        )
    )
    parser.add_argument(
        "h5_path",
        type=Path,
        help="Input .h5 (e.g. from nsm-crop)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help=(
            "Output directory, or a path ending in .png. "
            f"Default directory: {DEFAULT_PLOTS_DIR} (writes <stem>_kymograph.png)."
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
    h5_path = args.h5_path.expanduser().resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"not a file: {h5_path}")

    if args.no_save:
        dest: Path | None = None
    else:
        dest = resolve_png_destination(
            args.output,
            source_stem=h5_path.stem,
            filename=f"{h5_path.stem}_kymograph.png",
        )
    _plot(
        h5_path,
        out_path=dest,
        show=args.show or args.no_save,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
