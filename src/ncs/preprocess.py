"""Residual kymographs: raw minus per-column temporal median or mean."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt

from ncs.data import DATASET_DEFAULT, IMAGE_CMAP, discover_h5_files, load_kymograph, output_path_for_file
from ncs.statistics import temporal_mean_background, temporal_median_background


def subtract_background(arr: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Broadcast subtract per-column background: ``arr - background``."""
    if background.shape != (arr.shape[1],):
        raise ValueError(
            f"background length {background.shape} != width {arr.shape[1]}"
        )
    return arr - background


def contrast_limits(a: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    vmin, vmax = np.percentile(a, (lo, hi))
    return float(vmin), float(vmax)


def _prepare_imshow(arr_td: np.ndarray) -> tuple[np.ndarray, float, float]:
    vmin, vmax = contrast_limits(arr_td)
    disp = np.clip(arr_td.T, vmin, vmax)
    return disp, vmin, vmax


def _residual_output_paths(primary: Path) -> tuple[Path, Path]:
    """``…/stem.png`` → ``…/stem_median.png`` and ``…/stem_mean.png``."""
    median_p = primary.with_name(f"{primary.stem}_median{primary.suffix}")
    mean_p = primary.with_name(f"{primary.stem}_mean{primary.suffix}")
    return median_p, mean_p


def _save_one_figure(
    *,
    path: Path,
    full_shape: tuple[int, int],
    disp: np.ndarray,
    title_line: str,
    out_path: Path | None,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    arr_shape = (disp.shape[1], disp.shape[0])
    meta = f"loaded array {arr_shape} • on-disk {full_shape}"
    im = ax.imshow(
        disp,
        aspect="equal",
        origin="upper",
        cmap=IMAGE_CMAP,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{path.name} — {title_line}\n{meta}")
    ax.set_xlabel("time (axis 0)")
    ax.set_ylabel("position (pixels)")
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path.resolve()}")
    if not show:
        plt.close(fig)


def _plot_preprocessed_panel(
    path: Path,
    *,
    out_path_base: Path | None,
    show: bool,
    dataset_name: str,
) -> None:
    arr, full_shape = load_kymograph(path, dataset_name=dataset_name)

    corrected_med = subtract_background(arr, temporal_median_background(arr))
    corrected_mean = subtract_background(arr, temporal_mean_background(arr))

    disp_med, _, _ = _prepare_imshow(corrected_med)
    disp_mean, _, _ = _prepare_imshow(corrected_mean)

    if out_path_base is not None:
        median_p, mean_p = _residual_output_paths(out_path_base)
    else:
        median_p, mean_p = None, None

    _save_one_figure(
        path=path,
        full_shape=full_shape,
        disp=disp_med,
        title_line="raw − temporal median (per column)",
        out_path=median_p,
        show=show,
    )
    _save_one_figure(
        path=path,
        full_shape=full_shape,
        disp=disp_mean,
        title_line="raw − temporal mean (per column)",
        out_path=mean_p,
        show=show,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Two separate figures per file: raw − temporal median and "
            "raw − temporal mean (background = aggregate over time at each x). "
            "Saves …_median.png and …_mean.png."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=os.path.expanduser("~/data/ncs"),
        type=Path,
        help="Directory containing .h5 files (default: ~/data/ncs)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(os.path.expanduser("~/data/ncs/plots/ncs_preprocess.png")),
        help=(
            "PNG stem (default: …/ncs_preprocess.png): writes "
            "stem_median.png and stem_mean.png "
            "(with several .h5: stem_<file>_median.png, etc.)."
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
        help="Do not write PNG (use with --show)",
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
        _plot_preprocessed_panel(
            path,
            out_path_base=dest,
            show=args.show or args.no_save,
            dataset_name=args.dataset,
        )

    if args.show or args.no_save:
        plt.show()
        plt.close("all")


if __name__ == "__main__":
    main()
