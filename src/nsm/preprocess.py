"""Preprocess pipeline: temporal median subtract + spatial median along x; lpdiff PNGs only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nsm.data import (
    DATASET_DEFAULT,
    DEFAULT_DATA_ROOT,
    DEFAULT_PLOTS_DIR,
    FIGSIZE_INCHES,
    IMAGE_CMAP,
    discover_h5_files,
    load_kymograph,
    output_path_for_file,
)
from nsm.lpdiff import (
    equidistant_time_indices,
    kymograph_clip_for_imshow,
    lpdiff_residual,
    y_axis_minmax,
)


def _lpdiff_output_paths(template: Path, h5_path: Path, n_files: int) -> tuple[Path, Path]:
    """``lpdiff`` = preprocessed minus spatial-median-smoothed (along x); output PNG pair."""
    base = output_path_for_file(template, h5_path, n_files)
    stem, suf = base.stem, base.suffix
    return (
        base.with_name(f"{stem}_lpdiff_intensity{suf}"),
        base.with_name(f"{stem}_lpdiff_heatmap{suf}"),
    )


def _plot_lpdiff(
    path: Path,
    *,
    meta: str,
    residual: np.ndarray,
    smooth_caption: str,
    out_intensity: Path | None,
    out_heatmap: Path | None,
    show: bool,
) -> None:
    """``residual`` = preprocessed (raw − temporal median) minus spatial median along x."""
    n_time, _ = residual.shape
    t_rows = equidistant_time_indices(n_time, k=5)
    stacked = np.stack([residual[int(t)].astype(np.float32, copy=False) for t in t_rows], axis=0)
    ymin, ymax = y_axis_minmax(stacked)

    xs = np.arange(residual.shape[1], dtype=np.float32)

    disp, _, _ = kymograph_clip_for_imshow(residual)

    base_ylab = "raw − temporal median"
    base_heat = "raw − temporal median (per column)"
    ylab = f"{base_ylab} − smooth(x)"
    heat_title = f"{base_heat} − spatial median (x)"
    prefix = f"preprocessed − {smooth_caption}\n"
    slices_note = f"{len(t_rows)} equidistant time slice{'s' if len(t_rows) != 1 else ''} t ∈ {{{', '.join(str(int(t)) for t in t_rows)}}}"
    line_title = f"{path.name}\n{prefix}{ylab} vs x — {slices_note} • {meta}"
    map_title = f"{prefix}{heat_title}\n{meta}"
    ref_y = 0.0

    fig_line = plt.figure(figsize=FIGSIZE_INCHES, layout="constrained")
    ax_line = fig_line.subplots()
    for i, t in enumerate(t_rows):
        y = residual[int(t)].astype(np.float32, copy=False)
        ax_line.plot(
            xs,
            y,
            color=f"C{i}",
            linewidth=0.9,
            label=f"t = {int(t)}",
        )
    ax_line.set_xlim(float(xs[0]), float(xs[-1]))
    ax_line.set_ylim(ymin, ymax)
    ax_line.set_xlabel("position x (pixel index)")
    ax_line.set_ylabel(ylab)
    ax_line.axhline(ref_y, color="0.4", linestyle=":", linewidth=0.8)
    ax_line.grid(True, alpha=0.35)
    ax_line.set_title(line_title)
    ax_line.legend(loc="best", fontsize=9, framealpha=0.92)

    if out_intensity is not None:
        out_intensity.parent.mkdir(parents=True, exist_ok=True)
        fig_line.savefig(out_intensity, dpi=150)
        print(f"Wrote {out_intensity.resolve()}")
    if not show:
        plt.close(fig_line)

    fig_map = plt.figure(figsize=FIGSIZE_INCHES, layout="constrained")
    ax_map = fig_map.subplots()
    im = ax_map.imshow(
        disp,
        aspect="equal",
        origin="upper",
        cmap=IMAGE_CMAP,
        interpolation="nearest",
    )
    fig_map.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    ax_map.set_title(map_title)
    ax_map.set_xlabel("time (axis 0)")
    ax_map.set_ylabel("position (pixels)")

    if out_heatmap is not None:
        out_heatmap.parent.mkdir(parents=True, exist_ok=True)
        fig_map.savefig(out_heatmap, dpi=150)
        print(f"Wrote {out_heatmap.resolve()}")
    if not show:
        plt.close(fig_map)


def _plot_preprocess_outputs(
    path: Path,
    *,
    out_lpdiff: tuple[Path | None, Path | None],
    show: bool,
    dataset_name: str,
    median_kernel_x: int,
) -> None:
    arr, full_shape = load_kymograph(path, dataset_name=dataset_name)

    res, smooth_caption = lpdiff_residual(arr, median_kernel_x=median_kernel_x)

    meta = f"loaded array {tuple(arr.shape)} • on-disk {full_shape}"

    _plot_lpdiff(
        path,
        meta=meta,
        residual=res,
        smooth_caption=smooth_caption,
        out_intensity=out_lpdiff[0],
        out_heatmap=out_lpdiff[1],
        show=show,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Two PNGs per file: (preproc − spatial median along x) — kymograph and line panel "
            "(up to 5 equidistant time slices, distinct colors). "
            "Temporal median subtraction + spatial median smooth; only *_lpdiff_* outputs."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=DEFAULT_DATA_ROOT,
        type=Path,
        help=f"Directory containing .h5 files (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_PLOTS_DIR / "nsm_preprocess.png",
        help=(
            "Base PNG path; per .h5 writes only <stem>_lpdiff_{intensity,heatmap}.png "
            f"(default: {DEFAULT_PLOTS_DIR / 'nsm_preprocess.png'}). "
            "Multi-file: stem_<filestem>_…."
        ),
    )
    parser.add_argument(
        "--median-kernel-x",
        type=int,
        default=15,
        help=(
            "Spatial median window along x (odd length; even values bump +1; default: 15)"
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
    if not args.no_save:
        print(
            "nsm-preprocess: lpdiff PNGs only (2 per .h5); temporal median + spatial median (x)."
        )
    data_dir = args.directory.expanduser().resolve()
    paths = discover_h5_files(data_dir)
    n_files = len(paths)
    template = None if args.no_save else args.output.expanduser()

    for path in paths:
        if template is None:
            out_lp = (None, None)
        else:
            out_lp = _lpdiff_output_paths(template, path, n_files)
        _plot_preprocess_outputs(
            path,
            out_lpdiff=out_lp,
            show=args.show or args.no_save,
            dataset_name=args.dataset,
            median_kernel_x=args.median_kernel_x,
        )

    if args.show or args.no_save:
        plt.show()
        plt.close("all")


if __name__ == "__main__":
    main()
