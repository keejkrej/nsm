"""Preprocess: median-subtracted HDF5 (+ illumination median profile) + lpdiff PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from nsm.data import (
    DATASET_DEFAULT,
    DEFAULT_PLOTS_DIR,
    FIGSIZE_INCHES,
    IMAGE_CMAP,
    ILLUMINATION_DATASET,
    load_kymograph,
    resolve_output_directory,
)
from nsm.lpdiff import (
    equidistant_time_indices,
    kymograph_clip_for_imshow,
    lpdiff_residual,
    subtract_background,
    temporal_median_background,
    y_axis_minmax,
)


def _lpdiff_png_paths(out_dir: Path, stem: str) -> tuple[Path, Path]:
    """``lpdiff`` = preprocessed minus wavelet LP (along x); output PNG pair."""
    suf = ".png"
    return (
        out_dir / f"{stem}_lpdiff_intensity{suf}",
        out_dir / f"{stem}_lpdiff_heatmap{suf}",
    )


def _write_preprocessed_h5(
    path: Path,
    *,
    preprocessed: np.ndarray,
    illumination: np.ndarray,
    dataset_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fw:
        fw.create_dataset(
            dataset_name, data=preprocessed.astype(np.float32, copy=False)
        )
        fw.create_dataset(
            ILLUMINATION_DATASET,
            data=illumination.astype(np.float32, copy=False),
        )
    print(
        f"Wrote {path.resolve()} — {dataset_name!r} (median-subtracted), "
        f"{ILLUMINATION_DATASET!r} (per-x median over t)"
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
    """``residual`` = preprocessed (raw − temporal median) minus wavelet LP along x."""
    n_time, _ = residual.shape
    t_rows = equidistant_time_indices(n_time, k=5)
    stacked = np.stack([residual[int(t)].astype(np.float32, copy=False) for t in t_rows], axis=0)
    ymin, ymax = y_axis_minmax(stacked)

    xs = np.arange(residual.shape[1], dtype=np.float32)

    disp, _, _ = kymograph_clip_for_imshow(residual)

    base_ylab = "raw − temporal median"
    base_heat = "raw − temporal median (per column)"
    ylab = f"{base_ylab} − LP"
    heat_title = f"{base_heat} − wavelet LP along x"
    prefix = f"preprocessed − wavelet LP ({smooth_caption})\n"
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
    arr: np.ndarray | None,
    disk_shape: tuple[int, int] | None,
    out_lpdiff: tuple[Path | None, Path | None],
    show: bool,
    dataset_name: str,
    wavelet: str,
    wavelet_level: int,
) -> None:
    if arr is None or disk_shape is None:
        arr, disk_shape = load_kymograph(path, dataset_name=dataset_name)

    res, smooth_caption = lpdiff_residual(
        arr, wavelet=wavelet, wavelet_level=wavelet_level
    )

    meta = f"loaded array {tuple(arr.shape)} • on-disk {disk_shape}"

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
            "One input .h5 (full time axis): writes <stem>_preprocessed.h5 with the "
            "median-subtracted kymograph, per-x temporal median as illumination, "
            "plus two lpdiff PNGs."
        )
    )
    parser.add_argument(
        "h5_path",
        type=Path,
        help="Input .h5 (e.g. cropped file from nsm-crop)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help=f"Output directory (default: {DEFAULT_PLOTS_DIR})",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="db4",
        help="PyWavelets name; 1-D LP along **x** per time row (default: db4)",
    )
    parser.add_argument(
        "--wavelet-level",
        type=int,
        default=4,
        help="1-D decomposition depth along x (clamped; default: 4)",
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
        help="Do not write PNG or preprocessed .h5 (use with --show)",
    )

    args = parser.parse_args()
    h5_path = args.h5_path.expanduser().resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"not a file: {h5_path}")

    if not args.no_save:
        print(
            "nsm-preprocess: preprocessed .h5 (kymograph + illumination median) + "
            "2× lpdiff PNG (wavelet LP along x)."
        )

    out_dir: Path | None = None if args.no_save else resolve_output_directory(args.output)
    stem = h5_path.stem
    arr0: np.ndarray | None = None
    disk_shape0: tuple[int, int] | None = None
    if out_dir is not None:
        pre_h5 = out_dir / f"{stem}_preprocessed.h5"
        arr0, disk_shape0 = load_kymograph(h5_path, dataset_name=args.dataset)
        illumination = temporal_median_background(arr0)
        pre = subtract_background(arr0, illumination)
        _write_preprocessed_h5(
            pre_h5,
            preprocessed=pre,
            illumination=illumination,
            dataset_name=args.dataset,
        )

    out_lp: tuple[Path | None, Path | None]
    if out_dir is None:
        out_lp = (None, None)
    else:
        out_lp = _lpdiff_png_paths(out_dir, stem)

    _plot_preprocess_outputs(
        h5_path,
        arr=arr0,
        disk_shape=disk_shape0,
        out_lpdiff=out_lp,
        show=args.show or args.no_save,
        dataset_name=args.dataset,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
    )

    if args.show or args.no_save:
        plt.show()
        plt.close("all")


if __name__ == "__main__":
    main()
