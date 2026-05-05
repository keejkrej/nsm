"""Write preprocessed HDF5 (wavelet residual + illumination + I²) and PNG panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from nsm.kymograph_io import (
    DEFAULT_KYMOGRAPH_DATASET,
    DEFAULT_PLOTS_DIR,
    FIGSIZE_INCHES,
    I2_GAUSSIAN_DATASET,
    IMAGE_CMAP,
    ILLUMINATION_DATASET,
    load_kymograph,
    resolve_output_directory,
)
from nsm.wavelet_residual import (
    I2_GAUSSIAN_SIGMA,
    ILLUMINATION_GAUSSIAN_SIGMA,
    equidistant_time_indices,
    gaussian_smooth_profile_x,
    i2_from_lpdiff,
    lpdiff_residual,
    temporal_median_background,
    y_axis_minmax,
)


def _preprocess_png_paths(out_dir: Path, stem: str) -> tuple[Path, Path]:
    suf = ".png"
    return (
        out_dir / f"{stem}_i2_gauss_heatmap{suf}",
        out_dir / f"{stem}_i2_gauss_intensity{suf}",
    )


def _write_preprocessed_h5(
    path: Path,
    *,
    preprocessed: np.ndarray,
    i2_gaussian: np.ndarray,
    illumination: np.ndarray,
    dataset_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fw:
        fw.create_dataset(
            dataset_name, data=preprocessed.astype(np.float32, copy=False)
        )
        fw.create_dataset(
            I2_GAUSSIAN_DATASET,
            data=i2_gaussian.astype(np.float32, copy=False),
        )
        fw.create_dataset(
            ILLUMINATION_DATASET,
            data=illumination.astype(np.float32, copy=False),
        )
    print(
        f"Wrote {path.resolve()} — {dataset_name!r} "
        f"(median subtract − wavelet LP along x); "
        f"{I2_GAUSSIAN_DATASET!r} ((lpdiff)² / (smoothed {ILLUMINATION_DATASET})², "
        f"then Gaussian σ_x={I2_GAUSSIAN_SIGMA:g}); "
        f"{ILLUMINATION_DATASET!r} (median over t, Gaussian σ_x="
        f"{ILLUMINATION_GAUSSIAN_SIGMA:g} along x)"
    )


def _plot_preprocess_panels(
    path: Path,
    *,
    meta: str,
    residual: np.ndarray,
    i2_kymo: np.ndarray,
    smooth_caption: str,
    out_heatmap: Path | None,
    out_i2_intensity: Path | None,
    show: bool,
) -> None:
    """Line slices and heatmap show **i2_kymo** (illumination-rescaled Gaussian I²)."""
    n_time, nx = residual.shape
    assert i2_kymo.shape == (n_time, nx)
    t_rows = equidistant_time_indices(n_time, k=5)
    stacked_i2 = np.stack([i2_kymo[int(t)] for t in t_rows], axis=0)
    ymin2, ymax2 = y_axis_minmax(stacked_i2)

    xs = np.arange(nx, dtype=np.float32)
    prefix = f"preprocessed − wavelet LP ({smooth_caption})\n"
    slices_note = f"{len(t_rows)} equidistant time slice{'s' if len(t_rows) != 1 else ''} t ∈ {{{', '.join(str(int(t)) for t in t_rows)}}}"

    ylab_i2 = r"$I^2$"
    ylab_i2_sub = (
        f"(lpdiff)² / (Gaussian illumin, σ_x={ILLUMINATION_GAUSSIAN_SIGMA:g})², "
        f"then Gaussian σ_x={I2_GAUSSIAN_SIGMA:g} along x"
    )
    i2_line_title = (
        f"{path.name}\n{prefix}{ylab_i2} vs x — {slices_note} • {ylab_i2_sub}\n"
        rf"($I$ = lpdiff residual) • {meta}"
    )

    fig_i2 = plt.figure(figsize=FIGSIZE_INCHES, layout="constrained")
    ax_i2 = fig_i2.subplots()
    for i, t in enumerate(t_rows):
        y_sq = stacked_i2[i]
        ax_i2.plot(
            xs,
            y_sq,
            color=f"C{i}",
            linewidth=0.9,
            label=f"t = {int(t)}",
        )
    ax_i2.set_xlim(float(xs[0]), float(xs[-1]))
    ax_i2.set_ylim(ymin2, ymax2)
    ax_i2.set_xlabel("position x (pixel index)")
    ax_i2.set_ylabel(ylab_i2)
    ax_i2.grid(True, alpha=0.35)
    ax_i2.set_title(i2_line_title)
    ax_i2.legend(loc="best", fontsize=9, framealpha=0.92)

    if out_i2_intensity is not None:
        out_i2_intensity.parent.mkdir(parents=True, exist_ok=True)
        fig_i2.savefig(out_i2_intensity, dpi=150)
        print(f"Wrote {out_i2_intensity.resolve()}")
    if not show:
        plt.close(fig_i2)

    disp = i2_kymo.T.astype(np.float32, copy=False)
    heat_vmin, heat_vmax = np.percentile(i2_kymo, (1.0, 99.0))
    if (
        not np.isfinite(heat_vmin)
        or not np.isfinite(heat_vmax)
        or heat_vmax <= heat_vmin
    ):
        heat_vmin, heat_vmax = y_axis_minmax(i2_kymo)

    heat_title = (
        f"{prefix}{ylab_i2} kymograph • {ylab_i2_sub}\n{meta}"
    )

    fig_map = plt.figure(figsize=FIGSIZE_INCHES, layout="constrained")
    ax_map = fig_map.subplots()
    im = ax_map.imshow(
        disp,
        aspect="equal",
        origin="upper",
        cmap=IMAGE_CMAP,
        vmin=heat_vmin,
        vmax=heat_vmax,
        interpolation="nearest",
    )
    fig_map.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    ax_map.set_title(heat_title)
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
    bundle: tuple[np.ndarray, np.ndarray, str] | None,
    out_paths: tuple[Path | None, Path | None],
    show: bool,
    dataset_name: str,
    wavelet: str,
    wavelet_level: int,
) -> None:
    """``bundle`` = ``(lpdiff residual, i2_gaussian, smooth_caption)`` when available."""
    if bundle is None:
        if arr is None or disk_shape is None:
            arr, disk_shape = load_kymograph(path, dataset_name=dataset_name)
        res, smooth_caption = lpdiff_residual(
            arr, wavelet=wavelet, wavelet_level=wavelet_level
        )
        illum_s = gaussian_smooth_profile_x(
            temporal_median_background(arr), sigma=ILLUMINATION_GAUSSIAN_SIGMA
        )
        i2_k = i2_from_lpdiff(res, illum_s)
    else:
        res, i2_k, smooth_caption = bundle
        if arr is None or disk_shape is None:
            arr, disk_shape = load_kymograph(path, dataset_name=dataset_name)

    meta = f"loaded array {tuple(arr.shape)} • on-disk {disk_shape}"

    _plot_preprocess_panels(
        path,
        meta=meta,
        residual=res,
        i2_kymo=i2_k,
        smooth_caption=smooth_caption,
        out_heatmap=out_paths[0],
        out_i2_intensity=out_paths[1],
        show=show,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One input .h5: writes <stem>_preprocessed.h5 with lpdiff kymograph "
            "(raw − temporal median − wavelet LP along x), Gaussian-smoothed temporal-median "
            "illumination along x, illumination-rescaled Gaussian I² along x, plus two PNG "
            "panels (I² kymogram heatmap and multi-time I² vs x)."
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
        default=DEFAULT_KYMOGRAPH_DATASET,
        help=f"HDF5 dataset key (default: {DEFAULT_KYMOGRAPH_DATASET})",
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
            "nsm-preprocess: preprocessed .h5 (lpdiff + i2_gaussian + illumination) + "
            "2× PNG panels."
        )

    out_dir: Path | None = None if args.no_save else resolve_output_directory(args.output)
    stem = h5_path.stem
    arr0: np.ndarray | None = None
    disk_shape0: tuple[int, int] | None = None
    bundle: tuple[np.ndarray, np.ndarray, str] | None = None
    if out_dir is not None:
        pre_h5 = out_dir / f"{stem}_preprocessed.h5"
        arr0, disk_shape0 = load_kymograph(h5_path, dataset_name=args.dataset)
        illum_median = temporal_median_background(arr0)
        illumination_smooth = gaussian_smooth_profile_x(
            illum_median, sigma=ILLUMINATION_GAUSSIAN_SIGMA
        )
        res, capt = lpdiff_residual(
            arr0,
            wavelet=args.wavelet,
            wavelet_level=args.wavelet_level,
        )
        i2_g = i2_from_lpdiff(res, illumination_smooth)
        bundle = (res, i2_g, capt)
        _write_preprocessed_h5(
            pre_h5,
            preprocessed=res,
            i2_gaussian=i2_g,
            illumination=illumination_smooth,
            dataset_name=args.dataset,
        )

    out_p: tuple[Path | None, Path | None]
    if out_dir is None:
        out_p = (None, None)
    else:
        out_p = _preprocess_png_paths(out_dir, stem)

    _plot_preprocess_outputs(
        h5_path,
        arr=arr0,
        disk_shape=disk_shape0,
        bundle=bundle,
        out_paths=out_p,
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
