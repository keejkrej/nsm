"""Write preprocessed HDF5 (wavelet detail + Gaussian-smoothed detail²) and PNG panels."""

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
    IMAGE_CMAP,
    RESIDUAL_SQ_GAUSSIAN_DATASET,
    load_kymograph,
    resolve_output_directory,
)
from nsm.wavelet_residual import (
    RESIDUAL_SQ_GAUSSIAN_SIGMA,
    equidistant_time_indices,
    squared_residual_gaussian,
    wavelet_detail_residual,
    y_axis_minmax,
)


def _preprocess_png_paths(out_dir: Path, stem: str) -> tuple[Path, Path]:
    suf = ".png"
    return (
        out_dir / f"{stem}_residual_sq_gauss_heatmap{suf}",
        out_dir / f"{stem}_residual_sq_gauss_intensity{suf}",
    )


def _write_preprocessed_h5(
    path: Path,
    *,
    wavelet_detail: np.ndarray,
    residual_sq_gaussian: np.ndarray,
    dataset_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fw:
        fw.create_dataset(
            dataset_name, data=wavelet_detail.astype(np.float32, copy=False)
        )
        fw.create_dataset(
            RESIDUAL_SQ_GAUSSIAN_DATASET,
            data=residual_sq_gaussian.astype(np.float32, copy=False),
        )
    print(
        f"Wrote {path.resolve()} - {dataset_name!r} "
        "(median subtract - wavelet LP along x); "
        f"{RESIDUAL_SQ_GAUSSIAN_DATASET!r} (detail^2 then Gaussian sigma_x="
        f"{RESIDUAL_SQ_GAUSSIAN_SIGMA:g} along x)"
    )


def _plot_preprocess_panels(
    path: Path,
    *,
    meta: str,
    wavelet_detail: np.ndarray,
    residual_sq_kymo: np.ndarray,
    smooth_caption: str,
    out_heatmap: Path | None,
    out_sq_intensity: Path | None,
    show: bool,
) -> None:
    """Line slices and heatmap show **residual_sq_kymo** (Gaussian-smoothed squared wavelet detail)."""
    n_time, nx = wavelet_detail.shape
    assert residual_sq_kymo.shape == (n_time, nx)
    t_rows = equidistant_time_indices(n_time, k=5)
    stacked_sq = np.stack([residual_sq_kymo[int(t)] for t in t_rows], axis=0)
    ymin2, ymax2 = y_axis_minmax(stacked_sq)

    xs = np.arange(nx, dtype=np.float32)
    prefix = f"preprocessed − wavelet LP ({smooth_caption})\n"
    slices_note = f"{len(t_rows)} equidistant time slice{'s' if len(t_rows) != 1 else ''} t ∈ {{{', '.join(str(int(t)) for t in t_rows)}}}"

    ylab_sq = r"$\mathrm{detail}^{2}$"
    ylab_sq_sub = f"Gaussian σ_x={RESIDUAL_SQ_GAUSSIAN_SIGMA:g} along x"
    sq_line_title = (
        f"{path.name}\n{prefix}{ylab_sq} vs x — {slices_note} • {ylab_sq_sub}\n"
        "(detail = median-corrected − wavelet LP along x) • "
        f"{meta}"
    )

    fig_sq = plt.figure(figsize=FIGSIZE_INCHES, layout="constrained")
    ax_sq = fig_sq.subplots()
    for i, t in enumerate(t_rows):
        y_sq = stacked_sq[i]
        ax_sq.plot(
            xs,
            y_sq,
            color=f"C{i}",
            linewidth=0.9,
            label=f"t = {int(t)}",
        )
    ax_sq.set_xlim(float(xs[0]), float(xs[-1]))
    ax_sq.set_ylim(ymin2, ymax2)
    ax_sq.set_xlabel("position x (pixel index)")
    ax_sq.set_ylabel(ylab_sq)
    ax_sq.grid(True, alpha=0.35)
    ax_sq.set_title(sq_line_title)
    ax_sq.legend(loc="best", fontsize=9, framealpha=0.92)

    if out_sq_intensity is not None:
        out_sq_intensity.parent.mkdir(parents=True, exist_ok=True)
        fig_sq.savefig(out_sq_intensity, dpi=150)
        print(f"Wrote {out_sq_intensity.resolve()}")
    if not show:
        plt.close(fig_sq)

    disp = residual_sq_kymo.T.astype(np.float32, copy=False)
    heat_vmin, heat_vmax = np.percentile(residual_sq_kymo, (1.0, 99.0))
    if (
        not np.isfinite(heat_vmin)
        or not np.isfinite(heat_vmax)
        or heat_vmax <= heat_vmin
    ):
        heat_vmin, heat_vmax = y_axis_minmax(residual_sq_kymo)

    heat_title = f"{prefix}{ylab_sq} kymograph • {ylab_sq_sub}\n{meta}"

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
    """``bundle`` = ``(wavelet_detail, residual_sq_gaussian, smooth_caption)`` when available."""
    if bundle is None:
        if arr is None or disk_shape is None:
            arr, disk_shape = load_kymograph(path, dataset_name=dataset_name)
        detail, smooth_caption = wavelet_detail_residual(
            arr, wavelet=wavelet, wavelet_level=wavelet_level
        )
        sq_k = squared_residual_gaussian(detail)
    else:
        detail, sq_k, smooth_caption = bundle
        if arr is None or disk_shape is None:
            arr, disk_shape = load_kymograph(path, dataset_name=dataset_name)

    meta = f"loaded array {tuple(arr.shape)} • on-disk {disk_shape}"

    _plot_preprocess_panels(
        path,
        meta=meta,
        wavelet_detail=detail,
        residual_sq_kymo=sq_k,
        smooth_caption=smooth_caption,
        out_heatmap=out_paths[0],
        out_sq_intensity=out_paths[1],
        show=show,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One input .h5: writes <stem>_preprocessed.h5 with wavelet-detail kymograph "
            "(raw − temporal median − wavelet LP along x), Gaussian-smoothed squared detail, "
            "plus two PNG panels (heatmap and multi-time line plots)."
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
            f"nsm-preprocess: preprocessed .h5 (wavelet detail + {RESIDUAL_SQ_GAUSSIAN_DATASET}) + "
            "2 PNG panels."
        )

    out_dir: Path | None = None if args.no_save else resolve_output_directory(args.output)
    stem = h5_path.stem
    arr0: np.ndarray | None = None
    disk_shape0: tuple[int, int] | None = None
    bundle: tuple[np.ndarray, np.ndarray, str] | None = None
    if out_dir is not None:
        pre_h5 = out_dir / f"{stem}_preprocessed.h5"
        arr0, disk_shape0 = load_kymograph(h5_path, dataset_name=args.dataset)
        detail, capt = wavelet_detail_residual(
            arr0,
            wavelet=args.wavelet,
            wavelet_level=args.wavelet_level,
        )
        sq_g = squared_residual_gaussian(detail)
        bundle = (detail, sq_g, capt)
        _write_preprocessed_h5(
            pre_h5,
            wavelet_detail=detail,
            residual_sq_gaussian=sq_g,
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
