"""Preprocess views: raw − temporal median, and residual (preproc − wavelet LP along x)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt

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
from nsm.statistics import temporal_median_background


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


def wavelet_lowpass_along_x(
    arr: np.ndarray,
    *,
    wavelet: str = "db4",
    level: int = 4,
) -> tuple[np.ndarray, int]:
    """1-D approximation-only low-pass along axis **x** for each time row (detail coeffs zeroed).

    Assumes ``arr`` is shaped ``(time, x)``. No smoothing along time.

    Returns the filtered array and the decomposition depth used (0 if the input
    was returned unchanged).
    """
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D (time, x) array, got shape {x.shape}")
    n_time, width = x.shape
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(int(width), w)
    if max_level < 1:
        return arr.astype(np.float32, copy=False), 0
    lvl = max(1, min(int(level), max_level))
    out = np.empty_like(x)
    for i in range(n_time):
        coeffs = pywt.wavedec(x[i], wavelet, level=lvl, mode="symmetric")
        coeffs_lp = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        rec = pywt.waverec(coeffs_lp, wavelet, mode="symmetric")
        out[i] = rec[:width]
    return out.astype(np.float32), lvl


def _prepare_imshow(arr_td: np.ndarray) -> tuple[np.ndarray, float, float]:
    vmin, vmax = contrast_limits(arr_td)
    disp = np.clip(arr_td.T, vmin, vmax)
    return disp, vmin, vmax


def _preprocess_output_paths(
    template: Path, h5_path: Path, n_files: int
) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    """(raw − median plots, preproc−LP residual plots); each (intensity, heatmap).

    Wavelet LP fields are computed but not saved as their own PNG pair.
    """
    base = output_path_for_file(template, h5_path, n_files)
    stem, suf = base.stem, base.suffix
    diff = (
        base.with_name(f"{stem}_intensity{suf}"),
        base.with_name(f"{stem}_heatmap{suf}"),
    )
    diff_res = (
        base.with_name(f"{stem}_lpdiff_intensity{suf}"),
        base.with_name(f"{stem}_lpdiff_heatmap{suf}"),
    )
    return diff, diff_res


def _plot_preprocess(
    path: Path,
    *,
    meta: str,
    corrected: np.ndarray,
    out_intensity: Path | None,
    out_heatmap: Path | None,
    show: bool,
) -> None:
    row0 = corrected[0].astype(np.float32, copy=False)
    xs = np.arange(row0.shape[0], dtype=np.float32)
    ymin, ymax = np.percentile(row0, [1.0, 99.0])
    ymin, ymax = float(ymin), float(ymax)
    if ymax <= ymin:
        ymax = ymin + 1e-6

    disp, _, _ = _prepare_imshow(corrected)

    ylab = "raw − temporal median"
    heat_title = "raw − temporal median (per column)"
    line_title = f"{path.name}\nfirst frame (t=0): {ylab} vs x • {meta}"
    map_title = f"{heat_title}\n{meta}"
    ref_y = 0.0

    fig_line = plt.figure(figsize=FIGSIZE_INCHES, layout="constrained")
    ax_line = fig_line.subplots()
    ax_line.plot(xs, row0, color="C0", linewidth=0.9)
    ax_line.set_xlim(float(xs[0]), float(xs[-1]))
    ax_line.set_ylim(ymin, ymax)
    ax_line.set_xlabel("position x (pixel index)")
    ax_line.set_ylabel(ylab)
    ax_line.axhline(ref_y, color="0.4", linestyle=":", linewidth=0.8)
    ax_line.grid(True, alpha=0.35)
    ax_line.set_title(line_title)

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


def _plot_preprocess_residual(
    path: Path,
    *,
    meta: str,
    residual: np.ndarray,
    wavelet_desc: str,
    out_intensity: Path | None,
    out_heatmap: Path | None,
    show: bool,
) -> None:
    """``residual`` = preprocessed (raw − median) minus wavelet LP along x."""
    row0 = residual[0].astype(np.float32, copy=False)
    xs = np.arange(row0.shape[0], dtype=np.float32)
    ymin, ymax = np.percentile(row0, [1.0, 99.0])
    ymin, ymax = float(ymin), float(ymax)
    if ymax <= ymin:
        ymax = ymin + 1e-6

    disp, _, _ = _prepare_imshow(residual)

    base_ylab = "raw − temporal median"
    base_heat = "raw − temporal median (per column)"
    ylab = f"{base_ylab} − LP"
    heat_title = f"{base_heat} − wavelet LP along x"
    prefix = f"preprocessed − wavelet LP ({wavelet_desc})\n"
    line_title = f"{path.name}\n{prefix}first frame (t=0): {ylab} vs x • {meta}"
    map_title = f"{prefix}{heat_title}\n{meta}"
    ref_y = 0.0

    fig_line = plt.figure(figsize=FIGSIZE_INCHES, layout="constrained")
    ax_line = fig_line.subplots()
    ax_line.plot(xs, row0, color="C0", linewidth=0.9)
    ax_line.set_xlim(float(xs[0]), float(xs[-1]))
    ax_line.set_ylim(ymin, ymax)
    ax_line.set_xlabel("position x (pixel index)")
    ax_line.set_ylabel(ylab)
    ax_line.axhline(ref_y, color="0.4", linestyle=":", linewidth=0.8)
    ax_line.grid(True, alpha=0.35)
    ax_line.set_title(line_title)

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
    out_diff: tuple[Path | None, Path | None],
    out_diff_res: tuple[Path | None, Path | None],
    show: bool,
    dataset_name: str,
    wavelet: str,
    wavelet_level: int,
) -> None:
    arr, full_shape = load_kymograph(path, dataset_name=dataset_name)
    median = temporal_median_background(arr)
    corrected = subtract_background(arr, median)

    lp, lvl = wavelet_lowpass_along_x(
        corrected, wavelet=wavelet, level=wavelet_level
    )
    wavelet_caption = (
        f"{wavelet}, level {lvl}, axis=x"
        if lvl
        else f"{wavelet}, axis=x (no decomposition; LP equals preproc)"
    )
    res = corrected.astype(np.float32, copy=False) - lp

    meta = f"loaded array {tuple(arr.shape)} • on-disk {full_shape}"

    _plot_preprocess(
        path,
        meta=meta,
        corrected=corrected,
        out_intensity=out_diff[0],
        out_heatmap=out_diff[1],
        show=show,
    )
    _plot_preprocess_residual(
        path,
        meta=meta,
        residual=res,
        wavelet_desc=wavelet_caption,
        out_intensity=out_diff_res[0],
        out_heatmap=out_diff_res[1],
        show=show,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Four PNGs per file: raw − temporal median (line + kymograph), and the same for "
            "(preproc − wavelet LP along x). The LP-smoothed field itself is not plotted."
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
            "Base PNG path; per .h5 writes <stem>_{intensity,heatmap} and "
            "<stem>_lpdiff_{intensity,heatmap} (preproc − LP). "
            f"(default: {DEFAULT_PLOTS_DIR / 'nsm_preprocess.png'}). "
            "Multi-file: stem_<filestem>_…."
        ),
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
        help="Do not write PNG (use with --show)",
    )

    args = parser.parse_args()
    if not args.no_save:
        print(
            "nsm-preprocess: raw−median + (preproc−LP) residual; no standalone LP PNGs."
        )
    data_dir = args.directory.expanduser().resolve()
    paths = discover_h5_files(data_dir)
    n_files = len(paths)
    template = None if args.no_save else args.output.expanduser()

    for path in paths:
        if template is None:
            empty = (None, None)
            out_d = out_r = empty
        else:
            out_d, out_r = _preprocess_output_paths(template, path, n_files)
        _plot_preprocess_outputs(
            path,
            out_diff=out_d,
            out_diff_res=out_r,
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
