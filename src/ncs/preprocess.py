"""Preprocess views: difference and ratio vs median — line, spectrum, kymograph each."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ncs.data import (
    DATASET_DEFAULT,
    FIGSIZE_INCHES,
    IMAGE_CMAP,
    discover_h5_files,
    load_kymograph,
    output_path_for_file,
)
from ncs.statistics import temporal_median_background


def subtract_background(arr: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Broadcast subtract per-column background: ``arr - background``."""
    if background.shape != (arr.shape[1],):
        raise ValueError(
            f"background length {background.shape} != width {arr.shape[1]}"
        )
    return arr - background


def ratio_to_median(arr: np.ndarray, median: np.ndarray) -> np.ndarray:
    """Per-column ``raw / median`` with floors to avoid division by tiny values."""
    if median.shape != (arr.shape[1],):
        raise ValueError(
            f"median length {median.shape} != width {arr.shape[1]}"
        )
    m = np.maximum(median.astype(np.float64), 1e-12)
    return (arr.astype(np.float64) / m).astype(np.float32)


def contrast_limits(a: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    vmin, vmax = np.percentile(a, (lo, hi))
    return float(vmin), float(vmax)


def _spatial_rfft_spectrum_along_x(
    row_tx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Real FFT magnitude of one time slice vs position (``length X``).

    Mean is removed along x before the transform. Returns ``(freq_cycles_per_px, magnitude)``.
    """
    x = np.asarray(row_tx, dtype=np.float64)
    x = x - np.mean(x)
    n = x.shape[0]
    if n < 2:
        f = np.array([0.0], dtype=np.float64)
        m = np.array([np.abs(x[0])], dtype=np.float64)
        return f, m
    spec = np.fft.rfft(x)
    mag = np.abs(spec).astype(np.float64)
    freqs = np.fft.rfftfreq(n, d=1.0)
    return freqs.astype(np.float64), mag


def _prepare_imshow(arr_td: np.ndarray) -> tuple[np.ndarray, float, float]:
    vmin, vmax = contrast_limits(arr_td)
    disp = np.clip(arr_td.T, vmin, vmax)
    return disp, vmin, vmax


def _preprocess_output_paths(
    template: Path, h5_path: Path, n_files: int
) -> tuple[tuple[Path, Path, Path], tuple[Path, Path, Path]]:
    """Difference and ratio triples: (intensity, spectrum, heatmap) each."""
    base = output_path_for_file(template, h5_path, n_files)
    stem, suf = base.stem, base.suffix
    diff = (
        base.with_name(f"{stem}_intensity{suf}"),
        base.with_name(f"{stem}_spectrum{suf}"),
        base.with_name(f"{stem}_heatmap{suf}"),
    )
    rat = (
        base.with_name(f"{stem}_ratio_intensity{suf}"),
        base.with_name(f"{stem}_ratio_spectrum{suf}"),
        base.with_name(f"{stem}_ratio_heatmap{suf}"),
    )
    return diff, rat


def _plot_preprocess_variant(
    path: Path,
    *,
    meta: str,
    corrected: np.ndarray,
    out_intensity: Path | None,
    out_spectrum: Path | None,
    out_heatmap: Path | None,
    show: bool,
    ratio_mode: bool,
) -> None:
    row0 = corrected[0].astype(np.float32, copy=False)
    xs = np.arange(row0.shape[0], dtype=np.float32)
    ymin, ymax = np.percentile(row0, [1.0, 99.0])
    ymin, ymax = float(ymin), float(ymax)
    if ymax <= ymin:
        ymax = ymin + 1e-6

    freqs, mag = _spatial_rfft_spectrum_along_x(row0)

    disp, _, _ = _prepare_imshow(corrected)

    if ratio_mode:
        ylab = "raw / temporal median"
        heat_title = "raw / temporal median (per column)"
        line_title = f"{path.name}\nfirst frame (t=0): {ylab} vs x • {meta}"
        spec_title = f"{path.name}\nfirst frame (t=0): spatial spectrum of ratio slice • {meta}"
        ref_y = 1.0
    else:
        ylab = "raw − temporal median"
        heat_title = "raw − temporal median (per column)"
        line_title = f"{path.name}\nfirst frame (t=0): {ylab} vs x • {meta}"
        spec_title = (
            f"{path.name}\nfirst frame (t=0): spatial spectrum of difference slice • {meta}"
        )
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

    fig_spec = plt.figure(figsize=FIGSIZE_INCHES, layout="constrained")
    ax_spec = fig_spec.subplots()
    mag_plot = np.maximum(mag, 1e-20)
    ax_spec.plot(freqs, mag_plot, color="C1", linewidth=0.9)
    ax_spec.set_xlim(0.0, float(freqs[-1]) if freqs.size else 1.0)
    ax_spec.set_xlabel("spatial frequency (cycles / pixel)")
    ax_spec.set_ylabel("|FFT| (mean removed along x)")
    ax_spec.set_yscale("log")
    ax_spec.grid(True, alpha=0.35, which="both")
    ax_spec.set_title(spec_title)

    if out_spectrum is not None:
        out_spectrum.parent.mkdir(parents=True, exist_ok=True)
        fig_spec.savefig(out_spectrum, dpi=150)
        print(f"Wrote {out_spectrum.resolve()}")
    if not show:
        plt.close(fig_spec)

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
    ax_map.set_title(f"{heat_title}\n{meta}")
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
    out_diff: tuple[Path | None, Path | None, Path | None],
    out_ratio: tuple[Path | None, Path | None, Path | None],
    show: bool,
    dataset_name: str,
) -> None:
    arr, full_shape = load_kymograph(path, dataset_name=dataset_name)
    median = temporal_median_background(arr)
    corrected_diff = subtract_background(arr, median)
    corrected_ratio = ratio_to_median(arr, median)

    meta = f"loaded array {tuple(arr.shape)} • on-disk {full_shape}"

    _plot_preprocess_variant(
        path,
        meta=meta,
        corrected=corrected_diff,
        out_intensity=out_diff[0],
        out_spectrum=out_diff[1],
        out_heatmap=out_diff[2],
        show=show,
        ratio_mode=False,
    )
    _plot_preprocess_variant(
        path,
        meta=meta,
        corrected=corrected_ratio,
        out_intensity=out_ratio[0],
        out_spectrum=out_ratio[1],
        out_heatmap=out_ratio[2],
        show=show,
        ratio_mode=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Six PNGs per file: difference (raw − median) and ratio (raw / median), each with "
            "t=0 vs x, spatial FFT of that slice, and full kymograph."
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
            "Base PNG path; writes difference: <stem>_{intensity,spectrum,heatmap}.png and "
            "ratio: <stem>_ratio_{intensity,spectrum,heatmap}.png "
            "(default: ~/data/ncs/plots/ncs_preprocess.png). "
            "Several .h5 files: stem_<filestem>_… for each."
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
        if template is None:
            out_d = (None, None, None)
            out_r = (None, None, None)
        else:
            out_d, out_r = _preprocess_output_paths(template, path, n_files)
        _plot_preprocess_outputs(
            path,
            out_diff=out_d,
            out_ratio=out_r,
            show=args.show or args.no_save,
            dataset_name=args.dataset,
        )

    if args.show or args.no_save:
        plt.show()
        plt.close("all")


if __name__ == "__main__":
    main()
