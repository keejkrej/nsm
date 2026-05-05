"""SciPy peak picks on illumination-rescaled `i2_gaussian` with preprocess overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from nsm.data import (
    DEFAULT_PLOTS_DIR,
    FIGSIZE_INCHES,
    I2_GAUSSIAN_DATASET,
    IMAGE_CMAP,
    load_kymograph,
    resolve_output_directory,
)
from nsm.lpdiff import (
    I2_GAUSSIAN_SIGMA,
    ILLUMINATION_GAUSSIAN_SIGMA,
    equidistant_time_indices,
    y_axis_minmax,
)

PEAK_DATASET = "peaks_ix"
"""``(N, 2)`` uint32 storing ``(t, x_pixel)`` for each detected peak."""

PEAK_COLUMNS = np.dtype([("time", "<u4"), ("x", "<u4")])


def collect_peaks_1d_rows(
    z: np.ndarray,
    *,
    rel_prominence: float,
    distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``find_peaks`` on each time row along **x**; prominence = ``rel * row_max``.

    ``distance`` enforces separation between neighbouring peaks along **x** (pixels).
    """
    nt = int(z.shape[0])
    t_list: list[int] = []
    x_list: list[int] = []
    dist = max(1, int(distance))

    for t in range(nt):
        row = z[t].astype(np.float64, copy=False)
        mx = float(np.nanmax(row))
        if not np.isfinite(mx) or mx <= 1e-30:
            continue
        cand, _ = find_peaks(row, prominence=rel_prominence * mx, distance=dist)
        for xi in cand:
            t_list.append(t)
            x_list.append(int(xi))

    if not t_list:
        return np.array([], dtype=np.uint32), np.array([], dtype=np.uint32)
    peaks = np.zeros(len(t_list), dtype=PEAK_COLUMNS)
    peaks["time"] = np.asarray(t_list, dtype=np.uint32)
    peaks["x"] = np.asarray(x_list, dtype=np.uint32)
    return peaks["time"], peaks["x"]


def plot_detect_overlays(
    src_name: Path,
    i2_gauss: np.ndarray,
    *,
    peaks_t: np.ndarray,
    peaks_x: np.ndarray,
    peak_meta: str,
    out_heatmap: Path,
    out_intensity: Path,
) -> None:
    """Rebuild preprocess I² PNGs with scatter overlays (same layout as nsm-preprocess)."""
    n_time, nx = i2_gauss.shape
    t_rows = equidistant_time_indices(n_time, k=5)
    stacked_i2 = np.stack([i2_gauss[int(t)] for t in t_rows], axis=0)
    ymin2, ymax2 = y_axis_minmax(stacked_i2)

    xs = np.arange(nx, dtype=np.float32)
    prefix = (
        f"(lpdiff)² / (Gaussian illumin, σ_x={ILLUMINATION_GAUSSIAN_SIGMA:g})² "
        f"→ Gaussian σ_x={I2_GAUSSIAN_SIGMA:g} • {peak_meta}\n"
    )

    ylab_i2 = r"$I^2$"
    slices_note = (
        f"{len(t_rows)} equidistant time slices t ∈ {{{', '.join(str(int(t)) for t in t_rows)}}}"
    )
    title_line = (
        f"{src_name.name}\nsnm-detect overlays • {prefix}{ylab_i2} vs x — {slices_note}"
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
        mk = peaks_t == int(t)
        if np.any(mk):
            px = peaks_x[mk]
            py = i2_gauss[int(t), px.astype(np.int64)]
            ax_i2.scatter(
                px.astype(np.float32),
                py,
                color=f"C{i}",
                s=60.0,
                marker="s",
                edgecolors="black",
                linewidths=0.85,
                zorder=10,
                label="_nolegend_",
            )
    ax_i2.set_xlim(float(xs[0]), float(xs[-1]))
    ax_i2.set_ylim(ymin2, ymax2)
    ax_i2.set_xlabel("position x (pixel index)")
    ax_i2.set_ylabel(ylab_i2)
    ax_i2.grid(True, alpha=0.35)
    ax_i2.set_title(title_line)
    ax_i2.legend(loc="best", fontsize=9, framealpha=0.92)
    out_intensity.parent.mkdir(parents=True, exist_ok=True)
    fig_i2.savefig(out_intensity, dpi=150)
    plt.close(fig_i2)
    print(f"Wrote {out_intensity.resolve()}")

    disp = i2_gauss.T.astype(np.float32, copy=False)
    heat_vmin, heat_vmax = np.percentile(i2_gauss, (1.0, 99.0))
    if (
        not np.isfinite(heat_vmin)
        or not np.isfinite(heat_vmax)
        or heat_vmax <= heat_vmin
    ):
        heat_vmin, heat_vmax = y_axis_minmax(i2_gauss)

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
    ax_map.set_title(f"{src_name.name}\nsnm-detect overlays • {prefix}I² kymograph")
    ax_map.set_xlabel("time (axis 0)")
    ax_map.set_ylabel("position (pixels)")
    if peaks_t.size:
        ax_map.scatter(
            peaks_t.astype(np.float64),
            peaks_x.astype(np.float64),
            s=10.0,
            facecolors="cyan",
            edgecolors="navy",
            linewidths=0.35,
            alpha=0.9,
            marker="o",
        )
    out_heatmap.parent.mkdir(parents=True, exist_ok=True)
    fig_map.savefig(out_heatmap, dpi=150)
    plt.close(fig_map)
    print(f"Wrote {out_heatmap.resolve()}")


def write_peaks_h5(
    dest: Path,
    peaks: np.ndarray,
    *,
    source_path: Path,
    rel_prominence: float,
    distance: int,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dest, "w") as fw:
        ds = fw.create_dataset(
            PEAK_DATASET,
            data=peaks.astype(PEAK_COLUMNS, copy=False),
        )
        ds.attrs["columns"] = "time, x"
        ds.attrs["source"] = str(source_path.expanduser().resolve())
        ds.attrs["illumination_gaussian_sigma_x"] = float(ILLUMINATION_GAUSSIAN_SIGMA)
        ds.attrs["i2_gaussian_sigma_x"] = float(I2_GAUSSIAN_SIGMA)
        ds.attrs["i2_rescaled_by_illumination_squared"] = True
        ds.attrs["rel_prominence"] = float(rel_prominence)
        ds.attrs["peak_distance_px"] = int(distance)
    print(f"Wrote {dest.resolve()} — {len(peaks)} peaks")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"SciPy peaks on illumination-rescaled `{I2_GAUSSIAN_DATASET}` "
            "in *_preprocessed.h5 — scatter overlays matching nsm-preprocess."
        )
    )
    parser.add_argument(
        "preprocessed_h5",
        type=Path,
        help=(
            "nsm-preprocess output HDF5 (`kymograph`, `illumination`, "
            f"`{I2_GAUSSIAN_DATASET}`)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help=(
            "Output directory for <stem>_detect.h5 plus *_detect_*.png "
            f"(default: {DEFAULT_PLOTS_DIR})"
        ),
    )
    parser.add_argument(
        "--rel-prominence",
        type=float,
        default=0.06,
        help="SciPy prominence = this × max(signal) per row (default: 0.06)",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=8,
        help="SciPy minimal peak separation along **x** in pixels (default: 8)",
    )

    args = parser.parse_args()
    if args.rel_prominence <= 0:
        parser.error("--rel-prominence must be > 0")
    if args.distance < 1:
        parser.error("--distance must be >= 1")

    src = args.preprocessed_h5.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"not a file: {src}")

    i2_g, _ = load_kymograph(src, dataset_name=I2_GAUSSIAN_DATASET)
    pt, px = collect_peaks_1d_rows(
        i2_g,
        rel_prominence=float(args.rel_prominence),
        distance=int(args.distance),
    )
    peaks = np.zeros(len(pt), dtype=PEAK_COLUMNS)
    if peaks.size:
        peaks["time"] = pt
        peaks["x"] = px

    out_dir = resolve_output_directory(args.output)
    stem = src.stem
    dest_h5 = out_dir / f"{stem}_detect.h5"
    write_peaks_h5(
        dest_h5,
        peaks,
        source_path=src,
        rel_prominence=float(args.rel_prominence),
        distance=int(args.distance),
    )

    pk_meta = (
        f"{len(peaks)} peaks • scipy find_peaks "
        f"rel_prom={args.rel_prominence:g} • distance={args.distance}px"
    )

    plot_detect_overlays(
        src,
        i2_gauss=i2_g,
        peaks_t=pt,
        peaks_x=px,
        peak_meta=pk_meta,
        out_heatmap=out_dir / f"{stem}_detect_i2_gauss_heatmap.png",
        out_intensity=out_dir / f"{stem}_detect_i2_gauss_intensity.png",
    )


if __name__ == "__main__":
    main()
