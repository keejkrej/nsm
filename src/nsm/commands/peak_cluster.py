"""Peak-cluster command implementation.

This module now owns the wavelet preprocessing, peak detection, and clustering
helpers that were previously split across
``peak_cluster_wavelet.py``, ``peak_cluster_peaks.py``, and
``peak_cluster_clustering.py``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
from skimage.filters import gaussian
from sklearn.cluster import DBSCAN

from nsm.core import (
    DEFAULT_KYMOGRAPH_DATASET,
    DEFAULT_PLOTS_DIR,
    FIGSIZE_INCHES,
    IMAGE_CMAP,
    RESIDUAL_SQ_GAUSSIAN_DATASET,
    load_kymograph,
    resolve_output_directory,
    write_kymograph_png,
    write_raw_preview,
    write_trajectory_overlay_png,
)


NAME = "peak-cluster"
HELP = "Run peak detection and cluster-based trajectory extraction."

METHOD = "peak_cluster"

# ---------------------------------------------------------------------------
# Wavelet / pre-processing helpers (from peak_cluster_wavelet.py)

RESIDUAL_SQ_GAUSSIAN_SIGMA = 10.0
"""Gaussian σ in pixels along **x** on squared wavelet detail."""


def temporal_median_background(arr: np.ndarray) -> np.ndarray:
    """For each position x, median intensity over time. ``arr`` is (T, X)."""
    if arr.ndim != 2:
        raise ValueError(f"expected 2D (T, X), got shape {arr.shape}")
    return np.median(arr.astype(np.float32, copy=False), axis=0).astype(np.float32)


def subtract_background(arr: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Broadcast subtract per-column background: ``arr - background``."""
    if background.shape != (arr.shape[1],):
        raise ValueError(
            f"background length {background.shape} != width {arr.shape[1]}"
        )
    return arr - background


def wavelet_lowpass_along_x(
    arr: np.ndarray,
    *,
    wavelet: str = "db4",
    level: int = 4,
) -> tuple[np.ndarray, int]:
    """Low-pass reconstruction along **x** using wavelets.

    Returns filtered array and decomposition depth (0 when unchanged).
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


def equidistant_time_indices(n_time: int, *, k: int = 5) -> np.ndarray:
    """``k`` evenly spaced row indices covering ``0 … n_time−1``."""
    if n_time <= 0:
        raise ValueError(f"need positive time extent, got {n_time}")
    k_eff = min(k, n_time)
    if k_eff == 1:
        return np.zeros(1, dtype=np.int64)
    denom = k_eff - 1
    return (np.arange(k_eff, dtype=np.int64) * (n_time - 1) // denom).astype(np.int64)


def gaussian_smooth_along_x(arr_tx: np.ndarray, *, sigma: float) -> np.ndarray:
    """Smooth each time row along **x** only."""
    z = np.asarray(arr_tx, dtype=np.float32)
    if sigma <= 0:
        return z
    out = gaussian(
        np.asarray(z, dtype=np.float64),
        sigma=(0.0, float(sigma)),
        mode="reflect",
        preserve_range=True,
        truncate=4.0,
    )
    return out.astype(np.float32, copy=False)


def squared_residual_gaussian(
    wavelet_detail_tx: np.ndarray,
    *,
    gaussian_sigma: float = RESIDUAL_SQ_GAUSSIAN_SIGMA,
) -> np.ndarray:
    """Squared wavelet-detail kymograph, then Gaussian smooth along x."""
    sq = np.square(np.asarray(wavelet_detail_tx, dtype=np.float32))
    return gaussian_smooth_along_x(sq, sigma=float(gaussian_sigma))


def y_axis_minmax(a: np.ndarray) -> tuple[float, float]:
    """Return y-axis limits with tiny expansion if flat."""
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def wavelet_lp_caption(wavelet: str, lvl: int) -> str:
    if lvl:
        return f"{wavelet}, level {lvl}, axis=x"
    return f"{wavelet}, axis=x (no decomposition; LP equals preproc)"


def median_subtracted(arr_tx: np.ndarray) -> np.ndarray:
    """``(T, X)`` raw kymograph → ``raw − median`` per column."""
    median = temporal_median_background(arr_tx)
    return subtract_background(arr_tx, median)


def wavelet_detail_residual(
    arr_tx: np.ndarray,
    *,
    wavelet: str = "db4",
    wavelet_level: int = 4,
) -> tuple[np.ndarray, str]:
    """Temporal median correction + detail extraction along x + caption."""
    corrected = median_subtracted(arr_tx)
    lp, lvl = wavelet_lowpass_along_x(
        corrected, wavelet=wavelet, level=wavelet_level
    )
    detail = corrected.astype(np.float32, copy=False) - lp
    return detail, wavelet_lp_caption(wavelet, lvl)


# ---------------------------------------------------------------------------
# Peak detection helpers (from peak_cluster_peaks.py)

PEAK_DATASET = "peaks_ix"
"""``(N, 2)`` uint32 storing ``(time, x_pixel)`` for each detected peak."""

PEAK_COLUMNS = np.dtype([("time", "<u4"), ("x", "<u4")])


def gather_peaks_rowwise(
    z: np.ndarray,
    *,
    rel_prominence: float,
    distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find 1D peaks in every time row along x."""
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


def plot_peak_overlays(
    src_name: Path,
    residual_sq_gauss: np.ndarray,
    *,
    peaks_t: np.ndarray,
    peaks_x: np.ndarray,
    peak_meta: str,
    out_heatmap: Path,
    out_intensity: Path,
) -> None:
    """Rebuild preprocessed PNG overlays (line plots + heatmap)."""
    n_time, nx = residual_sq_gauss.shape
    t_rows = equidistant_time_indices(n_time, k=5)
    stacked_sq = np.stack([residual_sq_gauss[int(t)] for t in t_rows], axis=0)
    ymin2, ymax2 = y_axis_minmax(stacked_sq)

    xs = np.arange(nx, dtype=np.float32)
    ylab = r"$\mathrm{detail}^{2}$"
    prefix = f"Gaussian σ_x={RESIDUAL_SQ_GAUSSIAN_SIGMA:g} after squaring • {peak_meta}\n"
    slices_note = (
        f"{len(t_rows)} equidistant time slices t ∈ {{{', '.join(str(int(t)) for t in t_rows)}}}"
    )
    title_line = (
        f"{src_name.name}\nsnm-detect overlays • {prefix}{ylab} vs x — {slices_note}"
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
        mk = peaks_t == int(t)
        if np.any(mk):
            px = peaks_x[mk]
            py = residual_sq_gauss[int(t), px.astype(np.int64)]
            ax_sq.scatter(
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
    ax_sq.set_xlim(float(xs[0]), float(xs[-1]))
    ax_sq.set_ylim(ymin2, ymax2)
    ax_sq.set_xlabel("position x (pixel index)")
    ax_sq.set_ylabel(ylab)
    ax_sq.grid(True, alpha=0.35)
    ax_sq.set_title(title_line)
    ax_sq.legend(loc="best", fontsize=9, framealpha=0.92)
    out_intensity.parent.mkdir(parents=True, exist_ok=True)
    fig_sq.savefig(out_intensity, dpi=150)
    plt.close(fig_sq)
    print(f"Wrote {out_intensity.resolve()}")

    disp = residual_sq_gauss.T.astype(np.float32, copy=False)
    heat_vmin, heat_vmax = np.percentile(residual_sq_gauss, (1.0, 99.0))
    if (
        not np.isfinite(heat_vmin)
        or not np.isfinite(heat_vmax)
        or heat_vmax <= heat_vmin
    ):
        heat_vmin, heat_vmax = y_axis_minmax(residual_sq_gauss)

    fig_map = plt.figure(figsize=FIGSIZE_INCHES, layout="constrained")
    ax_map = fig_map.subplots()
    im = ax_map.imshow(
        disp,
        aspect="auto",
        origin="upper",
        cmap=IMAGE_CMAP,
        vmin=heat_vmin,
        vmax=heat_vmax,
        interpolation="nearest",
    )
    fig_map.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    ax_map.set_title(f"{src_name.name}\nsnm-detect overlays • {prefix}{ylab} kymograph")
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


def write_peaks_csv(
    dest: Path,
    *,
    x: np.ndarray,
    t: np.ndarray,
    intensity: np.ndarray,
) -> None:
    """Write detected peaks CSV ``x,t,intensity`` sorted by ``(t, x)``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    order = np.lexsort((x, t))
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "t", "intensity"])
        for i in order:
            w.writerow([float(x[i]), int(t[i]), float(intensity[i])])
    print(f"Wrote {dest.resolve()} — {len(t)} rows")


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
        ds.attrs["residual_sq_gaussian_sigma_x"] = float(RESIDUAL_SQ_GAUSSIAN_SIGMA)
        ds.attrs["rel_prominence"] = float(rel_prominence)
        ds.attrs["peak_distance_px"] = int(distance)
    print(f"Wrote {dest.resolve()} — {len(peaks)} peaks")


def run_peak_detect_cli(
    preprocessed_h5: Path,
    output: Path = DEFAULT_PLOTS_DIR,
    rel_prominence: float = 0.06,
    distance: int = 8,
) -> tuple[Path, Path, int]:
    """CLI-like helper retained for compatibility with old module behaviour."""
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocessed_h5")
    parser.add_argument("-o", "--output")
    parser.add_argument("--rel-prominence", type=float, default=0.06)
    parser.add_argument("--distance", type=int, default=8)
    args = parser.parse_args([str(preprocessed_h5), "-o", str(output), "--rel-prominence", str(rel_prominence), "--distance", str(distance)])

    src = Path(args.preprocessed_h5).expanduser().resolve()
    sq_g, _ = load_kymograph(src, dataset_name=RESIDUAL_SQ_GAUSSIAN_DATASET)
    pt, px = gather_peaks_rowwise(
        sq_g,
        rel_prominence=float(args.rel_prominence),
        distance=int(args.distance),
    )
    peaks = np.zeros(len(pt), dtype=PEAK_COLUMNS)
    if peaks.size:
        peaks["time"] = pt
        peaks["x"] = px
    out_dir = resolve_output_directory(output)
    stem = src.stem
    dest_h5 = out_dir / f"{stem}_detect.h5"
    write_peaks_h5(
        dest_h5,
        peaks,
        source_path=src,
        rel_prominence=float(args.rel_prominence),
        distance=int(args.distance),
    )
    dest_csv = out_dir / f"{stem}_peaks.csv"
    if len(peaks):
        inten = sq_g[pt.astype(np.int64), px.astype(np.int64)].astype(np.float64, copy=False)
        write_peaks_csv(
            dest_csv,
            x=px.astype(np.float64),
            t=pt.astype(np.int64),
            intensity=inten,
        )
    else:
        write_peaks_csv(
            dest_csv,
            x=np.array([]),
            t=np.array([], dtype=np.int64),
            intensity=np.array([]),
        )
    plot_peak_overlays(
        src,
        residual_sq_gauss=sq_g,
        peaks_t=pt,
        peaks_x=px,
        peak_meta=(
            f"{len(peaks)} peaks • scipy find_peaks "
            f"rel_prom={args.rel_prominence:g} • distance={args.distance}px"
        ),
        out_heatmap=out_dir / f"{stem}_detect_residual_sq_gauss_heatmap.png",
        out_intensity=out_dir / f"{stem}_detect_residual_sq_gauss_intensity.png",
    )
    return dest_csv, dest_h5, len(peaks)


# ---------------------------------------------------------------------------
# Clustering helpers (from peak_cluster_clustering.py)


def resolve_preprocessed_h5(peaks_csv: Path, explicit: Path | None) -> Path | None:
    """Find sibling ``<stem>.h5`` unless explicit preprocessed path is provided."""
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--preprocessed-h5 not a file: {p}")
        return p

    stem = peaks_csv.stem
    if not stem.endswith("_peaks"):
        return None
    base = stem[: -len("_peaks")]
    cand = peaks_csv.parent / f"{base}.h5"
    if cand.is_file():
        return cand
    return None


def plot_tracks_overlay(
    preprocessed_h5: Path,
    stacked: np.ndarray,
    *,
    out_png: Path,
    title_note: str,
) -> None:
    """Render trajectory overlay on residual heatmap from preprocessed HDF5."""
    z, _ = load_kymograph(preprocessed_h5, dataset_name=RESIDUAL_SQ_GAUSSIAN_DATASET)
    disp = z.T.astype(np.float32, copy=False)
    heat_vmin, heat_vmax = np.percentile(z, (1.0, 99.0))
    if (
        not np.isfinite(heat_vmin)
        or not np.isfinite(heat_vmax)
        or heat_vmax <= heat_vmin
    ):
        heat_vmin, heat_vmax = y_axis_minmax(z)

    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES)
    im = ax.imshow(
        disp,
        aspect="auto",
        origin="upper",
        cmap=IMAGE_CMAP,
        vmin=heat_vmin,
        vmax=heat_vmax,
        interpolation="nearest",
    )

    cid = stacked[:, 0]
    xt = stacked[:, 2].astype(np.float64, copy=False)
    xx = stacked[:, 1].astype(np.float64, copy=False)

    vmax_id = float(np.max(cid)) if cid.size else 1.0
    ax.scatter(
        xt,
        xx,
        c=cid,
        cmap="gist_ncar",
        vmin=-0.5,
        vmax=max(vmax_id + 0.5, 1.0),
        s=14.0,
        alpha=0.92,
        edgecolors="black",
        linewidths=0.25,
        zorder=5,
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    prefix = (
        f"Gaussian σ_x={RESIDUAL_SQ_GAUSSIAN_SIGMA:g} · particle id (peak_cluster) • {title_note}"
    )
    ax.set_title(f"{preprocessed_h5.name}\nsnm-track overlays • {prefix}")
    ax.set_xlabel("time (axis 0)")
    ax.set_ylabel("position (pixels)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png.resolve()}")


def read_peaks_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, t, intensity)`` from peaks CSV."""
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path.name}: empty file")
    header = [h.strip().lower() for h in rows[0]]
    required = {"x", "t", "intensity"}
    if required - set(header):
        raise ValueError(
            f"{path.name}: CSV must start with columns x, t, intensity "
            f"(any order); got {header!r}"
        )
    xi, ti, ii = header.index("x"), header.index("t"), header.index("intensity")
    xs_list: list[float] = []
    ts_list: list[int] = []
    ins_list: list[float] = []
    for ln in rows[1:]:
        if not ln or all(not str(c).strip() for c in ln):
            continue
        try:
            x = float(ln[xi])
            t_raw = float(ln[ti])
            iq = float(ln[ii])
        except (ValueError, IndexError):
            raise ValueError(f"{path.name}: bad row {ln!r}") from None
        t_int = int(round(t_raw))
        if abs(t_raw - t_int) > 1e-6:
            raise ValueError(
                f"{path.name}: column t must be integer frame indices; got {t_raw!r}"
            )
        xs_list.append(x)
        ts_list.append(t_int)
        ins_list.append(iq)
    if not xs_list:
        raise ValueError(f"{path.name}: no numeric rows below header")
    return (
        np.asarray(xs_list, dtype=np.float64),
        np.asarray(ts_list, dtype=np.int64),
        np.asarray(ins_list, dtype=np.float64),
    )


def _finalize_dbscan_labels(z_feat: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Attach DBSCAN noise points to nearest cluster in feature space."""
    z_feat = np.asarray(z_feat, dtype=np.float64)
    if z_feat.ndim == 1:
        z_feat = z_feat.reshape(-1, 1)
    labs = labels.astype(np.int64).copy()
    noise_mask = labs == -1
    finite = labs >= 0
    cluster_ids = np.unique(labs[finite])
    if cluster_ids.size == 0:
        return np.zeros(labs.shape[0], dtype=np.int64)
    medians = np.stack(
        [np.median(z_feat[labs == c], axis=0) for c in cluster_ids],
        axis=0,
    )
    if np.any(noise_mask):
        zn = z_feat[noise_mask]
        d = np.linalg.norm(zn[:, None, :] - medians[None, :, :], axis=2)
        nearest = cluster_ids[np.argmin(d, axis=1)]
        labs[noise_mask] = nearest.astype(np.int64)
    return labs


def _silhouette_mean_euclidean(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette for Euclidean distance labels."""
    X = np.asarray(X, dtype=np.float64)
    labs = labels.astype(np.int64)
    n = X.shape[0]
    uniq = np.unique(labs)
    if uniq.size < 2 or n < 2:
        return float("-inf")
    dist_sq = squareform(pdist(X, metric="euclidean"))
    sil = np.zeros(n, dtype=np.float64)
    idx_all = np.arange(n)
    for i in range(n):
        own = labs[i]
        same_mask = (labs == own) & (idx_all != i)
        if np.any(same_mask):
            a_i = float(np.mean(dist_sq[i, same_mask]))
        else:
            a_i = 0.0
        b_i = float("inf")
        for c in uniq:
            if int(c) == int(own):
                continue
            msk = labs == c
            if np.any(msk):
                b_i = min(b_i, float(np.mean(dist_sq[i, msk])))
        den = max(a_i, b_i)
        sil[i] = (b_i - a_i) / den if den > 0.0 else 0.0
    return float(np.mean(sil))


def _kmeans_best_labels(Xz: np.ndarray, k_eff: int, n_init: int) -> np.ndarray:
    """Best labels across multiple ``kmeans2`` seeds by min inertia."""
    best_labels: np.ndarray | None = None
    best_cost = float("inf")
    for seed in range(max(1, n_init)):
        centroids, labels = kmeans2(
            Xz, k_eff, iter=100, minit="points", seed=seed
        )
        if np.unique(labels).size < k_eff:
            continue
        li = labels.astype(np.int64)
        cost = float(np.sum((Xz - centroids[li]) ** 2))
        if cost < best_cost:
            best_cost = cost
            best_labels = li.astype(np.int32).copy()
    if best_labels is None:
        _centroids, labels = kmeans2(Xz, k_eff, iter=100, minit="random", seed=0)
        best_labels = labels.astype(np.int32)
    return best_labels


def _pca_tx_scores(ts: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA of ``(t, x)``, returning scores and explained-variance fractions."""
    P = np.column_stack((ts.astype(np.float64), xs.astype(np.float64)))
    Pc = P - P.mean(axis=0)
    n = int(Pc.shape[0])
    if n < 2:
        return np.zeros((n, 1), dtype=np.float64), np.array([1.0], dtype=np.float64)
    cov = np.cov(Pc.T)
    evals, evecs = np.linalg.eigh(cov)
    ord_ = np.argsort(evals)[::-1]
    evals = np.maximum(evals[ord_].astype(np.float64), 0.0)
    evecs = evecs[:, ord_]
    n_comp = min(2, evecs.shape[1])
    scores = Pc @ evecs[:, :n_comp]
    tot_var = float(np.sum(evals))
    if tot_var <= 0:
        var_frac = np.ones(n_comp, dtype=np.float64) / float(max(n_comp, 1))
    else:
        var_frac = (evals[:n_comp] / tot_var).astype(np.float64)
    return scores, var_frac


def _save_particle_cluster_diagnostic(
    ts: np.ndarray,
    xs: np.ndarray,
    labels: np.ndarray,
    pc_scores: np.ndarray,
    var_frac: np.ndarray,
    out_path: Path,
    *,
    n_clusters: int,
    silhouette: float | None,
    title_note: str,
    subtitle_detail: str,
) -> None:
    """PNG diagnostic of (t, x) assignments + PCA score clusters."""
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labs = labels.astype(np.int64)
    uniq = np.unique(labs)
    cmap = plt.cm.tab10
    mxlab = float(np.max(labs)) if labs.size else 1.0
    norm = plt.Normalize(vmin=-0.5, vmax=max(mxlab + 0.5, 1.0))

    fig = plt.figure(figsize=(11.0, 10.0), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0])
    ax_tx = fig.add_subplot(gs[0, 0])
    ax_pc = fig.add_subplot(gs[1, 0])

    for c in uniq:
        m = labs == int(c)
        color = cmap(norm(float(c)))
        ax_tx.scatter(
            ts[m],
            xs[m],
            s=36,
            color=color,
            edgecolors="black",
            linewidths=0.35,
            alpha=0.9,
            label=f"particle {int(c)}",
        )

    ax_tx.set_xlabel("time t (frame)")
    ax_tx.set_ylabel("position x (pixels)")
    ax_tx.set_title(f"{title_note}\nPeak assignments in (t, x) · {n_clusters} clusters")
    ax_tx.legend(loc="best", fontsize=9, framealpha=0.92)
    ax_tx.grid(True, alpha=0.35)
    ax_tx.set_aspect("equal", adjustable="box")

    sc = pc_scores.astype(np.float64)
    if sc.shape[1] >= 2:
        vx = float(var_frac[0]) if var_frac.size else 0.0
        vy = float(var_frac[1]) if var_frac.size > 1 else 0.0
        xlab = f"PC1 ({100 * vx:.0f}% var)"
        ylab = f"PC2 ({100 * vy:.0f}% var)"
        pc_x = sc[:, 0]
        pc_y = sc[:, 1]
    else:
        xlab = f"PC1 ({100 * float(var_frac[0]):.0f}% var)" if var_frac.size else "PC1"
        ylab = "PC2 (unused)"
        pc_x = sc[:, 0]
        pc_y = np.zeros(pc_x.shape[0], dtype=np.float64)

    for c in uniq:
        m = labs == int(c)
        color = cmap(norm(float(c)))
        ax_pc.scatter(
            pc_x[m],
            pc_y[m],
            s=36,
            color=color,
            edgecolors="black",
            linewidths=0.35,
            alpha=0.9,
        )
    suline = (
        f"mean silhouette = {silhouette:.3f}"
        if silhouette is not None and np.isfinite(float(silhouette))
        else ""
    )
    pc_title = subtitle_detail
    if suline:
        pc_title += " · " + suline
    ax_pc.set_title(pc_title)
    ax_pc.set_xlabel(xlab)
    ax_pc.set_ylabel(ylab)
    ax_pc.grid(True, alpha=0.35)

    fig.suptitle(
        "Particle ids from PCA(t,x) clustering — intensity excluded",
        fontsize=10,
        y=1.02,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}")


def cluster_peaks_to_particle_trails(
    xs: np.ndarray,
    ts: np.ndarray,
    ints: np.ndarray,
    *,
    method: str,
    dbscan_eps: float,
    dbscan_min_samples: int,
    k_min: int,
    k_max: int,
    k_penalty: float,
    n_init: int,
    cluster_plot_path: Path | None = None,
    plot_title: str = "",
) -> tuple[list[list[int]], int | None, float | None]:
    """Assign each peak to a particle trajectory.

    Supported ``method``: ``dbscan`` and ``kmeans``.
    """
    _ = ints
    n = int(xs.shape[0])
    if n == 0:
        return [], None, None
    if n == 1:
        return [[0]], None, None

    scores, var_frac = _pca_tx_scores(ts, xs)

    chosen_labels: np.ndarray | None = None
    chosen_n_clust: int | None = None
    chosen_sil: float | None = None
    subtitle_detail = ""

    if method == "dbscan":
        if scores.shape[1] >= 2:
            mu = scores.mean(axis=0)
            sig = scores.std(axis=0)
            sig = np.where(sig < 1e-12, 1.0, sig)
            z_feat = (scores - mu) / sig
            subtitle_detail = (
                f"DBSCAN on z(PC1)+z(PC2): eps={float(dbscan_eps):g}, "
                f"min_samples={int(dbscan_min_samples)}"
            )
        else:
            pc2_raw = scores[:, 0]
            mu2 = float(np.mean(pc2_raw))
            sig2 = float(np.std(pc2_raw))
            if sig2 < 1e-12:
                z_feat = np.zeros((n, 1), dtype=np.float64)
            else:
                z_feat = ((pc2_raw - mu2) / sig2).reshape(-1, 1)
            subtitle_detail = (
                f"DBSCAN on z(PC2): eps={float(dbscan_eps):g}, "
                f"min_samples={int(dbscan_min_samples)}"
            )
        raw = DBSCAN(
            eps=float(dbscan_eps),
            min_samples=int(dbscan_min_samples),
            metric="euclidean",
        ).fit_predict(z_feat)
        labs_dense = _finalize_dbscan_labels(z_feat, raw.astype(np.int64))
        uniq_dense = np.unique(labs_dense)
        chosen_n_clust = int(uniq_dense.size)
        chosen_labels = labs_dense
        sil = _silhouette_mean_euclidean(z_feat, chosen_labels)
        chosen_sil = sil if chosen_n_clust >= 2 and np.isfinite(sil) else None
    elif method == "kmeans":
        mu = scores.mean(axis=0)
        sig = scores.std(axis=0)
        sig = np.where(sig < 1e-12, 1.0, sig)
        Xz = (scores - mu) / sig

        k_hi = min(k_max, n)
        k_lo = max(2, min(k_min, k_hi))
        if k_lo > k_hi:
            trails = [[i] for i in range(n)]
            return trails, None, None

        best_score = float("-inf")
        best_k_pick = k_lo
        best_labels_pick: np.ndarray | None = None
        for kk in range(k_lo, k_hi + 1):
            labels_try = _kmeans_best_labels(Xz, kk, n_init)
            sil_try = _silhouette_mean_euclidean(Xz, labels_try)
            score_try = sil_try - float(k_penalty) * float(kk)
            if score_try > best_score + 1e-12 or (
                abs(score_try - best_score) <= 1e-12 and kk < best_k_pick
            ):
                best_score = score_try
                best_k_pick = kk
                best_labels_pick = labels_try
        assert best_labels_pick is not None
        chosen_labels = best_labels_pick
        chosen_n_clust = int(np.unique(chosen_labels).size)
        subtitle_detail = f"k-means (auto K={best_k_pick}) on z(PC1), z(PC2)"
        sil = _silhouette_mean_euclidean(Xz, chosen_labels)
        chosen_sil = (
            sil
            if chosen_n_clust >= 2 and np.isfinite(sil)
            else None
        )
    else:
        raise ValueError(f"unknown cluster method {method!r}")

    assert chosen_labels is not None and chosen_n_clust is not None

    labs_raw = chosen_labels.astype(np.int64)

    uniq_labels = sorted(np.unique(labs_raw).tolist())
    order_by_x = sorted(
        uniq_labels,
        key=lambda cid: float(np.median(xs[labs_raw == cid])),
    )
    remap = {int(old): j for j, old in enumerate(order_by_x)}
    labs = np.array([remap[int(a)] for a in labs_raw], dtype=np.int64)

    n_parts = int(labs.max()) + 1 if labs.size else 0
    buckets: list[list[int]] = [[] for _ in range(n_parts)]
    for pi in range(n):
        buckets[int(labs[pi])].append(pi)

    trails = [sorted(set(ix), key=lambda ri: int(ts[ri])) for ix in buckets]
    trails = [t for t in trails if t]

    if cluster_plot_path is not None:
        _save_particle_cluster_diagnostic(
            ts,
            xs,
            labs,
            scores,
            var_frac,
            cluster_plot_path,
            n_clusters=int(chosen_n_clust),
            silhouette=chosen_sil,
            title_note=plot_title,
            subtitle_detail=subtitle_detail,
        )

    return trails, chosen_n_clust, chosen_sil


def trails_to_rows(
    trails: list[list[int]],
    xs: np.ndarray,
    ts: np.ndarray,
    ints: np.ndarray,
) -> np.ndarray:
    rows: list[tuple[int, float, int, float]] = []
    for tid, ixlist in enumerate(trails):
        for pi in ixlist:
            rows.append((tid, float(xs[pi]), int(ts[pi]), float(ints[pi])))
    return np.asarray(rows, dtype=np.float64)


def write_tracks_csv(dest: Path, rows: np.ndarray) -> None:
    """Write trajectory CSV with columns ``id,x,t,intensity``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "t", "intensity"])
        for r in rows:
            pid, xx, tt, iq = int(r[0]), float(r[1]), int(r[2]), float(r[3])
            w.writerow([pid, xx, tt, iq])


def _lexsort_stacked(stacked: np.ndarray) -> np.ndarray:
    arr = np.asarray(stacked, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"expected (N, 4) rows, got {arr.shape}")
    order = np.lexsort((arr[:, 3], arr[:, 1], arr[:, 2], arr[:, 0]))
    return arr[order]


def _pick_winner_by_silhouette(sil_db: float | None, sil_km: float | None) -> str:
    """Prefer higher silhouette, tie to ``dbscan``."""

    def score(s: float | None) -> float:
        if s is None or not np.isfinite(float(s)):
            return float("-inf")
        return float(s)

    if score(sil_km) > score(sil_db):
        return "kmeans"
    return "dbscan"


def _sil_txt(sil: float | None) -> str:
    if sil is None or not np.isfinite(float(sil)):
        return "sil=n/a"
    return f"sil={float(sil):.3f}"


def run_cluster_cli(
    peaks_csv: Path,
    output: Path = DEFAULT_PLOTS_DIR,
    preprocessed_h5: Path | None = None,
    no_overlay: bool = False,
    cluster_method: str = "both",
    cluster_dbscan_eps: float = 1.0,
    cluster_dbscan_min_samples: int = 4,
    cluster_min_k: int = 2,
    cluster_max_k: int = 12,
    cluster_k_penalty: float = 0.045,
    cluster_init: int = 48,
    cluster_plot: Path | None = None,
    no_cluster_plot: bool = False,
) -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("peaks_csv")
    parser.add_argument("-o", "--output", default=str(DEFAULT_PLOTS_DIR))
    parser.add_argument("--cluster-method", choices=("both", "dbscan", "kmeans"), default="both")
    # other args intentionally ignored in this helper wrapper
    args = parser.parse_args([str(peaks_csv), "-o", str(output), "--cluster-method", str(cluster_method)])

    inp = args.peaks_csv
    inp = Path(inp).expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"not a file: {inp}")

    stem_csv = inp.stem
    stem = stem_csv[: -len("_peaks")] if stem_csv.endswith("_peaks") else stem_csv
    out_dir = resolve_output_directory(output)

    xs, ts, ints = read_peaks_csv(inp)
    n_peaks = int(xs.shape[0])
    dest = out_dir / f"{stem}_tracks.csv"
    h5 = resolve_preprocessed_h5(inp, preprocessed_h5)

    common_kw = dict(
        dbscan_eps=float(cluster_dbscan_eps),
        dbscan_min_samples=int(cluster_dbscan_min_samples),
        k_min=cluster_min_k,
        k_max=cluster_max_k,
        k_penalty=float(cluster_k_penalty),
        n_init=int(cluster_init),
        plot_title=str(inp.name),
    )

    def run_overlay(
        stacked_arr: np.ndarray,
        trails_list: list[list[int]],
        nc: int | None,
        sil: float | None,
        method_tag: str,
        out_png: Path,
    ) -> None:
        if h5 is None:
            return
        note = f"{len(trails_list)} particles, {stacked_arr.shape[0]} peaks"
        if nc is not None:
            if method_tag == "dbscan":
                note += (
                    f" (DBSCAN: {nc} clusters, "
                    f"eps={float(cluster_dbscan_eps):g}, "
                    f"min_samples={int(cluster_dbscan_min_samples)}"
                )
            else:
                note += f" (k-means auto K={nc}"
            if sil is not None:
                note += f", sil={sil:.3f}"
            note += f", from {n_peaks} peaks)"
        plot_tracks_overlay(
            h5,
            stacked_arr,
            out_png=out_png,
            title_note=note,
        )

    if cluster_method == "both":
        if no_cluster_plot:
            plot_db = None
            plot_km = None
            plot_winner: Path | None = None
        elif cluster_plot is not None:
            bp = Path(cluster_plot).expanduser().resolve()
            if not bp.suffix:
                bp = bp.with_suffix(".png")
            plot_winner = bp
            plot_db = bp.with_name(f"{bp.stem}_dbscan{bp.suffix}")
            plot_km = bp.with_name(f"{bp.stem}_kmeans{bp.suffix}")
        else:
            plot_db = out_dir / f"{stem}_particle_clusters_dbscan.png"
            plot_km = out_dir / f"{stem}_particle_clusters_kmeans.png"
            plot_winner = out_dir / f"{stem}_particle_clusters.png"

        trails_db, nc_db, sil_db = cluster_peaks_to_particle_trails(
            xs,
            ts,
            ints,
            method="dbscan",
            cluster_plot_path=plot_db,
            **common_kw,
        )
        trails_km, nc_km, sil_km = cluster_peaks_to_particle_trails(
            xs,
            ts,
            ints,
            method="kmeans",
            cluster_plot_path=plot_km,
            **common_kw,
        )

        stacked_db = _lexsort_stacked(trails_to_rows(trails_db, xs, ts, ints))
        stacked_km = _lexsort_stacked(trails_to_rows(trails_km, xs, ts, ints))
        dest_db = out_dir / f"{stem}_tracks_dbscan.csv"
        dest_km = out_dir / f"{stem}_tracks_kmeans.csv"
        write_tracks_csv(dest_db, stacked_db)
        write_tracks_csv(dest_km, stacked_km)
        print(f"Wrote {dest_db.resolve()}")
        print(f"Wrote {dest_km.resolve()}")

        winner = _pick_winner_by_silhouette(sil_db, sil_km)
        if winner == "dbscan":
            trails, stacked, nc_win, sil_win = trails_db, stacked_db, nc_db, sil_db
        else:
            trails, stacked, nc_win, sil_win = trails_km, stacked_km, nc_km, sil_km

        write_tracks_csv(dest, stacked)

        if plot_winner is not None and plot_db is not None and plot_km is not None:
            shutil.copy2(plot_db if winner == "dbscan" else plot_km, plot_winner)
            print(f"Wrote {plot_winner.resolve()} (winner={winner})")

        if not no_overlay:
            if h5 is None:
                print(
                    "Skipping overlay: no preprocessed .h5 found (use --preprocessed-h5 PATH).",
                    file=sys.stderr,
                )
            else:
                o_db = out_dir / f"{stem}_tracks_overlay_dbscan.png"
                o_km = out_dir / f"{stem}_tracks_overlay_kmeans.png"
                o_win = out_dir / f"{stem}_tracks_overlay.png"
                run_overlay(stacked_db, trails_db, nc_db, sil_db, "dbscan", o_db)
                run_overlay(stacked_km, trails_km, nc_km, sil_km, "kmeans", o_km)
                shutil.copy2(o_db if winner == "dbscan" else o_km, o_win)
                print(f"Wrote {o_win.resolve()} (winner={winner})")

        print(
            "Compared: "
            f"dbscan clusters={nc_db} {_sil_txt(sil_db)} | "
            f"k-means K={nc_km} {_sil_txt(sil_km)} -> picked {winner}"
        )
        msg = (
            f"Wrote {dest.resolve()} - {len(trails)} ids (winner={winner}), "
            f"{stacked.shape[0]} peaks, lengths (desc, top<=10)= "
            f"{sorted((len(t) for t in trails), reverse=True)[:min(10, len(trails))]}"
        )
        if nc_win is not None:
            sil_txt = f", sil={sil_win:.3f}" if sil_win is not None else ""
            if winner == "dbscan":
                msg += (
                    f" | peak_cluster (dbscan) clusters={nc_win}, "
                    f"eps={float(cluster_dbscan_eps):g}, "
                    f"min_samples={int(cluster_dbscan_min_samples)}"
                    f"{sil_txt} ({n_peaks} peaks)"
                )
            else:
                msg += (
                    f" | peak_cluster (k-means auto) K={nc_win}{sil_txt} ({n_peaks} peaks)"
                )
        print(msg)
        return dest

    plot_path: Path | None = None
    if not no_cluster_plot:
        if cluster_plot is not None:
            plot_path = Path(cluster_plot).expanduser().resolve()
        else:
            plot_path = out_dir / f"{stem}_particle_clusters.png"

    trails, cluster_k_out, cluster_sil = cluster_peaks_to_particle_trails(
        xs,
        ts,
        ints,
        method=cluster_method,
        cluster_plot_path=plot_path,
        **common_kw,
    )

    stacked = _lexsort_stacked(trails_to_rows(trails, xs, ts, ints))
    write_tracks_csv(dest, stacked)

    if not no_overlay:
        if h5 is None:
            print(
                "Skipping overlay: no preprocessed .h5 found (use --preprocessed-h5 PATH).",
                file=sys.stderr,
            )
        else:
            run_overlay(
                stacked,
                trails,
                cluster_k_out,
                cluster_sil,
                cluster_method,
                out_dir / f"{stem}_tracks_overlay.png",
            )

    preview = sorted((len(t) for t in trails), reverse=True)
    msg = (
        f"Wrote {dest.resolve()} - {len(trails)} ids, "
        f"{stacked.shape[0]} peaks, lengths (desc, top<=10)= {preview[:min(10, len(preview))]}"
    )
    if cluster_k_out is not None:
        sil_txt = f", sil={cluster_sil:.3f}" if cluster_sil is not None else ""
        if cluster_method == "dbscan":
            msg += (
                f" | peak_cluster (dbscan) clusters={cluster_k_out}, "
                f"eps={float(cluster_dbscan_eps):g}, "
                f"min_samples={int(cluster_dbscan_min_samples)}"
                f"{sil_txt} ({n_peaks} peaks)"
            )
        else:
            msg += (
                f" | peak_cluster (k-means auto) K={cluster_k_out}{sil_txt} "
                f"({n_peaks} peaks)"
            )
    print(msg)
    return dest


# ---------------------------------------------------------------------------
# Existing command entrypoint (Typer)


def run_command(
    raw_h5: Path,
    output: Path = DEFAULT_PLOTS_DIR,
    max_time: int = 1024,
    cluster_method: str = "both",
    overlay: bool = True,
) -> None:
    """Run end-to-end: preprocess peaks, cluster to tracks, optional overlay."""
    raw_path = raw_h5.expanduser().resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"not a file: {raw_path}")
    if max_time < 1:
        raise ValueError("--max-time must be >= 1")
    if cluster_method not in {"both", "dbscan", "kmeans"}:
        raise ValueError("--cluster-method must be both, dbscan, or kmeans")

    out_dir = resolve_output_directory(output)
    stem = raw_path.stem

    raw, _ = load_kymograph(raw_path, dataset_name=DEFAULT_KYMOGRAPH_DATASET, max_time=max_time)
    raw_png = out_dir / f"{stem}_raw.png"
    if not raw_png.exists():
        write_raw_preview(
            raw,
            raw_png,
            title=(
                f"{raw_path.name}\n"
                f"Cropped raw ({tuple(raw.shape)}) dataset={DEFAULT_KYMOGRAPH_DATASET}"
            ),
        )
        print(f"Wrote {raw_png.resolve()}")

    detail, _ = wavelet_detail_residual(
        raw,
        wavelet="db4",
        wavelet_level=4,
    )
    residual_sq = squared_residual_gaussian(detail)

    pre_h5 = out_dir / f"{stem}_{METHOD}_preprocessed.h5"
    with h5py.File(pre_h5, "w") as fw:
        fw.create_dataset("kymograph", data=detail.astype(np.float32, copy=False))
        fw.create_dataset(
            RESIDUAL_SQ_GAUSSIAN_DATASET,
            data=residual_sq.astype(np.float32, copy=False),
        )
    pre_png = out_dir / f"{stem}_{METHOD}_preprocessed.png"
    write_kymograph_png(
        residual_sq,
        pre_png,
        title=(
            f"{raw_path.name}\n"
            f"{METHOD} preprocessing: residual^2 smoothed Gaussian σ_x={RESIDUAL_SQ_GAUSSIAN_SIGMA:g}\n"
            f"wavelet=db4 · level=4 · dataset={DEFAULT_KYMOGRAPH_DATASET}"
        ),
    )
    print(f"Wrote {pre_h5.resolve()} - residual^2 map")

    pt, px = gather_peaks_rowwise(
        residual_sq,
        rel_prominence=0.06,
        distance=8,
    )
    peaks_csv = out_dir / f"{stem}_{METHOD}_peaks.csv"
    intensity = residual_sq[pt.astype(np.int64), px.astype(np.int64)] if pt.size else np.array([])
    write_peaks_csv(
        peaks_csv,
        x=px.astype(np.float64),
        t=pt.astype(np.int64),
        intensity=intensity.astype(np.float64, copy=False),
    )
    print(f"Wrote {peaks_csv.resolve()} — {int(pt.size)} peaks")

    xs = px.astype(np.float64)
    ts = pt.astype(np.int64)
    ints = intensity.astype(np.float64, copy=False)

    common_kw = dict(
        dbscan_eps=1.0,
        dbscan_min_samples=4,
        k_min=2,
        k_max=12,
        k_penalty=0.045,
        n_init=24,
        plot_title=str(raw_path.name),
    )

    if cluster_method == "both":
        trails_db, nc_db, sil_db = cluster_peaks_to_particle_trails(
            xs,
            ts,
            ints,
            method="dbscan",
            **common_kw,
        )
        trails_km, nc_km, sil_km = cluster_peaks_to_particle_trails(
            xs,
            ts,
            ints,
            method="kmeans",
            **common_kw,
        )

        db_rows = trails_to_rows(trails_db, xs, ts, ints)
        km_rows = trails_to_rows(trails_km, xs, ts, ints)
        db_stacked = _lexsort_stacked(db_rows)
        km_stacked = _lexsort_stacked(km_rows)

        winner = _pick_winner_by_silhouette(sil_db, sil_km)
        if winner == "dbscan":
            tracks = db_stacked
            n_clusters = nc_db
            sil = sil_db
            winner_label = "dbscan"
        else:
            tracks = km_stacked
            n_clusters = nc_km
            sil = sil_km
            winner_label = "kmeans"

        msg_w = f"dbscan clusters={nc_db} sil={sil_db if sil_db is not None else float('nan'):.3g}"
        msg_k = f"kmeans clusters={nc_km} sil={sil_km if sil_km is not None else float('nan'):.3g}"
        print(f"{METHOD}: compared {msg_w} | {msg_k} -> winner={winner}")
    else:
        trails, n_clusters, sil = cluster_peaks_to_particle_trails(
            xs,
            ts,
            ints,
            method=cluster_method,
            **common_kw,
        )
        tracks = _lexsort_stacked(trails_to_rows(trails, xs, ts, ints))
        winner_label = cluster_method

    tracks_stacked = _lexsort_stacked(tracks)
    out_tracks = out_dir / f"{stem}_{METHOD}_tracks.csv"
    write_tracks_csv(out_tracks, tracks_stacked)
    print(f"Wrote {out_tracks.resolve()} — {int(n_clusters) if n_clusters is not None else None} ids")

    if overlay:
        overlay_png = out_dir / f"{stem}_{METHOD}_tracks_overlay.png"
        write_trajectory_overlay_png(
            residual_sq,
            tracks_stacked,
            overlay_png,
            title=(
                f"{raw_path.name} · {METHOD} trajectory overlay · "
                f"method={winner_label} id={int(n_clusters) if n_clusters is not None else 'n/a'} "
                f"sil={sil if sil is not None else float('nan'):.3g}"
            ),
        )
        print(f"Wrote {overlay_png.resolve()}")

    n_tracks = int(np.unique(tracks_stacked[:, 0]).size) if tracks_stacked.size else 0
    print(
        "Done: peak_cluster generated "
        f"{n_tracks} tracks with {int(tracks_stacked.shape[0])} points"
    )
