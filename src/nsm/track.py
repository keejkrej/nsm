"""Assign particle IDs by clustering peaks in `(t, x)` via PCA (`nsm-track`).

Peaks are centered in time–position, projected onto the **first two PCA axes** of
`(t, x)` only. **k-means** partitions peaks in that PC space (auto-``K`` uses the same
silhouette − λ·K rule unless ``--cluster-k`` fixes ``K``). Trajectories are peaks per
cluster sorted by ``t``. Intensity is **not** used for grouping (still written to CSV).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist, squareform

from nsm.kymograph_io import (
    DEFAULT_PLOTS_DIR,
    FIGSIZE_INCHES,
    IMAGE_CMAP,
    RESIDUAL_SQ_GAUSSIAN_DATASET,
    load_kymograph,
    resolve_output_directory,
)
from nsm.wavelet_residual import RESIDUAL_SQ_GAUSSIAN_SIGMA, y_axis_minmax


def resolve_preprocessed_h5(peaks_csv: Path, explicit: Path | None) -> Path | None:
    """Find ``*_preprocessed.h5`` next to peaks CSV unless ``explicit`` path is given."""
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
    """``stacked`` columns: id, x, t, intensity — scatter colored by particle id."""
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
        aspect="equal",
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
        f"Gaussian σ_x={RESIDUAL_SQ_GAUSSIAN_SIGMA:g} · particle id (tx-cluster) • {title_note}"
    )
    ax.set_title(f"{preprocessed_h5.name}\nsnm-track overlays • {prefix}")
    ax.set_xlabel("time (axis 0)")
    ax.set_ylabel("position (pixels)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png.resolve()}")


def read_peaks_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, t, intensity)`` arrays (integer frame indices ``t``)."""
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


def _silhouette_mean_euclidean(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (higher is better separated clusters)."""
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
    """Lowest-inertia labels among multi-seed ``kmeans2`` runs."""
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
    """PCA of centered `(t, x)`. Returns score matrix ``(n, min(2, d))`` and variance fractions."""
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
    chosen_k: int,
    silhouette: float | None,
    title_note: str,
) -> None:
    """PNG: ``t`` vs ``x`` assignments + PCA score scatter."""
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
    ax_tx.set_title(f"{title_note}\nPeak assignments in (t, x) · K={chosen_k}")
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
    ax_pc.set_title(
        "PCA(t,x) scores (raw); k-means uses z-scaled columns · " + suline
        if suline
        else "PCA(t,x) scores (raw); k-means uses z-scaled columns"
    )
    ax_pc.set_xlabel(xlab)
    ax_pc.set_ylabel(ylab)
    ax_pc.grid(True, alpha=0.35)

    fig.suptitle(
        "Particle ids from k-means on PCA scores of (t, x) — intensity excluded",
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
    k: int | None,
    k_min: int,
    k_max: int,
    k_penalty: float,
    n_init: int,
    cluster_plot_path: Path | None = None,
    plot_title: str = "",
) -> tuple[list[list[int]], int | None, float | None]:
    """Cluster every peak into particles; one trajectory list per cluster (indices sorted by ``t``).

    Features are **PCA scores** of centered ``(t, x)`` only (typically PC1 and PC2),
    z-scaled before k-means. Intensity is kept only for CSV output.
    """
    n = int(xs.shape[0])
    if n == 0:
        return [], None, None
    if n == 1:
        return [[0]], None, None

    scores, var_frac = _pca_tx_scores(ts, xs)
    mu = scores.mean(axis=0)
    sig = scores.std(axis=0)
    sig = np.where(sig < 1e-12, 1.0, sig)
    Xz = (scores - mu) / sig

    k_hi = min(k_max, n)
    k_lo = max(2, min(k_min, k_hi))
    if k_lo > k_hi:
        trails = [[i] for i in range(n)]
        return trails, None, None

    chosen_labels: np.ndarray | None = None
    chosen_k_eff: int | None = None
    chosen_sil: float | None = None

    if k is not None:
        k_eff = max(2, min(k, n))
        chosen_labels = _kmeans_best_labels(Xz, k_eff, n_init)
        chosen_k_eff = k_eff
        chosen_sil = _silhouette_mean_euclidean(Xz, chosen_labels)
    else:
        best_score = float("-inf")
        best_k_pick = k_lo
        best_labels_pick: np.ndarray | None = None
        best_sil_at_pick = float("-inf")
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
                best_sil_at_pick = sil_try
        chosen_labels = best_labels_pick
        chosen_k_eff = best_k_pick
        chosen_sil = (
            best_sil_at_pick if np.isfinite(best_sil_at_pick) else None
        )

    assert chosen_labels is not None and chosen_k_eff is not None
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
            chosen_k=int(chosen_k_eff),
            silhouette=chosen_sil,
            title_note=plot_title,
        )

    return trails, chosen_k_eff, chosen_sil


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
    """Write trajectory CSV: columns id,x,t,intensity."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "t", "intensity"])
        for r in rows:
            pid, xx, tt, iq = int(r[0]), float(r[1]), int(r[2]), float(r[3])
            w.writerow([pid, xx, tt, iq])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read peaks CSV (x,t,intensity) and assign particle ids by k-means on PCA scores "
            "of (t,x) only (intensity is not used for grouping). Writes *_tracks.csv and overlay PNG. "
            "Use nsm-diffusion on *_tracks.csv for diffusion estimates."
        )
    )
    parser.add_argument(
        "peaks_csv",
        type=Path,
        help="Peak list CSV with columns x,t,intensity",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Output directory for <stem>_tracks.csv and <stem>_tracks_overlay.png",
    )
    parser.add_argument(
        "--preprocessed-h5",
        type=Path,
        default=None,
        help=(
            "Preprocessed HDF5 for heatmap backdrop (loads residual_sq_gaussian). "
            "Default: sibling <basename>.h5 when peaks path ends with _peaks.csv"
        ),
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Do not write the PNG trajectory overlay",
    )
    parser.add_argument(
        "--cluster-k",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Fix particle count to K (≥2). If omitted, K is chosen automatically "
            "(silhouette − λ×K over [--cluster-min-k, --cluster-max-k])."
        ),
    )
    parser.add_argument(
        "--cluster-min-k",
        type=int,
        default=2,
        help="Minimum K when auto-selecting particle count (default: 2)",
    )
    parser.add_argument(
        "--cluster-max-k",
        type=int,
        default=12,
        help="Maximum K when auto-selecting (default: 12)",
    )
    parser.add_argument(
        "--cluster-k-penalty",
        type=float,
        default=0.045,
        help="Parsimony λ for auto-K: maximize silhouette − λ×K (default: 0.045)",
    )
    parser.add_argument(
        "--cluster-init",
        type=int,
        default=48,
        help="Random seeds for k-means (default: 48)",
    )
    parser.add_argument(
        "--cluster-plot",
        type=Path,
        default=None,
        metavar="PNG",
        help=(
            "Write clustering diagnostic PNG here. "
            "Default: <stem>_particle_clusters.png under -o."
        ),
    )
    parser.add_argument(
        "--no-cluster-plot",
        action="store_true",
        help="Skip clustering diagnostic PNG.",
    )
    args = parser.parse_args()

    if args.cluster_k is not None and args.cluster_k < 2:
        parser.error("--cluster-k must be >= 2 when given")
    if args.cluster_min_k < 2:
        parser.error("--cluster-min-k must be >= 2")
    if args.cluster_max_k < args.cluster_min_k:
        parser.error("--cluster-max-k must be >= --cluster-min-k")
    if args.cluster_k_penalty < 0:
        parser.error("--cluster-k-penalty must be >= 0")
    if args.cluster_init < 1:
        parser.error("--cluster-init must be >= 1")

    inp = args.peaks_csv.expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"not a file: {inp}")

    stem_csv = inp.stem
    stem = stem_csv[: -len("_peaks")] if stem_csv.endswith("_peaks") else stem_csv
    out_dir = resolve_output_directory(args.output)

    plot_path: Path | None = None
    if args.cluster_plot is not None:
        plot_path = args.cluster_plot.expanduser().resolve()
    elif not args.no_cluster_plot:
        plot_path = out_dir / f"{stem}_particle_clusters.png"

    xs, ts, ints = read_peaks_csv(inp)
    n_peaks = int(xs.shape[0])
    trails, cluster_k_out, cluster_sil = cluster_peaks_to_particle_trails(
        xs,
        ts,
        ints,
        k=args.cluster_k,
        k_min=args.cluster_min_k,
        k_max=args.cluster_max_k,
        k_penalty=args.cluster_k_penalty,
        n_init=args.cluster_init,
        cluster_plot_path=plot_path,
        plot_title=str(inp.name),
    )

    stacked = trails_to_rows(trails, xs, ts, ints)
    order = np.lexsort((stacked[:, 3], stacked[:, 1], stacked[:, 2], stacked[:, 0]))
    stacked = stacked[order]

    dest = out_dir / f"{stem}_tracks.csv"
    write_tracks_csv(dest, stacked)

    h5 = resolve_preprocessed_h5(inp, args.preprocessed_h5)
    if not args.no_overlay:
        if h5 is None:
            print(
                "Skipping overlay: no preprocessed .h5 found (use --preprocessed-h5 PATH).",
                file=sys.stderr,
            )
        else:
            note = f"{len(trails)} particles, {stacked.shape[0]} peaks"
            if cluster_k_out is not None:
                note += f" (cluster K={cluster_k_out}"
                if cluster_sil is not None:
                    note += f", sil={cluster_sil:.3f}"
                note += f", from {n_peaks} peaks)"
            plot_tracks_overlay(
                h5,
                stacked,
                out_png=out_dir / f"{stem}_tracks_overlay.png",
                title_note=note,
            )

    lengths = sorted((len(t) for t in trails), reverse=True)
    preview = lengths[: min(10, len(lengths))]
    msg = (
        f"Wrote {dest.resolve()} - {len(trails)} ids, "
        f"{stacked.shape[0]} peaks, lengths (desc, top<=10)= {preview}"
    )
    if cluster_k_out is not None:
        sil_txt = f", sil={cluster_sil:.3f}" if cluster_sil is not None else ""
        mode = "fixed" if args.cluster_k is not None else "auto"
        msg += (
            f" | tx-cluster ({mode}) K={cluster_k_out}{sil_txt} "
            f"({n_peaks} peaks)"
        )
    print(msg)


if __name__ == "__main__":
    main()
