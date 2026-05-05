"""Link fragmented peak CSVs into trajectories (`nsm-track`)."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from nsm.kymograph_io import (
    DEFAULT_PLOTS_DIR,
    FIGSIZE_INCHES,
    IMAGE_CMAP,
    RESIDUAL_SQ_GAUSSIAN_DATASET,
    load_kymograph,
    resolve_output_directory,
)
from nsm.wavelet_residual import RESIDUAL_SQ_GAUSSIAN_SIGMA, y_axis_minmax

INF_COST = 1e18


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
    """``stacked`` columns: id, x, t, intensity — scatter colored by id on residual heatmap."""
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
        f"Gaussian σ_x={RESIDUAL_SQ_GAUSSIAN_SIGMA:g} · colored by linkage id • {title_note}"
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


def displacement_budget(max_pixel_step: float, lag: int, *, scaling: str) -> float:
    lag = max(1, lag)
    if scaling == "sqrt":
        return float(max_pixel_step) * float(np.sqrt(lag))
    if scaling == "linear":
        return float(max_pixel_step) * float(lag)
    raise ValueError(f"scaling must be sqrt|linear, got {scaling!r}")


def link_peaks(
    xs: np.ndarray,
    ts: np.ndarray,
    *,
    max_gap: int,
    max_pixel_step: float,
    scaling: str,
) -> list[list[int]]:
    """Attach peaks across frames via Hungarian assignment with gap-limited predecessors.

    Returns trail index lists referencing rows of ``xs`` / ``ts`` sorted by ``t``.
    """
    if xs.size == 0:
        return []

    raw_n = xs.shape[0]
    by_time: defaultdict[int, list[int]] = defaultdict(list)
    for pi in range(raw_n):
        by_time[int(ts[pi])].append(pi)

    trails: dict[int, list[int]] = {}
    active: dict[int, int] = {}
    next_tid = 0

    times_sorted = sorted(by_time.keys())
    budget = displacement_budget

    for tf in times_sorted:
        peak_ix = sorted(by_time[tf], key=lambda p: xs[p])

        active = {
            tid: pi
            for tid, pi in active.items()
            if tf - int(ts[pi]) <= max_gap
        }

        cand: list[tuple[int, int]] = []
        for tid, pi0 in active.items():
            lag = tf - int(ts[pi0])
            if lag < 1 or lag > max_gap:
                continue
            lim = budget(max_pixel_step, lag, scaling=scaling)
            x0 = float(xs[pi0])
            if any(abs(xs[pj] - x0) <= lim for pj in peak_ix):
                cand.append((tid, pi0))

        if cand and peak_ix:
            n_tr, n_p = len(cand), len(peak_ix)
            cmat = np.full((n_tr, n_p), INF_COST)
            for i, (_tid, pi0) in enumerate(cand):
                lag = tf - int(ts[pi0])
                lim = budget(max_pixel_step, lag, scaling=scaling)
                x0 = float(xs[pi0])
                for j, pj in enumerate(peak_ix):
                    dx = abs(float(xs[pj]) - x0)
                    if dx <= lim:
                        cmat[i, j] = (dx * dx) / float(max(1, lag))
            rows, cols = linear_sum_assignment(cmat)
            used_peaks: set[int] = set()

            for i, j in zip(rows.tolist(), cols.tolist()):
                if cmat[i, j] >= INF_COST * 0.5:
                    continue
                tid = cand[i][0]
                pj = peak_ix[j]
                used_peaks.add(pj)
                trail = trails[tid]
                trail.append(pj)
                active[tid] = pj

            for pj in peak_ix:
                if pj not in used_peaks:
                    tid = next_tid
                    next_tid += 1
                    trails[tid] = [pj]
                    active[tid] = pj

        elif peak_ix:
            for pj in peak_ix:
                tid = next_tid
                next_tid += 1
                trails[tid] = [pj]
                active[tid] = pj

    out: list[list[int]] = []
    for tid in sorted(trails):
        idxs = trails[tid]
        uniq_ordered = sorted(set(idxs), key=lambda ri: ts[ri])
        if len(uniq_ordered) >= 1:
            out.append(uniq_ordered)
    return out


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
    """Separate file from peaks CSV: adds ``id`` (trajectory index); columns id,x,t,intensity."""
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
            "Read peaks-only CSV (x,t,intensity) and write id,x,t,intensity tracks CSV "
            "plus a heatmap PNG with peaks colored by track id (requires sibling "
            "preprocessed .h5 or --preprocessed-h5)."
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
        help=(
            "Output directory for <stem>_tracks.csv and <stem>_tracks_overlay.png"
        ),
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=48,
        help="Maximum frame lag allowed between bridged peaks (default: 48)",
    )
    parser.add_argument(
        "--max-pixel-step",
        type=float,
        default=12.0,
        help="Displacement scale (px): budget grows ×√lag by default (default 12 px)",
    )
    parser.add_argument(
        "--scaling",
        choices=("sqrt", "linear"),
        default="sqrt",
        help="Displacement budget scales as sqrt(#frames) [default] vs linear multiply",
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
    args = parser.parse_args()

    if args.max_gap < 1:
        parser.error("--max-gap must be >= 1")
    if args.max_pixel_step <= 0:
        parser.error("--max-pixel-step must be > 0")

    inp = args.peaks_csv.expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"not a file: {inp}")

    xs, ts, ints = read_peaks_csv(inp)
    trails = link_peaks(
        xs,
        ts,
        max_gap=args.max_gap,
        max_pixel_step=args.max_pixel_step,
        scaling=args.scaling,
    )
    stacked = trails_to_rows(trails, xs, ts, ints)
    order = np.lexsort((stacked[:, 3], stacked[:, 1], stacked[:, 2], stacked[:, 0]))
    stacked = stacked[order]

    out_dir = resolve_output_directory(args.output)
    stem = inp.stem
    if stem.endswith("_peaks"):
        stem = stem[: -len("_peaks")]
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
            plot_tracks_overlay(
                h5,
                stacked,
                out_png=out_dir / f"{stem}_tracks_overlay.png",
                title_note=f"{len(trails)} tracks, {stacked.shape[0]} points",
            )

    lengths = sorted((len(t) for t in trails), reverse=True)
    preview = lengths[: min(10, len(lengths))]
    print(
        f"Wrote {dest.resolve()} — {len(trails)} ids, "
        f"{stacked.shape[0]} points, lengths (desc, top≤10)= {preview}"
    )


if __name__ == "__main__":
    main()
