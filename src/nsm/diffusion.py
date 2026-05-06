"""Squared displacement vs lag scatter and 1D diffusion estimate from tracks (`nsm-diffusion`)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nsm.kymograph_io import DEFAULT_PLOTS_DIR, FIGSIZE_INCHES, resolve_output_directory


def resolve_tracks_csv(path: Path) -> Path:
    """Use ``*_tracks.csv``; if ``*_peaks.csv`` is given, use sibling tracks file."""
    p = path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"not a file: {p}")
    stem = p.stem
    if stem.endswith("_tracks"):
        return p
    if stem.endswith("_peaks"):
        cand = p.parent / f"{stem[: -len('_peaks')]}_tracks.csv"
        if cand.is_file():
            return cand
        raise FileNotFoundError(
            f"{p.name}: expected sibling {cand.name!r} — run nsm-track on peaks first"
        )
    return p


def read_tracks_csv(path: Path) -> np.ndarray:
    """Return float array shape ``(n, 4)`` columns id, x, t, intensity."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path.name}: empty file")
    header = [h.strip().lower() for h in rows[0]]
    need = {"id", "x", "t"}
    if not need <= set(header):
        raise ValueError(
            f"{path.name}: need columns id, x, t (optional intensity); got {header!r}"
        )
    idi, xi, ti = header.index("id"), header.index("x"), header.index("t")
    rows_out: list[list[float]] = []
    for ln in rows[1:]:
        if not ln or all(not str(c).strip() for c in ln):
            continue
        try:
            pid = int(float(ln[idi]))
            x = float(ln[xi])
            t_raw = float(ln[ti])
        except (ValueError, IndexError):
            raise ValueError(f"{path.name}: bad row {ln!r}") from None
        t_int = int(round(t_raw))
        if abs(t_raw - t_int) > 1e-6:
            raise ValueError(
                f"{path.name}: column t must be integer frame indices; got {t_raw!r}"
            )
        rows_out.append([float(pid), x, t_int, 0.0])
    if not rows_out:
        raise ValueError(f"{path.name}: no numeric rows below header")
    return np.asarray(rows_out, dtype=np.float64)


def drift_ols_fit(
    x: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """OLS ``x ≈ v t + b`` on full track. Returns ``(residual, x_fitted, v, b, R²_x)``.

    Residual is **trajectory minus drift**: ``x_raw - (vt + b)``, used for Δx² vs lag pairs.
    """
    if x.size < 2:
        xx = x.astype(np.float64, copy=False)
        return xx.copy(), xx.copy(), float("nan"), float("nan"), float("nan")
    tt = t.astype(np.float64, copy=False)
    xx = x.astype(np.float64, copy=False)
    A = np.column_stack((tt, np.ones(tt.shape[0])))
    coef, _, _, _ = np.linalg.lstsq(A, xx, rcond=None)
    v_px_per_frame = float(coef[0])
    intercept = float(coef[1])
    x_fit = (A @ coef).astype(np.float64, copy=False)
    residual = xx - x_fit
    xm = float(np.mean(xx))
    ss_tot = float(np.dot(xx - xm, xx - xm))
    ss_res = float(np.dot(residual, residual))
    r2_traj = (
        float("nan")
        if ss_tot <= 0.0 or not np.isfinite(ss_res)
        else 1.0 - ss_res / ss_tot
    )
    return residual, x_fit, v_px_per_frame, intercept, r2_traj


def track_arrays_by_id(rows: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Grouped by trajectory id → ``(x, t)`` sorted by time with duplicate frames collapsed."""
    ids = rows[:, 0].astype(np.int64, copy=False)
    uniq = np.unique(ids)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for tid in uniq.tolist():
        m = ids == tid
        sub = rows[m]
        order = np.lexsort((sub[:, 1], sub[:, 2]))
        xt = sub[order]
        ts = xt[:, 2].astype(np.int64, copy=False)
        xs = xt[:, 1].astype(np.float64, copy=False)
        if ts.size <= 1:
            out[int(tid)] = (xs, ts)
            continue
        uniq_t, inv = np.unique(ts, return_inverse=True)
        if uniq_t.size == ts.size:
            out[int(tid)] = (xs, ts)
            continue
        x_mean = np.bincount(inv, weights=xs) / np.bincount(inv)
        out[int(tid)] = (x_mean.astype(np.float64), uniq_t.astype(np.int64))
    return out


def lag_squared_displacements(
    x: np.ndarray,
    t: np.ndarray,
    *,
    max_lag: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """All ordered pairs with positive integer lag; returns ``(lag, Δx²)``."""
    n = int(t.size)
    if n < 2:
        return np.array([]), np.array([])
    ii, jj = np.triu_indices(n, k=1)
    dt = t[jj].astype(np.float64) - t[ii].astype(np.float64)
    ok = dt >= 1.0
    if max_lag is not None:
        ok &= dt <= float(max_lag)
    ii, jj = ii[ok], jj[ok]
    dt = dt[ok]
    dx = x[jj] - x[ii]
    dsq = dx * dx
    return dt, dsq


def per_lag_mean_delta_x_sq(
    lag: np.ndarray, dsq: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per integer lag τ: arithmetic mean of all pair ``Δx²`` values, and pair count ``n``."""
    li = lag.astype(np.int64, copy=False)
    uniq = np.unique(li)
    if uniq.size == 0:
        return np.array([]), np.array([]), np.array([])
    means: list[float] = []
    counts: list[int] = []
    for t in uniq:
        msk = li == int(t)
        means.append(float(np.mean(dsq[msk])))
        counts.append(int(np.sum(msk)))
    return (
        uniq.astype(np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
    )


def diffusion_D_from_pairs(
    lag: np.ndarray, dsq: np.ndarray, *, max_fit_lag: int | None
) -> tuple[float, float, float]:
    """``D`` from per-lag mean ``Δx²``; fit ``⟨Δx²⟩_τ ≈ 2 D τ`` weighted by pair count."""
    tau, ms, w = per_lag_mean_delta_x_sq(lag, dsq)
    if max_fit_lag is not None:
        m = tau <= float(max_fit_lag)
        tau, ms, w = tau[m], ms[m], w[m]
    if tau.size < 2:
        return float("nan"), float("nan"), float("nan")
    num = float(np.sum(w * tau * ms))
    den = float(np.sum(w * tau * tau))
    if den <= 0.0 or not np.isfinite(num):
        return float("nan"), float("nan"), float("nan")
    m = num / den
    D = 0.5 * m
    yhat = m * tau
    resid = ms - yhat
    ss_res = float(np.dot(w, resid * resid))
    ybar = float(np.sum(w * ms) / np.sum(w))
    ss_tot = float(np.dot(w, (ms - ybar) ** 2))
    if ss_tot <= 0.0 or not np.isfinite(ss_res):
        r2 = float("nan")
    else:
        r2 = 1.0 - ss_res / ss_tot
    return D, m, r2


def physical_D_um2_s(D_px2_per_frame: float, *, pixel_um: float, dt_s: float) -> float:
    return D_px2_per_frame * (pixel_um**2) / dt_s


def write_one_track_diffusion_png(
    *,
    src_name: str,
    stem: str,
    row: tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        int,
    ],
    out_dir: Path,
    scatter_cap: int,
    pixel_um: float | None,
    dt_s: float | None,
    rng: np.random.Generator,
) -> Path:
    """Two-panel figure for one trajectory; returns path written."""
    tid, ts, xs, x_fit, v_drift, r2_drift, lag, dsq, D, m, r2, fit_cap = row
    tid_int = int(tid)

    fig_w = min(13.5, FIGSIZE_INCHES[0] * 1.35)
    fig_h = min(11.0, FIGSIZE_INCHES[1] * 1.12)
    fig, axes_arr = plt.subplots(
        1,
        2,
        figsize=(fig_w, fig_h),
        squeeze=False,
        layout="constrained",
    )
    ax_left = axes_arr[0, 0]
    ax_ms = axes_arr[0, 1]
    cap = int(scatter_cap)

    nx = int(xs.size)
    travis: slice | np.ndarray = slice(None)
    if nx > cap:
        travis = np.sort(rng.choice(nx, size=cap, replace=False))
    ax_left.scatter(ts[travis], xs[travis], s=7.0, alpha=0.45, edgecolors="none")
    if np.isfinite(v_drift) and xs.size >= 2:
        ax_left.plot(
            ts.astype(np.float64),
            x_fit,
            color="C3",
            linewidth=1.65,
            label=r"$x \approx vt + b$",
        )
        ax_left.legend(loc="best", fontsize=7)
    v_txt = f"{v_drift:.4g}" if np.isfinite(v_drift) else "n/a"
    rd_txt = f"{r2_drift:.3f}" if np.isfinite(r2_drift) else "n/a"
    ax_left.set_title(
        f"id {tid_int} · drift · v={v_txt} px/frame · $R_x^2$={rd_txt}",
        fontsize=9,
    )
    ax_left.set_xlabel("t (frames)")
    ax_left.set_ylabel(r"$x$ (px)")
    ax_left.grid(True, alpha=0.35)

    vis: slice | np.ndarray = slice(None)
    if lag.size > cap:
        vis = rng.choice(lag.size, size=cap, replace=False)
    ax_ms.scatter(
        lag[vis],
        dsq[vis],
        s=5.0,
        alpha=0.12,
        edgecolors="none",
        color="0.3",
        label=r"pair $\Delta x^2$",
        zorder=1,
    )
    tau_m, ms_m, _w_m = per_lag_mean_delta_x_sq(lag, dsq)
    if tau_m.size:
        fit_span = float(min(float(fit_cap), float(np.max(tau_m))))
        ax_ms.axvspan(
            0.0,
            fit_span,
            alpha=0.08,
            color="0.5",
            zorder=0,
            linewidth=0,
        )
        ax_ms.plot(
            tau_m,
            ms_m,
            color="C2",
            linewidth=1.35,
            alpha=0.92,
            zorder=4,
            label=r"mean $\langle\Delta x^2\rangle_\tau$",
        )
        ax_ms.scatter(
            tau_m,
            ms_m,
            s=18.0,
            facecolors="C2",
            edgecolors="black",
            linewidths=0.35,
            zorder=5,
            label="_nolegend_",
        )
    t_line_max = float(np.max(lag)) if lag.size else 0.0
    if np.isfinite(D) and np.isfinite(m) and t_line_max > 0:
        xs_line = np.array([0.0, t_line_max], dtype=np.float64)
        ax_ms.plot(
            xs_line,
            m * xs_line,
            color="C3",
            linewidth=1.75,
            zorder=6,
            label=rf"fit $2D\tau$ ($\tau\leq{fit_cap}$)",
        )
    unit = "µm²/s" if pixel_um is not None else "px²/frame"
    d_show = D
    if pixel_um is not None and np.isfinite(D):
        d_show = physical_D_um2_s(D, pixel_um=pixel_um, dt_s=float(dt_s))
    d_txt = f"{d_show:.3g} {unit}" if np.isfinite(d_show) else "n/a"
    r_txt = f"{r2:.3f}" if np.isfinite(r2) else "n/a"
    ax_ms.set_title(
        rf"id {tid_int} · $\Delta x^2$ vs $\tau$: pairs + mean per $\tau$ + $2D\tau$ fit · "
        f"D≈{d_txt} · $R^2$={r_txt}",
        fontsize=9,
    )
    ax_ms.set_xlabel("lag (frames)")
    ax_ms.set_ylabel(r"$\Delta x^2$ (px², after drift removal)")
    ax_ms.grid(True, alpha=0.35)
    ax_ms.legend(loc="upper left", fontsize=7, framealpha=0.93)

    title_bits = [
        f"{src_name} · track id {tid_int}",
        "nsm-diffusion · drift (left); pairs + per-τ mean + 2Dτ fit (right)",
    ]
    if pixel_um is not None:
        title_bits.append(f"calibrated D: {pixel_um:g} µm/px, {float(dt_s):g} s/frame")
    fig.suptitle("\n".join(title_bits), fontsize=11)

    png_path = out_dir / f"{stem}_diffusion_id{tid_int}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read trajectory CSV (id,x,t from nsm-track). Fit OLS drift x≈vt+b; pairwise "
            "Δx² vs lag on residuals (faint cloud); D from weighted per-lag ⟨Δx²⟩τ≈2Dτ. "
            "Writes `{stem}_diffusion.csv` plus `{stem}_diffusion_id#.png` per track."
        )
    )
    parser.add_argument(
        "tracks_or_peaks_csv",
        type=Path,
        help=(
            "Tracks CSV (id,x,t,intensity) or peaks CSV (resolves sibling *_tracks.csv)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Output directory for summary CSV and one PNG per track",
    )
    parser.add_argument(
        "--max-fit-lag",
        type=int,
        default=None,
        help=(
            "Use only lag ≤ this value when fitting ⟨Δx²⟩≈2Dτ (weighted per-lag means). "
            "Default for each track: 25 %% of its frame span (t_max−t_min), at least 1."
        ),
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=None,
        help="Cap frame lag for pairs (default: no cap)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=3,
        help="Skip tracks with fewer than this many time points (default: 3)",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=4,
        help="Skip tracks with fewer displacement pairs (default: 4)",
    )
    parser.add_argument(
        "--max-plots",
        "--max-subplots",
        dest="max_plots",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap how many PNGs to write (longest tracks first by pair count); "
            "omit for all qualifying tracks"
        ),
    )
    parser.add_argument(
        "--pixel-um",
        type=float,
        default=None,
        help="Pixel size in µm — with --dt-s, reports D in µm²/s",
    )
    parser.add_argument(
        "--dt-s",
        type=float,
        default=None,
        help="Time per frame in seconds — with --pixel-um, reports D in µm²/s",
    )
    parser.add_argument(
        "--scatter-cap",
        type=int,
        default=6000,
        help="Max pairwise points drawn in the Δx² cloud per track",
    )
    args = parser.parse_args()

    if args.min_frames < 2:
        parser.error("--min-frames must be >= 2")
    if args.min_pairs < 1:
        parser.error("--min-pairs must be >= 1")
    if args.max_plots is not None and args.max_plots < 1:
        parser.error("--max-plots must be >= 1 when set")
    if args.max_fit_lag is not None and args.max_fit_lag < 1:
        parser.error("--max-fit-lag must be >= 1 when set")
    if args.max_lag is not None and args.max_lag < 1:
        parser.error("--max-lag must be >= 1 when set")
    if (args.pixel_um is None) ^ (args.dt_s is None):
        parser.error("set both --pixel-um and --dt-s for physical D, or neither")
    if args.pixel_um is not None and args.pixel_um <= 0:
        parser.error("--pixel-um must be > 0")
    if args.dt_s is not None and args.dt_s <= 0:
        parser.error("--dt-s must be > 0")

    src = resolve_tracks_csv(args.tracks_or_peaks_csv)
    rows = read_tracks_csv(src)
    by_id = track_arrays_by_id(rows)

    out_dir = resolve_output_directory(args.output)
    stem = src.stem
    if stem.endswith("_tracks"):
        stem = stem[: -len("_tracks")]
    csv_path = out_dir / f"{stem}_diffusion.csv"

    records: list[
        tuple[int, int, int, float, float, float | None, str]
    ] = []  # id, frames, pairs, D, R2, D_phys?, note

    plot_candidates: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            float,
            np.ndarray,
            np.ndarray,
            float,
            float,
            float,
            int,
        ]
    ] = []
    # tuple: tid, ts, xs, x_fit, v_drift, r2_drift, lag, dsq, D, m, r2, fit_cap

    total_warn_dup = False
    for tid, (xs, ts) in sorted(by_id.items()):
        note = ""
        if ts.size < int(args.min_frames):
            records.append((tid, int(ts.size), 0, float("nan"), float("nan"), None, "< min-frames"))
            continue
        if ts.size != rows[rows[:, 0] == tid].shape[0]:
            total_warn_dup = True
        _x_resid, x_fit, v_drift, _b_drift, r2_drift = drift_ols_fit(xs, ts)
        lag, dsq = lag_squared_displacements(_x_resid, ts, max_lag=args.max_lag)
        n_pairs = int(lag.size)
        if n_pairs < int(args.min_pairs):
            records.append((tid, int(ts.size), n_pairs, float("nan"), float("nan"), None, "< min-pairs"))
            continue
        span = max(1, int(ts.max()) - int(ts.min()))
        fit_cap = (
            int(args.max_fit_lag)
            if args.max_fit_lag is not None
            else max(1, int(0.25 * span))
        )
        D, m, r2 = diffusion_D_from_pairs(lag, dsq, max_fit_lag=fit_cap)
        d_phys = None
        if args.pixel_um is not None and np.isfinite(D):
            d_phys = physical_D_um2_s(D, pixel_um=args.pixel_um, dt_s=float(args.dt_s))
        plot_candidates.append(
            (tid, ts, xs, x_fit, v_drift, r2_drift, lag, dsq, D, m, r2, fit_cap)
        )
        records.append((tid, int(ts.size), n_pairs, D, r2, d_phys, note))

    if total_warn_dup:
        print(
            "Note: duplicate frame indices in a track were averaged in x.",
            file=sys.stderr,
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if args.pixel_um is not None:
            w.writerow(
                [
                    "id",
                    "n_frames",
                    "n_pairs",
                    "D_px2_per_frame",
                    "R2_fit",
                    "D_um2_per_s",
                    "note",
                ]
            )
            for r in records:
                tid, nf, npair, D, r2, dph, note = r
                dph_s = "" if dph is None or not np.isfinite(dph) else f"{dph:.6g}"
                w.writerow(
                    [
                        tid,
                        nf,
                        npair,
                        f"{D:.6g}" if np.isfinite(D) else "",
                        f"{r2:.6g}" if np.isfinite(r2) else "",
                        dph_s,
                        note,
                    ]
                )
        else:
            w.writerow(["id", "n_frames", "n_pairs", "D_px2_per_frame", "R2_fit", "note"])
            for r in records:
                tid, nf, npair, D, r2, _dph, note = r
                w.writerow(
                    [
                        tid,
                        nf,
                        npair,
                        f"{D:.6g}" if np.isfinite(D) else "",
                        f"{r2:.6g}" if np.isfinite(r2) else "",
                        note,
                    ]
                )

    print(f"Wrote {csv_path.resolve()}")

    plot_candidates.sort(key=lambda row: -row[6].size)
    plot_rows = plot_candidates
    if args.max_plots is not None:
        plot_rows = plot_rows[: int(args.max_plots)]
    if not plot_rows:
        print(
            "No tracks met --min-frames / --min-pairs; skipping PNG.",
            file=sys.stderr,
        )
        return

    rng = np.random.default_rng(0)
    for row in plot_rows:
        p = write_one_track_diffusion_png(
            src_name=src.name,
            stem=stem,
            row=row,
            out_dir=out_dir,
            scatter_cap=int(args.scatter_cap),
            pixel_um=args.pixel_um,
            dt_s=args.dt_s,
            rng=rng,
        )
        print(f"Wrote {p.resolve()}")


if __name__ == "__main__":
    main()
