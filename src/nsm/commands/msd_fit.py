"""MSD/MLE fit command implementation.

This module contains the full trajectory fitting pipeline that was previously
kept in ``src/nsm/diffusion_impl.py``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from nsm.commands.bayesian_fit import (
    consecutive_increments,
    fit_langevin_pymc,
    langevin_mle,
    summarize_langevin_idata,
)
from nsm.core import resolve_output_directory


# ---------------------------------------------------------------------------
# Utilities for loading and grouping trajectories


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


def track_arrays_by_id(rows: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Group rows by trajectory id and return ``(x, t)`` sorted by time.

    Duplicate time indices within a trajectory are averaged in place.
    """
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


def physical_D_um2_s(D_px2_per_frame: float, *, pixel_um: float, dt_s: float) -> float:
    return D_px2_per_frame * (pixel_um**2) / dt_s



def write_posterior_arviz_png(
    idata: az.InferenceData,
    *,
    src_name: str,
    stem: str,
    tid: int,
    out_dir: Path,
    pixel_um: float | None,
    dt_s: float | None,
) -> Path:
    """Write ArviZ joint/marginal posterior KDE for ``v`` and ``D``."""
    tid_int = int(tid)
    if (
        pixel_um is not None
        and dt_s is not None
        and "v_um_s" in idata.posterior
    ):
        var_names = ["v_um_s", "D_um2_s"]
    else:
        var_names = ["v", "D"]

    az.plot_pair(
        idata,
        var_names=var_names,
        kind="kde",
        divergences=True,
        figsize=(7.5, 6.8),
        textsize=9.5,
    )
    fig = plt.gcf()
    cap = "posterior: " + ", ".join(var_names) + " · PyMC NUTS + ArviZ"
    fig.suptitle(
        f"{src_name} · track id {tid_int}\n{cap}",
        fontsize=10,
        y=1.02,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}_diffusion_id{tid_int}_posterior.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path



def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read trajectory CSV (id,x,t from nsm-track). Estimate drift v and diffusion D "
            "from consecutive Gaussian increments (MLE). "
            "With --increment-bayes: PyMC NUTS posterior, CSV summaries, and ArviZ pair plots "
            "of v and D (PNG output requires this flag)."
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
        required=True,
        metavar="DIR",
        help="Output directory for summary CSV and posterior PNGs (required)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=3,
        help="Skip tracks with fewer than this many time points (default: 3)",
    )
    parser.add_argument(
        "--min-increments",
        type=int,
        default=1,
        help="Skip tracks with fewer consecutive increments (default: 1)",
    )
    parser.add_argument(
        "--max-plots",
        "--max-subplots",
        dest="max_plots",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap posterior PNGs (longest tracks first by frame count); only with --increment-bayes"
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
        "--increment-bayes",
        action="store_true",
        help=(
            "PyMC NUTS posterior on consecutive increments; adds columns to CSV and writes "
            "ArviZ `plot_pair` figures (`*_posterior.png`). Required for PNG output."
        ),
    )
    parser.add_argument(
        "--bayes-draws",
        type=int,
        default=1000,
        metavar="N",
        help="NUTS posterior draws per chain after tuning (default: 1000)",
    )
    parser.add_argument(
        "--bayes-warmup",
        type=int,
        default=1000,
        metavar="N",
        help="NUTS tuning steps (default: 1000)",
    )
    parser.add_argument(
        "--bayes-seed",
        type=int,
        default=0,
        help="RNG seed for increment-Bayes MCMC",
    )
    parser.add_argument(
        "--bayes-sigma-v",
        type=float,
        default=50.0,
        metavar="SIGMA",
        help="Gaussian prior sigma on drift v px/frame at t (default: 50, very wide)",
    )
    parser.add_argument(
        "--bayes-chains",
        type=int,
        default=4,
        metavar="C",
        help="Parallel NUTS chains (≥2 recommended for r̂ / ESS; default: 4)",
    )
    parser.add_argument(
        "--bayes-cores",
        type=int,
        default=1,
        metavar="J",
        help=(
            "Worker processes for sampling (default: 1 avoids multiprocessing overhead on small models)"
        ),
    )
    parser.add_argument(
        "--bayes-target-accept",
        type=float,
        default=0.92,
        metavar="P",
        help="NUTS target acceptance (default: 0.92; increase if many divergences)",
    )
    args = parser.parse_args(argv)

    if args.min_frames < 2:
        parser.error("--min-frames must be >= 2")
    if args.min_increments < 1:
        parser.error("--min-increments must be >= 1")
    if args.max_plots is not None and args.max_plots < 1:
        parser.error("--max-plots must be >= 1 when set")
    if (args.pixel_um is None) ^ (args.dt_s is None):
        parser.error("set both --pixel-um and --dt-s for physical D, or neither")
    if args.pixel_um is not None and args.pixel_um <= 0:
        parser.error("--pixel-um must be > 0")
    if args.dt_s is not None and args.dt_s <= 0:
        parser.error("--dt-s must be > 0")
    if args.bayes_draws < 50:
        parser.error("--bayes-draws must be >= 50")
    if args.bayes_warmup < 0:
        parser.error("--bayes-warmup must be >= 0")
    if args.bayes_sigma_v <= 0:
        parser.error("--bayes-sigma-v must be > 0")
    if args.bayes_chains < 1:
        parser.error("--bayes-chains must be >= 1")
    if args.bayes_cores < 1:
        parser.error("--bayes-cores must be >= 1")
    if not (0.5 < args.bayes_target_accept < 1.0):
        parser.error("--bayes-target-accept must be between 0.5 and 1.0")

    src = resolve_tracks_csv(args.tracks_or_peaks_csv)
    rows = read_tracks_csv(src)
    by_id = track_arrays_by_id(rows)

    out_dir = resolve_output_directory(args.output)
    stem = src.stem
    if stem.endswith("_tracks"):
        stem = stem[: -len("_tracks")]
    csv_path = out_dir / f"{stem}_diffusion.csv"

    records: list[tuple] = []

    posterior_plot_rows: list[tuple[int, int, az.InferenceData]] = []

    def empty_bayes() -> tuple[
        float, float, float, float, float, float, float, float, float, float
    ]:
        x = float("nan")
        return (x, x, x, x, x, x, x, x, x, x)

    total_warn_dup = False
    for tid, (xs, ts) in sorted(by_id.items()):
        note = ""
        (
            vb_m,
            vb_s,
            db_m,
            db_s,
            db_q05,
            db_q95,
            v_r,
            d_r,
            ess_v,
            ess_d,
        ) = empty_bayes()
        if ts.size < int(args.min_frames):
            records.append(
                (
                    tid,
                    int(ts.size),
                    0,
                    float("nan"),
                    float("nan"),
                    None,
                    "< min-frames",
                    *empty_bayes(),
                )
            )
            continue
        if ts.size != rows[rows[:, 0] == tid].shape[0]:
            total_warn_dup = True

        try:
            dx_i, dt_i = consecutive_increments(xs, ts)
        except ValueError:
            records.append(
                (
                    tid,
                    int(ts.size),
                    0,
                    float("nan"),
                    float("nan"),
                    None,
                    "non-increasing time",
                    *empty_bayes(),
                )
            )
            continue

        n_inc = int(dx_i.size)
        if n_inc < int(args.min_increments):
            records.append(
                (
                    tid,
                    int(ts.size),
                    n_inc,
                    float("nan"),
                    float("nan"),
                    None,
                    "< min-increments",
                    *empty_bayes(),
                )
            )
            continue

        v_mle, d_mle = langevin_mle(dx_i, dt_i)
        if not np.isfinite(d_mle):
            note = ((note + "; ") if note else "") + "D MLE nan"
            d_phys = None
        elif args.pixel_um is not None:
            d_phys = physical_D_um2_s(d_mle, pixel_um=float(args.pixel_um), dt_s=float(args.dt_s))
        else:
            d_phys = None

        if args.increment_bayes:
            try:
                if n_inc >= 2:
                    idata = fit_langevin_pymc(
                        dx_i,
                        dt_i,
                        draws=int(args.bayes_draws),
                        tune=int(args.bayes_warmup),
                        chains=int(args.bayes_chains),
                        cores=int(args.bayes_cores),
                        target_accept=float(args.bayes_target_accept),
                        sigma_v_prior=float(args.bayes_sigma_v),
                        pixel_um=float(args.pixel_um) if args.pixel_um is not None else None,
                        dt_s=float(args.dt_s) if args.dt_s is not None else None,
                        random_seed=int(args.bayes_seed),
                        progressbar=False,
                    )
                    summ = summarize_langevin_idata(idata)
                    vb_m, vb_s = summ.v_mean, summ.v_sd
                    db_m, db_s = summ.d_mean, summ.d_sd
                    db_q05, db_q95 = summ.d_q05, summ.d_q95
                    v_r, d_r = summ.v_r_hat, summ.d_r_hat
                    ess_v, ess_d = summ.ess_v, summ.ess_D
                    posterior_plot_rows.append((tid, int(ts.size), idata))
                else:
                    note = (note + "; " if note else "") + "<2 increments for bayes"
            except (ValueError, OSError, RuntimeError) as exc:
                note = (note + "; " if note else "") + f"increment-bayes: {exc}"

        records.append(
            (
                tid,
                int(ts.size),
                n_inc,
                v_mle,
                d_mle,
                d_phys,
                note,
                vb_m,
                vb_s,
                db_m,
                db_s,
                db_q05,
                db_q95,
                v_r,
                d_r,
                ess_v,
                ess_d,
            )
        )

    if total_warn_dup:
        print(
            "Note: duplicate frame indices in a track were averaged in x.",
            file=sys.stderr,
        )

    def _bayes_cells(
        vb_m: float,
        vb_s: float,
        db_m: float,
        db_s: float,
        db_q05: float,
        db_q95: float,
        v_r: float,
        d_r: float,
        ess_v: float,
        ess_d: float,
    ) -> list[str]:
        def g(x: float) -> str:
            return f"{x:.6g}" if np.isfinite(x) else ""

        return [
            g(vb_m),
            g(vb_s),
            g(db_m),
            g(db_s),
            g(db_q05),
            g(db_q95),
            g(v_r),
            g(d_r),
            g(ess_v),
            g(ess_d),
        ]

    bayes_cols = [
        "v_bayes_mean_px_per_frame",
        "v_bayes_sd_px_per_frame",
        "D_bayes_mean_px2_per_frame",
        "D_bayes_sd_px2_per_frame",
        "D_bayes_q05_px2_per_frame",
        "D_bayes_q95_px2_per_frame",
        "v_r_hat",
        "D_r_hat",
        "ess_bulk_v",
        "ess_bulk_D",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if args.pixel_um is not None:
            head = [
                "id",
                "n_frames",
                "n_increments",
                "v_mle_px_per_frame",
                "D_mle_px2_per_frame",
                "D_mle_um2_per_s",
                "note",
            ]
            if args.increment_bayes:
                head.extend(bayes_cols)
                head.append("D_bayes_mean_um2_per_s")
            w.writerow(head)
            for r in records:
                (
                    tid,
                    nf,
                    n_inc,
                    v_mle,
                    d_mle,
                    dph,
                    note,
                    vb_m,
                    vb_s,
                    db_m,
                    db_s,
                    db_q05,
                    db_q95,
                    v_r,
                    d_r,
                    ess_v,
                    ess_d,
                ) = r
                dph_s = "" if dph is None or not np.isfinite(dph) else f"{dph:.6g}"
                row = [
                    tid,
                    nf,
                    n_inc,
                    f"{v_mle:.6g}" if np.isfinite(v_mle) else "",
                    f"{d_mle:.6g}" if np.isfinite(d_mle) else "",
                    dph_s,
                    note,
                ]
                if args.increment_bayes:
                    row.extend(
                        _bayes_cells(
                            vb_m,
                            vb_s,
                            db_m,
                            db_s,
                            db_q05,
                            db_q95,
                            v_r,
                            d_r,
                            ess_v,
                            ess_d,
                        )
                    )
                    d_b_um = ""
                    if args.pixel_um is not None and np.isfinite(db_m):
                        d_b_um_val = physical_D_um2_s(
                            db_m, pixel_um=float(args.pixel_um), dt_s=float(args.dt_s)
                        )
                        d_b_um = f"{d_b_um_val:.6g}" if np.isfinite(d_b_um_val) else ""
                    row.append(d_b_um)
                w.writerow(row)
        else:
            head = [
                "id",
                "n_frames",
                "n_increments",
                "v_mle_px_per_frame",
                "D_mle_px2_per_frame",
                "note",
            ]
            if args.increment_bayes:
                head.extend(bayes_cols)
            w.writerow(head)
            for r in records:
                (
                    tid,
                    nf,
                    n_inc,
                    v_mle,
                    d_mle,
                    _dph,
                    note,
                    vb_m,
                    vb_s,
                    db_m,
                    db_s,
                    db_q05,
                    db_q95,
                    v_r,
                    d_r,
                    ess_v,
                    ess_d,
                ) = r
                row = [
                    tid,
                    nf,
                    n_inc,
                    f"{v_mle:.6g}" if np.isfinite(v_mle) else "",
                    f"{d_mle:.6g}" if np.isfinite(d_mle) else "",
                    note,
                ]
                if args.increment_bayes:
                    row.extend(
                        _bayes_cells(
                            vb_m,
                            vb_s,
                            db_m,
                            db_s,
                            db_q05,
                            db_q95,
                            v_r,
                            d_r,
                            ess_v,
                            ess_d,
                        )
                    )
                w.writerow(row)

    print(f"Wrote {csv_path.resolve()}")

    if not args.increment_bayes:
        print(
            "Note: posterior PNGs require --increment-bayes (PyMC NUTS + ArviZ).",
            file=sys.stderr,
        )
    else:
        posterior_plot_rows.sort(key=lambda r: -r[1])
        plot_rows = posterior_plot_rows
        if args.max_plots is not None:
            plot_rows = plot_rows[: int(args.max_plots)]
        if not plot_rows:
            print(
                "No posterior plots (no successful sampling runs).",
                file=sys.stderr,
            )
        else:
            for plot_tid, _nf, idata in plot_rows:
                try:
                    path = write_posterior_arviz_png(
                        idata,
                        src_name=src.name,
                        stem=stem,
                        tid=int(plot_tid),
                        out_dir=out_dir,
                        pixel_um=args.pixel_um,
                        dt_s=args.dt_s,
                    )
                    print(f"Wrote {path.resolve()}")
                except (ValueError, OSError, RuntimeError) as exc:
                    print(
                        f"Posterior plot skipped for id {plot_tid}: {exc}",
                        file=sys.stderr,
                    )


# ----------------------------------------------------------------------------
# CLI adapter expected by Typer command entrypoint


NAME = "msd-fit"
HELP = "Fit per-trajectory drift and diffusion from displacement increments (MLE)."


def run_command(
    tracks_or_peaks_csv: Path,
    output: Path,
    min_frames: int = 3,
) -> None:
    """Run the diffusion workflow with MLE-only settings."""
    run(
        [
            str(tracks_or_peaks_csv),
            "--output",
            str(output),
            "--min-frames",
            str(min_frames),
        ]
    )
