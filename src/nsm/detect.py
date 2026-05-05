"""Binary particle mask from preprocessed kymograph + illumination profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian, threshold_otsu, threshold_sauvola

from nsm.data import (
    DATASET_DEFAULT,
    DEFAULT_PLOTS_DIR,
    FIGSIZE_INCHES,
    IMAGE_CMAP,
    ILLUMINATION_DATASET,
    MAX_TIME_SAMPLES,
    PARTICLE_MASK_DATASET,
    resolve_output_directory,
    resolve_video_destination,
)
from nsm.lpdiff import y_axis_minmax
from nsm.movie import _FRAME_PX_WH, _frame_rgb, _to_video_frame, _write_movie

# Along-x Gaussian before Otsu/Sauvola: σ such that 2·truncate·σ ≈ 50 px (truncate=4 in skimage).
_BLUR_SIGMA_DEFAULT = 50.0 / (2.0 * 4.0)


def load_preprocessed_with_illumination(
    path: Path,
    *,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load residual kymograph ``(T, X)`` and 1-D illumination ``(X,)``."""
    path = path.expanduser().resolve()
    with h5py.File(path, "r") as hf:
        if dataset_name not in hf:
            raise KeyError(f"{path.name}: missing '{dataset_name}' dataset")
        if ILLUMINATION_DATASET not in hf:
            raise KeyError(
                f"{path.name}: missing '{ILLUMINATION_DATASET}' — "
                "use nsm-preprocess output .h5"
            )
        proc = np.asarray(hf[dataset_name][...], dtype=np.float32)
        illum = np.asarray(hf[ILLUMINATION_DATASET][...], dtype=np.float64)
        if illum.ndim != 1:
            raise ValueError(f"illumination must be 1-D, shape {illum.shape}")
        if proc.ndim != 2:
            raise ValueError(f"kymograph must be 2-D (T, X), shape {proc.shape}")
        if illum.shape[0] != proc.shape[1]:
            raise ValueError(
                f"illumination length {illum.shape[0]} != width {proc.shape[1]}"
            )
        return proc, illum


def energy_squared(proc: np.ndarray) -> np.ndarray:
    """Bipolar residuals → nonnegative energy."""
    return np.square(proc.astype(np.float64))


def illuminate_flatten_energy(
    energy: np.ndarray,
    illumination: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    """Compensate when contrast scales roughly with illumination (divide by illumin²).

    Columns with very low illumination are stabilized with ``eps``.
    """
    base = illumination**2
    denom = np.maximum(base, float(eps))
    return energy / denom


def gaussian_blur_along_x(z: np.ndarray, *, sigma: float) -> np.ndarray:
    """Smooth each time row along **x** only (σ in pixels; ``σ≤0`` returns ``z`` unchanged)."""
    if sigma <= 0:
        return z
    zf = np.asarray(z, dtype=np.float64)
    return gaussian(
        zf,
        sigma=(0.0, float(sigma)),
        mode="reflect",
        preserve_range=True,
        truncate=4.0,
    )


def clamp_odd_window(requested: int, width: int) -> int | None:
    """Largest usable odd Sauvola window, or None if width < 3."""
    if width < 3:
        return None
    wmax = width if width % 2 == 1 else width - 1
    w = max(3, requested | 1)
    return min(w, wmax)


def mask_rows_global_otsu(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One Otsu threshold per time slice (whole line). Returns ``mask, thresholds`` shaped (T,)."""
    t, _ = z.shape
    masks = np.zeros(z.shape, dtype=np.uint8)
    thresholds = np.full(t, np.nan, dtype=np.float64)
    for ti in range(t):
        row = z[ti]
        vmin, vmax = float(np.nanmin(row)), float(np.nanmax(row))
        if not np.isfinite(vmin) or vmax <= vmin + 1e-15:
            continue
        thresh = threshold_otsu(row)
        thresholds[ti] = thresh
        masks[ti] = (row > thresh).astype(np.uint8)
    return masks, thresholds


def mask_rows_sauvola(
    z: np.ndarray,
    *,
    window_size: int,
    k: float,
) -> tuple[np.ndarray, None]:
    """Sauvola threshold along **x** on each slice (handles local illumination / gain).

    Threshold at each pixel depends on mean and variance in a sliding window —
    adaptive relative to local statistics (stronger than a single global Otsu on the row).
    """
    t, width = z.shape
    masks = np.zeros((t, width), dtype=np.uint8)
    ws_eff = clamp_odd_window(window_size, width)
    if ws_eff is None:
        raise ValueError(f"need width ≥ 3 for Sauvola, got width={width}")

    for ti in range(t):
        row = z[ti].astype(np.float64, copy=False)
        vmin, vmax = float(np.nanmin(row)), float(np.nanmax(row))
        if not np.isfinite(vmin) or vmax <= vmin + 1e-15:
            continue
        row2d = row.reshape(1, -1)
        th = threshold_sauvola(row2d, window_size=ws_eff, k=k)
        masks[ti] = (row2d > th).astype(np.uint8).ravel()
    return masks, None


def save_kymograph_mask_overlay(
    proc: np.ndarray,
    mask: np.ndarray,
    *,
    out_path: Path,
    meta: str,
) -> None:
    """``imshow``: preprocessed residual (T×X) with semi-transparent mask overlay."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES, layout="constrained")
    disp = proc.T.astype(np.float64)
    m = mask.T.astype(np.float64)
    vmin, vmax = np.percentile(proc, (2.0, 98.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(proc)), float(np.nanmax(proc))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    ax.imshow(
        disp,
        aspect="equal",
        origin="upper",
        cmap=IMAGE_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    rgba = np.zeros((*disp.shape, 4), dtype=np.float64)
    rgba[..., 0] = 0.25
    rgba[..., 1] = 0.85
    rgba[..., 2] = 1.0
    rgba[..., 3] = np.clip(m, 0.0, 1.0) * 0.6
    ax.imshow(rgba, aspect="equal", origin="upper", interpolation="nearest")
    ax.set_title(f"preprocessed + particle mask overlay (cyan)\n{meta}")
    ax.set_xlabel("time (axis 0)")
    ax.set_ylabel("position (pixels)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}")


def save_first_frame_intensity_masked(
    proc: np.ndarray,
    mask: np.ndarray,
    *,
    out_path: Path,
    meta: str,
) -> None:
    """Line plot at ``t=0``: residual vs **x** with particle band shading."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y = proc[0].astype(np.float64)
    m = mask[0].astype(bool)
    xs = np.arange(y.shape[0], dtype=np.float32)
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    if ymax <= ymin:
        ymax = ymin + 1e-6

    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES, layout="constrained")
    ax.fill_between(
        xs,
        ymin,
        ymax,
        where=m,
        alpha=0.4,
        facecolor="0.45",
        interpolate=True,
        linewidth=0,
        label="mask (particle)",
    )
    ax.plot(xs, y, color="C0", linewidth=1.0, label="preprocessed t = 0")
    ax.set_xlim(float(xs[0]), float(xs[-1]))
    ax.set_ylim(ymin, ymax)
    ax.axhline(0.0, color="0.55", linestyle=":", linewidth=0.85)
    ax.set_xlabel("position x (pixel index)")
    ax.set_ylabel("preprocessed intensity")
    ax.set_title(f"{meta}\nfirst time slice + mask")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9, framealpha=0.92)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}")


def frames_detect_line_movie(
    proc: np.ndarray,
    mask: np.ndarray,
    *,
    dpi: int,
    max_frames: int,
    title_stem: str,
    method_label: str,
) -> list[np.ndarray]:
    """Per-frame line vs **x** with mask shading; PNG-style layout for libx264."""
    t_max, x_size = proc.shape
    cap = max(1, min(max_frames, t_max))
    chunk = proc[:cap]
    ymin, ymax = y_axis_minmax(chunk)
    xs = np.arange(x_size, dtype=np.float32)

    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES)
    fig.subplots_adjust(left=0.09, bottom=0.17, top=0.78, right=0.96)
    frames_rgb: list[np.ndarray] = []
    try:
        for t in range(cap):
            ax.clear()
            mt = mask[t].astype(bool)
            ax.fill_between(
                xs,
                ymin,
                ymax,
                where=mt,
                alpha=0.4,
                facecolor="0.45",
                interpolate=True,
                linewidth=0,
            )
            ax.plot(xs, proc[t], color="C0", linewidth=0.85)
            ax.set_xlim(float(xs[0]), float(xs[-1]))
            ax.set_ylim(ymin, ymax)
            ax.axhline(0.0, color="0.55", linestyle=":", linewidth=0.75)
            ax.set_xlabel("position x (pixel index)")
            ax.set_ylabel("preprocessed intensity")
            ax.grid(True, alpha=0.35)
            ax.set_title(
                f"{title_stem}\npreprocessed + mask • {method_label}\n"
                f"frame t = {t} / {cap - 1}  (cap {max_frames} of {t_max} rows)",
                fontsize=10,
            )
            rgb = _frame_rgb(fig, dpi)
            frames_rgb.append(_to_video_frame(rgb, _FRAME_PX_WH))
    finally:
        plt.close(fig)
    return frames_rgb


def write_mask_h5(
    dest: Path,
    mask: np.ndarray,
    *,
    source_path: Path,
    method: str,
    window_size: int | None,
    sauvola_k: float | None,
    illumin_norm: bool,
    eps: float,
    blur_sigma: float,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dest, "w") as fw:
        ds = fw.create_dataset(PARTICLE_MASK_DATASET, data=mask, dtype=np.uint8)
        ds.attrs["source_preprocessed"] = str(source_path)
        ds.attrs["method"] = method
        ds.attrs["illumination_normalization"] = illumin_norm
        ds.attrs["eps"] = eps
        ds.attrs["blur_sigma_x"] = blur_sigma
        if window_size is not None:
            ds.attrs["sauvola_window"] = window_size
        if sauvola_k is not None:
            ds.attrs["sauvola_k"] = sauvola_k
    print(f"Wrote {dest.resolve()} — dataset {PARTICLE_MASK_DATASET!r} shape {mask.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Particle mask per time × position from nsm-preprocess *_preprocessed.h5. "
            "Squares residuals (energy), optionally divides by illumin² where signal tracks "
            "illumination, then Gaussian blur along x (default σ ≈ 6.25 → ~50 px span), "
            "then Sauvola (adaptive along x; default) or per-slice global Otsu. "
            "Writes overlay + first-slice intensity PNGs (like nsm-preprocess); "
            "use --movie for a line-scan video with mask shading."
        )
    )
    parser.add_argument(
        "preprocessed_h5",
        type=Path,
        help="Output of nsm-preprocess (datasets: kymograph + illumination)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help=f"Output directory for <stem>_detect.h5 (default: {DEFAULT_PLOTS_DIR})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_DEFAULT,
        help=f"Preprocessed residual dataset name (default: {DATASET_DEFAULT})",
    )
    parser.add_argument(
        "--method",
        choices=("sauvola", "otsu"),
        default="sauvola",
        help="sauvola = local adaptive along x (default); otsu = one global threshold per row",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help=(
            "Sauvola window along x (odd; auto: ~8%% of width, min 7, max 65). Ignored for otsu."
        ),
    )
    parser.add_argument(
        "--sauvola-k",
        type=float,
        default=0.2,
        help="Sauvola sensitivity (typical ~0.2–0.5; default 0.2)",
    )
    parser.add_argument(
        "--no-illum-norm",
        action="store_true",
        help="Do not divide energy by illumin² (only use squared residual)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help=(
            "Denominator floor for illumin² normalization. "
            "Default: max(illum²)×1e-6 so dark columns stay stable."
        ),
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=_BLUR_SIGMA_DEFAULT,
        help=(
            "Gaussian σ along x on the thresholding image (after squaring / illumin norm), "
            f"before Otsu/Sauvola; 0 disables. Default {_BLUR_SIGMA_DEFAULT:.4g} px "
            "(~50 px effective scale with skimage truncate=4)."
        ),
    )
    parser.add_argument(
        "--movie",
        action="store_true",
        help="Also write <stem>_detect_movie.mp4 (preprocessed line vs x + mask per frame)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frames per second when --movie (default: 24)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_TIME_SAMPLES,
        help=f"Movie length cap (default: {MAX_TIME_SAMPLES})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="matplotlib DPI for --movie rasterization (default: 100)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG overlays and --movie (HDF5 mask only)",
    )

    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be >= 1")
    if args.blur_sigma < 0:
        parser.error("--blur-sigma must be >= 0")

    src = args.preprocessed_h5.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"not a file: {src}")

    proc, illum = load_preprocessed_with_illumination(src, dataset_name=args.dataset)
    energy = energy_squared(proc)

    illumin_norm = not args.no_illum_norm
    if illumin_norm:
        ref = float(np.nanmax(illum**2))
        eps_eff = (
            args.eps
            if args.eps is not None
            else (max(ref * 1e-6, 1e-12) if ref > 0 else 1e-12)
        )
        z = illuminate_flatten_energy(energy, illum, eps=eps_eff)
    else:
        eps_eff = 0.0
        z = energy

    z_blur = gaussian_blur_along_x(z, sigma=float(args.blur_sigma))

    if args.method == "otsu":
        mask, _ = mask_rows_global_otsu(z_blur)
        window_report: int | None = None
        k_report: float | None = None
    else:
        width = z.shape[1]
        if args.window is None:
            w_auto = max(7, min(65, int(round(0.08 * width)) | 1))
        else:
            w_auto = args.window | 1
            if w_auto < 7:
                w_auto = 7 | 1
        mask, _ = mask_rows_sauvola(z_blur, window_size=w_auto, k=float(args.sauvola_k))
        window_report = w_auto
        k_report = float(args.sauvola_k)

    out_dir = resolve_output_directory(args.output)
    stem = src.stem
    dest = out_dir / f"{stem}_detect.h5"

    method_parts = [args.method]
    if window_report is not None:
        method_parts.append(f"w={window_report}")
    if k_report is not None:
        method_parts.append(f"k={k_report}")
    method_label = ", ".join(method_parts) + (
        " • illumin² norm" if illumin_norm else " • energy only"
    )
    if args.blur_sigma > 0:
        method_label += f", blur σx={float(args.blur_sigma):g}"

    write_mask_h5(
        dest,
        mask,
        source_path=src,
        method=args.method,
        window_size=window_report,
        sauvola_k=k_report,
        illumin_norm=illumin_norm,
        eps=float(eps_eff),
        blur_sigma=float(args.blur_sigma),
    )

    if not args.no_plot:
        meta_plot = f"{src.name} • shape {tuple(proc.shape)} • {method_label}"
        save_kymograph_mask_overlay(
            proc,
            mask,
            out_path=out_dir / f"{stem}_detect_overlay.png",
            meta=meta_plot,
        )
        save_first_frame_intensity_masked(
            proc,
            mask,
            out_path=out_dir / f"{stem}_detect_intensity_t0.png",
            meta=meta_plot,
        )
        if args.movie:
            frames_rgb = frames_detect_line_movie(
                proc,
                mask,
                dpi=args.dpi,
                max_frames=args.max_frames,
                title_stem=src.name,
                method_label=method_label,
            )
            mov_dest = resolve_video_destination(
                out_dir,
                source_stem=stem,
                suffix="_detect_movie.mp4",
            )
            _write_movie(frames_rgb, mov_dest, fps=args.fps)
            print(f"Wrote {mov_dest.resolve()} ({len(frames_rgb)} frames)")


if __name__ == "__main__":
    main()
