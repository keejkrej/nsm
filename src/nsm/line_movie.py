"""MP4 line movies: raw intensity or preprocessed residuals vs *x* (`nsm-movie-*`)."""

from __future__ import annotations

import argparse
import io
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nsm.kymograph_io import (
    DEFAULT_KYMOGRAPH_DATASET,
    DEFAULT_LEADING_TIME_ROWS,
    DEFAULT_PLOTS_DIR,
    FIGSIZE_INCHES,
    I2_GAUSSIAN_DATASET,
    load_kymograph,
    resolve_video_destination,
)
from nsm.wavelet_residual import I2_GAUSSIAN_SIGMA, y_axis_minmax

# Output pixels: keep **square** aspect to match ``FIGSIZE_INCHES`` (line movies use a square
# figure). A wide target (e.g. 1280×512) was squeezing the plot vertically after resize.
# Side length is a multiple of 16 for libx264.
_FRAME_PX_SIDE = 1280
_FRAME_PX_WH = (_FRAME_PX_SIDE, _FRAME_PX_SIDE)


def _frame_rgb(fig: plt.Figure, dpi: int) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        pad_inches=0.0,
        facecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _to_video_frame(rgb: np.ndarray, px_wh: tuple[int, int]) -> np.ndarray:
    w, h = px_wh
    im = Image.fromarray(rgb).convert("RGB")
    return np.asarray(im.resize((w, h), Image.Resampling.LANCZOS), dtype=np.uint8)


def _write_movie(frames_rgb: list[np.ndarray], uri: Path, *, fps: float) -> None:
    import imageio.v2 as imageio

    uri.parent.mkdir(parents=True, exist_ok=True)
    if uri.suffix.lower() == ".mp4":
        writer = imageio.get_writer(
            uri,
            fps=fps,
            codec="libx264",
            ffmpeg_log_level="error",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        try:
            for fm in frames_rgb:
                writer.append_data(fm)
        finally:
            writer.close()
    else:
        imageio.mimsave(uri, frames_rgb, fps=fps)


def _make_movie_frames_line(
    path: Path,
    *,
    dpi: int,
    max_frames: int,
    data_td: np.ndarray,
    disk_shape: tuple[int, int],
    title_mode: str,
    y_axis_label: str,
    max_frames_arg: int,
) -> list[np.ndarray]:
    """Animate ``data_td[t, :]`` vs **x** for ``t`` in ``[0 … cap)``, ``cap ≤ max_frames``."""
    t_max, x_size = data_td.shape
    frames_cap = max(1, min(max_frames, t_max))

    chunk = data_td[:frames_cap]
    vmin, vmax = y_axis_minmax(chunk)

    xs = np.arange(x_size, dtype=np.float32)
    frames_rgb: list[np.ndarray] = []

    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES)
    fig.subplots_adjust(left=0.09, bottom=0.17, top=0.78, right=0.96)

    try:
        for t in range(frames_cap):
            ax.clear()
            ax.plot(xs, data_td[t], color="C0", linewidth=0.8)
            ax.set_xlim(float(xs[0]), float(xs[-1]))
            ax.set_ylim(vmin, vmax)
            ax.set_xlabel("position x (pixel index)")
            ax.set_ylabel(y_axis_label)
            ax.grid(True, alpha=0.35)
            ax.set_title(
                f"{path.name}\n{title_mode}\nframe t = {t} / {frames_cap - 1}  "
                f"(movie uses {frames_cap} of {tuple(data_td.shape)} time rows; "
                f"full file on-disk {disk_shape}; --max-frames cap {max_frames_arg})",
                fontsize=10,
            )
            raw = _frame_rgb(fig, dpi)
            frames_rgb.append(_to_video_frame(raw, _FRAME_PX_WH))
    finally:
        plt.close(fig)

    return frames_rgb


def _movie_argparser(description: str, default_output: Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "h5_path",
        type=Path,
        help="One .h5 (cropped raw for raw movie, *_preprocessed.h5 with lpdiff kymograph for preprocess movie)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output,
        help="Output video file (.mp4 / .gif) or directory (writes <stem>_movie_*.mp4)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frames per second (default: 24)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="matplotlib render DPI (default: 100)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_KYMOGRAPH_DATASET,
        help=f"HDF5 dataset key (default: {DEFAULT_KYMOGRAPH_DATASET})",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_LEADING_TIME_ROWS,
        help=f"At most this many frames (default: {DEFAULT_LEADING_TIME_ROWS})",
    )
    return p


def _run_movie_line(
    path: Path,
    *,
    dest: Path,
    transform: Callable[[np.ndarray], np.ndarray],
    title_mode: str,
    y_axis_label: str,
    fps: float,
    dpi: int,
    dataset_name: str,
    max_frames: int,
) -> None:
    arr, disk_shape = load_kymograph(path, dataset_name=dataset_name)
    data_td = np.asarray(transform(arr), dtype=np.float32)
    frames = _make_movie_frames_line(
        path,
        dpi=dpi,
        max_frames=max_frames,
        data_td=data_td,
        disk_shape=disk_shape,
        title_mode=title_mode,
        y_axis_label=y_axis_label,
        max_frames_arg=max_frames,
    )
    _write_movie(frames, dest, fps=fps)
    print(
        f"Wrote {dest.resolve()} ({len(frames)} frames; "
        f"array {tuple(data_td.shape)}; on-disk {disk_shape})"
    )


def main_movie_raw() -> None:
    parser = _movie_argparser(
        (
            "Animate raw intensity vs position-x for each time row of one .h5 "
            "(full file loaded; playback length capped by --max-frames, default "
            f"{DEFAULT_LEADING_TIME_ROWS}). Pass the cropped HDF5 from nsm-crop."
        ),
        DEFAULT_PLOTS_DIR,
    )
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be >= 1")

    path = args.h5_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    dest = resolve_video_destination(args.output, source_stem=path.stem, suffix="_movie_raw.mp4")

    _run_movie_line(
        path,
        dest=dest,
        transform=lambda a: a,
        title_mode="Raw intensity vs x",
        y_axis_label="intensity",
        fps=args.fps,
        dpi=args.dpi,
        dataset_name=args.dataset,
        max_frames=args.max_frames,
    )


def _main_movie_from_preprocessed_h5(*, squared: bool) -> None:
    if squared:
        description = (
            "Animate illumination-rescaled Gaussian I² from *_preprocessed.h5 "
            f"(`{I2_GAUSSIAN_DATASET}`, precomputed by nsm-preprocess; playback capped by "
            f"--max-frames, default {DEFAULT_LEADING_TIME_ROWS})."
        )
        suffix = "_movie_preprocess_I2.mp4"
        title_mode = (
            f"'{I2_GAUSSIAN_DATASET}' (lpdiff²/illum² • Gauss σ_x={I2_GAUSSIAN_SIGMA:g})"
        )
        y_axis_label = r"$I^2$"

        def transform(a: np.ndarray) -> np.ndarray:
            return np.asarray(a, dtype=np.float32)

    else:
        description = (
            "Animate (median subtract − wavelet LP along x) vs x from nsm-preprocess "
            "*_preprocessed.h5 "
            f"(loads full array; playback capped by --max-frames, default {DEFAULT_LEADING_TIME_ROWS})."
        )
        suffix = "_movie_preprocess.mp4"
        title_mode = "Lpdiff residual vs x (from file)"
        y_axis_label = "lpdiff residual"

        def transform(a: np.ndarray) -> np.ndarray:
            return a

    parser = _movie_argparser(description, DEFAULT_PLOTS_DIR)
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be >= 1")

    path = args.h5_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    dest = resolve_video_destination(args.output, source_stem=path.stem, suffix=suffix)

    _run_movie_line(
        path,
        dest=dest,
        transform=transform,
        title_mode=title_mode,
        y_axis_label=y_axis_label,
        fps=args.fps,
        dpi=args.dpi,
        dataset_name=(I2_GAUSSIAN_DATASET if squared else args.dataset),
        max_frames=args.max_frames,
    )


def main_movie_preprocess() -> None:
    _main_movie_from_preprocessed_h5(squared=False)


def main_movie_preprocess_i2() -> None:
    _main_movie_from_preprocessed_h5(squared=True)


if __name__ == "__main__":
    main_movie_raw()
