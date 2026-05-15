"""NSM CLI entrypoint."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from nsm import core
from nsm.commands import bayesian_fit as bayesian_fit_command
from nsm.commands import crop as crop_command
from nsm.commands import msd_fit as msd_fit_command
from nsm.commands import peak_cluster as peak_cluster_command
from nsm.commands import segmented_viterbi as segmented_viterbi_command
from nsm.core import DEFAULT_LEADING_TIME_ROWS
from nsm.core import DEFAULT_PLOTS_DIR

app = typer.Typer(add_completion=False, no_args_is_help=True, help=core.HELP)


@app.command(crop_command.NAME, help=crop_command.HELP)
def crop(
    h5_path: Path,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for cropped output .h5"),
    ],
    max_time: Annotated[
        int,
        typer.Option("--max-time", help=f"Number of leading time rows to keep (default: {DEFAULT_LEADING_TIME_ROWS})"),
    ] = DEFAULT_LEADING_TIME_ROWS,
    preview: Annotated[
        bool,
        typer.Option("--preview/--no-preview", help="Write or skip preview image"),
    ] = True,
) -> None:
    crop_command.run_command(
        h5_path=h5_path,
        output_dir=output_dir,
        max_time=max_time,
        preview=preview,
    )


@app.command(peak_cluster_command.NAME, help=peak_cluster_command.HELP)
def peak_cluster(
    raw_h5: Path,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = DEFAULT_PLOTS_DIR,
    max_time: Annotated[
        int,
        typer.Option("--max-time", help="Keep only first N frames"),
    ] = 1024,
    cluster_method: Annotated[
        str,
        typer.Option("--cluster-method", help="both/dbscan/kmeans"),
    ] = "both",
    overlay: Annotated[
        bool,
        typer.Option("--overlay/--no-overlay", help="Write trajectory overlay PNG"),
    ] = True,
) -> None:
    peak_cluster_command.run_command(
        raw_h5=raw_h5,
        output=output,
        max_time=max_time,
        cluster_method=cluster_method,
        overlay=overlay,
    )


@app.command(segmented_viterbi_command.NAME, help=segmented_viterbi_command.HELP)
def segmented_viterbi(
    raw_h5: Path,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = DEFAULT_PLOTS_DIR,
    max_time: Annotated[
        int,
        typer.Option("--max-time", help="Keep only first N frames"),
    ] = 1024,
    overlay: Annotated[
        bool,
        typer.Option("--overlay/--no-overlay", help="Write trajectory overlay PNG"),
    ] = True,
    keep_preprocessed: Annotated[
        bool,
        typer.Option("--preprocessed/--no-preprocessed", help="Keep preprocessed probability map"),
    ] = True,
) -> None:
    segmented_viterbi_command.run_command(
        raw_h5=raw_h5,
        output=output,
        max_time=max_time,
        overlay=overlay,
        keep_preprocessed=keep_preprocessed,
    )


@app.command(msd_fit_command.NAME, help=msd_fit_command.HELP)
def msd_fit(
    tracks_or_peaks_csv: Path,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ],
    min_frames: Annotated[
        int,
        typer.Option("--min-frames", help="Drop short tracks"),
    ] = 3,
) -> None:
    msd_fit_command.run_command(
        tracks_or_peaks_csv=tracks_or_peaks_csv,
        output=output,
        min_frames=min_frames,
    )


@app.command(bayesian_fit_command.NAME, help=bayesian_fit_command.HELP)
def bayesian_fit(
    tracks_or_peaks_csv: Path,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ],
    min_frames: Annotated[
        int,
        typer.Option("--min-frames", help="Drop short tracks"),
    ] = 3,
    max_plots: Annotated[
        int | None,
        typer.Option("--max-plots", "--max-subplots", "--bayes-max-plots", help="Cap posterior tracks"),
    ] = None,
) -> None:
    bayesian_fit_command.run_command(
        tracks_or_peaks_csv=tracks_or_peaks_csv,
        output=output,
        min_frames=min_frames,
        max_plots=max_plots,
    )


def main(argv: Sequence[str] | None = None) -> None:
    app(args=core.normalize_argv(argv), prog_name=core.PROG_NAME)


if __name__ == "__main__":
    main()
