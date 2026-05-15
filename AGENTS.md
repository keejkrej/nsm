# NSM — Nanoparticle Single-Molecule Kymograph Analysis Toolkit

## Cursor Cloud specific instructions

### Overview

Pure Python CLI pipeline (no web services, databases, or Docker). All I/O is local HDF5/CSV/PNG files.
Six CLI entry points are installed by `uv sync`: `nsm-crop`, `nsm-raw`, `nsm-preprocess`, `nsm-detect`, `nsm-track`, `nsm-diffusion`.

### Environment

- **Python 3.12** (pinned in `.python-version`).
- **uv** is the package manager (`uv.lock` present). Run `uv sync` to install all deps into `.venv/`.
- No `README.md`, `Makefile`, or CI config exists in the repo.

### Running the pipeline

Each CLI tool is documented via `--help`. The typical pipeline order is:

1. `uv run nsm-crop <input.h5> -o <dir>` — trim to a time window
2. `uv run nsm-raw <cropped.h5> -o <dir>` — visualize raw kymograph as PNG
3. `uv run nsm-preprocess <cropped.h5> -o <dir>` — wavelet processing, outputs `*_preprocessed.h5`
4. `uv run nsm-detect <preprocessed.h5> -o <dir>` — peak detection, outputs `*_peaks.csv`
5. `uv run nsm-track <peaks.csv> -o <dir> --preprocessed-h5 <preprocessed.h5>` — trajectory clustering
6. `uv run nsm-diffusion <tracks.csv> -o <dir> --increment-bayes` — Bayesian drift/diffusion estimation

### Gotchas

- Set `MPLBACKEND=Agg` when running headless (no display server). All PNG-generation commands work headless with this env var.
- `nsm-track` takes a **CSV** file (the peaks CSV from `nsm-detect`), not an HDF5 file. Pass `--preprocessed-h5` for overlay plots.
- `nsm-diffusion` with `--increment-bayes` runs PyMC NUTS sampling which is CPU-intensive; expect minutes per track.
- No automated test suite exists in this repo. Validate changes by running the pipeline end-to-end on sample HDF5 data.
- The repo has no linter configuration. Standard `pyright` or `mypy` can be used for type-checking if desired.
