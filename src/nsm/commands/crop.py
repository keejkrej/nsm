"""Crop command implementation."""

from __future__ import annotations

from pathlib import Path

import h5py

from nsm.core import DEFAULT_KYMOGRAPH_DATASET, DEFAULT_LEADING_TIME_ROWS, write_raw_preview

NAME = "crop"
HELP = "Create a leading-time cropped kymograph for all later steps."


def crop_kymograph_h5(
    src: Path,
    dst: Path,
    *,
    dataset_name: str,
    max_time: int,
) -> None:
    """Copy ``dataset_name`` from ``src`` into ``dst`` with at most ``max_time`` rows."""
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src, "r") as fr:
        if dataset_name not in fr:
            avail = ", ".join(sorted(fr.keys()))
            raise KeyError(f"{src.name}: missing '{dataset_name}'. Available: {avail or '(empty)'}")
        ds = fr[dataset_name]
        raw_shape = (int(ds.shape[0]), int(ds.shape[1]))
        if max_time < 1:
            raise ValueError(f"max_time must be >= 1, got {max_time}")
        t_end = min(raw_shape[0], max_time)
        block = ds[:t_end, :][...]

    with h5py.File(dst, "w") as fw:
        fw.create_dataset(dataset_name, data=block, dtype=block.dtype)

    print(
        f"Wrote {dst.resolve()} — time 0:{t_end} of {raw_shape[0]} "
        f"({dataset_name} shape on disk {raw_shape})"
    )


def run_command(
    h5_path: Path,
    output_dir: Path,
    max_time: int = DEFAULT_LEADING_TIME_ROWS,
    preview: bool = True,
) -> None:
    src = h5_path.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"not a file: {src}")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dst = output_dir / f"{src.stem}_crop.h5"
    crop_kymograph_h5(src, dst, dataset_name=DEFAULT_KYMOGRAPH_DATASET, max_time=max_time)

    if preview:
        with h5py.File(dst, "r") as fr:
            arr = fr[DEFAULT_KYMOGRAPH_DATASET][...]
        preview_png = output_dir / f"{dst.stem}_raw.png"
        write_raw_preview(
            arr,
            preview_png,
            title=(
                f"{dst.name}\n"
                f"cropped raw ({tuple(arr.shape)}) dataset={DEFAULT_KYMOGRAPH_DATASET}"
            ),
        )
        print(f"Wrote {preview_png.resolve()}")
