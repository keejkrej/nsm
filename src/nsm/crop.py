"""Write a leading time window of a kymograph HDF5 to a new file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from nsm.kymograph_io import DEFAULT_KYMOGRAPH_DATASET, DEFAULT_LEADING_TIME_ROWS


def crop_kymograph_h5(
    src: Path,
    dst: Path,
    *,
    dataset_name: str,
    max_time: int,
) -> None:
    """Copy ``dataset_name`` from ``src`` into ``dst`` with at most ``max_time`` leading rows."""
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src, "r") as fr:
        if dataset_name not in fr:
            avail = ", ".join(sorted(fr.keys()))
            raise KeyError(
                f"{src.name}: missing '{dataset_name}'. Available: {avail or '(empty)'}"
            )
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a smaller HDF5 by keeping only the leading time rows "
            f"(default {DEFAULT_LEADING_TIME_ROWS}). Use before nsm-raw / nsm-preprocess when you want "
            "a fixed time window; downstream tools load the full extent of their input file."
        )
    )
    parser.add_argument(
        "h5_path",
        type=Path,
        help="Source .h5 file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the cropped output .h5",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Output filename (default: <input_stem>_crop.h5). "
            "Written under --output-dir."
        ),
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=DEFAULT_LEADING_TIME_ROWS,
        help=f"Number of leading time rows to keep (default: {DEFAULT_LEADING_TIME_ROWS})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_KYMOGRAPH_DATASET,
        help=f"HDF5 dataset key (default: {DEFAULT_KYMOGRAPH_DATASET})",
    )
    args = parser.parse_args()

    src = args.h5_path.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"not a file: {src}")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.output_name or f"{src.stem}_crop.h5"
    if not name.endswith(".h5"):
        name = f"{name}.h5"
    dst = out_dir / name

    crop_kymograph_h5(
        src,
        dst,
        dataset_name=args.dataset,
        max_time=args.max_time,
    )


if __name__ == "__main__":
    main()
