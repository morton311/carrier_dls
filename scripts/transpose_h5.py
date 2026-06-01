#!/usr/bin/env python3
"""
Transpose every dataset in an HDF5 file without loading full arrays into memory.

This script walks all groups/datasets in the input file, creates a mirrored structure
in an output file, and writes each dataset transposed (default axes are reversed,
matching NumPy's .T behavior for N-D arrays).

Designed for very large datasets (tens of GB): data is copied in chunks/blocks.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transpose every dataset in an HDF5 file into a new output file, "
            "using chunked/block I/O for large arrays."
        )
    )
    parser.add_argument("input", type=Path, help="Path to input .h5 file")
    parser.add_argument("output", type=Path, help="Path to output .h5 file")
    parser.add_argument(
        "--max-block-mb",
        type=int,
        default=1024,
        help="Target max in-memory block size in MB when manual blocking is needed (default: 256)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    return parser.parse_args()


def _copy_attrs(src_obj: h5py.HLObject, dst_obj: h5py.HLObject) -> None:
    for key, value in src_obj.attrs.items():
        dst_obj.attrs[key] = value


def _transpose_slices(
    src_sel: Tuple[slice, ...], axes: Tuple[int, ...]
) -> Tuple[slice, ...]:
    axis_to_slice = {axis: src_sel[axis] for axis in range(len(src_sel))}
    return tuple(axis_to_slice[a] for a in axes)


def _safe_block_shape(shape: Sequence[int], itemsize: int, max_bytes: int) -> Tuple[int, ...]:
    if not shape:
        return ()

    block = [max(1, int(dim)) for dim in shape]

    def block_bytes() -> int:
        return int(math.prod(block)) * itemsize

    if block_bytes() <= max_bytes:
        return tuple(block)

    # Repeatedly reduce the largest current block axis until the block fits.
    while block_bytes() > max_bytes:
        idx = max(range(len(block)), key=lambda i: block[i])
        if block[idx] == 1:
            # All axes already at 1 and still too large only if itemsize > max_bytes.
            break
        block[idx] = max(1, block[idx] // 2)

    return tuple(block)


def _iter_manual_blocks(shape: Sequence[int], block_shape: Sequence[int]) -> Iterable[Tuple[slice, ...]]:
    ranges_per_axis = [
        range(0, int(dim), int(step)) for dim, step in zip(shape, block_shape)
    ]
    for starts in itertools.product(*ranges_per_axis):
        sel = []
        for start, dim, step in zip(starts, shape, block_shape):
            stop = min(start + int(step), int(dim))
            sel.append(slice(start, stop))
        yield tuple(sel)


def _dataset_create_kwargs(src_ds: h5py.Dataset, dst_shape: Tuple[int, ...]) -> dict:
    kwargs: dict = {}

    if src_ds.chunks is not None and dst_shape:
        kwargs["chunks"] = tuple(min(src_ds.chunks[a], dst_shape[i]) for i, a in enumerate(reversed(range(src_ds.ndim))))

    if src_ds.compression is not None:
        kwargs["compression"] = src_ds.compression
    if src_ds.compression_opts is not None:
        kwargs["compression_opts"] = src_ds.compression_opts
    if src_ds.shuffle:
        kwargs["shuffle"] = src_ds.shuffle
    if src_ds.fletcher32:
        kwargs["fletcher32"] = src_ds.fletcher32
    if src_ds.scaleoffset is not None:
        kwargs["scaleoffset"] = src_ds.scaleoffset
    if src_ds.fillvalue is not None:
        kwargs["fillvalue"] = src_ds.fillvalue

    return kwargs


def transpose_dataset(
    src_ds: h5py.Dataset,
    dst_parent: h5py.Group,
    name: str,
    max_block_bytes: int,
) -> None:
    shape = src_ds.shape
    ndim = src_ds.ndim

    if ndim == 0:
        dst_ds = dst_parent.create_dataset(name, data=src_ds[()], dtype=src_ds.dtype)
        _copy_attrs(src_ds, dst_ds)
        return

    axes = tuple(reversed(range(ndim)))
    dst_shape = tuple(shape[a] for a in axes)
    kwargs = _dataset_create_kwargs(src_ds, dst_shape)
    dst_ds = dst_parent.create_dataset(name, shape=dst_shape, dtype=src_ds.dtype, **kwargs)

    if src_ds.chunks is not None:
        # Efficient path for chunked datasets: iterate storage chunks directly.
        for src_sel in src_ds.iter_chunks():
            block = src_ds[src_sel]
            dst_sel = _transpose_slices(src_sel, axes)
            dst_ds[dst_sel] = np.transpose(block, axes=axes)
    else:
        # Fallback for contiguous datasets: manual block traversal sized by memory budget.
        block_shape = _safe_block_shape(shape, src_ds.dtype.itemsize, max_block_bytes)
        for src_sel in _iter_manual_blocks(shape, block_shape):
            block = src_ds[src_sel]
            dst_sel = _transpose_slices(src_sel, axes)
            dst_ds[dst_sel] = np.transpose(block, axes=axes)

    _copy_attrs(src_ds, dst_ds)


def transpose_h5(input_path: Path, output_path: Path, max_block_mb: int, overwrite: bool) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )

    max_block_bytes = max(1, max_block_mb) * 1024 * 1024

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        _copy_attrs(src, dst)

        def visitor(path: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Group):
                grp = dst.require_group(path)
                _copy_attrs(obj, grp)
                return

            if isinstance(obj, h5py.Dataset):
                parent_path = str(Path(path).parent)
                parent = dst if parent_path == "." else dst.require_group(parent_path)
                ds_name = Path(path).name
                print(f"Transposing dataset: {path} | shape={obj.shape} dtype={obj.dtype}")
                transpose_dataset(obj, parent, ds_name, max_block_bytes)
                return

        src.visititems(visitor)


def main() -> None:
    args = parse_args()
    transpose_h5(
        input_path=args.input,
        output_path=args.output,
        max_block_mb=args.max_block_mb,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
