"""Diagnose (and optionally fix) the x/y axis orientation of an HDF5 source file.

lib/dls_2d.py indexes every spatial array as [y_index, x_index]:

    x_val = x_grid[ky, kx]              # local_modemat_over_elem
    q_u   = f[field][snaps, :, :, 0]    # -> (ny, nx) per snapshot

so a source file must store x varying along axis 1 and y varying along axis 0
(numpy meshgrid 'xy' indexing).  data/ldc_15k.h5 is stored that way and works.

A file written by a column-major language (MATLAB/Fortran) that built its arrays
as (ny, nx) comes back out of h5py as (nx, ny) -- every spatial array arrives
transposed.  dxpt (or dypt) is then identically zero, FEM_shape_calculator
divides by zero, and the GFEM mass matrix is exactly singular.
data/Challenge2_1_train.h5 is in that state.

The fix is to transpose the two spatial axes of every spatial array, *without*
renaming x_grid <-> y_grid: x_grid.T still holds x-coordinates, and after the
transpose they vary along axis 1, which is what the solver wants.  Swapping the
names instead (x_grid = y_grid.T) also produces a non-singular system -- the
element geometry is just reflected across the diagonal -- but it silently
mislabels the axes, so contour plots come out with x and y interchanged.

Usage:
    python scripts/flip_grids.py path/to/file.h5                 # diagnose only
    python scripts/flip_grids.py path/to/file.h5 --transpose     # rewrite in place
    python scripts/flip_grids.py path/to/file.h5 --transpose -o out.h5
"""
import argparse
import shutil
import sys

import h5py
import numpy as np

BATCH = 256


def describe(xg, yg):
    return (np.ptp(xg[:, 0]), np.ptp(xg[0, :]), np.ptp(yg[:, 0]), np.ptp(yg[0, :]))


def needs_transpose(xg, yg):
    """True if x varies down axis 0 and y across axis 1 (column-major layout)."""
    x_a0, x_a1, y_a0, y_a1 = describe(xg, yg)
    if x_a1 > 0 and y_a0 > 0:
        return False
    if x_a0 > 0 and y_a1 > 0:
        return True
    raise SystemExit(
        "ERROR: cannot classify grid orientation (a grid is constant along both axes?)"
    )


def spatial_axes(shape, gshape):
    """Index of the first adjacent axis pair matching the stored grid shape."""
    for i in range(len(shape) - 1):
        if (shape[i], shape[i + 1]) == gshape:
            return i
    return None


def transpose_dataset(f, name, ax):
    """Swap axes ax/ax+1 of f[name] in place, batching to bound memory."""
    src = f[name]
    perm = list(range(src.ndim))
    perm[ax], perm[ax + 1] = perm[ax + 1], perm[ax]
    shape = tuple(src.shape[p] for p in perm)

    tmp = name + "__transpose_tmp"
    if tmp in f:
        del f[tmp]
    dst = f.create_dataset(tmp, shape=shape, dtype=src.dtype)

    if ax == 0:
        dst[...] = src[...].transpose(perm)
    else:
        for s in range(0, src.shape[0], BATCH):
            dst[s : s + BATCH] = src[s : s + BATCH].transpose(perm)

    del f[name]
    f.move(tmp, name)
    return src.shape, shape


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="data/ldc_15k.h5")
    ap.add_argument("--transpose", action="store_true",
                    help="swap the two spatial axes of every spatial array")
    ap.add_argument("-o", "--output",
                    help="write to a copy instead of modifying PATH in place")
    args = ap.parse_args()

    path = args.path
    with h5py.File(path, "r") as f:
        xg, yg = f["x_grid"][:], f["y_grid"][:]
    x_a0, x_a1, y_a0, y_a1 = describe(xg, yg)
    print(f"{path}: x_grid {xg.shape}, y_grid {yg.shape}")
    print(f"  x_grid variation  axis0: {x_a0:.6g}  axis1: {x_a1:.6g}")
    print(f"  y_grid variation  axis0: {y_a0:.6g}  axis1: {y_a1:.6g}")

    if not needs_transpose(xg, yg):
        print("  layout OK: x along axis 1, y along axis 0 (row-major / 'xy') "
              "- nothing to do.")
        return
    print("  layout is COLUMN-MAJOR: x along axis 0, y along axis 1 "
          "- spatial arrays need transposing.")

    if not args.transpose:
        print("\nRe-run with --transpose to fix (add -o OUT.h5 to write a copy).")
        return

    if args.output:
        print(f"\nCopying {path} -> {args.output}")
        shutil.copyfile(path, args.output)
        path = args.output

    gshape = xg.shape
    with h5py.File(path, "r+") as f:
        names = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        skipped = []
        for name in names:
            ax = spatial_axes(f[name].shape, gshape)
            if ax is None:
                skipped.append((name, f[name].shape))
                continue
            old, new = transpose_dataset(f, name, ax)
            print(f"  transposed {name:52s} {old} -> {new}")

        if skipped:
            print("\n  left untouched (no adjacent "
                  f"{gshape} axis pair):")
            for name, shape in skipped:
                print(f"    {name:52s} {shape}")

        xg, yg = f["x_grid"][:], f["y_grid"][:]

    x_a0, x_a1, y_a0, y_a1 = describe(xg, yg)
    print(f"\n{path}: x_grid {xg.shape}, y_grid {yg.shape}")
    print(f"  x_grid variation  axis0: {x_a0:.6g}  axis1: {x_a1:.6g}")
    print(f"  y_grid variation  axis0: {y_a0:.6g}  axis1: {y_a1:.6g}")
    print("  layout OK" if not needs_transpose(xg, yg) else "  STILL WRONG")


if __name__ == "__main__":
    main()
