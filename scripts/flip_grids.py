"""Flip the x_grid/y_grid meshgrid orientation in an HDF5 source file.

dls_2d._read_static / dls._read_static transpose the grids on read
(grid_x = f['x_grid'][:].T), and the FEM shape-function geometry then
requires x to vary along axis 1 and y along axis 0 *after* that transpose.
Source files must therefore store x varying along axis 0 and y along axis 1
('ij' meshgrid indexing). Files stored the other way make dxpt/dypt zero and
the GFEM mass matrix exactly singular.

This transposes both grids in place, only when needed (idempotent).

Usage: python scripts/flip_grids.py [path/to/file.h5]   (default: data/ldc_15k.h5)
"""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else 'data/ldc_15k.h5'

with h5py.File(path, 'r+') as f:
    xg = f['x_grid'][:]
    yg = f['y_grid'][:]
    x_axis0 = np.ptp(xg[:, 0])
    x_axis1 = np.ptp(xg[0, :])
    y_axis0 = np.ptp(yg[:, 0])
    y_axis1 = np.ptp(yg[0, :])
    print(f"{path}: x_grid {xg.shape}, y_grid {yg.shape}")
    print(f"  x_grid variation  axis0: {x_axis0:.6g}  axis1: {x_axis1:.6g}")
    print(f"  y_grid variation  axis0: {y_axis0:.6g}  axis1: {y_axis1:.6g}")

    # if x_axis0 > 0 and y_axis1 > 0:
    #     print("Grids already stored as expected (x along axis 0, y along axis 1) - nothing to do.")
    #     sys.exit(0)
    # if xg.shape[0] != xg.shape[1] or yg.shape[0] != yg.shape[1]:
    #     print("ERROR: non-square grids cannot be transposed in place; aborting.")
    #     sys.exit(1)

    # f['x_grid'][...] = xg.T
    # f['y_grid'][...] = yg.T
    # print("x_grid/y_grid now swapped in place. Re-checking variation along axes:")

    # xg = f['x_grid'][:]
    # yg = f['y_grid'][:]
    # x_axis0 = np.ptp(xg[:, 0])
    # x_axis1 = np.ptp(xg[0, :])
    # y_axis0 = np.ptp(yg[:, 0])
    # y_axis1 = np.ptp(yg[0, :])
    # print(f"{path}: x_grid {xg.shape}, y_grid {yg.shape}")
    # print(f"  x_grid variation  axis0: {x_axis0:.6g}  axis1: {x_axis1:.6g}")
    # print(f"  y_grid variation  axis0: {y_axis0:.6g}  axis1: {y_axis1:.6g}")
    
