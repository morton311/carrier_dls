"""Split a snapshot dataset into a global POD part and a local residual.

    u(x,t) = mean(x)  +  sum_{i<K} a_i(t) phi_i(x)  +  r(x,t)

Writes the residual as a dataset in the *same* layout as the source, so the
existing DLS pipeline consumes it with no code changes: point
``latent_params.source_path`` at the residual file and compress as usual.
The POD modes and temporal coefficients are written alongside, to drive the
global model and to rebuild the full field at evaluation time.

Modes are computed by the method of snapshots on a strided subset (the basis
converges long before the full record is needed), then *all* snapshots are
projected onto that basis.

Usage:
    python scripts/pod_decompose.py data/Challenge2_1_train.h5 --modes 25
"""
from __future__ import annotations

import argparse
import logging
import os

import h5py
import numpy as np

logger = logging.getLogger(__name__)

COPY_KEYS = ('x_grid', 'y_grid', 'z_grid', 'x', 'y', 'z')
CHUNK = 500


def pod_basis(path: str, field: str, mean: np.ndarray, num_modes: int,
              stride: int, dtype=np.float64):
    """POD modes by the method of snapshots on every `stride`-th snapshot.

    Returns (modes, sigma) with modes of shape (n_points, num_modes).
    """
    with h5py.File(path, 'r') as f:
        idx = np.arange(0, f[field].shape[0], stride)
        block = (f[field][idx] - mean[None]).astype(dtype)
    snaps = block.reshape(len(idx), -1)
    logger.info("POD basis from %d snapshots (stride %d), %d points",
                len(idx), stride, snaps.shape[1])
    # correlation matrix is (n_snap, n_snap) -- far smaller than the spatial one
    corr = snaps @ snaps.T
    evals, evecs = np.linalg.eigh(corr)
    order = np.argsort(evals)[::-1][:num_modes]
    evals = np.clip(evals[order], 0.0, None)
    sigma = np.sqrt(evals)
    modes = snaps.T @ evecs[:, order]
    nz = sigma > sigma.max() * 1e-12
    modes[:, nz] /= sigma[nz]
    modes[:, ~nz] = 0.0
    total = float((snaps ** 2).sum())
    logger.info("energy captured by %d modes: %.2f%%",
                num_modes, evals.sum() / max(total, 1e-30) * 100)
    return modes, sigma


def decompose(path: str, field: str, num_modes: int, stride: int, out_dir: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    res_path = os.path.join(out_dir, f"{stem}_pod{num_modes}_residual.h5")
    pod_path = os.path.join(out_dir, f"{stem}_pod{num_modes}_global.h5")

    with h5py.File(path, 'r') as f:
        mean = f['mean'][:]
        shape = f[field].shape
        statics = {k: f[k][:] for k in COPY_KEYS if k in f}
    n_snaps = shape[0]

    modes, sigma = pod_basis(path, field, mean, num_modes, stride)

    with h5py.File(path, 'r') as f, \
            h5py.File(res_path, 'w') as fr, h5py.File(pod_path, 'w') as fp:
        for k, v in statics.items():
            fr.create_dataset(k, data=v)
            fp.create_dataset(k, data=v)
        # the residual's own mean is ~0 by construction; carry a zero field so
        # the DLS compression's mean subtraction is a no-op
        fr.create_dataset('mean', data=np.zeros_like(mean))
        fr.create_dataset('source_mean', data=mean)
        res = fr.create_dataset(field, shape=shape, dtype=np.float64)
        coeff = fp.create_dataset('coeffs', shape=(n_snaps, num_modes), dtype=np.float64)
        fp.create_dataset('modes', data=modes.reshape(*shape[1:], num_modes))
        fp.create_dataset('sigma', data=sigma)
        fp.create_dataset('mean', data=mean)

        kept = 0.0
        total = 0.0
        for lo in range(0, n_snaps, CHUNK):
            hi = min(lo + CHUNK, n_snaps)
            block = (f[field][lo:hi] - mean[None]).astype(np.float64)
            flat = block.reshape(hi - lo, -1)
            a = flat @ modes
            rem = (flat - a @ modes.T).reshape(block.shape)
            coeff[lo:hi] = a
            res[lo:hi] = rem
            total += float((flat ** 2).sum())
            kept += float((a ** 2).sum())
            logger.info("  %d/%d snapshots", hi, n_snaps)

    logger.info("POD-%d captured %.2f%% of fluctuation energy; residual holds %.2f%%",
                num_modes, kept / total * 100, (total - kept) / total * 100)
    logger.info("residual -> %s", res_path)
    logger.info("global   -> %s", pod_path)
    return res_path, pod_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', help='source HDF5 snapshot file')
    p.add_argument('--field', default='UV', help='field dataset name (default UV)')
    p.add_argument('--modes', type=int, default=25, help='number of POD modes')
    p.add_argument('--stride', type=int, default=4,
                   help='snapshot stride used to build the basis (default 4)')
    p.add_argument('--out-dir', default=None, help='output dir (default: source dir)')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    decompose(args.source, args.field, args.modes, args.stride,
              args.out_dir or os.path.dirname(args.source) or '.')


if __name__ == '__main__':
    main()
