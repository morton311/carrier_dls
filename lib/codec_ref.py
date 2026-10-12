"""One shared codec reference per latent space, linked into each run.

The codec reference is the reconstruction of the TRUE latent coefficients: eval()
scores predictions against it so the latent codec's own error is not charged to
the model. It depends only on the latent space and the snapshots, not on the
model, yet each run used to write its own copy -- ~4.8 GB per run on the 512x512
cavity, 20 identical copies in one latent directory.

Now one file per (latent space, dataset) holds the contiguous prefix of
snapshots [0, n) and grows on demand. Each run's old path holds a small HDF5
virtual dataset mapping its rows onto that file, so readers are unchanged. The
stub stores its source path relative to its own directory (HDF5 resolves it
from there), and fill value NaN, so a missing source shows up as NaN metrics
rather than silent zeros.
"""
import fcntl
import logging
import os
import time
from contextlib import contextmanager

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# snapshots reconstructed per batch while growing the shared file
_BATCH = 250
# a reader holding the shared file open blocks HDF5's write lock; growth is rare,
# so wait it out rather than fail
_OPEN_RETRIES = 60
_OPEN_WAIT_S = 10


def shared_path(latent_dir: str, name: str) -> str:
    return os.path.join(latent_dir, f"{name}_codec_ref_rec.h5")


@contextmanager
def _lock(path):
    """Serialize growth of `path` across processes (two eval chains can share a latent)."""
    with open(path + '.lock', 'w') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _open(path, mode):
    for attempt in range(_OPEN_RETRIES):
        try:
            return h5py.File(path, mode)
        except OSError:
            if attempt == _OPEN_RETRIES - 1:
                raise
            time.sleep(_OPEN_WAIT_S)


def _valid_rows(path):
    if not os.path.exists(path):
        return 0
    with _open(path, 'r') as f:
        return int(f.attrs.get('n_valid', 0))


def grow(path: str, n: int, recon) -> None:
    """Make the shared file hold valid rows for snapshots [0, n).

    `recon(lo, hi)` returns the reconstructed snapshots lo..hi-1 as an array
    (count, *spatial, ncomp). `n_valid` is advanced only after a batch is
    written, so an interrupted build resumes where it stopped.
    """
    with _lock(path):
        # read-only first: the common case is that every row already exists, and
        # a write open would block (and be blocked by) readers in other processes
        have = _valid_rows(path)
        if have >= n:
            return
        with _open(path, 'a') as f:
            for lo in range(have, n, _BATCH):
                hi = min(lo + _BATCH, n)
                q = np.asarray(recon(lo, hi), dtype=np.float32)
                if 'Q_rec' not in f:
                    f.create_dataset('Q_rec', shape=(0,) + q.shape[1:], maxshape=(None,) + q.shape[1:],
                                     dtype=np.float32, chunks=(1,) + q.shape[1:])
                ds = f['Q_rec']
                if ds.shape[0] < hi:
                    ds.resize(hi, axis=0)
                ds[lo:hi] = q
                f.attrs['n_valid'] = hi
        logger.info("Shared codec reference %s now holds snapshots [0, %d)", path, n)


def _runs(ids):
    """Split ids into (stub_row, src_lo, src_hi) runs of consecutive snapshots."""
    out, start = [], 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[i - 1] + 1:
            out.append((start, int(ids[start]), int(ids[i - 1]) + 1))
            start = i
    return out


def _stub_matches(stub, ids):
    try:
        with h5py.File(stub, 'r') as f:
            return (f['Q_rec'].is_virtual and 'ids' in f
                    and np.array_equal(f['ids'][:], ids))
    except (OSError, KeyError):
        return False


def _check_copy(old, shared, ids, rtol=1e-4):
    """True if a regular per-run file holds the same rows as the shared one (spot check)."""
    rows = sorted({0, len(ids) // 2, len(ids) - 1})
    with h5py.File(old, 'r') as fo, h5py.File(shared, 'r') as fs:
        if fo['Q_rec'].shape[0] != len(ids):
            return False
        for r in rows:
            a, b = fo['Q_rec'][r], fs['Q_rec'][int(ids[r])]
            if not np.allclose(a, b, rtol=rtol, atol=rtol * float(np.abs(b).max())):
                return False
    return True


def link(stub: str, shared: str, ids, recon) -> str:
    """Point `stub` at rows `ids` of the shared reference, building rows as needed.

    A regular file already at `stub` (the old per-run copy) is replaced only if
    a spot check shows it holds the same reconstruction; otherwise it is left
    in place and used as before.
    """
    ids = np.asarray(ids, dtype=np.int64)
    grow(shared, int(ids.max()) + 1, recon)
    if _stub_matches(stub, ids):
        return stub
    if os.path.exists(stub):
        with h5py.File(stub, 'r') as f:
            regular = not f['Q_rec'].is_virtual
        if regular and not _check_copy(stub, shared, ids):
            logger.warning("Existing codec reference %s differs from the shared one; keeping it", stub)
            return stub
        os.remove(stub)

    with h5py.File(shared, 'r') as fs:
        full = fs['Q_rec'].shape
    layout = h5py.VirtualLayout(shape=(len(ids),) + full[1:], dtype=np.float32)
    src = h5py.VirtualSource(os.path.relpath(shared, os.path.dirname(os.path.abspath(stub))),
                             'Q_rec', shape=full)
    for row, lo, hi in _runs(ids):
        layout[row:row + hi - lo] = src[lo:hi]
    with h5py.File(stub, 'w') as f:
        f.create_virtual_dataset('Q_rec', layout, fillvalue=np.nan)
        f.create_dataset('ids', data=ids)

    with h5py.File(stub, 'r') as f, h5py.File(shared, 'r') as fs:   # the link must resolve
        last = len(ids) - 1
        if not np.array_equal(f['Q_rec'][last], fs['Q_rec'][int(ids[last])]):
            raise RuntimeError(f"codec reference stub {stub} does not resolve to {shared}")
    return stub
