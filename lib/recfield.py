"""Decoded fields on demand, in place of stored reconstructions.

A decoded rollout is about 165 times larger than the latent coefficients it comes
from (2.1 MB against 12.7 kB per 512 x 512 LDC snapshot, 4.2 GB against 25 MB for
a 2,000-step rollout), and decoding is about as fast as writing and re-reading
it (dls_2d.decode_fields: ~5 ms per 512 x 512 snapshot). So by default
(config['save_rec'] false) prediction stores only the coefficients, and eval
decodes what it reads, when it reads it.

Consumers keep passing file paths. runner.eval registers, for each path it would
have written (``*_pred_rec.h5``, ``*_codec_ref_rec.h5``), a LazyField that
decodes from the coefficients; ``open_rec(path)`` then returns a RecFile that
behaves like the open HDF5 file did: ``f['Q_rec']`` slices like the dataset, and
the small cached statistics (``TKE_rec``, ``RMS_rec``) live in a sidecar file,
``*_stats.h5``. Paths that were not registered open as stored files, so runs
with save_rec true and older runs read exactly as before.
"""
import os

import h5py
import numpy as np

_REGISTRY = {}


def stats_path(rec_path: str) -> str:
    """Sidecar file for the statistics cached with a reconstruction."""
    return rec_path[:-len('_rec.h5')] + '_stats.h5' if rec_path.endswith('_rec.h5') else rec_path + '.stats.h5'


class LazyField:
    """Read-only stand-in for a ``Q_rec`` dataset that decodes rows on access.

    ``decode_rows(rows)`` takes a 1-D array of row indices (increasing, no
    repeats) and returns their fields, ``(len(rows), *frame_shape)``. Indexing
    supports what the consumers use: an int, a slice, a list or array of rows
    (any order, repeats allowed), optionally followed by indices into the
    spatial axes. Rows are decoded ``chunk`` at a time, so memory is bounded by
    the request, not by the rollout length.
    """

    def __init__(self, decode_rows, n_rows: int, frame_shape, chunk: int = 128):
        self._decode = decode_rows
        self.shape = (int(n_rows),) + tuple(int(n) for n in frame_shape)
        self.dtype = np.dtype(np.float32)
        self.chunk = chunk

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        k0, rest = key[0], key[1:]
        scalar = isinstance(k0, (int, np.integer))
        rows = np.atleast_1d(np.arange(self.shape[0])[k0])
        uniq, inv = np.unique(rows, return_inverse=True)
        out = np.empty((len(uniq),) + self.shape[1:], dtype=self.dtype)
        for s in range(0, len(uniq), self.chunk):
            out[s:s + self.chunk] = self._decode(uniq[s:s + self.chunk])
        out = out[inv]
        if rest:
            out = out[(slice(None),) + rest]
        return out[0] if scalar else out

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:], dtype=dtype)


class RecFile:
    """File-like view of a lazily decoded reconstruction (see the module docstring)."""

    def __init__(self, field: LazyField, sidecar: str, mode: str = 'r'):
        self.field = field
        self.sidecar = sidecar
        self.writable = mode != 'r'

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        pass

    def __contains__(self, key):
        if key == 'Q_rec':
            return True
        if not os.path.exists(self.sidecar):
            return False
        with h5py.File(self.sidecar, 'r') as f:
            return key in f

    def keys(self):
        names = ['Q_rec']
        if os.path.exists(self.sidecar):
            with h5py.File(self.sidecar, 'r') as f:
                names += list(f.keys())
        return names

    def __getitem__(self, key):
        if key == 'Q_rec':
            return self.field
        with h5py.File(self.sidecar, 'r') as f:
            return f[key][()]

    def create_dataset(self, key, data):
        if not self.writable:
            raise OSError(f"{self.sidecar} is open read-only")
        with h5py.File(self.sidecar, 'a') as f:
            if key in f:
                del f[key]
            f.create_dataset(key, data=data)


def register(rec_path: str, field: LazyField) -> None:
    """Serve ``rec_path`` by decoding instead of from disk (until unregistered)."""
    _REGISTRY[os.path.abspath(rec_path)] = field


def unregister(rec_path: str) -> None:
    _REGISTRY.pop(os.path.abspath(rec_path), None)


def is_lazy(rec_path: str) -> bool:
    return os.path.abspath(rec_path) in _REGISTRY


def open_rec(rec_path: str, mode: str = 'r'):
    """Open a reconstruction for reading: the registered lazy view, else the stored file."""
    field = _REGISTRY.get(os.path.abspath(rec_path))
    if field is not None:
        return RecFile(field, stats_path(rec_path), mode)
    return h5py.File(rec_path, mode)


def clear(rec_path: str) -> None:
    """Remove a stored reconstruction and its statistics sidecar (stale once the coefficients change)."""
    for p in (rec_path, stats_path(rec_path)):
        if os.path.exists(p):
            os.remove(p)
