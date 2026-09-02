"""Metrics for scoring predicted flow fields.

`l2_err_norm` scores a trajectory pointwise. Everything below scores the
*statistics* of a rollout instead, which is what stays meaningful after a
chaotic trajectory has decorrelated from the reference.

Field arrays are (T, *spatial, ncomp) fluctuation velocity, matching the
`Q_rec` layout written by the DLS/POD reconstruction. The spatial FFTs
allocate complex128 over the whole array, so pass a strided subset of
snapshots for large 3D cases rather than every frame.
"""
import numpy as np
from typing import Optional, Union


def l2_err_norm(
    true: np.ndarray,
    pred: np.ndarray,
    axis: Optional[Union[int, tuple]] = None,
) -> np.ndarray:
    """Relative L2 error: ||true - pred|| / ||true||."""
    return np.linalg.norm(true - pred, axis=axis) / np.linalg.norm(true, axis=axis)


def _hann_nd(spatial: tuple) -> np.ndarray:
    """Separable Hann window over `spatial`, normalized to preserve variance."""
    w = np.ones(spatial, dtype=np.float64)
    for i, n in enumerate(spatial):
        shape = [1] * len(spatial)
        shape[i] = n
        w = w * np.hanning(n).reshape(shape)
    return w / np.sqrt(np.mean(w ** 2))


def energy_spectrum(Q: np.ndarray, window: str = 'hann', nbins: int = None,
                    chunk: int = 16):
    """Radially binned kinetic-energy spectrum E(k), averaged over snapshots.

    Args:
        Q: (T, *spatial, ncomp) fluctuation velocity field.
        window: 'hann' (default) or None. The LDC and carrier domains are NOT
            periodic; an unwindowed FFT is leakage-dominated at high k, which
            mimics the very high-k deficit this metric exists to detect.
        nbins: radial bins (default half the smallest spatial extent).
        chunk: snapshots per FFT batch, to bound the complex128 allocation.

    Returns:
        (k, E): bin-center wavenumbers (cycles per domain) and mean E(k).
    """
    Q = np.asarray(Q)
    spatial = Q.shape[1:-1]
    nd = len(spatial)
    axes = tuple(range(1, nd + 1))

    w = _hann_nd(spatial)[None, ..., None] if window == 'hann' else None

    kgrids = np.meshgrid(*[np.fft.fftfreq(n) * n for n in spatial], indexing='ij')
    kmag = np.sqrt(sum(g ** 2 for g in kgrids)).ravel()
    kmax = min(spatial) // 2
    nbins = nbins or kmax
    edges = np.linspace(0.0, float(kmax), nbins + 1)
    idx = np.digitize(kmag, edges) - 1
    keep = (idx >= 0) & (idx < nbins)
    idx_keep = idx[keep]

    norm = float(np.prod(spatial)) ** 2
    E = np.zeros(nbins)
    for start in range(0, Q.shape[0], chunk):
        block = np.asarray(Q[start:start + chunk], dtype=np.float64)
        if w is not None:
            block = block * w
        power = (np.abs(np.fft.fftn(block, axes=axes)) ** 2).sum(axis=-1) / norm
        flat = power.reshape(power.shape[0], -1)[:, keep]
        for row in flat:
            E += np.bincount(idx_keep, weights=row, minlength=nbins)[:nbins]
    E /= Q.shape[0]
    return 0.5 * (edges[:-1] + edges[1:]), E


def velocity_pdf(Q: np.ndarray, bins: int = 101, rng: tuple = None):
    """Per-component PDF of fluctuation values.

    Returns (centers, pdf) with pdf of shape (ncomp, bins), normalized to
    integrate to 1. Mean regression narrows these and grows a spike at zero.
    """
    Q = np.asarray(Q)
    ncomp = Q.shape[-1]
    flat = Q.reshape(-1, ncomp)
    if rng is None:
        lim = float(np.abs(flat).max())
        rng = (-lim, lim)
    pdf = np.zeros((ncomp, bins))
    for c in range(ncomp):
        counts, edges = np.histogram(flat[:, c], bins=bins, range=rng, density=True)
        pdf[c] = counts
    return 0.5 * (edges[:-1] + edges[1:]), pdf


def temporal_autocorr(series: np.ndarray, max_lag: int = None):
    """Probe-averaged temporal autocorrelation rho(tau).

    Args:
        series: (T, n_probe) time series at fixed spatial probes.
        max_lag: longest lag returned (default T // 2).

    Returns:
        (lags, rho). A rollout drifting to a steady state keeps rho near 1 for
        all lags; one emitting noise collapses at lag 1. Variance-matching
        losses can confuse those two failures, so this separates them.
    """
    series = np.asarray(series, dtype=np.float64)
    T = series.shape[0]
    max_lag = max_lag or T // 2
    x = series - series.mean(axis=0, keepdims=True)
    var = (x ** 2).mean(axis=0)
    var[var == 0] = 1.0

    nfft = int(2 ** np.ceil(np.log2(2 * T)))
    F = np.fft.rfft(x, n=nfft, axis=0)
    ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=0)[:max_lag + 1]
    ac /= np.arange(T, T - max_lag - 1, -1)[:, None]
    rho = (ac / var).mean(axis=1)
    return np.arange(max_lag + 1), rho


def fluctuation_energy(Q: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """Per-snapshot fluctuation kinetic energy, 0.5 * sum(w * u^2). Shape (T,)."""
    Q = np.asarray(Q)
    if weights is not None:
        Q = Q * weights[np.newaxis, ..., np.newaxis]
    return 0.5 * (Q.reshape(Q.shape[0], -1) ** 2).sum(axis=-1)


def field_l2_vs_time(Q_gt: np.ndarray, Q_rec: np.ndarray,
                     chunk: int = 32) -> np.ndarray:
    """Per-snapshot relative L2 error of the reconstructed field. Shape (T,).

    Saturates near sqrt(2) once the prediction has fully decorrelated from the
    reference; undershooting that indicates the rollout has lost energy.
    """
    T = min(Q_gt.shape[0], Q_rec.shape[0])
    out = np.zeros(T)
    for start in range(0, T, chunk):
        stop = min(start + chunk, T)
        a = np.asarray(Q_gt[start:stop], dtype=np.float64).reshape(stop - start, -1)
        b = np.asarray(Q_rec[start:stop], dtype=np.float64).reshape(stop - start, -1)
        out[start:stop] = l2_err_norm(a, b, axis=1)
    return out


def spectrum_error(E_gt: np.ndarray, E_rec: np.ndarray, eps: float = 1e-30) -> float:
    """Mean absolute log-ratio between two spectra: mean |log(E_rec/E_gt)|.

    Scale-free and symmetric, so a uniform energy deficit and a uniform excess
    of the same factor score equally.
    """
    gt = np.asarray(E_gt, dtype=np.float64)
    rec = np.asarray(E_rec, dtype=np.float64)
    m = (gt > eps) & (rec > eps)
    if not m.any():
        return float('nan')
    return float(np.abs(np.log(rec[m] / gt[m])).mean())


def spectrum_kl(E_gt: np.ndarray, E_rec: np.ndarray, eps: float = 1e-30) -> float:
    """KL divergence between spectra normalized to unit total energy.

    Measures how energy is *distributed* across scales, independent of how much
    total energy there is (which energy_retention covers separately).
    """
    gt = np.asarray(E_gt, dtype=np.float64).clip(min=eps)
    rec = np.asarray(E_rec, dtype=np.float64).clip(min=eps)
    p = gt / gt.sum()
    q = rec / rec.sum()
    return float((p * np.log(p / q)).sum())
