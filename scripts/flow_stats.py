"""Turbulence statistics of free rollouts, as distances to the truth with sampling floors.

Each measure compares the statistics of a model's free rollout over the held-out block with those of the true flow over
the same block, so none of them depends on timing. Each is reported twice: as a distance, and divided by its sampling
floor, the distance between the held-out truth and equally long segments of the training record. A ratio near 1
means the rollout is as close to the truth as another stretch of the true flow would be. The basis's own projection of
the truth gives a second reference: what a perfect forecast in that basis would score.

Measures (all on mean-removed velocity):
    energy        mean fluctuation energy, rollout / truth (1 is right; reported, not a distance)
    rms_map       relative L2 error of the RMS-velocity map (where the fluctuations are)
    spec_x_db     log-spectral distance, dB, of the streamwise wavenumber spectrum in octave bands (which scales)
    spec_t_db     log-spectral distance, dB, of the frequency spectrum (Welch) in octave bands (which time scales; tones)
    shape_x_db    spec_x_db after normalizing each spectrum by its total: distribution across scales, amplitude removed
    shape_t_db    the same for the frequency spectrum
    pdf_w1        Wasserstein-1 distance of one-point velocity PDFs over the true local standard deviation, averaged over points
    pdf_shape     the same after standardizing each record by its own local standard deviation: skewness, tails, amplitude removed
    edist_pod     energy distance between state distributions in the truth's 10 leading POD coordinates, standardized
    t_int         integral time scale of the leading POD coefficients, rollout / truth (1 is right; reported)

    python scripts/flow_stats.py      # cavity runs; prints a table, writes results/flow_stats_cavity.json
"""
import json
import os
import pickle
import sys

import h5py
import numpy as np
import torch
from scipy.signal import welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import lib.dls_2d as dls       # noqa: E402

DATA = 'data/Challenge2_1_train.h5'
NAME = 'cavity'
BLOCK = (11520, 12800)            # held-out block; rollout row r is snapshot BLOCK[0] + r
TRAIN_END = 10368                 # floors come from segments of the training record
TL = 4
LC = 'results/cavity_dls_p13m5_fca0_1odd/'
L0 = 'results/cavity_dls_p13m5_fc/'
L10 = 'results/cavity_dls_p13m10_fca0_1odd/'
RUNS = {
    'DLS, alpha 0.1, s1': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1',
    'DLS, alpha 0.1, s2': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_s2',
    'DLS, alpha 0.1, s3': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_s3',
    'DLS, alpha 0, d128': L0 + 'spatial_gram_rollout_galerkin_fc_all_d128',
    'DLS, alpha 0, d64': L0 + 'spatial_gram_rollout_galerkin_fc_all_d64',
    'DLS, alpha 0.01': 'results/cavity_dls_p13m5_fca0_01odd/spatial_gram_rollout_galerkin_fc_all_d128_ca0_01',
    'DLS, truncated mesh': 'results/cavity_dls_p13m5/spatial_gram_rollout_galerkin',
    'DLS, alpha 0, s2': L0 + 'spatial_gram_rollout_galerkin_fc_all_d128_s2',
    'DLS, alpha 0, s3': L0 + 'spatial_gram_rollout_galerkin_fc_all_d128_s3',
    'DLS, 10 modes, s1': L10 + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_m10',
    'DLS, 10 modes, s2': L10 + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_m10_s2',
    'DLS, 10 modes, s3': L10 + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_m10_s3',
    'DLS, energy term 0.1': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_fen0_1',
    'DLS, energy term 0.3': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_fen0_3',
    'DLS, energy term 1': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_fen1',
    'DLS, energy term 3': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_fen3',
    'DLS, field energy 0.1': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_fen0_1_field',
    'DLS, field energy 0.3': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_fen0_3_field',
    'DLS, field energy 1': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_fen1_field',
    'DLS, field energy 3': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_fen3_field',
    'DLS, loss ridge 1e-3': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_gl1e-3',
    'DLS, loss ridge 1e-4': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_gl1e-4',
    'DLS, loss ridge 0': LC + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_gl0',
    'Hybrid ms8, alpha 0.1 (one sample)': LC + 'spatial_diff_res_fc_all_d128_ca0_1_ms8',
    'Hybrid ms8, alpha 0 (one sample)': L0 + 'spatial_diff_res_fc_all_d128_ms8',
    'Hybrid one-step, alpha 0 (one sample)': L0 + 'spatial_diff_res_fc_all_d128',
    'Pure diffusion, alpha 0 (one sample)': L0 + 'spatial_diff_fc_all_d128',
}
X_BANDS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 78)]
T_BANDS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 65)]
K_POD = 10


def truth(lo, hi, mean):
    with h5py.File(DATA, 'r') as f:
        return (f['UV'][lo:hi] - mean[None]).astype(np.float32)


def decode(lat, du, dv, shape):
    cfg = pickle.load(open(f'{lat}/latent_config.pkl', 'rb'))
    out = np.zeros((du.shape[0], *shape, 2), np.float32)
    for lo in range(0, du.shape[0], 128):
        Q = dls.decode_fields(cfg, du[lo:lo + 128], dv[lo:lo + 128])[:, :shape[0], :shape[1], :]
        out[lo:lo + 128, :Q.shape[1], :Q.shape[2]] = Q          # a truncated mesh leaves the strip at the mean flow
    return out


class Reference:
    """Statistics of one truth record, against which any record of the same length is scored."""

    def __init__(self, q, basis):
        self.q = q
        self.basis = basis                                   # (K, nx*ny*2) truth POD modes from training data
        self.stats = self.describe(q)

    def describe(self, q):
        n = q.shape[0]
        flat = q.reshape(n, -1)
        a = flat @ self.basis.T                              # (n, K)
        w = np.hanning(q.shape[1])[None, :, None, None]
        Sx = (np.abs(np.fft.rfft(q * w, axis=1)) ** 2).mean(axis=(0, 2, 3))
        f, P = welch(flat[:, ::4].astype(np.float64), nperseg=128, axis=0)
        St = P.mean(axis=1)
        return dict(energy=float((flat.astype(np.float64) ** 2).sum(1).mean()), rms=np.sqrt((q.astype(np.float64) ** 2).mean(0)),
                    sx=np.array([Sx[a_:b_].sum() for a_, b_ in X_BANDS]), st=np.array([St[a_:b_].sum() for a_, b_ in T_BANDS]),
                    sorted=np.sort(flat[:, ::3], axis=0), a=a, t_int=integral_time(a))

    def score(self, q):
        s, r = self.describe(q), self.stats
        lsd = lambda p, t: float(np.sqrt(np.mean((10 * np.log10(np.maximum(p, 1e-30) / t)) ** 2)))
        sd = np.maximum(r['sorted'].std(0), 1e-12)
        norm = lambda v: v / v.sum()
        sd_s = np.maximum(s['sorted'].std(0), 1e-12)
        return dict(energy=s['energy'] / r['energy'],
                    shape_x_db=lsd(norm(s['sx']), norm(r['sx'])), shape_t_db=lsd(norm(s['st']), norm(r['st'])),
                    pdf_shape=float(np.abs((s['sorted'] - s['sorted'].mean(0)) / sd_s - (r['sorted'] - r['sorted'].mean(0)) / sd).mean()),
                    rms_map=float(np.linalg.norm(s['rms'] - r['rms']) / np.linalg.norm(r['rms'])),
                    spec_x_db=lsd(s['sx'], r['sx']), spec_t_db=lsd(s['st'], r['st']),
                    pdf_w1=float((np.abs(s['sorted'] - r['sorted']).mean(0) / sd).mean()),
                    edist_pod=energy_distance(s['a'], r['a']), t_int=s['t_int'] / r['t_int'])


def integral_time(a, max_lag=200):
    """Integral time scale (steps) of the K leading POD coefficients, pooled: integral of the autocorrelation to its first zero."""
    a = a - a.mean(0)
    n = a.shape[0]
    F = np.fft.rfft(a, n=2 * n, axis=0)
    ac = np.fft.irfft(np.abs(F) ** 2, axis=0)[:max_lag].sum(1)
    rho = ac / ac[0]
    z = np.flatnonzero(rho <= 0)
    return float(rho[:z[0] if len(z) else max_lag].sum())


def energy_distance(x, y):
    """Energy distance between two samples, in units of the reference's per-coordinate standard deviation."""
    s = y.std(0, keepdims=True) + 1e-12
    X, Y = torch.from_numpy(x / s).double(), torch.from_numpy(y / s).double()
    return float(2 * torch.cdist(X, Y).mean() - torch.cdist(X, X).mean() - torch.cdist(Y, Y).mean())


def main():
    with h5py.File(DATA, 'r') as f:
        mean = f['mean'][:].astype(np.float32)
    shape = mean.shape[:2]
    n = BLOCK[1] - BLOCK[0] - TL
    # truth POD basis from the training record, every 5th frame
    with h5py.File(DATA, 'r') as f:
        train = (f['UV'][0:TRAIN_END:5] - mean[None]).reshape(-1, mean.size).astype(np.float64)
    _, _, Vt = np.linalg.svd(train, full_matrices=False)
    basis = Vt[:K_POD].astype(np.float32)
    del train
    ref = Reference(truth(BLOCK[0] + TL, BLOCK[1], mean), basis)
    print(f'held-out truth: {n} frames, integral time {ref.stats["t_int"]:.1f} steps', flush=True)

    rows = {}
    segs = [ref.score(truth(lo, lo + n, mean)) for lo in range(0, TRAIN_END - n + 1, n)]
    floor = {k: float(np.mean([s[k] for s in segs])) for k in segs[0]}
    rows['Sampling floor (true flow, other segments)'] = dict(floor, spread={k: float(np.std([s[k] for s in segs])) for k in segs[0]}, n_segments=len(segs))
    with h5py.File(f'{LC}latent_coeff.h5', 'r') as h:
        du, dv = h[NAME]['dof_u'][BLOCK[0] + TL:BLOCK[1]], h[NAME]['dof_v'][BLOCK[0] + TL:BLOCK[1]]
    rows['Basis alone (5 modes, alpha 0.1)'] = ref.score(decode(LC, du, dv, shape))
    with h5py.File(f'{L10}latent_coeff.h5', 'r') as h:
        du, dv = h[NAME]['dof_u'][BLOCK[0] + TL:BLOCK[1]], h[NAME]['dof_v'][BLOCK[0] + TL:BLOCK[1]]
    rows['Basis alone (10 modes, alpha 0.1)'] = ref.score(decode(L10, du, dv, shape))
    for label, run in RUNS.items():
        p = f'{run}/pred/{NAME}_eval_data_pred.h5'
        if not os.path.exists(p):
            print(label, 'no rollout; skipped')
            continue
        with h5py.File(p, 'r') as h:
            du, dv = h['dof_u'][TL:], h['dof_v'][TL:]
        rows[label] = ref.score(decode(os.path.dirname(run), du, dv, shape))
    keys = ['energy', 'rms_map', 'spec_x_db', 'shape_x_db', 'spec_t_db', 'shape_t_db', 'pdf_w1', 'pdf_shape', 'edist_pod', 't_int']
    print(f"{'':40}" + ''.join(f'{k[:10]:>11}' for k in keys))
    for label, r in rows.items():
        print(f'{label[:42]:42}' + ''.join(f'{r[k]:>11.3f}' for k in keys))
    print(f"{'  floor spread (std over segments)':42}" + ''.join(f"{rows['Sampling floor (true flow, other segments)']['spread'][k]:>11.3f}" for k in keys))
    print('ratio to the sampling floor (distances only):')
    for label, r in rows.items():
        if label.startswith('Sampling'):
            continue
        print(f'{label[:42]:42}' + ''.join(f'{r[k] / floor[k]:>11.1f}' if k not in ('energy', 't_int') else f'{"":>11}' for k in keys))
    json.dump(dict(rows=rows, keys=keys, floor=floor), open('results/flow_stats_cavity.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
