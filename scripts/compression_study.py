"""Compression study, Challenge 2.1 cavity: which element size and enrichment best keep the turbulence statistics.

For each (patch, modes) the DLS basis is built as for training (local modes from 10,000 random patches of training
snapshot 0, full-coverage mesh, cut weight 0.1 with the odd extension). The held-out block is projected onto it and
the projection is scored with the statistics of scripts/flow_stats.py, against the same sampling floors. No forecasting
model is involved, so each row is the best any forecast in that latent space could score. Global POD fit on the
training block, at the same number of coefficients, is the reference compression.

Also recorded per basis: coefficients, tokens (elements) and values per token, the compression NMSE (challenge
definition, mean over frames), energy kept per streamwise wavenumber band (as scripts/tradeoff_scales.py), and how
many directions of an interior element's mass matrix lie below the 1e-2 ridge of the loss and assembly.

    python scripts/compression_study.py                 # main grid, seed 0; cached in results/compression_study/
    python scripts/compression_study.py --seeds 1 2     # basis-sampling repeats of the candidate subset
    python scripts/compression_study.py --multi 200     # candidates with modes from patches of 200 training snapshots
    python scripts/compression_study.py --report        # print the tables from the cache
"""
import argparse
import dataclasses
import json
import logging
import os
import sys
import time

import h5py
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

import lib.dls_2d as dls        # noqa: E402
import flow_stats as fs         # noqa: E402

logging.getLogger().setLevel(logging.WARNING)
DATA = fs.DATA
LO, HI = fs.BLOCK[0] + fs.TL, fs.BLOCK[1]   # the frames flow_stats scores (rollouts start after 4 seed frames)
TRAIN_END = fs.TRAIN_END
OUT = 'results/compression_study'
CACHE = f'{OUT}/cavity.json'
NX, NY = 154, 51
NSKIP = [3, 4, 5, 6, 8, 10, 12, 16, 20]
MODES = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 66, 80, 100]
CMAX = 8000                                  # coefficient budget of the grid (p13m5 has 3,240)
POD_K = [100, 200, 300, 500, 700, 1000, 1500, 2000, 2500, 3240, 4000, 5000, 6000, 8000]
# repeats and the multi-snapshot basis: near 3,240 coefficients, plus the smaller latents a ROM would want
CANDIDATES = [(6, 5), (6, 2), (6, 3), (5, 3), (8, 8), (8, 4), (10, 15), (10, 8), (12, 20), (12, 10), (16, 30),
              (16, 15), (4, 2)]
BANDS = {'large (>31 pts)': (0, 5), 'medium (8-31 pts)': (5, 20), 'small (<8 pts)': (20, 78)}


def nodes(k):
    """GFEM nodes of the full-coverage mesh with nodes every k points."""
    return (-(-(NX - 1) // k) + 1) * (-(-(NY - 1) // k) + 1)


def n_coeff(k, m):
    return 2 * nodes(k) * (m + 1)


def grid():
    out = []
    for k in NSKIP:
        for m in MODES:
            # not over-complete (4 (m + 1) element DOFs for (k + 1)^2 points), and patches must fit the 51-point height
            if 4 * (m + 1) <= (k + 1) ** 2 and 2 * k + 1 < NY and n_coeff(k, m) <= CMAX:
                out.append((k, m))
    return out


def spectrum(Q):
    w = np.hanning(Q.shape[1])[None, :, None, None]
    return (np.abs(np.fft.rfft(Q * w, axis=1)) ** 2).mean(axis=(0, 2, 3))


def basis_modes(field, patch, seed, n_snap=1):
    """Local POD modes of one velocity component from 10,000 random patches.

    n_snap 1 reproduces lib/dls_2d.Modal_decomp exactly (training snapshot 0, global RNG); n_snap > 1 spreads the
    patches over that many evenly spaced training snapshots (field is then (n_snap, nx, ny))."""
    np.random.seed(seed)
    if n_snap == 1:
        return dls.Modal_decomp(field, patch)[0]
    X = np.zeros((patch * patch, 10000))
    for j in range(10000):
        s = np.random.randint(0, n_snap)
        d1 = np.random.randint(0, NX - patch)
        d2 = np.random.randint(0, NY - patch)
        t = field[s, d1:d1 + patch, d2:d2 + patch].reshape(-1, order='F')
        X[:, j] = t - t.mean()
    return np.linalg.svd(X, full_matrices=False)[0]


class Study:
    def __init__(self):
        with h5py.File(DATA, 'r') as f:
            self.mean = f['mean'][:]
            self.xg, self.yg = f['x_grid'][:], f['y_grid'][:]
            self.snap0 = f['UV'][0] - self.mean
            self.block = f['UV'][LO:HI]                                   # raw, as gfem_compress reads it
            train = (f['UV'][0:TRAIN_END:5] - self.mean[None]).reshape(-1, self.mean.size).astype(np.float64)
        self.T = (self.block - self.mean[None]).astype(np.float32)        # mean-removed truth
        _, _, Vt = np.linalg.svd(train, full_matrices=False)
        del train
        self.ref = fs.Reference(self.T, Vt[:fs.K_POD].astype(np.float32))
        self.St = spectrum(self.T)
        self.cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    def floors(self):
        if 'floor' not in self.cache:
            n = HI - LO
            with h5py.File(DATA, 'r') as f:
                segs = []
                for lo in range(0, TRAIN_END - n + 1, n):
                    Q = (f['UV'][lo:lo + n] - self.mean[None]).astype(np.float32)
                    segs.append(dict(self.ref.score(Q), **self.resolved(Q)))
            self.cache['floor'] = {k: float(np.mean([s[k] for s in segs])) for k in segs[0]}
            self.cache['floor_spread'] = {k: float(np.std([s[k] for s in segs])) for k in segs[0]}
            self.save()
        return self.cache['floor']

    def resolved(self, Q):
        """Statistics the PIV noise does not dominate (noise is white in time and flat beyond ~4-point wavelengths).

        shape_x_res, shape_t_res: spectral shape over the bands with wavelength over 4.8 points (k < 32) and the
        frequency bins below 32 of 64; rms_coh: relative error of the map sqrt(lag-1 autocovariance), the RMS of the
        temporally coherent part, which white noise does not enter."""
        if not hasattr(self, '_r'):
            self._r = self.describe_res(self.T)
        s, r = self.describe_res(Q), self._r
        lsd = lambda p, t: float(np.sqrt(np.mean((10 * np.log10(np.maximum(p, 1e-30) / t)) ** 2)))
        nx_, nt_ = 5, 5
        norm = lambda v: v / v.sum()
        return dict(shape_x_res=lsd(norm(s['sx'][:nx_]), norm(r['sx'][:nx_])),
                    shape_t_res=lsd(norm(s['st'][:nt_]), norm(r['st'][:nt_])),
                    rms_coh=float(np.linalg.norm(s['rc'] - r['rc']) / np.linalg.norm(r['rc'])))

    def describe_res(self, Q):
        d = self.ref.describe(Q)
        c1 = (Q[1:].astype(np.float64) * Q[:-1]).mean(0)
        return dict(sx=d['sx'], st=d['st'], rc=np.sqrt(np.maximum(c1, 0)))

    def save(self):
        os.makedirs(OUT, exist_ok=True)
        json.dump(self.cache, open(CACHE + '.tmp', 'w'), indent=1)
        os.replace(CACHE + '.tmp', CACHE)

    def score(self, Q):
        r = self.ref.score(Q)
        r.update(self.resolved(Q))
        E = Q - self.T
        r['nmse'] = float(((E ** 2).mean(axis=(1, 2, 3)) / (self.T ** 2).mean(axis=(1, 2, 3))).mean())
        # the same from lag-1 covariances: noise that is white in time drops out, so this is the error in the
        # temporally coherent field; resid_rho1 is the lag-1 correlation of what the basis discards (noise: ~0)
        lag1 = lambda a, b: float((a[1:].astype(np.float64) * b[:-1]).sum())
        r['nmse_coh'] = lag1(E, E) / lag1(self.T, self.T)
        r['resid_rho1'] = lag1(E, E) / float((E.astype(np.float64) ** 2)[:-1].sum())
        S = spectrum(Q)
        r['bands'] = {b: float(S[a:c].sum() / self.St[a:c].sum()) for b, (a, c) in BANDS.items()}
        d, t = self.ref.describe(Q), self.ref.stats
        r['sx_ratio'] = (d['sx'] / t['sx']).tolist()          # per X_BANDS octave, projection / truth
        r['st_ratio'] = (d['st'] / t['st']).tolist()
        return r

    def dls_row(self, k, m, seed=0, n_snap=1):
        key = f'p{2 * k + 1}m{m}_s{seed}' + (f'_snaps{n_snap}' if n_snap > 1 else '')
        if key in self.cache:
            return self.cache[key]
        t0 = time.time()
        p = 2 * k + 1
        if n_snap == 1:
            fields = (self.snap0[..., 0], self.snap0[..., 1])
        else:
            idx = np.linspace(0, TRAIN_END - 1, n_snap).astype(int)
            with h5py.File(DATA, 'r') as f:
                S = f['UV'][idx] - self.mean[None]
            fields = (S[..., 0], S[..., 1])
        if n_snap == 1:
            # as gfem_compress_flexible: u then v from one global RNG stream
            np.random.seed(seed)
            mu = dls.Modal_decomp(fields[0], p)[0]
            mv = dls.Modal_decomp(fields[1], p)[0]
        else:
            mu = basis_modes(fields[0], p, seed, n_snap)
            mv = basis_modes(fields[1], p, seed + 7919, n_snap)
        Bu, _ = dls.local_modemat_over_elem(self.xg, self.yg, k, mu, m, p)
        Bv, _ = dls.local_modemat_over_elem(self.xg, self.yg, k, mv, m, p)
        cfg = dls.dls_long_Config_Flexible(num_snaps=HI - LO, nx=NX, ny=NY, num_vars=2, patch_size=p, num_modes=m,
                                           modemat_local_u=Bu, modemat_local_v=Bv, full_cover=True, cut_weight=0.1,
                                           cut_extension='odd')
        src = dict(UV=self.block, mean=self.mean, x_grid=self.xg, y_grid=self.yg)
        cfg, du, dv = dls.gfem_compress_flexible(src, 'UV', p, m, latent_target=None, batch_size=HI - LO, dls_config=cfg)
        Q = dls.decode_fields(cfg, du, dv)[:, :NX, :NY, :]
        row = self.score(Q)
        # interior element mass matrix: directions under the 1e-2 ridge of the loss and assembly, condition number
        w = dls._build_wt_vec(k, k)
        ev = []
        for B in (Bu, Bv):
            M = (B * w[:, None]).T @ B
            e = np.linalg.eigvalsh(M)
            ev.append((int((e < 1e-2 * np.trace(M) / len(e)).sum()), float(e.max() / max(e.min(), 1e-300))))
        row.update(nskip=k, patch=p, modes=m, seed=seed, n_snap=n_snap, coefficients=int(du.shape[1] + dv.shape[1]),
                   tokens=int(cfg.num_gfem_elems), per_token=int(2 * cfg.dof_elem), dof_elem=int(cfg.dof_elem),
                   weak_dirs=[e[0] for e in ev], cond=[e[1] for e in ev], seconds=round(time.time() - t0, 1))
        self.cache[key] = row
        self.save()
        return row

    def pod_rows(self):
        todo = [K for K in POD_K if f'pod{K}' not in self.cache]
        if not todo:
            return
        with h5py.File(DATA, 'r') as f:
            X = torch.from_numpy((f['UV'][0:TRAIN_END] - self.mean[None]).reshape(TRAIN_END, -1).astype(np.float32))
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        _, _, Vh = torch.linalg.svd(X.to(dev), full_matrices=False)
        del X
        T = torch.from_numpy(self.T.reshape(self.T.shape[0], -1)).to(dev)
        for K in todo:
            V = Vh[:K]
            Q = ((T @ V.T) @ V).cpu().numpy().reshape(self.T.shape)
            row = self.score(Q)
            row.update(coefficients=K, modes=K)
            self.cache[f'pod{K}'] = row
            self.save()
            print(f'POD {K}: nmse {row["nmse"]:.4f}', flush=True)


KEYS = ['rms_map', 'rms_coh', 'shape_x_db', 'shape_x_res', 'shape_t_db', 'shape_t_res', 'pdf_shape']


def report(c, suffix='_s0'):
    """Distances as multiples of the sampling floor; nmse_coh and resid_rho1 are noise-immune (see score)."""
    fl = c['floor']
    rows = sorted([(v['coefficients'], kk, v) for kk, v in c.items() if kk.startswith('p') and kk.endswith(suffix)])
    print(f"{'basis':10}{'coeff':>6}{'tok':>5}{'nmse':>7}{'coh':>7}{'rho1':>6}{'energy':>7}"
          + ''.join(f'{k[:9]:>10}' for k in KEYS) + f"{'small':>6}{'weak':>5}")
    print(f"{'floor (abs)':48}" + ''.join(f'{fl[k]:>10.3f}' for k in KEYS))
    pr = lambda name, n, tok, v, weak: print(
        f"{name:10}{n:>6}{tok:>5}{v['nmse']:>7.4f}{v['nmse_coh']:>7.4f}{v['resid_rho1']:>6.2f}{v['energy']:>7.3f}"
        + ''.join(f'{v[k] / fl[k]:>10.2f}' for k in KEYS) + f"{v['bands']['small (<8 pts)']:>6.2f}{weak:>5}")
    for n, kk, v in rows:
        pr(kk[:-len(suffix)], n, v['tokens'], v, v['weak_dirs'][0])
    for K in POD_K:
        v = c.get(f'pod{K}')
        if v:
            pr('POD', K, '', v, '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='*', default=[])
    ap.add_argument('--multi', type=int, default=0)
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--only', nargs='*', default=None, help='e.g. 6:5 12:20 (nskip:modes)')
    a = ap.parse_args()
    if a.report:
        report(json.load(open(CACHE)))
        return
    if a.check:
        # this script's projection path against the production latent's stored coefficients, same basis
        import pickle
        lat = 'results/cavity_dls_p13m5_fca0_1odd/'
        cfg = pickle.load(open(lat + 'latent_config.pkl', 'rb'))
        with h5py.File(DATA, 'r') as f:
            src = dict(UV=f['UV'][LO:LO + 200], mean=f['mean'][:], x_grid=f['x_grid'][:], y_grid=f['y_grid'][:])
        _, du, dv = dls.gfem_compress_flexible(src, 'UV', 13, 5, latent_target=None, batch_size=200, dls_config=cfg)
        with h5py.File(lat + 'latent_coeff.h5', 'r') as h:
            ru, rv = h['cavity']['dof_u'][LO:LO + 200], h['cavity']['dof_v'][LO:LO + 200]
        Q = dls.decode_fields(cfg, du, dv)
        R = dls.decode_fields(cfg, ru, rv)
        print('max relative field difference vs stored latent:', float(np.abs(Q - R).max() / np.abs(R).max()))
        return
    st = Study()
    st.floors()
    combos = [tuple(map(int, s.split(':'))) for s in a.only] if a.only else grid()
    if a.seeds or a.multi:
        combos = [c for c in (combos if a.only else CANDIDATES)]
    print(f'{len(combos)} bases', flush=True)
    for k, m in combos:
        for seed in (a.seeds or [0]):
            r = st.dls_row(k, m, seed, a.multi or 1)
            print(f"p{2 * k + 1}m{m} s{seed}{' snaps' + str(a.multi) if a.multi else ''}: {r['coefficients']} coeff, "
                  f"nmse {r['nmse']:.4f}, rms_map {r['rms_map']:.4f}, shape_x {r['shape_x_db']:.2f} dB "
                  f"({r['seconds']} s)", flush=True)
    if not (a.seeds or a.multi or a.only):
        st.pod_rows()
    report(st.cache)


if __name__ == '__main__':
    main()
