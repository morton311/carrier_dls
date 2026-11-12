"""Open-cavity baselines on the multi-start protocol used for every flow in the annual report.

lib/multistart.py scores trained runs on 50 forecasts x 30 steps of the held-out block, each
relative to the latent's projection of the truth. This scores what it does not run on the same
forecasts, in that measure ('in_basis') and against the measured field ('raw'):

    DMD-25       port of the official baseline, fit to the 70 frames before each forecast
    persistence  hold the last frame; in_basis must reproduce the eval's ms_persist

Encoding the true frames must also reproduce the stored coefficients; both checks are printed.

    python scripts/cavity_baselines.py    # prints a table, writes results/cavity_baselines.json
"""
import json
import logging
import os
import pickle
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

import lib.dls_2d as dls_2d       # noqa: E402
from challenge_eval import dmd_forecast, _to_grid   # noqa: E402

DATA = 'data/Challenge2_1_train.h5'
NAME = 'cavity'
LAT = 'results/cavity_dls_p13m5_fca0_1odd'
RUN = 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1'
TL, HORIZON, N_IN = 4, 30, 70


def rel(err, truth):
    """(frames,) relative error of (frames, nx, ny, 2) fields."""
    return np.sqrt((err ** 2).sum(axis=(1, 2, 3)) / (truth ** 2).sum(axis=(1, 2, 3)))


def summary(e, pers=None):
    out = dict(err=float(e.mean()), by_step=e.mean(0).tolist(), at1=float(e[:, 0].mean()), at30=float(e[:, -1].mean()),
               worst_start=float(e.mean(1).max()), start_std=float(e.mean(1).std(ddof=1)))
    if pers is not None:
        out['skill'] = float(1 - e.mean() / pers.mean())
    return out


def main():
    logging.basicConfig(level=logging.WARNING)
    cfg = pickle.load(open(f'{LAT}/latent_config.pkl', 'rb'))
    with h5py.File(f'{LAT}/{RUN}/saved_metrics/{NAME}_eval_data.h5', 'r') as g:
        starts, ms_persist = g['ms_start_ids'][:], g['ms_persist'][:]
    first = starts + TL                                     # first forecast frame of each start
    with h5py.File(DATA, 'r') as f:
        lo = int(first[0]) - N_IN
        q = f['UV'][lo:].astype(np.float64) - f['mean'][:][None]
        xg, yg = f['x_grid'][:], f['y_grid'][:]
    shape = q.shape[1:3]
    with h5py.File(f'{LAT}/latent_coeff.h5', 'r') as h:
        du, dv = h[NAME]['dof_u'][:], h[NAME]['dof_v'][:]

    def decode(a, b):
        return _to_grid(dls_2d.decode_fields(cfg, a, b)[:, :shape[0], :shape[1], :].astype(np.float64), shape)

    def encode(F):
        src = {'UV': F.astype(np.float32), 'mean': np.zeros(F.shape[1:], np.float32), 'x_grid': xg, 'y_grid': yg}
        out = {}
        dls_2d.gfem_compress_flexible(src, 'UV', cfg.patch_size, cfg.num_modes, dls_config=cfg,
                                      batch_size=256, latent_target=out)
        return out['dof_u'], out['dof_v']

    n, h = len(first), HORIZON
    X = np.stack([q[t - lo:t - lo + h] for t in first])     # (n, h, nx, ny, 2)
    F = np.stack([dmd_forecast(q[t - lo - N_IN:t - lo].transpose(0, 3, 1, 2).reshape(N_IN, -1).T)
                  .T.reshape(h, 2, *shape).transpose(0, 2, 3, 1) for t in first])
    ids = np.concatenate([np.arange(t, t + h) for t in first])
    tu, tv = du[ids], dv[ids]

    eu, ev = encode(X.reshape(n * h, *X.shape[2:]))
    print(f'encoding the truth vs stored coefficients: relative difference '
          f'{np.linalg.norm(eu - tu) / np.linalg.norm(tu):.2e} (u), {np.linalg.norm(ev - tv) / np.linalg.norm(tv):.2e} (v)')
    fu, fv = encode(F.reshape(n * h, *F.shape[2:]))

    P = decode(tu, tv)                                      # projected truth
    dmd_b = rel(decode(fu - tu, fv - tv), P).reshape(n, h)
    dmd_r = rel((F - X).reshape(n * h, *X.shape[2:]), X.reshape(n * h, *X.shape[2:])).reshape(n, h)
    last = first - 1
    lu, lv = np.repeat(du[last], h, 0), np.repeat(dv[last], h, 0)
    pers_b = rel(decode(lu - tu, lv - tv), P).reshape(n, h)
    held = np.repeat(np.stack([q[t - lo] for t in last]), h, 0)
    pers_r = rel(held - X.reshape(n * h, *X.shape[2:]), X.reshape(n * h, *X.shape[2:])).reshape(n, h)
    print(f'persistence in basis {pers_b.mean():.4f} (eval {ms_persist.mean():.4f}), raw {pers_r.mean():.4f}')

    out = {'DMD-25': dict(in_basis=summary(dmd_b, ms_persist), raw=summary(dmd_r, pers_r)),
           'persistence': dict(in_basis=summary(ms_persist), raw=summary(pers_r)),
           'mean': dict(in_basis=dict(err=1.0, skill=float(1 - 1 / ms_persist.mean())), raw=dict(err=1.0))}
    for k in ('DMD-25', 'persistence'):
        for m in ('in_basis', 'raw'):
            print(k, m, {a: round(v, 4) for a, v in out[k][m].items() if a != 'by_step'})
    print('mean flow skill', round(out['mean']['in_basis']['skill'], 4))
    json.dump(out, open('results/cavity_baselines.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
