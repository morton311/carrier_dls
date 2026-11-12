"""Multi-start forecast error against the raw velocity field, and each basis's compression error.

Covers Re 30k (POD vs DLS; these keys have no prefix), Re 15k and the open cavity (DLS).

The eval's ms_err compares a forecast with its own latent's projection of the truth,
which hides the truncation error. With e the in-span error and r = X - proj(X):

    ||forecast - X||^2 = ||e||^2 + ||r||^2 + 2 e.r

e.r = 0 exactly for POD (orthonormal modes, orthogonal projection). It is also 0 for an
uncut DLS mesh; for the alpha = 0.1 latent the cut elements break it slightly, so the
DLS figure drops a small cross term and is approximate. Each model is scored on its own
50 starts (its window length sets them), spread over the same held-out block.

Also reports persistence against the raw field on the same starts, and for POD the RMS
error of the basis's own projection over the eval window.

    python scripts/raw_field_score.py    # prints a table, writes results/raw_field_score.json
"""
import json
import os
import pickle
import sys

import h5py
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import lib.dls_2d as dls_2d       # noqa: E402
import lib.plotting as pl         # noqa: E402

HORIZON = 30
RMS_30K = ('ldc_30k_10k', (8016, 10000), 'RMS_gt_ldc_30k_10k_8016-9999_ecfbf7aa')
L30, L15, LC = ('results/ldc_30k_10k_dls_p99m10_fca0_1odd', 'results/ldc_15k_dls_p99m10_fca0_1odd',
                'results/cavity_dls_p13m5_fca0_1odd')
GL = 'spatial_gram_rollout_galerkin'
CASES = [    # data file, group, source stored [y, x], {label: (latent, run, time lag)}
    ('data/ldc_30k_60ksnaps.h5', 'ldc_30k_10k', True, {
        **{f'POD {k}': (f'results/ldc_30k_10k_pod_m{k}_tr', f'pod_30k_10k_m{k}', 16) for k in (400, 100, 50)},
        'DLS s1': (L30, f'{GL}_30k_10k_fca0_1', 4),
        'DLS s2': (L30, f'{GL}_30k_10k_fca0_1_s2', 4),
        'DLS s3': (L30, f'{GL}_30k_10k_fca0_1_s3', 4),
        'DLS hybrid': (L30, 'spatial_diff_res_30k_10k_fca0_1', 4)}),
    ('data/ldc_15k.h5', 'ldc_15k', True, {
        **{f'Re 15k DLS s{k}': (L15, f'{GL}_heldout_s{k}_fca0_1', 4) for k in (1, 2, 3)},
        'Re 15k DLS hybrid': (L15, 'spatial_diff_res_heldout_s1_fca0_1_ms8', 4)}),
    ('data/Challenge2_1_train.h5', 'cavity', False, {
        'Cavity DLS s1': (LC, f'{GL}_fc_all_d128_ca0_1', 4),
        'Cavity DLS s2': (LC, f'{GL}_fc_all_d128_ca0_1_s2', 4),
        'Cavity DLS s3': (LC, f'{GL}_fc_all_d128_ca0_1_s3', 4),
        'Cavity DLS hybrid': (LC, 'spatial_diff_res_fc_all_d128_ca0_1_ms8', 4)}),
]


def raw_frames(f, mean, frames, swap):
    """Fluctuating field at the given snapshots, [x, y]."""
    q = (f['UV'][frames] - mean[None]).astype(np.float64)
    return q.transpose(0, 2, 1, 3) if swap else q


def pod_rms_floor(f, lat):
    """RMS error (%) of the basis's own projection over the eval window; the coefficient Gram gives the per-point sum."""
    name, (lo, hi), key = RMS_30K
    cfg = pickle.load(open(f'{lat}/latent_config.pkl', 'rb'))
    rms_gt = f[key][:].transpose(1, 0, 2)
    with h5py.File(f'{lat}/latent_coeff.h5', 'r') as h:
        coeff = [h[name][k][lo:hi] for k in ('dof_u', 'dof_v')]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rms = []
    for M, A in zip(cfg.modes, coeff):
        M = torch.from_numpy(np.asarray(M, dtype=np.float32)).to(dev)
        A = torch.from_numpy(A.astype(np.float32)).to(dev)
        per_point = ((M @ (A.T @ A)) * M).sum(1) / A.shape[0]
        rms.append(per_point.clamp_min(0).sqrt().cpu().numpy().reshape(rms_gt.shape[:2]))
    return 100 * float(pl.l2_err_norm(rms_gt, np.stack(rms, -1)))


def score_case(data, name, swap, runs, out):
    cache = {}
    with h5py.File(data, 'r') as f:
        mean = f['mean'][:]
        for label, (lat, run, tl) in runs.items():
            h5 = f'{lat}/{run}/saved_metrics/{name}_eval_data.h5'
            if not os.path.exists(h5):
                print(label, 'not evaluated yet; skipped', flush=True)
                continue
            with h5py.File(h5, 'r') as g:
                err, starts = g['ms_err'][:], g['ms_start_ids'][:]
            ckey = (lat, tl)
            if ckey not in cache:
                cfg = pickle.load(open(f'{lat}/latent_config.pkl', 'rb'))
                with h5py.File(f'{lat}/latent_coeff.h5', 'r') as h:
                    du, dv = h[name]['dof_u'][:], h[name]['dof_v'][:]
                e_raw, e_proj, e_res, pers = (np.zeros((len(starts), HORIZON)) for _ in range(4))
                for k, s in enumerate(starts):
                    t = np.arange(s + tl, s + tl + HORIZON)
                    X = raw_frames(f, mean, t, swap)
                    e_raw[k] = (X ** 2).sum(axis=(1, 2, 3))
                    last = raw_frames(f, mean, [s + tl - 1], swap)
                    pers[k] = np.sqrt(((X - last) ** 2).sum(axis=(1, 2, 3)) / e_raw[k])
                    if 'pod' in lat:      # orthonormal modes: ||proj(X)||^2 = sum a^2
                        e_proj[k] = (du[t].astype(np.float64) ** 2).sum(1) + (dv[t].astype(np.float64) ** 2).sum(1)
                        e_res[k] = e_raw[k] - e_proj[k]
                    else:
                        nx, ny = X.shape[1:3]
                        Q = dls_2d.decode_fields(cfg, du[t], dv[t])[:, :nx, :ny, :].astype(np.float64)
                        e_proj[k] = (Q ** 2).sum(axis=(1, 2, 3))
                        e_res[k] = ((X - Q) ** 2).sum(axis=(1, 2, 3))
                cache[ckey] = (e_raw, e_proj, e_res, pers)
            e_raw, e_proj, e_res, pers = cache[ckey]
            # ms_err is relative to the projected truth's energy
            raw = np.sqrt(((err ** 2) * e_proj + e_res) / e_raw)
            floor = np.sqrt(e_res / e_raw)
            out[label] = dict(in_span=float(err.mean()), raw=float(raw.mean()), floor=float(floor.mean()),
                              raw_worst_start=float(raw.mean(1).max()),
                              raw_at1=float(raw[:, 0].mean()), raw_at30=float(raw[:, -1].mean()),
                              raw_persistence=float(pers.mean()), raw_skill=float(1 - raw.mean() / pers.mean()))
            if 'pod' in lat:
                out[label]['rms_floor_pct'] = pod_rms_floor(f, lat)
            print(label, {k: round(v, 4) for k, v in out[label].items()}, flush=True)


def main():
    out = {}
    for case in CASES:
        score_case(*case, out)
    json.dump(out, open('results/raw_field_score.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
