"""Frequency-extrapolation baseline of the 2026 paper under the current protocol, and RMS floors.

The 2026 model takes the FFT of each latent coefficient over the training block and
extends it as a sum of cosines. With every frequency kept that is the training
record repeated with period T, so at integer steps it is d(t mod T); it needs no seed
frames. It is applied here to the DLS (alpha = 0.1) coefficients, so its truth is the
same projected field the DLS models are scored against, on the same 50 starts.

Checks printed on the way: persistence must reproduce the eval's ms_persist_mean, and
the eval's rms_error_pct must be reproduced from the run's stats sidecar.

    python scripts/extrap_baseline.py            # writes results/extrap_baseline.json/.npz
    python scripts/extrap_baseline.py --paper    # the paper's truncated-mesh bases -> results/extrap_baseline_paper.json
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

CASES = {
    're15k': dict(data='data/ldc_15k.h5', latent='results/ldc_15k_dls_p99m10_fca0_1odd',
                  run='spatial_gram_rollout_galerkin_heldout_s1_fca0_1', name='ldc_15k',
                  T=1800, tl=4, rms_key='RMS_gt_ldc_15k_2003-2498_b285d46e', window=(2003, 2499)),
    're30k': dict(data='data/ldc_30k_60ksnaps.h5', latent='results/ldc_30k_10k_dls_p99m10_fca0_1odd',
                  run='spatial_gram_rollout_galerkin_30k_10k_fca0_1', name='ldc_30k_10k',
                  T=7200, tl=4, rms_key='RMS_gt_ldc_30k_10k_8004-9999_17aec0d4', window=(8004, 10000)),
}
HORIZON = 30


def decode(cfg, du, dv, nx, ny):
    return dls_2d.decode_fields(cfg, du, dv)[:, :nx, :ny, :]


def main():
    out, arrays = {}, {}
    for key, c in CASES.items():
        cfg = pickle.load(open(c['latent'] + '/latent_config.pkl', 'rb'))
        with h5py.File(c['latent'] + '/latent_coeff.h5', 'r') as f:
            du, dv = f[c['name']]['dof_u'][:], f[c['name']]['dof_v'][:]
        with h5py.File(c['data'], 'r') as f:
            rms_gt = f[c['rms_key']][:].transpose(1, 0, 2)    # source is [y, x]
            nx, ny = rms_gt.shape[:2]
            weights = f['weights'][:].T[:nx, :ny, None]
            tke_gt = f[f"TKE_gt_{os.path.basename(c['latent'])}_{c['name']}eval_data"][np.arange(*c['window'])]
        run = f"{c['latent']}/{c['run']}"
        with h5py.File(f"{run}/saved_metrics/{c['name']}_eval_data.h5", 'r') as f:
            starts, ms_persist = f['ms_start_ids'][:], f['ms_persist'][:]
        metrics = json.load(open(f"{run}/saved_metrics/{c['name']}_eval_data.json"))['metrics']
        with h5py.File(f"{run}/pred/{c['name']}_eval_data_pred_stats.h5", 'r') as f:
            check = 100 * pl.l2_err_norm(rms_gt, f['RMS_rec'][:])
        print(f"{key}: eval rms_error_pct {metrics['rms_error_pct']:.4f}, reproduced {check:.4f}", flush=True)

        err, pers = np.zeros((len(starts), HORIZON)), np.zeros((len(starts), HORIZON))
        for k, s in enumerate(starts):
            t = np.arange(s + c['tl'], s + c['tl'] + HORIZON)
            truth = decode(cfg, du[t], dv[t], nx, ny)
            den = (truth ** 2).sum(axis=(1, 2, 3))
            e = decode(cfg, du[t % c['T']] - du[t], dv[t % c['T']] - dv[t], nx, ny)
            err[k] = np.sqrt((e ** 2).sum(axis=(1, 2, 3)) / den)
            last = np.full(HORIZON, s + c['tl'] - 1)
            p = decode(cfg, du[last] - du[t], dv[last] - dv[t], nx, ny)
            pers[k] = np.sqrt((p ** 2).sum(axis=(1, 2, 3)) / den)
        print(f"  persistence {pers.mean():.4f} (eval {ms_persist.mean():.4f}); "
              f"extrapolation {err.mean():.4f}, step 1 {err[:, 0].mean():.4f}, step 30 {err[:, -1].mean():.4f}, "
              f"skill {1 - err.mean() / pers.mean():.3f}", flush=True)

        # long-run statistics over the eval window: extrapolation, and the latent's own floor
        t = np.arange(*c['window'])
        sq_x, sq_f, tke_x = 0.0, 0.0, []
        for lo in range(0, len(t), 128):
            b = t[lo:lo + 128]
            qx = decode(cfg, du[b % c['T']], dv[b % c['T']], nx, ny).astype(np.float64)
            qf = decode(cfg, du[b], dv[b], nx, ny).astype(np.float64)
            sq_x, sq_f = sq_x + (qx ** 2).sum(0), sq_f + (qf ** 2).sum(0)
            tke_x.append(0.5 * ((qx * weights) ** 2).sum(axis=(1, 2, 3)))    # as runner.compute_TKE
        rms_x, rms_f = np.sqrt(sq_x / len(t)), np.sqrt(sq_f / len(t))
        tke_x = np.concatenate(tke_x)
        out[key] = dict(
            ms_err_mean=float(err.mean()), ms_err_at1=float(err[:, 0].mean()), ms_err_at30=float(err[:, -1].mean()),
            ms_err_worst_start=float(err.mean(1).max()), ms_skill=float(1 - err.mean() / pers.mean()),
            persistence=float(pers.mean()), persistence_eval=float(ms_persist.mean()),
            rms_error_pct=100 * float(pl.l2_err_norm(rms_gt, rms_x)),
            rms_floor_pct=100 * float(pl.l2_err_norm(rms_gt, rms_f)),
            tke_error_pct=100 * float(pl.l2_err_norm(tke_gt, tke_x)),
        )
        arrays[f'{key}_ms_err'] = err
        arrays[f'{key}_rms'] = rms_x.astype(np.float32)
        arrays[f'{key}_rms_floor'] = rms_f.astype(np.float32)
        arrays[f'{key}_tke'] = tke_x / tke_gt.mean()      # normalized like the eval's tke_rec_t
        print(' ', out[key], flush=True)

    # POD floor, Re 30k: only the per-point sum of squares is needed, via the coefficient Gram
    lat = 'results/ldc_30k_10k_pod_m400_tr'
    cfg = pickle.load(open(lat + '/latent_config.pkl', 'rb'))
    t = np.arange(8016, 10000)
    with h5py.File('data/ldc_30k_60ksnaps.h5', 'r') as f:
        rms_gt = f['RMS_gt_ldc_30k_10k_8016-9999_ecfbf7aa'][:].transpose(1, 0, 2)
    with h5py.File(lat + '/latent_coeff.h5', 'r') as f:
        coeff = [f['ldc_30k_10k'][k][t[0]:t[-1] + 1] for k in ('dof_u', 'dof_v')]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rms = []
    for M, A in zip(cfg.modes, coeff):
        M = torch.from_numpy(np.asarray(M, dtype=np.float32)).to(dev)
        A = torch.from_numpy(A.astype(np.float32)).to(dev)
        G = A.T @ A
        per_point = ((M @ G) * M).sum(1) / A.shape[0]
        rms.append(per_point.clamp_min(0).sqrt().cpu().numpy().reshape(rms_gt.shape[:2]))
    out['re30k']['pod400_rms_floor_pct'] = 100 * float(pl.l2_err_norm(rms_gt, np.stack(rms, -1)))
    print('POD 400 RMS floor', out['re30k']['pod400_rms_floor_pct'], flush=True)

    json.dump(out, open('results/extrap_baseline.json', 'w'), indent=2)
    np.savez_compressed('results/extrap_baseline.npz', **arrays)


PAPER = {   # the truncated-mesh bases and runs that the paper's lid-driven cavity tables use
    're15k': dict(data='data/ldc_15k.h5', latent='results/ldc_15k_dls_p99m10', run='spatial_gram_rollout_galerkin_heldout_s1',
                  name='ldc_15k', T=1800, tl=4, rms_key='RMS_gt_ldc_15k_2003-2498_b285d46e', window=(2003, 2499)),
    're30k': dict(data='data/ldc_30k_60ksnaps.h5', latent='results/ldc_30k_10k_dls_p99m10', run='spatial_gram_rollout_galerkin_30k_10k',
                  name='ldc_30k_10k', T=7200, tl=4, rms_key='RMS_gt_ldc_30k_10k_8004-9999_17aec0d4', window=(8004, 10000)),
}


def paper():
    """The same baseline on the paper's truncated-mesh bases, scored on the region the mesh covers.

    Writes results/extrap_baseline_paper.json. Energy and TKE use the DLS projection of the truth as reference.
    """
    out = {}
    for key, c in PAPER.items():
        cfg = pickle.load(open(c['latent'] + '/latent_config.pkl', 'rb'))
        with h5py.File(c['latent'] + '/latent_coeff.h5', 'r') as f:
            du, dv = f[c['name']]['dof_u'][:], f[c['name']]['dof_v'][:]
        dec = lambda a, b: dls_2d.decode_fields(cfg, a, b).astype(np.float64)
        n = dec(du[:1], dv[:1]).shape[1]
        with h5py.File(c['data'], 'r') as f:
            rms_gt = f[c['rms_key']][:].transpose(1, 0, 2)[:n, :n]            # source is [y, x]
        with h5py.File(f"{c['latent']}/{c['run']}/saved_metrics/{c['name']}_eval_data.h5", 'r') as f:
            starts, ms_persist = f['ms_start_ids'][:], f['ms_persist'][:]
        err, pers = np.zeros((len(starts), HORIZON)), np.zeros((len(starts), HORIZON))
        for k, s in enumerate(starts):
            t = np.arange(s + c['tl'], s + c['tl'] + HORIZON)
            den = (dec(du[t], dv[t]) ** 2).sum(axis=(1, 2, 3))
            err[k] = np.sqrt((dec(du[t % c['T']] - du[t], dv[t % c['T']] - dv[t]) ** 2).sum(axis=(1, 2, 3)) / den)
            last = np.full(HORIZON, s + c['tl'] - 1)
            pers[k] = np.sqrt((dec(du[last] - du[t], dv[last] - dv[t]) ** 2).sum(axis=(1, 2, 3)) / den)
        t = np.arange(*c['window'])
        sq_x, sq_f, e_x, e_f = 0.0, 0.0, [], []
        for lo in range(0, len(t), 128):
            b = t[lo:lo + 128]
            qx, qf = dec(du[b % c['T']], dv[b % c['T']]), dec(du[b], dv[b])
            sq_x, sq_f = sq_x + (qx ** 2).sum(0), sq_f + (qf ** 2).sum(0)
            e_x.append((qx ** 2).sum(axis=(1, 2, 3)))
            e_f.append((qf ** 2).sum(axis=(1, 2, 3)))
        e_x, e_f = np.concatenate(e_x), np.concatenate(e_f)
        l2 = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
        out[key] = dict(ms_err_mean=float(err.mean()), ms_err_at1=float(err[:, 0].mean()), ms_err_at30=float(err[:, -1].mean()),
                        ms_skill=float(1 - err.mean() / pers.mean()), persistence=float(pers.mean()),
                        persistence_eval=float(ms_persist.mean()),
                        rms_error_pct=100 * l2(np.sqrt(sq_x / len(t)), rms_gt), rms_floor_pct=100 * l2(np.sqrt(sq_f / len(t)), rms_gt),
                        tke_error_pct_vs_projection=100 * l2(e_x, e_f), energy_ratio=float(e_x.mean() / e_f.mean()))
        print(key, out[key], flush=True)
    json.dump(out, open('results/extrap_baseline_paper.json', 'w'), indent=2)


if __name__ == '__main__':
    paper() if '--paper' in sys.argv else main()
