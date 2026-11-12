"""Challenge 2.1 NMSE over the cavity multi-start forecasts: deterministic, pure and hybrid diffusion.

The multi-start evaluation (lib/multistart.py) launches 50 forecasts of 30 steps,
spread evenly over the held-out block (snapshots 11,520-12,799), and scores them
in the latent field-energy metric. This scores the same starts with the official
Challenge 2.1 metric instead (scripts/challenge_eval.py::nmse, as in
Challenge2_1_evaluate.m): per-step NMSE of the mean-removed fields on the full
154 x 51 grid, mean(dq^2) / mean(q^2) over both velocity components, against the
PIV fields themselves. Models are seeded with the latent coefficients of the
last time_lag frames before each window, as in the multi-start evaluation.

Methods (cache keys):
    deterministic     rollout-trained element-token transformer, d128 (1.9M)
    pure_*            diffusion over the next-frame increment (1.4M)
    hybrid_*          the deterministic forecast plus a diffused residual (1.4M + the frozen 1.9M)
    hybridms_*        the same, fine-tuned with the 8-step energy score (lib/diffusion.py, ms_steps)
                      *_mean8 / *_mean32: ensemble mean of 8 / 32 members; *_sample: one member
    dmd25             DMD with 25 modes, fit to the 70 frames before the window
    persistence       the last frame before the window
The figure leaves persistence out (it reaches NMSE 2.3 and flattens the rest); the
printed table has it.

    python scripts/fig_multistart_nmse.py compute   # results/<latent>/multistart_nmse.npz (only missing keys)
    python scripts/fig_multistart_nmse.py plot      # figures/multistart_nmse.pdf / .png
"""
import json
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

import challenge_eval as ce  # noqa: E402  (nmse, dmd_forecast, _to_grid, DATA, NAME)

LATENT = 'cavity_dls_p13m5_fc'
RUNS = {'deterministic': 'spatial_gram_rollout_galerkin_fc_all_d128',
        'pure': 'spatial_diff_fc_all_d128',
        'hybrid': 'spatial_diff_res_fc_all_d128',
        'hybridms': 'spatial_diff_res_fc_all_d128_ms8'}
HORIZON, STARTS, MEMBERS = 30, 50, 32
CACHE = f'results/{LATENT}/multistart_nmse.npz'
OUT = 'figures/multistart_nmse'


def _runner(run):
    from lib.runner import runner
    c = json.load(open(f'results/{LATENT}/{run}/{run}.json'))
    c.update(overwrite='x', mode='eval', name=run, log='terminal', distributed=False, device='cuda')
    return runner(c)


def _decode(r, d, shape):
    """(H, 2, N) coefficients -> (H, nx, ny, 2) on the full grid."""
    return ce._to_grid(r.dls.gfem_recon_flexible(config=r.l_config, dof_u=d[:, 0].astype(np.float32),
                                                 dof_v=d[:, 1].astype(np.float32), rec_target=None), shape)


def compute():
    import logging
    import lib.diffusion as diffusion
    import lib.multistart as multistart
    logging.basicConfig(level=logging.WARNING)

    E = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
    # the starts follow the evaluation split, which every cavity run shares
    r0 = _runner(RUNS['hybrid'])
    tl = r0.config['model_params']['time_lag']
    ids = np.asarray(r0.indices['eval_data'][ce.NAME]['val_indices'])
    first = ids[multistart.start_positions(ids, tl + HORIZON, STARTS)] + tl   # first forecast snapshot
    if 'starts' in E and not np.array_equal(E['starts'], first):
        raise ValueError(f"{CACHE} holds other starts; delete it to recompute")
    E['starts'] = first

    with h5py.File(ce.DATA, 'r') as f:
        mean = f['mean'][:].astype(np.float64)
        lo, hi = first.min() - ce.N_IN, first.max() + HORIZON
        q = f['UV'][lo:hi].astype(np.float64) - mean                          # mean-removed PIV fields
    shape = q.shape[1:3]
    true = np.stack([q[s - lo:s - lo + HORIZON] for s in first])
    with h5py.File(r0.paths_bib.latent_path, 'r') as f:
        g = f[ce.NAME]
        seeds = [(g['dof_u'][s - tl:s], g['dof_v'][s - tl:s], None) for s in first]

    def score(key, fc):
        E[key] = np.stack([ce.nmse(p, t) for p, t in zip(fc, true)])         # (starts, steps)

    for kind in ('pure', 'hybrid', 'hybridms'):
        if f'{kind}_mean32' in E:
            continue
        r = r0 if kind == 'hybrid' else _runner(RUNS[kind])
        P = diffusion.forecast(r, ce.NAME, seeds, HORIZON, members=MEMBERS)[:, :, tl:]   # (S, M, H, 2, N)
        # decoding is linear: the decoded ensemble mean is the mean of the decoded members
        score(f'{kind}_mean8', [_decode(r, p[:8].mean(0), shape) for p in P])
        score(f'{kind}_mean32', [_decode(r, p.mean(0), shape) for p in P])
        score(f'{kind}_sample', [_decode(r, p[0], shape) for p in P])
        del P
    if 'deterministic' not in E:
        rd = _runner(RUNS['deterministic'])
        fc = []
        for s, (du, dv, _) in zip(first, seeds):
            pu, pv, _ = rd._predict_dofs_forward(name=ce.NAME, ids=np.arange(s - tl, s + HORIZON),
                                                 dof_u=du, dof_v=dv, total_steps=tl + HORIZON)
            fc.append(_decode(rd, np.stack([pu[tl:], pv[tl:]], axis=1), shape))
        score('deterministic', fc)
    if 'dmd25' not in E:
        fc = []
        for s in first:
            Q = q[s - lo - ce.N_IN:s - lo].transpose(0, 3, 1, 2).reshape(ce.N_IN, -1).T
            fc.append(ce.dmd_forecast(Q, n_fore=HORIZON).T.reshape(HORIZON, 2, *shape).transpose(0, 2, 3, 1))
        score('dmd25', fc)
    if 'persistence' not in E:
        score('persistence', [np.repeat(q[s - lo - 1][None], HORIZON, 0) for s in first])

    np.savez(CACHE, **E)
    print(f"{len(first)} starts, first forecast snapshots {first.min()}-{first.max()}; wrote {CACHE}")
    print(f"{'method':16}" + ''.join(f"{'step ' + str(k):>9}" for k in (1, 5, 10, 20, 30))
          + f"{'1-30':>8}{'start std':>11}")
    for k, e in E.items():
        if k == 'starts':
            continue
        m = e.mean(0)
        print(f"{k:16}" + ''.join(f"{m[j - 1]:9.3f}" for j in (1, 5, 10, 20, 30))
              + f"{e.mean():8.3f}{e.mean(1).std(ddof=1):11.3f}")


def plot():
    import matplotlib.pyplot as plt
    import lib.plotting as pl
    d = np.load(CACHE)
    t = np.arange(1, HORIZON + 1)
    fig, ax = plt.subplots(figsize=(0.62 * pl.width, 0.45 * pl.width))
    ax.axhline(1.0, color='0.45', lw=1.0, label='mean flow')
    # one hue per model; the line style separates the ensemble mean (solid) from a
    # single sample (dotted). DMD and deterministic keep their colors from
    # figures/challenge_heldout.pdf (C1, C3)
    # the one-step hybrid is drawn dashed in the multi-step hybrid's hue: same model, earlier training
    series = [('dmd25', 'DMD, 25 modes', 'C1', '-', 1.0),
              ('deterministic', 'deterministic (1.9M)', 'C3', '-', 1.6),
              ('pure_mean32', 'pure diffusion, mean of 32', 'C2', '-', 1.6),
              ('hybrid_mean32', 'hybrid, one-step training, mean of 32', 'C4', '--', 1.2),
              ('hybridms_mean32', 'hybrid, multi-step training, mean of 32', 'C4', '-', 1.6),
              ('hybridms_sample', 'hybrid, multi-step training, one sample', 'C4', ':', 1.2)]
    for key, lab, col, ls, lw in series:
        ax.plot(t, d[key].mean(0), color=col, ls=ls, lw=lw, label=lab)
    ax.set_xlim(1, HORIZON)
    ax.set_ylim(0, None)
    ax.set_xlabel('forecast step')
    ax.set_ylabel('NMSE')
    ax.grid(visible=True, which='major', linestyle='--', linewidth=0.3)
    ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.savefig(OUT + '.pdf', bbox_inches='tight')
    fig.savefig(OUT + '.png', dpi=200, bbox_inches='tight')
    print(f"wrote {OUT}.pdf / .png ({len(d['starts'])} starts, mean over starts)")


if __name__ == '__main__':
    {'compute': compute, 'plot': plot}[sys.argv[1] if len(sys.argv) > 1 else 'plot']()
