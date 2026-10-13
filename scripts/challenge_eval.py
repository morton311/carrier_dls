"""Challenge 2.1 check on a held-out block of the training record, with the official metric.

The challenge withholds its test outputs, so this scores forecasts on sequences cut
from the end of the 12,800-snapshot training record. Every cavity run here holds that
block out (eval block = last 10%, snapshots 11,520-12,799). Each sequence is 100
consecutive snapshots: the first 70 are given and the next 30 are forecast, as in the
challenge. Starts are spaced 40 snapshots apart, so the 30 sequences have
non-overlapping forecast windows. The metric is the one in Challenge2_1_evaluate.m:
per-step NMSE of the mean-removed fields, mean(dq^2) / mean(q^2) over both velocity
components, then the mean and standard deviation across sequences.

Methods scored on the same sequences:
    persistence   hold input frame 70
    mean          zero fluctuation (NMSE = 1 by construction)
    DMD-25        port of the official baseline (papers/challenge_2.1/DMD), fit to the 70 inputs
    DLS floor     the latent's projection of the true future frames (compression floor)
    <latent>/<run>  trained runs given on the command line, seeded with the last
                  time_lag of the 70 given frames

The DLS bases cover 151x49 of the 154x51 grid, so every method is scored on that region.

    python scripts/challenge_eval.py cavity_dls_p13m5/spatial_gram_rollout_galerkin ...
    python scripts/challenge_eval.py --check-dmd dmd_matlab.h5   # port vs MATLAB (scripts/challenge/dmd_check.m)
    python scripts/challenge_eval.py --plot                      # figures/challenge_heldout.pdf
"""
import json
import logging
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

DATA = 'data/Challenge2_1_train.h5'
CH = 'papers/challenge_2.1/'
NAME = 'cavity'
BLOCK = (11520, 12800)          # held out by every cavity run (pred_split 0.1)
N_IN, N_OUT, STRIDE = 70, 30, 40
OUT = 'results/cavity_challenge_eval.json'
REPORT = (1, 2, 5, 10, 20, 30)
PLOT_KEYS = ['persistence', 'mean', 'DMD-25', 'cavity_dls_p13m5/local_gram',
             'cavity_dls_p13m5/spatial_gram_rollout_galerkin', 'DLS floor (cavity_dls_p13m5)']
PLOT_LABELS = ['persistence', 'mean flow', 'DMD, 25 modes', 'element-local temporal transformer',
               'spatial transformer, Galerkin', 'DLS floor (p13m5)']


def dmd_forecast(Q, n_modes=25, n_fore=N_OUT):
    """Exact DMD of the columns of Q (official utils/dmd.m), then Psi (b .* lambda^k)."""
    Q1, Q2 = Q[:, :-1], Q[:, 1:]
    U, s, Vh = np.linalg.svd(Q1, full_matrices=False)
    tol = max(len(s) * np.spacing(s[0]), 1e-12 * s[0])
    r = min(n_modes, int((s > tol).sum()), U.shape[1])
    U, s, V = U[:, :r], s[:r], Vh[:r].conj().T
    Vs = V / s
    L, W = np.linalg.eig(U.conj().T @ Q2 @ Vs)
    Psi = Q2 @ Vs @ W
    b = np.linalg.lstsq(Psi, Q[:, -1].astype(complex), rcond=None)[0]   # MATLAB Psi \ q
    return np.stack([np.real(Psi @ (b * L ** k)) for k in range(1, n_fore + 1)], axis=1)


def nmse(pred, true):
    """(steps,) NMSE for (steps, ...) arrays, both mean-removed."""
    ax = tuple(range(1, true.ndim))
    return ((pred - true) ** 2).mean(axis=ax) / ((true ** 2).mean(axis=ax) + np.finfo(float).eps)


def check_dmd(matlab_file):
    """The port against the official MATLAB steps run on the first test sequences.

    matlab_file comes from scripts/challenge/dmd_check.m (Challenge2_1_DMD.m's forecast
    loop on test sequences 1-3). The shipped DMD/Challenge2_1_test_DMD.h5 is not a usable
    reference: it stores full fields (mean included), lacks the ux_mean dataset the
    current script writes, and differs from what that script produces, so it came from
    an earlier version of it.
    """
    with h5py.File(DATA, 'r') as f:
        mean = f['mean'][:]
    with h5py.File(CH + 'data/Challenge2_1_test_input.h5', 'r') as f:
        ux, uy = f['ux'][:3], f['uy'][:3]                   # (seq, 70, 154, 51)
    with h5py.File(matlab_file, 'r') as f:
        ref = f['out'][:]                                   # (seq, 30, 2P)
    for i in range(ref.shape[0]):
        # MATLAB's column-major flattening of [ny nx] equals C-order flattening of (x, y)
        Q = np.concatenate([(ux[i] - mean[..., 0]).reshape(N_IN, -1),
                            (uy[i] - mean[..., 1]).reshape(N_IN, -1)], axis=1).T
        F = dmd_forecast(Q).T
        d = np.linalg.norm(F - ref[i]) / np.linalg.norm(ref[i])
        print(f"test sequence {i + 1}: DMD port vs MATLAB, relative difference {d:.2e}")


def plot(keys, labels, out='figures/challenge_heldout'):
    """Mean NMSE per step +/- std across sequences for the chosen methods."""
    import matplotlib.pyplot as plt
    import lib.plotting as pl
    res = json.load(open(OUT))
    fig, ax = plt.subplots(figsize=(0.62 * pl.width, 0.45 * pl.width))
    t = np.arange(1, N_OUT + 1)
    for k, lab in zip(keys, labels):
        m, sd = np.array(res[k]['mean']), np.array(res[k]['std'])
        if k == 'mean' or k.startswith('DLS floor'):
            ax.plot(t, m, color='0.45', ls=':' if k == 'mean' else '-.', lw=1.2, label=lab)
            continue
        line, = ax.plot(t, m, lw=1.5, label=lab, ls='--' if k == 'persistence' else '-')
        ax.fill_between(t, m - sd, m + sd, color=line.get_color(), alpha=0.15, lw=0)
    ax.set_xlim(1, N_OUT)
    ax.set_yscale('log')
    ax.set_xlabel('forecast step')
    ax.set_ylabel('NMSE')
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.3)
    ax.legend(frameon=False, fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
    print(f"wrote {out}.pdf/.png")


def model_forecasts(spec, starts, nx, ny):
    """(n_seq, 30, nx, ny, 2) decoded forecasts of a trained run, seeded from the given frames."""
    import torch  # noqa: F401  (runner needs CUDA)
    from lib.runner import runner
    latent, run = spec.split('/')
    c = json.load(open(f'results/{latent}/{run}/{run}.json'))
    c.update(overwrite='x', mode='eval', name=run, log='terminal', distributed=False, device='cuda')
    r = runner(c)
    tl = c['model_params']['time_lag']
    out = []
    with h5py.File(r.paths_bib.latent_path, 'r') as f:
        g = f[NAME]
        for s in starts:
            lo = s + N_IN - tl
            du, dv = g['dof_u'][lo:s + N_IN], g['dof_v'][lo:s + N_IN]
            pu, pv, _ = r._predict_dofs_forward(name=NAME, ids=np.arange(lo, s + N_IN + N_OUT),
                                                dof_u=du, dof_v=dv, total_steps=tl + N_OUT)
            Q = r.dls.gfem_recon_flexible(config=r.l_config, dof_u=pu[tl:].astype(np.float32),
                                          dof_v=pv[tl:].astype(np.float32), rec_target=None)
            out.append(Q[:, :nx, :ny])
    return np.stack(out), c['model_params']['model_type'], tl


def summarize(e):
    """e: (n_seq, steps) NMSE -> scalars."""
    m, sd = e.mean(0), e.std(0, ddof=1)
    above = np.flatnonzero(m > 1.0)
    return dict(mean=m.tolist(), std=sd.tolist(), window_mean=float(e.mean()),
                window_mean_seq_std=float(e.mean(1).std(ddof=1)),
                first_step_above_1=int(above[0]) + 1 if len(above) else -1)


def main(specs):
    logging.basicConfig(level=logging.WARNING)
    a, b = BLOCK
    starts = np.arange(a, b - (N_IN + N_OUT) + 1, STRIDE)
    with h5py.File(DATA, 'r') as f:
        UV = f['UV'][a:b].astype(np.float64)
        mean = f['mean'][:].astype(np.float64)
    q = UV - mean                                            # mean-removed truth
    nx, ny = 151, 49                                         # region every DLS basis covers
    true = np.stack([q[s - a + N_IN:s - a + N_IN + N_OUT, :nx, :ny] for s in starts])

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res['_protocol'] = dict(block=list(BLOCK), starts=starts.tolist(), n_in=N_IN, n_out=N_OUT,
                            region=[nx, ny], metric='official NMSE of mean-removed fields')
    E = {}
    E['persistence'] = np.stack([nmse(np.repeat(q[s - a + N_IN - 1][None, :nx, :ny], N_OUT, 0), t)
                                 for s, t in zip(starts, true)])
    E['mean'] = np.ones((len(starts), N_OUT))
    dmd = []
    for s, t in zip(starts, true):
        hist = q[s - a:s - a + N_IN]                          # full 154x51 grid, as the baseline
        Q = hist.transpose(0, 3, 1, 2).reshape(N_IN, -1).T
        F = dmd_forecast(Q).T.reshape(N_OUT, 2, *hist.shape[1:3]).transpose(0, 2, 3, 1)
        dmd.append(nmse(F[:, :nx, :ny], t))
    E['DMD-25'] = np.stack(dmd)
    for latent in sorted({sp.split('/')[0] for sp in specs}):
        with h5py.File(f'results/{latent}/{NAME}_codec_ref_rec.h5', 'r') as f:
            E[f'DLS floor ({latent})'] = np.stack([nmse(f['Q_rec'][s + N_IN:s + N_IN + N_OUT].astype(np.float64), t)
                                                   for s, t in zip(starts, true)])
    for sp in specs:
        F, mtype, tl = model_forecasts(sp, starts, nx, ny)
        E[sp] = np.stack([nmse(p.astype(np.float64), t) for p, t in zip(F, true)])
        res.setdefault('_runs', {})[sp] = dict(model_type=mtype, time_lag=tl)
    for k, e in E.items():
        res[k] = summarize(e)
    json.dump(res, open(OUT, 'w'), indent=1)

    print(f"{len(starts)} held-out sequences in snapshots {a}-{b - 1}; NMSE mean (std across sequences)")
    print(f"{'method':44}" + ''.join(f"{'step ' + str(k):>15}" for k in REPORT) + f"{'1-30':>9}{'>1 at':>7}")
    for k in E:
        s = res[k]
        print(f"{k:44}" + ''.join(f"{s['mean'][j - 1]:>8.3f} ({s['std'][j - 1]:.2f})" for j in REPORT)
              + f"{s['window_mean']:>9.3f}{s['first_step_above_1']:>7}")
    print(f"wrote {OUT}")


if __name__ == '__main__':
    if '--check-dmd' in sys.argv:
        check_dmd(sys.argv[sys.argv.index('--check-dmd') + 1])
    elif '--plot' in sys.argv:
        plot(PLOT_KEYS, PLOT_LABELS)
    else:
        main([a for a in sys.argv[1:] if not a.startswith('-')])
