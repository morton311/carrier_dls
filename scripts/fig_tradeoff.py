"""NMSE against fluctuation energy on the Challenge 2.1 cavity (figures/tradeoff_cavity.{pdf,png}).

For any forecast F of a mean-removed field q, with E = |F|^2 / |q|^2 and c the pattern correlation,

    NMSE = 1 + E - 2 c sqrt(E)

holds exactly for every frame (the law of cosines behind a Taylor diagram). At fixed c the error is
smallest at E = c^2, where NMSE = 1 - c^2, and matching the energy (E = 1) costs NMSE = 2 (1 - c).

Multiplying a saved forecast by a lead-dependent gain g(h) moves it along its own curve at no training cost, so
this is the reference a statistically trained model has to beat. The gains are set from the seven
episodes in our validation block (snapshots before 11,520) and scored on the 30 held-out sequences.

Panel (b) collects every cavity run on the current split with a valid eval: official NMSE against the
energy kept in the free rollout. For generative models the two numbers come from different products:
NMSE from the ensemble mean, the energy from one sampled trajectory, so each also gets a point for one sample's NMSE.

    python scripts/slick_comparison.py infer      # needed once
    python scripts/fig_tradeoff.py                # prints a table, writes the figure and results/tradeoff_cavity.json
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

from challenge_eval import dmd_forecast            # noqa: E402
from slick_baseline import load_model, predict     # noqa: E402
from slick_comparison import CLEAN, DATA, N_IN, N_OUT, OUT, OURS, THEIRS, field_metrics, load_truth   # noqa: E402

CAL = THEIRS[~CLEAN]                  # forecast windows inside our validation block
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#6f6e69'
THETA = np.linspace(-0.6, 1.1, 171)   # gain exponent: 0 is the forecast as it is, 1 matches the energy at each lead


def read_block(lo):
    with h5py.File(DATA, 'r') as f:
        return f['UV'][lo:].astype(np.float32)


def slick_fields(model, starts, shape, mean):
    uv = read_block(int(starts.min()))
    lo = int(starts.min())
    q = uv.transpose(3, 0, 1, 2).reshape(2, uv.shape[0], -1).transpose(0, 2, 1).reshape(-1, uv.shape[0])
    return np.stack([predict(model, q[:, s - lo:s - lo + N_IN]).T.reshape(N_OUT, 2, *shape).transpose(0, 2, 3, 1) for s in starts]) - mean


def summary(m):
    return dict(nmse=float(m['nmse'].mean()), energy=float(m['energy'].mean()), corr=float(m['corr'].mean()))


def scaled(test, cal):
    """(energy, NMSE) over gains g(h) = E_cal(h)^(-theta/2), with E_cal the forecast's mean energy ratio on the calibration episodes."""
    Ecal = cal['energy'].mean(0)
    out = []
    for th in THETA:
        g = (Ecal ** (-th / 2))[None, :]
        E2 = g ** 2 * test['energy']
        out.append((E2.mean(), (1 + E2 - 2 * g * test['corr'] * np.sqrt(test['energy'])).mean()))
    return np.array(out)


def collect():
    z = np.load(f'{OUT}/models.npz')
    idx = np.searchsorted(z['starts'], OURS)
    icl = np.searchsorted(z['starts'], CAL)
    grab = lambda key, ii: {k: z[f'{key}_{k}'][ii] for k in ('nmse', 'corr', 'energy')}
    test = {f'DLS s{i}': grab(f's{i}', idx) for i in (1, 2, 3)}
    cal = {f'DLS s{i}': grab(f's{i}', icl) for i in (1, 2, 3)}
    test['hybrid'], cal['hybrid'] = grab('hybrid', idx), grab('hybrid', icl)
    test['hybrid sample'], cal['hybrid sample'] = grab('hybrid_sample', idx), grab('hybrid_sample', icl)

    true, mean, shape = load_truth(OURS)
    tcal, _, _ = load_truth(CAL)
    model = load_model()
    test['SLICK'] = field_metrics(slick_fields(model, OURS, shape, mean), true)
    cal['SLICK'] = field_metrics(slick_fields(model, CAL, shape, mean), tcal)

    uv = read_block(int(OURS.min()))
    lo = int(OURS.min())
    q = uv.astype(np.float64) - mean
    dmd = np.stack([dmd_forecast(q[s - lo:s - lo + N_IN].transpose(0, 3, 1, 2).reshape(N_IN, -1).T).T.reshape(N_OUT, 2, *shape)
                    .transpose(0, 2, 3, 1) for s in OURS])
    test['DMD-25'] = field_metrics(dmd, true)
    pers = np.stack([np.repeat(q[s - lo + N_IN - 1][None], N_OUT, 0) for s in OURS])
    test['persistence'] = field_metrics(pers, true)
    return test, cal


RUNS = 'cavity_dls_p13m5{}/spatial_gram_rollout_galerkin{}'
GROUPS = [   # label, marker, keys of results/cavity_challenge_eval.json
    (r'Deterministic, cut elements unregularized', 'o', [RUNS.format('_fc', s) for s in
     ('_fc', '_fc_all', '_fc_all_d128', '_fc_all_d64', '_fc_all_es', '_fc_all_tl8', '_fc_all_tl16', '_fc_all_d128_tl16')]),
    (r'Deterministic, $\alpha=0.01$', 's', ['cavity_dls_p13m5_fca0_01odd/spatial_gram_rollout_galerkin_fc_all_d128_ca0_01']),
    (r'Deterministic, $\alpha=0.1$, 3 seeds', 'D', ['cavity_dls_p13m5_fca0_1odd/spatial_gram_rollout_galerkin_fc_all_d128_ca0_1' + s
                                                     for s in ('', '_s2', '_s3')]),
    (r'Deterministic, truncated mesh', '^', [RUNS.format('', '')]),
]
GENERATIVE = [
    (r'Hybrid, multi-step', 'cavity_dls_p13m5_fc/spatial_diff_res_fc_all_d128_ms8'),
    (r'Hybrid, multi-step, $\alpha=0.1$', 'cavity_dls_p13m5_fca0_1odd/spatial_diff_res_fc_all_d128_ca0_1_ms8'),
    (r'Hybrid, one-step', 'cavity_dls_p13m5_fc/spatial_diff_res_fc_all_d128'),
    (r'Pure diffusion', 'cavity_dls_p13m5_fc/spatial_diff_fc_all_d128'),
]


def run_metrics(key):
    lat, run = key.split('/')
    return json.load(open(f'results/{lat}/{run}/saved_metrics/cavity_eval_data.json'))['metrics']


def main():
    test, cal = collect()
    # the identity, frame by frame
    worst = max(np.abs(m['nmse'] - (1 + m['energy'] - 2 * m['corr'] * np.sqrt(m['energy']))).max() for m in test.values())
    print(f'identity NMSE = 1 + E - 2 c sqrt(E): largest residual over all frames {worst:.2e}')

    dls_keys = [f'DLS s{i}' for i in (1, 2, 3)]
    avg = lambda ms: {k: np.mean([m[k] for m in ms], axis=0) for k in ms[0]}
    curves = {'DLS': np.mean([scaled(test[k], cal[k]) for k in dls_keys], axis=0), 'SLICK': scaled(test['SLICK'], cal['SLICK']),
              'hybrid': scaled(test['hybrid'], cal['hybrid'])}
    res = {k: summary(m) for k, m in test.items()}
    res['DLS mean'] = summary(avg([test[k] for k in dls_keys]))
    for k, c in curves.items():
        j, nat, full = int(np.argmin(c[:, 1])), int(np.argmin(np.abs(THETA))), int(np.argmin(np.abs(THETA - 1.0)))
        res[f'{k} scaled'] = dict(as_is_nmse=float(c[nat, 1]), as_is_energy=float(c[nat, 0]), min_nmse=float(c[j, 1]), energy_at_min=float(c[j, 0]),
                                  nmse_energy_matched=float(c[full, 1]), energy_matched=float(c[full, 0]))
    for k in ('DLS', 'SLICK'):
        m = avg([test[i] for i in dls_keys]) if k == 'DLS' else test[k]
        res[f'{k} identity'] = dict(lowest_possible_given_c=float((1 - m['corr'].mean(0) ** 2).mean()), energy_matched_given_c=float((2 * (1 - m['corr'].mean(0))).mean()))
    for k, v in res.items():
        print(k, {a: round(b, 3) for a, b in v.items()})

    ch = json.load(open('results/cavity_challenge_eval.json'))
    import matplotlib.pyplot as plt
    import lib.plotting as pl
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(pl.width, 0.66 * pl.width))

    # (a) the geometry: every forecast lies on the curve of its own correlation
    E = np.linspace(0, 1.3, 300)
    for c in (0.2, 0.4, 0.6, 0.8):
        ax1.plot(E, 1 + E - 2 * c * np.sqrt(E), color='0.8', lw=0.7, zorder=1)
        top = 1 + 1.3 - 2 * c * np.sqrt(1.3)
        if top <= 1.5:      # label at the right edge, or where the curve leaves through the top
            ax1.annotate(rf'$c={c}$', (1.3, top), xytext=(2, 0), textcoords='offset points', fontsize=6.5, va='center',
                         color='0.4', annotation_clip=False)
        else:
            x = (c + np.sqrt(c ** 2 + 0.5)) ** 2
            ax1.annotate(rf'$c={c}$', (x, 1.5), xytext=(0, 2), textcoords='offset points', fontsize=6.5, ha='center',
                         va='bottom', color='0.4', annotation_clip=False)
    cs = np.linspace(0, 1, 200)
    ax1.plot(cs ** 2, 1 - cs ** 2, color='k', lw=1.1, zorder=2, label=r'Lowest error for a given $c$')
    ax1.plot(curves['DLS'][:, 0], curves['DLS'][:, 1], color=BLUE, lw=1.3, zorder=3, label='DLS transformer, rescaled')
    ax1.plot(curves['SLICK'][:, 0], curves['SLICK'][:, 1], color=AQUA, lw=1.3, ls='--', zorder=3, label='SLICK, rescaled')
    mark = dict(ms=6, mec='white', mew=0.6, ls='none', zorder=5)
    d = res['DLS mean']
    ax1.plot(d['energy'], d['nmse'], 'D', color=BLUE, label='DLS transformer', **mark)
    ax1.plot(res['SLICK']['energy'], res['SLICK']['nmse'], 'v', color=AQUA, label='SLICK', **mark)
    ax1.plot(res['hybrid']['energy'], res['hybrid']['nmse'], 'o', color=ORANGE, label='Hybrid, ensemble mean', **mark)
    ax1.plot(res['hybrid sample']['energy'], res['hybrid sample']['nmse'], 'P', color=ORANGE, label='Hybrid, one sample', **mark)
    ax1.plot(res['DMD-25']['energy'], res['DMD-25']['nmse'], 'X', color=GRAY, label='DMD, 25 modes', **mark)
    ax1.set_xlim(0, 1.3)
    ax1.set_ylim(0.5, 1.5)
    ax1.set_xlabel(r'Energy kept over the 30 steps, $E$')
    ax1.set_ylabel('30-step NMSE')
    ax1.set_title('(a) Error is set by correlation and energy', fontsize=9, pad=11)
    ax1.grid(True, ls='--', lw=0.3, color='0.88')
    ax1.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncols=2, columnspacing=0.8, handletextpad=0.4)

    # (b) the runs we already have
    for label, marker, keys in GROUPS:
        pts = np.array([(ch[k]['window_mean'], run_metrics(k)['energy_retention_mean']) for k in keys])
        res.setdefault('runs', {})[label] = pts.tolist()
        ax2.plot(pts[:, 0], pts[:, 1], marker, color=BLUE, label=label, ms=5, mec='white', mew=0.5, ls='none', zorder=4)
    for (label, key), marker in zip(GENERATIVE, 'osPX'):
        m, one = ch[f'{key} (M=32)']['window_mean'], ch[f'{key} (M=32) [one sample]']['window_mean']
        en = run_metrics(key)['energy_retention_mean']
        res['runs'][label] = dict(ensemble_mean_nmse=m, one_sample_nmse=one, energy=en)
        ax2.plot([m, one], [en, en], color=ORANGE, lw=0.8, zorder=2)
        ax2.plot(m, en, marker, mfc='white', mec=ORANGE, ms=5, mew=1.0, ls='none', zorder=4)
        ax2.plot(one, en, marker, color=ORANGE, mec='white', mew=0.5, ms=5.5, ls='none', zorder=4, label=label)
    ax2.axvline(1, color='0.55', lw=0.7, ls=':')
    ax2.text(1.005, 0.015, 'mean flow', fontsize=6.5, color='0.4', rotation=90, va='bottom')
    ax2.set_xlim(0.7, 1.35)
    ax2.set_ylim(0, 0.8)
    ax2.set_xlabel('30-step NMSE')
    ax2.set_ylabel('Energy kept, free rollout')
    ax2.set_title('(b) Scored runs (open: ensemble mean)', fontsize=9, pad=11)
    ax2.grid(True, ls='--', lw=0.3, color='0.88')
    ax2.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncols=2, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(w_pad=1.4)
    fig.savefig('figures/tradeoff_cavity.pdf', bbox_inches='tight')
    fig.savefig('figures/tradeoff_cavity.png', dpi=250, bbox_inches='tight')
    json.dump(res, open('results/tradeoff_cavity.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
