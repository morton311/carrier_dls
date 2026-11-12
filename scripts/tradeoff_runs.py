"""Score study runs on the forecast-energy identity: NMSE, energy kept E, correlation c, and the free-gain baseline.

For each run, on the cavity's 30 held-out sequences: the 30-step NMSE, E and c; the energy kept per wavenumber band;
the NMSE of the parent model rescaled to the same E (the gain family of fig_tradeoff.py, set on the 7 calibration
episodes), so 'vs gain' < 0 means the run beat a free gain; and, for energy-term runs (loss mse + beta * frame_energy),
the field energy that the per-frame optimum sqrt(E) = (c + beta) / (1 + beta) predicts from the run's own correlation.
Per-frame metrics are cached in results/tradeoff_runs.npz, so a rerun only scores new runs.

    python scripts/tradeoff_runs.py      # prints a table, writes results/tradeoff_runs.json
"""
import json
import logging
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

from challenge_eval import model_forecasts   # noqa: E402
from slick_comparison import CLEAN, OURS, THEIRS, field_metrics, load_truth   # noqa: E402
from tradeoff_scales import bands, spectrum  # noqa: E402

CAL = THEIRS[~CLEAN]
CACHE = 'results/tradeoff_runs.npz'
L1 = 'cavity_dls_p13m5_fca0_1odd/spatial_gram_rollout_galerkin_fc_all_d128_ca0_1'
L0 = 'cavity_dls_p13m5_fc/spatial_gram_rollout_galerkin_fc_all_d128'
L10 = 'cavity_dls_p13m10_fca0_1odd/spatial_gram_rollout_galerkin_fc_all_d128_ca0_1_m10'
PARENT = [L1, L1 + '_s2', L1 + '_s3']           # the paper's model, 3 seeds: the gain baseline is its rescaled curve
RUNS = {   # label: (run, beta of the energy term or None)
    'alpha 0.1, s1': (L1, None), 'alpha 0.1, s2': (L1 + '_s2', None), 'alpha 0.1, s3': (L1 + '_s3', None),
    'alpha 0, s1': (L0, None), 'alpha 0, s2': (L0 + '_s2', None), 'alpha 0, s3': (L0 + '_s3', None),
    '10 modes, s1': (L10, None), '10 modes, s2': (L10 + '_s2', None), '10 modes, s3': (L10 + '_s3', None),
    'energy term 0.1': (L1 + '_fen0_1', 0.1), 'energy term 0.3': (L1 + '_fen0_3', 0.3),
    'energy term 1': (L1 + '_fen1', 1.0), 'energy term 3': (L1 + '_fen3', 3.0),
    'loss ridge 1e-3': (L1 + '_gl1e-3', None), 'loss ridge 1e-4': (L1 + '_gl1e-4', None), 'loss ridge 0': (L1 + '_gl0', None),
    'field energy 0.1': (L1 + '_fen0_1_field', 0.1), 'field energy 0.3': (L1 + '_fen0_3_field', 0.3),
    'field energy 1': (L1 + '_fen1_field', 1.0), 'field energy 3': (L1 + '_fen3_field', 3.0),
}
THETA = np.linspace(-1.0, 2.5, 351)


def gain_curve(test, cal):
    """(E, NMSE) of the forecast times g(h) = E_cal(h)^(-theta/2), over THETA."""
    Ecal = cal['energy'].mean(0)
    out = []
    for th in THETA:
        g = (Ecal ** (-th / 2))[None, :]
        E2 = g ** 2 * test['energy']
        out.append((E2.mean(), (1 + E2 - 2 * g * test['corr'] * np.sqrt(test['energy'])).mean()))
    return np.array(out)


def main():
    logging.basicConfig(level=logging.WARNING)
    true, mean, shape = load_truth(OURS)
    tcal, _, _ = load_truth(CAL)
    St = spectrum(true)
    both = np.concatenate([CAL, OURS])
    n = len(CAL)
    cache = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
    for run in sorted({r for r, _ in RUNS.values()}):
        if f'{run}|nmse' in cache:
            continue
        if not os.path.exists(f'results/{run}/model.pth'):
            print(run, 'not trained; skipped')
            continue
        F = model_forecasts(run, both, shape, None)[0][''].astype(np.float64)
        for part, Fp, T in (('', F[n:], true), ('cal_', F[:n], tcal)):
            for k, v in field_metrics(Fp, T).items():
                cache[f'{run}|{part}{k}'] = v
        cache[f'{run}|bands'] = np.array(list(bands(spectrum(F[n:]), St).values()))
        np.savez(CACHE, **cache)
        print('scored', run, flush=True)

    grab = lambda run, part='': {k: cache[f'{run}|{part}{k}'] for k in ('nmse', 'corr', 'energy')}
    curve = np.mean([gain_curve(grab(r), grab(r, 'cal_')) for r in PARENT], axis=0)
    res = dict(parent=PARENT, gain_curve=curve.tolist(), runs={})
    print(f"{'':18}{'NMSE':>7}{'E':>7}{'c':>7}{'vs gain':>9}{'E pred':>8}{'c@1':>7}{'c@10':>7}{'c@30':>7}"
          f"{'>31pt':>7}{'8-31':>7}{'<8pt':>7}")
    for label, (run, beta) in RUNS.items():
        if f'{run}|nmse' not in cache:
            continue
        m = grab(run)
        E, c = float(m['energy'].mean()), float(m['corr'].mean())
        r = dict(run=run, beta=beta, nmse=float(m['nmse'].mean()), energy=E, corr=c,
                 nmse_vs_gain=float(m['nmse'].mean() - np.interp(E, curve[:, 0], curve[:, 1])),
                 corr_by_lead=m['corr'].mean(0).tolist(), energy_by_lead=m['energy'].mean(0).tolist(),
                 nmse_by_lead=m['nmse'].mean(0).tolist(),
                 bands=dict(zip(('over 31 pts', '8-31 pts', 'under 8 pts'), cache[f'{run}|bands'].tolist())))
        if beta is not None:
            r['energy_predicted'] = float((((np.clip(m['corr'], -1, 1) + beta) / (1 + beta)) ** 2).mean())
        res['runs'][label] = r
        cl = r['corr_by_lead']
        print(f"{label:18}{r['nmse']:>7.3f}{E:>7.3f}{c:>7.3f}{r['nmse_vs_gain']:>+9.3f}"
              + (f"{r['energy_predicted']:>8.3f}" if beta is not None else f"{'':>8}")
              + f"{cl[0]:>7.3f}{cl[9]:>7.3f}{cl[29]:>7.3f}" + ''.join(f'{v:>7.2f}' for v in r['bands'].values()))
    json.dump(res, open('results/tradeoff_runs.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
