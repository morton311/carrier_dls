"""Which scales are recovered: energy by streamwise wavenumber band on the cavity's 30 held-out sequences.

Compares forecasts (as they are, and rescaled to match the energy at each lead) with the truth, and measures how much of
each band each GFEM enrichment level can represent at all (its projection of the truth; no model involved).

    python scripts/tradeoff_scales.py     # prints a table, writes results/tradeoff_scales.json
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

import lib.dls_2d as dls       # noqa: E402
import lib.plotting as pl      # noqa: E402
from challenge_eval import _to_grid, model_forecasts  # noqa: E402
from slick_comparison import CLEAN, N_IN, N_OUT, OUT, OURS, SEEDS, HYBRID, THEIRS, load_truth  # noqa: E402

CAL = THEIRS[~CLEAN]
BANDS = {'k<5 (wavelength over 31 pts)': (0, 5), '5-19 (8-31 pts)': (5, 20), '20+ (under 8 pts)': (20, 78)}
LATENTS = {'5 modes, alpha 0.1 (paper)': 'cavity_dls_p13m5_fca0_1odd', '5 modes, alpha 0': 'cavity_dls_p13m5_fc',
           '10 modes, alpha 0.1': 'cavity_dls_p13m10_fca0_1odd',
           '10 modes, truncated mesh (older)': 'cavity_dls_p13m10', '20 modes, truncated mesh (older)': 'cavity_dls_p13m20'}


def spectrum(F):
    """Mean energy per streamwise wavenumber of (..., nx, ny, 2) fields, Hann-windowed in x."""
    w = np.hanning(F.shape[-3])[:, None, None]
    return (np.abs(np.fft.rfft(F.reshape(-1, *F.shape[-3:]) * w, axis=1)) ** 2).mean(axis=(0, 2, 3))


def bands(S, St):
    return {k: float(S[a:b].sum() / St[a:b].sum()) for k, (a, b) in BANDS.items()}


def rms_err(F, T):
    a, b = np.sqrt((T.reshape(-1, *T.shape[-3:]) ** 2).mean(0)), np.sqrt((F.reshape(-1, *F.shape[-3:]) ** 2).mean(0))
    return 100 * float(pl.l2_err_norm(a, b))


def main():
    logging.basicConfig(level=logging.WARNING)
    true, mean, shape = load_truth(OURS)
    tcal, _, _ = load_truth(CAL)
    St = spectrum(true)
    res = dict(truth_energy_share={k: float(St[a:b].sum() / St.sum()) for k, (a, b) in BANDS.items()})
    print('share of the true energy per band', {k: round(v, 3) for k, v in res['truth_energy_share'].items()}, flush=True)

    # what each enrichment space can hold
    ceil = {}
    for label, lat in LATENTS.items():
        path = f'results/{lat}/latent_config.pkl'
        if not os.path.exists(path):
            print(label, 'no latent; skipped')
            continue
        cfg = pickle.load(open(path, 'rb'))
        t = np.concatenate([np.arange(s + N_IN, s + N_IN + N_OUT) for s in OURS])
        with h5py.File(f'results/{lat}/latent_coeff.h5', 'r') as h:
            du, dv = h['cavity']['dof_u'][t], h['cavity']['dof_v'][t]
        Q = _to_grid(dls.decode_fields(cfg, du, dv)[:, :shape[0], :shape[1], :].astype(np.float64), shape)
        T = true.reshape(-1, *true.shape[2:])
        ceil[label] = dict(bands(spectrum(Q), St), compression_nmse=float(((Q - T) ** 2).sum() / (T ** 2).sum()), coefficients=int(du.shape[1] + dv.shape[1]))
        print('ceiling', label, {k: round(v, 3) for k, v in ceil[label].items()}, flush=True)
    res['ceiling'] = ceil

    # forecasts, as they are and energy-matched per lead (gain set on the calibration episodes)
    both = np.concatenate([CAL, OURS])
    rows = {}
    out = {}
    fc, _, _ = model_forecasts(SEEDS[0], both, shape, None)
    out['DLS transformer, seed 1'] = fc['']
    fc, _, _ = model_forecasts(HYBRID, both, shape, 8)
    for k, v in fc.items():
        out['Hybrid, one sample' if 'sample' in k else 'Hybrid, ensemble mean'] = v
    n = len(CAL)
    for name, F in out.items():
        F = F.astype(np.float64)
        Ecal = ((F[:n] ** 2).sum((2, 3, 4)) / (tcal ** 2).sum((2, 3, 4))).mean(0)
        G = F[n:]
        H = G * (Ecal ** -0.5)[None, :, None, None, None]
        e = lambda X: float(((X - true) ** 2).sum((2, 3, 4)).__truediv__((true ** 2).sum((2, 3, 4))).mean())
        rows[name] = dict(as_is=dict(nmse=e(G), rms_err=rms_err(G, true), **bands(spectrum(G), St)),
                          energy_matched=dict(nmse=e(H), rms_err=rms_err(H, true), **bands(spectrum(H), St)))
        print(name, {a: {b: round(c, 3) for b, c in d.items()} for a, d in rows[name].items()}, flush=True)
    res['forecasts'] = rows
    json.dump(res, open('results/tradeoff_scales.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
