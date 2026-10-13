"""Short-horizon forecast error of the held-out LDC runs (paper figure).

Pools the multi-start forecasts saved by eval (saved_metrics/*_eval_data.h5:
ms_err, ms_persist; 50 forecasts x 30 steps per run) over the three seeds of
each recipe, and plots mean +/- std across forecasts against persistence.

    python scripts/fig_multistart.py      # writes figures/multistart_heldout.pdf/.png
"""
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

LAT = 'results/ldc_15k_dls_p99m10/'
RECIPES = {
    'nodal average': ('spatial_gram_rollout_heldout_s{}', '#2a78d6', '-'),
    'Galerkin, trained in the loop': ('spatial_gram_rollout_galerkin_heldout_s{}', '#1baf7a', '-.'),
}


def load(pattern):
    err, pers = [], []
    for s in (1, 2, 3):
        with h5py.File(LAT + pattern.format(s) + '/saved_metrics/ldc_15k_eval_data.h5', 'r') as f:
            err.append(f['ms_err'][:])
            pers.append(f['ms_persist'][:])
    return np.concatenate(err), np.concatenate(pers)


def main():
    import matplotlib.pyplot as plt
    import lib.plotting as pl          # repo figure style (paper size, LaTeX text)

    fig, ax = plt.subplots(figsize=(0.62 * pl.width, 0.45 * pl.width))
    persist = None
    for label, (pattern, color, ls) in RECIPES.items():
        err, pers = load(pattern)
        persist = pers if persist is None else persist
        t = np.arange(1, err.shape[1] + 1)
        m, s = err.mean(0), err.std(0)
        ax.fill_between(t, m - s, m + s, color=color, alpha=0.18, lw=0)
        ax.plot(t, m, color=color, ls=ls, lw=1.6, label=label)
        print(f"{label:32} step 1 {m[0]:.4f}  step 10 {m[9]:.4f}  step 30 {m[29]:.4f}  "
              f"mean 1-30 {err.mean():.4f}  ({err.shape[0]} forecasts)")
    pm = persist.mean(0)
    ax.plot(t, pm, color='0.45', ls='--', lw=1.4, label='persistence')
    print(f"{'persistence':32} step 1 {pm[0]:.4f}  step 10 {pm[9]:.4f}  step 30 {pm[29]:.4f}  mean 1-30 {persist.mean():.4f}")
    ax.set_xlim(1, t[-1])
    # log scale: persistence grows to ~0.6 while the forecasts stay below ~0.07
    ax.set_yscale('log')
    ax.set_xlabel('forecast step')
    ax.set_ylabel(r'$\|\mathbf{u}-\hat{\mathbf{u}}\|/\|\mathbf{u}\|$')
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.3)
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/multistart_heldout.pdf', bbox_inches='tight')
    fig.savefig('figures/multistart_heldout.png', dpi=200, bbox_inches='tight')
    print("wrote figures/multistart_heldout.pdf/.png")


if __name__ == '__main__':
    main()
