"""Global POD modes of the Re 30k lid-driven cavity (paper figure).

Plots selected spatial modes of the training-block POD used by the POD baseline
(results/ldc_30k_10k_pod_m400_tr: 400 modes per velocity component, fit to the
first 7,200 snapshots). The u and v fluctuations have separate bases, so mode k
of u and mode k of v are unrelated. Each panel is scaled by its own max |value|
(the sign of a mode is arbitrary). Panel titles give the mode's share of the
training-block fluctuating energy of that component.

    python scripts/fig_pod_modes.py [k1 k2 ...]   # default 1 2 5 10 100
    # writes figures/pod_modes_30k.pdf/.png
"""
import os
import pickle
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

LAT = 'results/ldc_30k_10k_pod_m400_tr/'
DATA = 'data/ldc_30k_60ksnaps.h5'
FIT = 7200                     # training block [0, FIT) the basis was fit to


def main():
    import matplotlib.pyplot as plt
    import lib.plotting as pl          # repo figure style (paper size, LaTeX text)

    ks = [int(a) for a in sys.argv[1:]] or [1, 2, 5, 10, 100]
    cfg = pickle.load(open(LAT + 'latent_config.pkl', 'rb'))
    nx, ny = cfg.spatial
    with h5py.File(DATA, 'r') as f:
        # the source is stored [y, x]; POD transposes on read, so the modes are [x, y]
        xg, yg = f['x_grid'][:].T, f['y_grid'][:].T
    # modal energies: training-block sums of squared coefficients (orthonormal modes);
    # 400 modes hold all but ~1e-5 of the training-block energy, so the sum is the total
    with h5py.File(LAT + 'latent_coeff.h5', 'r') as f:
        g = f['ldc_30k_10k']
        lam = [(g[key][:FIT].astype(np.float64) ** 2).sum(0) for key in ('dof_u', 'dof_v')]

    fig, axs = plt.subplots(2, len(ks), figsize=(pl.width, 0.43 * pl.width),
                            sharex=True, sharey=True)
    for r, (comp, M) in enumerate(zip(('u', 'v'), cfg.modes)):
        for c, k in enumerate(ks):
            q = np.asarray(M[:, k - 1]).reshape(nx, ny)
            q = q / np.abs(q).max()
            ax = axs[r, c]
            im = ax.pcolormesh(xg, yg, q, cmap='RdBu_r', vmin=-1, vmax=1,
                               shading='auto', rasterized=True)
            ax.set_aspect('equal')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            share = 100 * lam[r][k - 1] / lam[r].sum()
            ax.set_title(rf'${comp}$, mode {k} ({share:.2g}\%)', fontsize=8)
            if r == 1:
                ax.set_xlabel('$x$')
            if c == 0:
                ax.set_ylabel('$y$')
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02)
    cb.set_label(r'$\phi_k / \max|\phi_k|$')
    cb.set_ticks([-1, 0, 1])
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/pod_modes_30k.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('figures/pod_modes_30k.png', dpi=200, bbox_inches='tight')
    for comp, l in zip(('u', 'v'), lam):
        cum = np.cumsum(l) / l.sum()
        print(comp, 'energy share', {k: round(float(l[k - 1] / l.sum()), 4) for k in ks},
              'cumulative', {k: round(float(cum[k - 1]), 4) for k in ks})


if __name__ == '__main__':
    main()
