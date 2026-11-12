"""Figure: compression of the Challenge 2.1 PIV by DLS element size and enrichment, against global POD.

(a) energy share and frame-to-frame coherence of the held-out truth by streamwise wavelength band;
(b) compression NMSE against the number of coefficients; (c) the same in the temporally coherent field
(lag-1 covariances, which white measurement noise does not enter). Data: results/compression_study/cavity.json
(scripts/compression_study.py).

    python scripts/fig_compression.py        # writes figures/compression_study.pdf
"""
import json
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
LO, HI = 11524, 12800
BANDS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 78)]
# one hue, light to dark with element size (an ordered quantity); POD in neutral ink
RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#0d366b']
INK, MUTED = '#0b0b0b', '#52514e'


def band_coherence():
    with h5py.File('data/Challenge2_1_train.h5', 'r') as f:
        T = (f['UV'][LO:HI] - f['mean'][:][None]).astype(np.float64)
    F = np.fft.rfft(T * np.hanning(T.shape[1])[None, :, None, None], axis=1)
    E = (np.abs(F) ** 2).sum(0)
    C = np.abs((F[1:] * np.conj(F[:-1])).sum(0))
    share = np.array([E[a:b].sum() for a, b in BANDS]) / E[1:].sum()
    coh = np.array([C[a:b].sum() / E[a:b].sum() for a, b in BANDS])
    return share, coh, np.sqrt(np.pi / 4 / F.shape[0])


def main():
    c = json.load(open('results/compression_study/cavity.json'))
    rows = [v for k, v in c.items() if k.startswith('p') and k.endswith('_s0')]
    patches = sorted({r['patch'] for r in rows})
    pod = sorted([v for k, v in c.items() if k.startswith('pod')], key=lambda v: v['coefficients'])
    share, coh, bias = band_coherence()

    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5), constrained_layout=True)

    a = ax[0]
    x = np.arange(len(BANDS))
    a.bar(x, 100 * share, width=0.7, color='#86b6ef', edgecolor='white', linewidth=1)
    a.set_ylabel('share of fluctuation energy (%)')
    a.set_xticks(x, [f'{154 / lo:.0f}' if lo < 32 else f'{154 / lo:.1f}' for lo, _ in BANDS])
    a.set_xlabel('longest wavelength in band (grid points)')
    for xi, s, r in zip(x, share, coh):
        a.text(xi, 100 * s + 1, f'{r:.2f}', ha='center', va='bottom', fontsize=6.5, color=INK)
    a.set_ylim(0, 50)
    a.set_title('(a) truth by wavelength band', fontsize=8, loc='left', color=INK)
    a.text(0.98, 0.62, f'labels: lag-1 coherence\n(white noise {bias:.3f})', transform=a.transAxes, ha='right', fontsize=6.5, color=MUTED)

    for panel, key, title in ((ax[1], 'nmse', '(b) compression NMSE'),
                              (ax[2], 'nmse_coh', '(c) coherent-field NMSE')):
        for i, p in enumerate(patches):
            rr = sorted([r for r in rows if r['patch'] == p], key=lambda r: r['coefficients'])
            n = [r['coefficients'] for r in rr]
            y = [max(r[key], 1e-6) for r in rr]
            panel.plot(n, y, color=RAMP[i], lw=1.5, label=f'{p}$^2$')
        panel.plot([v['coefficients'] for v in pod], [max(v[key], 1e-6) for v in pod], color=INK, lw=1.5, ls='--',
                   label='global POD')
        ref = c['p13m5_s0']
        panel.plot(ref['coefficients'], max(ref[key], 1e-6), 'o', ms=6, mfc='white', mec=INK, mew=1.5, zorder=5)
        panel.annotate('p13m5', (ref['coefficients'], max(ref[key], 1e-6)), xytext=(-36, -9) if key == 'nmse_coh' else (6, 4),
                       textcoords='offset points', fontsize=6.5, color=INK)
        panel.set_xscale('log')
        panel.set_yscale('log')
        panel.set_xlabel('latent coefficients')
        panel.set_title(title, fontsize=8, loc='left', color=INK)
        panel.grid(True, which='major', color='#e5e4e0', lw=0.5)
    ax[2].set_ylim(5e-5, 0.4)
    ax[1].set_ylim(5e-3, 0.5)
    ax[1].legend(title='patch', fontsize=6, title_fontsize=6.5, ncol=2, frameon=False, loc='lower left',
                 handlelength=1.5, columnspacing=0.8)
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/compression_study.pdf')
    fig.savefig('figures/compression_study.png', dpi=200)
    print('wrote figures/compression_study.pdf')


if __name__ == '__main__':
    main()
