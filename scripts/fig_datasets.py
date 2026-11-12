"""Figure: the flows used in the paper, one instantaneous snapshot each, with the DLS element mesh.

(a, b) lid-driven cavity vorticity at Re 15,000 and 30,000 (DNS); (c, d) Challenge 2.1 open-cavity PIV, streamwise
velocity and wall-normal velocity fluctuation. Thin lines are the finite elements (tokens) of the DLS mesh.

    python scripts/fig_datasets.py        # writes figures/datasets.pdf
"""
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
INK, MUTED, MESH = '#0b0b0b', '#52514e', '#3a3936'
LDC = [('data/ldc_15k.h5', 2000, r'(a) $\mathrm{Re}=15{,}000$'),
       ('data/ldc_30k_60ksnaps.h5', 9000, r'(b) $\mathrm{Re}=30{,}000$')]
CAVITY, CAV_SNAP = 'data/Challenge2_1_train.h5', 12000
LDC_NSKIP, CAV_NSKIP = 49, 6


def mesh_lines(ax, xs, ys, **kw):
    """Element edges at the given node coordinates, clipped to the axes."""
    for x in xs:
        ax.plot([x, x], [ys.min(), ys.max()], **kw)
    for y in ys:
        ax.plot([xs.min(), xs.max()], [y, y], **kw)


def main():
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED})
    fig = plt.figure(figsize=(7.0, 4.75), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.9, 1.0])
    ldc_axes = []
    kw = dict(color=MESH, lw=0.4, alpha=0.55)

    for k, (path, snap, title) in enumerate(LDC):
        with h5py.File(path, 'r') as f:                      # stored as [y, x, component]
            q, x, y = f['UV'][snap], f['x'][:], f['y'][:]
        w = np.gradient(q[..., 1], x, axis=1) - np.gradient(q[..., 0], y, axis=0)
        lim = 12.0
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(w, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='RdBu_r', vmin=-lim, vmax=lim,
                       interpolation='bilinear', rasterized=True)
        nodes = np.arange(0, len(x), LDC_NSKIP)
        mesh_lines(ax, x[nodes], y[nodes], **kw)
        ax.annotate('lid', xy=(0.97, 1.035), xytext=(0.72, 1.035), xycoords='axes fraction', fontsize=7, color=INK,
                    va='center', arrowprops=dict(arrowstyle='->', color=INK, lw=0.8))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=8, loc='left', color=INK)
        ldc_axes.append(ax)
    cb = fig.colorbar(im, ax=ldc_axes, shrink=0.8, pad=0.02, extend='both')
    cb.set_label('vorticity', fontsize=7)
    cb.ax.tick_params(labelsize=7)

    with h5py.File(CAVITY, 'r') as f:                        # stored as [x, y, component], x decreasing
        q, mean, x, y = f['UV'][CAV_SNAP], f['mean'][:], f['x'][:, 0], f['y'][:, 0]
    dx, dy = x[1] - x[0], y[1] - y[0]
    xn = x[0] + dx * np.arange(0, len(x) + CAV_NSKIP, CAV_NSKIP)      # full-coverage mesh: one node past the edge
    yn = y[0] + dy * np.arange(0, len(y) + CAV_NSKIP, CAV_NSKIP)
    panels = [(q[..., 0], '(c) open-cavity PIV, streamwise velocity', r'$u$, m/s', 250.0),
              ((q - mean)[..., 1], "(d) wall-normal velocity fluctuation", r"$v'$, m/s", 80.0)]
    for k, (field, title, label, lim) in enumerate(panels):
        ax = fig.add_subplot(gs[1, k])
        im = ax.pcolormesh(x, y, field.T, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='nearest', rasterized=True)
        mesh_lines(ax, xn, yn, **kw)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_aspect('equal')
        ax.set_xlabel('$x/D$')
        ax.set_ylabel('$y/D$')
        ax.set_title(title, fontsize=8, loc='left', color=INK)
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, extend='both')
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=7)

    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/datasets.pdf', dpi=300)
    fig.savefig('figures/datasets.png', dpi=200)
    print('wrote figures/datasets.pdf')


if __name__ == '__main__':
    main()
