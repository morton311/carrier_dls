"""DLS reconstructions with and without full domain coverage, against the true snapshots.

The original DLS mesh stops at the last full element and drops the trailing rows and
columns of the grid; the full-coverage mesh (latent_params.full_cover) overhangs the
edge with zero weight outside the domain (lib/dls_2d.element_masks). This reconstructs
a block of held-out snapshots with both latents and compares them with the truth:

    full grid   relative L2 error over the whole grid; the original mesh contributes
                zero fluctuation (the mean flow) where it has no elements
    covered     the region the original mesh covers
    strip       the rows/columns only the full-coverage mesh represents

    python scripts/fig_full_cover.py compute   # writes results/full_cover_compare.npz
    python scripts/fig_full_cover.py plot      # writes figures/full_cover.pdf/.png
"""
import os
import pickle
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

CASES = {   # name: (source file, latent group, original latent, full-coverage latent, held-out block, label)
    'ldc': ('data/ldc_30k_60ksnaps.h5', 'ldc_30k_10k', 'ldc_30k_10k_dls_p99m10', 'ldc_30k_10k_dls_p99m10_fc',
            (8000, 8200), r'Lid-driven cavity, $\mathrm{Re}=30{,}000$'),
    'cavity': ('data/Challenge2_1_train.h5', 'cavity', 'cavity_dls_p13m5', 'cavity_dls_p13m5_fc',
               (11520, 11720), r'Challenge 2.1 cavity PIV'),
}
OUT = 'results/full_cover_compare.npz'
SHOW = 100                      # index within the block of the snapshot drawn


def rel(p, t):
    """Per-snapshot relative L2 error over all axes but the first."""
    ax = tuple(range(1, t.ndim))
    return np.sqrt(((p - t) ** 2).sum(ax) / (t ** 2).sum(ax))


def compute():
    import lib.dls_2d as dls
    res = {}
    for name, (src, grp, lat_o, lat_f, (a, b), _) in CASES.items():
        with h5py.File(src, 'r') as f:
            yx = dls._is_yx_ordered(f['x_grid'][:], f['y_grid'][:])
            q = f['UV'][a:b].astype(np.float64) - f['mean'][:].astype(np.float64)
        if yx:
            q = q.transpose(0, 2, 1, 3)                       # into the [x, y] layout of the reconstruction
        recs = {}
        for tag, lat in (('orig', lat_o), ('full', lat_f)):
            cfg = pickle.load(open(f'results/{lat}/latent_config.pkl', 'rb'))
            with h5py.File(f'results/{lat}/latent_coeff.h5', 'r') as f:
                du, dv = f[grp]['dof_u'][a:b], f[grp]['dof_v'][a:b]
            r = dls.gfem_recon_flexible(rec_target=None, config=cfg, dof_u=du, dof_v=dv).astype(np.float64)
            recs[tag] = (r, cfg.nx_t, cfg.ny_t)
        r_o, nxo, nyo = recs['orig']
        r_f = recs['full'][0]
        assert r_f.shape[1:3] == q.shape[1:3], (r_f.shape, q.shape)
        pad = np.zeros_like(q)                                # original mesh: mean flow outside its elements
        pad[:, :nxo, :nyo] = r_o
        strip = np.ones(q.shape[1:3], bool)
        strip[:nxo, :nyo] = False
        res[f'{name}_err_full_orig'] = rel(pad, q)
        res[f'{name}_err_full_full'] = rel(r_f, q)
        res[f'{name}_err_cov_orig'] = rel(r_o, q[:, :nxo, :nyo])
        res[f'{name}_err_cov_full'] = rel(r_f[:, :nxo, :nyo], q[:, :nxo, :nyo])
        res[f'{name}_err_strip_full'] = rel(r_f[:, strip], q[:, strip])
        res[f'{name}_strip_energy'] = (q[:, strip] ** 2).sum((1, 2)) / (q ** 2).sum((1, 2, 3))
        res[f'{name}_true'] = q[SHOW]
        res[f'{name}_orig'] = pad[SHOW]
        res[f'{name}_full'] = r_f[SHOW]
        res[f'{name}_cover'] = np.array([nxo, nyo])
        print(f"{name}: {b - a} held-out snapshots {a}-{b - 1}, grid {q.shape[1]}x{q.shape[2]}, "
              f"original mesh covers {nxo}x{nyo}; strip holds {100 * res[f'{name}_strip_energy'].mean():.1f}% of the energy")
        for k in ('full', 'cov'):
            print(f"   {k:5} error  original {100 * res[f'{name}_err_{k}_orig'].mean():6.2f}%   "
                  f"full coverage {100 * res[f'{name}_err_{k}_full'].mean():6.2f}%")
        print(f"   strip error, full coverage {100 * res[f'{name}_err_strip_full'].mean():.2f}% (original: 100%)")
    np.savez_compressed(OUT, **res)
    print(f"wrote {OUT}")


def plot(name='ldc'):
    import matplotlib.pyplot as plt
    import lib.plotting as pl  # noqa: F401  (repo figure style: LaTeX text)
    plt.rcParams['figure.constrained_layout.use'] = False
    d = np.load(OUT)
    nxo, nyo = d[f'{name}_cover']
    tru, org, ful = d[f'{name}_true'][..., 0], d[f'{name}_orig'][..., 0], d[f'{name}_full'][..., 0]
    nx, ny = tru.shape
    org_show = org.copy()
    org_show[nxo:, :] = np.nan                                # not represented by the original mesh
    org_show[:, nyo:] = np.nan
    lim = np.percentile(np.abs(tru), 99.5)
    e_o, e_f = np.abs(org - tru), np.abs(ful - tru)
    elim = np.percentile(np.concatenate([e_o.ravel(), e_f.ravel()]), 99.5)

    W = pl.width
    fig = plt.figure(figsize=(W, 0.74 * W))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.82')
    ecmap = plt.get_cmap('Greys')
    left, gap = 0.035, 0.012
    pw = (0.86 - left - 2 * gap) / 3
    ph = pw * W / (0.74 * W)                                  # square panels
    rows_y = [0.875 - ph, 0.875 - 2 * ph - 0.075]
    xs = [left + i * (pw + gap) for i in range(3)]
    ext = (0, (nx - 1) / (max(nx, ny) - 1), 0, (ny - 1) / (max(nx, ny) - 1))
    xc, yc = (nxo - 1) / (nx - 1) * ext[1], (nyo - 1) / (ny - 1) * ext[3]

    def panel(x, y, a, cm, vmin, vmax, title, edge=False):
        ax = fig.add_axes([x, y, pw, ph])
        ax.imshow(a.T, origin='lower', cmap=cm, vmin=vmin, vmax=vmax, extent=ext, interpolation='nearest')
        if edge:                                              # where the original mesh stops
            ax.plot([xc, xc], [0, yc], color='k', lw=0.6, ls='--')
            ax.plot([0, xc], [yc, yc], color='k', lw=0.6, ls='--')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
        ax.set_title(title, fontsize=7.6, pad=3)
        return ax

    panel(xs[0], rows_y[0], tru, cmap, -lim, lim, r"truth, $u'$")
    panel(xs[1], rows_y[0], org_show, cmap, -lim, lim,
          r'original mesh' '\n' rf'(${nxo}\times{nyo}$ of ${nx}\times{ny}$ points)', edge=True)
    panel(xs[2], rows_y[0], ful, cmap, -lim, lim, r'full coverage' '\n' rf'(all ${nx}\times{ny}$ points)')
    b1 = panel(xs[1], rows_y[1], e_o, ecmap, 0, elim, r"$|u'_{\mathrm{rec}}-u'|$, original mesh", edge=True)
    b2 = panel(xs[2], rows_y[1], e_f, ecmap, 0, elim, r"$|u'_{\mathrm{rec}}-u'|$, full coverage", edge=True)
    for ax, key in ((b1, 'orig'), (b2, 'full')):
        txt = (r'this snapshot:' '\n' rf"full grid {100 * d[f'{name}_err_full_{key}'][SHOW]:.1f}\%" '\n'
               rf"covered region {100 * d[f'{name}_err_cov_{key}'][SHOW]:.1f}\%")
        ax.text(0.05, 0.55, txt, transform=ax.transAxes, fontsize=6.0, va='center', ha='left', color='0.1',
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.2))

    # colorbars
    cax = fig.add_axes([xs[2] + pw + 0.012, rows_y[0] + 0.05 * ph, 0.012, 0.9 * ph])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-lim, lim), cmap='RdBu_r'), cax=cax)
    cb.ax.tick_params(labelsize=5.8, length=2); cb.set_label(r"$u'$", fontsize=7, labelpad=2)
    cax2 = fig.add_axes([xs[2] + pw + 0.012, rows_y[1] + 0.05 * ph, 0.012, 0.9 * ph])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, elim), cmap=ecmap), cax=cax2)
    cb2.ax.tick_params(labelsize=5.8, length=2); cb2.set_label(r'error', fontsize=7, labelpad=2)

    # summary over the held-out blocks, both datasets
    ax = fig.add_axes([xs[0], rows_y[1], pw, ph])
    ax.axis('off')
    lines = [r'\textbf{Relative $L^2$ error}', r'held-out snapshots, both', r'velocity components', '']
    for c, (src, grp, lo, lf, (a, b), lab) in CASES.items():
        lines += [lab.replace(r'Lid-driven cavity, ', r'LDC, '),
                  rf"\quad original mesh \ {100 * d[f'{c}_err_full_orig'].mean():.1f}\%",
                  rf"\quad full coverage \ \ {100 * d[f'{c}_err_full_full'].mean():.1f}\%",
                  rf"\quad (strip: {100 * d[f'{c}_strip_energy'].mean():.1f}\% of energy)", '']
    ax.text(0.0, 1.0, '\n'.join(lines[:-1]), fontsize=6.4, va='top', ha='left', color='0.1', linespacing=1.4,
            transform=ax.transAxes)
    fig.text(left, 0.985, CASES[name][5] + r'; gray: not represented by the original mesh; dashed: where it stops',
             fontsize=7.6, va='top')
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/full_cover.pdf')
    fig.savefig('figures/full_cover.png', dpi=220)
    print('wrote figures/full_cover.pdf/.png')


if __name__ == '__main__':
    if sys.argv[1:] == ['compute']:
        compute()
    elif sys.argv[1:] == ['plot']:
        plot()
    else:
        print(__doc__)
