"""Figures: long-run statistics of free rollouts against the true flow.

figures/ldc_statistics.pdf     lid-driven cavity: RMS-velocity maps (DNS, DLS projection, forecast, difference) at
                               Re 15,000, and the fluctuating kinetic energy over the rollout at Re 15,000 and 30,000
figures/cavity_statistics.pdf  Challenge 2.1 PIV: RMS-velocity maps (PIV, DLS projection, forecast), the fluctuating
                               kinetic energy over the rollout, and the frequency spectrum

The fields are decoded once and the reduced statistics cached in results/fig_statistics.npz.

    python scripts/fig_statistics.py              # plot (computes the cache if missing)
    python scripts/fig_statistics.py --recompute
"""
import os
import pickle
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from scipy.signal import welch    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

CACHE = 'results/fig_statistics.npz'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e5e4e0'
C_AVG, C_PROJ, C_FLOOR = '#2a78d6', '#1baf7a', '#8a8984'      # nodal average, projection in the loop, DLS projection
TL = 4                                                         # seed frames at the head of every stored rollout

L15 = 'results/ldc_15k_dls_p99m10'
R15 = dict(avg=L15 + '/spatial_gram_rollout_heldout_s1', proj=L15 + '/spatial_gram_rollout_galerkin_heldout_s1')
B15 = 1999                                                     # rollout row r is snapshot B15 + r
L30 = 'results/ldc_30k_10k_dls_p99m10'
R30 = [L30 + '/spatial_gram_rollout_galerkin_30k_10k' + s for s in ('', '_s2', '_s3')]
CAV_FS = 16.0                                                  # kHz


def compute():
    import lib.dls_2d as dls
    import flow_stats as fs
    out = {}

    # --- lid-driven cavity, Re 15,000 (file stores [y, x]; the decoder returns [x, y])
    cfg = pickle.load(open(f'{L15}/latent_config.pkl', 'rb'))
    dec = lambda du, dv: np.concatenate([dls.decode_fields(cfg, du[i:i + 64], dv[i:i + 64]).transpose(0, 2, 1, 3)
                                         for i in range(0, len(du), 64)])
    with h5py.File(f'{R15["proj"]}/pred/ldc_15k_eval_data_pred.h5', 'r') as h:
        n = h['dof_u'].shape[0]
    lo, hi = B15 + TL, B15 + n
    with h5py.File('data/ldc_15k.h5', 'r') as f:
        x, y = f['x'][:], f['y'][:]
        T = (f['UV'][lo:hi] - f['mean'][:][None]).astype(np.float32)
    with h5py.File(f'{L15}/latent_coeff.h5', 'r') as f:
        P = dec(f['ldc_15k/dof_u'][lo:hi], f['ldc_15k/dof_v'][lo:hi])
    m = P.shape[1]
    T = T[:, :m, :m]
    rms = lambda Q: np.sqrt((Q.astype(np.float64) ** 2).mean(0))            # (ny, nx, 2)
    tke = lambda Q: 0.5 * (Q.astype(np.float64) ** 2).sum(-1).mean((1, 2))
    err = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    rt = rms(T)
    out.update(l15_x=x[:m], l15_y=y[:m], l15_rms_truth=rt, l15_rms_floor=rms(P), l15_tke_truth=tke(T), l15_tke_floor=tke(P))
    print(f'Re 15k RMS error of the DLS projection: {100 * err(rms(P), rt):.2f} %')
    for key, run in R15.items():
        with h5py.File(f'{run}/pred/ldc_15k_eval_data_pred.h5', 'r') as h:
            Q = dec(h['dof_u'][TL:], h['dof_v'][TL:])
        out[f'l15_rms_{key}'], out[f'l15_tke_{key}'] = rms(Q), tke(Q)
        print(f'Re 15k {key}: RMS error {100 * err(rms(Q), rt):.2f} %, energy ratio {tke(Q).mean() / tke(T).mean():.3f}')
    del T, P, Q

    # --- lid-driven cavity, Re 30,000: stored energy histories of the three seeds
    for k, run in enumerate(R30):
        with h5py.File(f'{run}/saved_metrics/ldc_30k_10k_eval_data.h5', 'r') as h:
            out['l30_tke_truth'], out[f'l30_tke_s{k + 1}'] = h['tke_gt_t'][:], h['tke_rec_t'][:]
        print(f'Re 30k seed {k + 1}: energy ratio {out[f"l30_tke_s{k + 1}"].mean() / out["l30_tke_truth"].mean():.3f}')

    # --- Challenge 2.1 cavity PIV
    with h5py.File(fs.DATA, 'r') as f:
        mean = f['mean'][:].astype(np.float32)
        out.update(cav_x=f['x'][:, 0], cav_y=f['y'][:, 0])
    lo, hi = fs.BLOCK[0] + fs.TL, fs.BLOCK[1]
    T = fs.truth(lo, hi, mean)
    lat = fs.LC.rstrip('/')
    with h5py.File(f'{lat}/latent_coeff.h5', 'r') as f:
        P = fs.decode(lat, f['cavity/dof_u'][lo:hi], f['cavity/dof_v'][lo:hi], mean.shape[:2])

    def spectrum(Q):
        f, S = welch(Q.reshape(len(Q), -1)[:, ::4].astype(np.float64), fs=CAV_FS, nperseg=256, axis=0)
        return f, S.mean(1)
    ft, st = spectrum(T)
    rt = rms(T)
    out.update(cav_rms_truth=rt, cav_rms_floor=rms(P), cav_tke_truth=tke(T), cav_tke_floor=tke(P), cav_f=ft, cav_s_truth=st,
               cav_s_floor=spectrum(P)[1])
    for k in (1, 2, 3):
        run = fs.RUNS[f'DLS, alpha 0.1, s{k}']
        with h5py.File(f'{run}/pred/{fs.NAME}_eval_data_pred.h5', 'r') as h:
            Q = fs.decode(lat, h['dof_u'][fs.TL:], h['dof_v'][fs.TL:], mean.shape[:2])
        out[f'cav_rms_s{k}'], out[f'cav_tke_s{k}'], out[f'cav_s_s{k}'] = rms(Q), tke(Q), spectrum(Q)[1]
        print(f'cavity seed {k}: RMS-map error {100 * err(rms(Q), rt):.1f} %, energy ratio {tke(Q).mean() / tke(T).mean():.3f}')
    np.savez_compressed(CACHE, **out)
    return out


def style():
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'legend.frameon': False, 'legend.fontsize': 7})


def clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5)


def mag(r):
    return np.sqrt((r ** 2).sum(-1))


def fig_ldc(d):
    fig = plt.figure(figsize=(7.0, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0])
    x, y = d['l15_x'], d['l15_y']
    ext = [x[0], x[-1], y[0], y[-1]]
    truth, proj = mag(d['l15_rms_truth']), mag(d['l15_rms_proj'])
    vmax = float(np.percentile(truth, 99.5))
    maps = [(truth, '(a) DNS'), (mag(d['l15_rms_floor']), '(b) DLS projection'), (proj, '(c) forecast')]
    axes = []
    for k, (f, title) in enumerate(maps):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(f, origin='lower', extent=ext, cmap='Blues', vmin=0, vmax=vmax, rasterized=True)
        axes.append(ax)
        ax.set_title(title, fontsize=8, loc='left', color=INK)
    cb = fig.colorbar(im, ax=axes, location='bottom', shrink=0.5, aspect=35, pad=0.03)
    cb.set_label('RMS velocity fluctuation', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[0, 3])
    diff = 1e3 * (proj - truth)
    lim = float(np.percentile(np.abs(diff), 99.5))
    im = ax.imshow(diff, origin='lower', extent=ext, cmap='RdBu_r', vmin=-lim, vmax=lim, rasterized=True)
    ax.set_title('(d) forecast $-$ DNS', fontsize=8, loc='left', color=INK)
    cb = fig.colorbar(im, ax=ax, location='bottom', shrink=0.9, aspect=18, pad=0.03)
    cb.set_label(r'difference, $\times10^{-3}$', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    for ax in axes + [ax]:
        ax.set_xticks([0, 0.5])
        ax.set_yticks([0, 0.5])
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal')

    a = fig.add_subplot(gs[1, :2])
    k0 = d['l15_tke_truth'].mean()
    t = np.arange(1, len(d['l15_tke_truth']) + 1)
    a.plot(t, d['l15_tke_truth'] / k0, color=INK, lw=1.4, label='DNS')
    a.plot(t, d['l15_tke_floor'] / k0, color=C_FLOOR, lw=1.0, ls=':', label='DLS projection')
    a.plot(t, d['l15_tke_avg'] / k0, color=C_AVG, lw=1.2, label='forecast, nodal average')
    a.plot(t, d['l15_tke_proj'] / k0, color=C_PROJ, lw=1.2, ls='-.', label='forecast, projection in the loop')
    a.set_xlabel('rollout step')
    a.set_ylabel('kinetic energy / DNS mean')
    a.set_title(r'(e) fluctuating kinetic energy, $\mathrm{Re}=15{,}000$', fontsize=8, loc='left', color=INK)
    a.set_xlim(0, len(t))
    a.set_ylim(0.88, 1.32)
    a.legend(loc='upper right', ncol=2, columnspacing=1.0, handlelength=1.8)
    clean(a)

    b = fig.add_subplot(gs[1, 2:])
    k0 = d['l30_tke_truth'].mean()
    t = np.arange(1, len(d['l30_tke_truth']) + 1)
    b.plot(t, d['l30_tke_truth'] / k0, color=INK, lw=1.4, label='DNS')
    for k in (1, 2, 3):
        b.plot(t, d[f'l30_tke_s{k}'] / k0, color=C_PROJ, lw=1.0, ls='-.', alpha=0.9,
               label='forecast, projection in the loop (3 seeds)' if k == 1 else None)
    b.set_xlabel('rollout step')
    b.set_title(r'(f) $\mathrm{Re}=30{,}000$', fontsize=8, loc='left', color=INK)
    b.set_xlim(0, len(t))
    b.set_ylim(0, 2.2)
    b.legend(loc='upper left', ncol=2, columnspacing=1.0, handlelength=1.8)
    clean(b)
    fig.savefig('figures/ldc_statistics.pdf', dpi=300)
    fig.savefig('figures/ldc_statistics.png', dpi=200)
    print('wrote figures/ldc_statistics.pdf')


def fig_cavity(d):
    fig = plt.figure(figsize=(7.0, 4.2), constrained_layout=True)
    left, right = fig.subfigures(1, 2, width_ratios=[1.0, 1.0])
    x, y = d['cav_x'], d['cav_y']
    truth = mag(d['cav_rms_truth'])
    vmax = float(truth.max())
    maps = [(truth, '(a) PIV'), (mag(d['cav_rms_floor']), '(b) DLS projection'), (mag(d['cav_rms_s1']), '(c) forecast')]
    axes = left.subplots(3, 1, sharex=True)
    for ax, (f, title) in zip(axes, maps):
        im = ax.pcolormesh(x, y, f.T, cmap='Blues', vmin=0, vmax=vmax, shading='nearest', rasterized=True)
        ax.set_aspect('equal')
        ax.set_ylabel('$y/D$')
        ax.set_title(title, fontsize=8, loc='left', color=INK)
    axes[-1].set_xlabel('$x/D$')
    cb = left.colorbar(im, ax=axes, location='bottom', shrink=0.8, aspect=35, pad=0.03)
    cb.set_label('RMS velocity fluctuation, m/s', fontsize=7)
    cb.ax.tick_params(labelsize=7)

    a, b = right.subplots(2, 1)
    k0 = d['cav_tke_truth'].mean()
    w = 25
    smooth = lambda v: np.convolve(v, np.ones(w) / w, mode='valid')
    t = np.arange(1, len(d['cav_tke_truth']) + 1)[w - 1:]
    a.plot(t, smooth(d['cav_tke_truth']) / k0, color=INK, lw=1.3, label='PIV')
    a.plot(t, smooth(d['cav_tke_floor']) / k0, color=C_FLOOR, lw=1.0, ls=':', label='DLS projection')
    for k in (1, 2, 3):
        a.plot(t, smooth(d[f'cav_tke_s{k}']) / k0, color=C_PROJ, lw=1.0, ls='-.', label='forecast (3 seeds)' if k == 1 else None)
    a.set_xlabel('rollout step')
    a.set_ylabel('kinetic energy / PIV mean')
    a.set_title('(d) fluctuating kinetic energy, 25-step mean', fontsize=8, loc='left', color=INK)
    a.set_xlim(0, t[-1])
    a.set_ylim(0, 2.4)
    a.legend(loc='upper left', ncol=3, columnspacing=0.8, handlelength=1.6, borderaxespad=0.2)
    clean(a)

    f = d['cav_f'][1:]
    b.loglog(f, d['cav_s_truth'][1:], color=INK, lw=1.3, label='PIV')
    b.loglog(f, d['cav_s_floor'][1:], color=C_FLOOR, lw=1.0, ls=':', label='DLS projection')
    b.loglog(f, d['cav_s_s1'][1:], color=C_PROJ, lw=1.2, ls='-.', label='forecast')
    b.set_xlabel('frequency, kHz')
    b.set_ylabel(r'PSD, m$^2\,$s$^{-2}\,$kHz$^{-1}$')
    b.set_title('(e) frequency spectrum', fontsize=8, loc='left', color=INK)
    b.set_ylim(1e-3, 1e6)
    b.legend(loc='upper left', ncol=3, columnspacing=0.8, handlelength=1.6, borderaxespad=0.2)
    clean(b)
    fig.savefig('figures/cavity_statistics.pdf', dpi=300)
    fig.savefig('figures/cavity_statistics.png', dpi=200)
    print('wrote figures/cavity_statistics.pdf')


def main():
    d = compute() if '--recompute' in sys.argv or not os.path.exists(CACHE) else dict(np.load(CACHE))
    style()
    os.makedirs('figures', exist_ok=True)
    fig_ldc(d)
    fig_cavity(d)


if __name__ == '__main__':
    main()
