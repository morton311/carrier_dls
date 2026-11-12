"""Figures of the 2026 accomplishments report (figures/report_2026/*.png).

Reads saved results only; run scripts/extrap_baseline.py first.

    python scripts/fig_report_2026.py
"""
import json
import os
import sys

import h5py
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.signal import welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = 'figures/report_2026/'
L15 = 'results/ldc_15k_dls_p99m10_fca0_1odd/'
L30 = 'results/ldc_30k_10k_dls_p99m10_fca0_1odd/'
POD = 'results/ldc_30k_10k_pod_m400_tr/pod_30k_10k_m400'
FS = 100                 # snapshots per nondimensional time unit

# colour follows the entity in every figure; line style is the secondary encoding
BLUE, ORANGE, AQUA, VIOLET, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#eda100'
GRAY = '#6f6e69'
STYLE = {
    'truth': dict(color='k', ls='-', lw=1.1, label='Ground truth'),
    'dls': dict(color=BLUE, ls='-', lw=1.4, label='DLS transformer'),
    'hybrid': dict(color=VIOLET, ls=(0, (5, 1.5)), lw=1.4, label='DLS hybrid'),
    'pod': dict(color=ORANGE, ls=':', lw=1.6, label='POD transformer, 400 modes'),
    'extrap': dict(color=AQUA, ls='-.', lw=1.3, label='Frequency extrapolation'),
    'persist': dict(color=GRAY, ls='--', lw=1.2, label='Persistence'),
    'dmd': dict(color=YELLOW, ls=(0, (3, 1, 1, 1)), lw=1.4, label='DMD, 25 modes'),
    'mean': dict(color=GRAY, ls=':', lw=1.1, label='Mean flow'),
}

CASES = {
    're15k': dict(name='ldc_15k', data='data/ldc_15k.h5', rms_key='RMS_gt_ldc_15k_2003-2498_b285d46e',
                  dls=[L15 + f'spatial_gram_rollout_galerkin_heldout_s{s}_fca0_1' for s in (1, 2, 3)],
                  hybrid=L15 + 'spatial_diff_res_heldout_s1_fca0_1_ms8'),
    're30k': dict(name='ldc_30k_10k', data='data/ldc_30k_60ksnaps.h5', rms_key='RMS_gt_ldc_30k_10k_8004-9999_17aec0d4',
                  dls=[L30 + 'spatial_gram_rollout_galerkin_30k_10k_fca0_1'] +
                      [L30 + f'spatial_gram_rollout_galerkin_30k_10k_fca0_1_s{s}' for s in (2, 3)],
                  hybrid=L30 + 'spatial_diff_res_30k_10k_fca0_1', pod=POD),
}


def eval_h5(run, name):
    return h5py.File(f'{run}/saved_metrics/{name}_eval_data.h5', 'r')


def metrics(run, name):
    return json.load(open(f'{run}/saved_metrics/{name}_eval_data.json'))['metrics']


def rms_rec(run, name):
    with h5py.File(f'{run}/pred/{name}_eval_data_pred_stats.h5', 'r') as f:
        return f['RMS_rec'][:]


def fig_forecast_error(plt, pl, ex):
    fig, axs = plt.subplots(1, 2, figsize=(pl.width, 0.42 * pl.width), sharex=True)
    for ax, (key, c), title in zip(axs, CASES.items(), (r'$Re = 15{,}000$', r'$Re = 30{,}000$')):
        steps = np.arange(1, 31)
        dls = []
        for r in c['dls']:
            with eval_h5(r, c['name']) as f:
                dls.append(f['ms_err'][:])
                pers = f['ms_persist'][:]
        dls = np.concatenate(dls)
        ax.plot(steps, 100 * pers.mean(0), **STYLE['persist'])
        ax.plot(steps, 100 * ex[f'{key}_ms_err'].mean(0), **STYLE['extrap'])
        if 'pod' in c:
            with eval_h5(c['pod'], c['name']) as f:
                ax.plot(steps, 100 * f['ms_err'][:].mean(0), **STYLE['pod'])
        ax.fill_between(steps, 100 * (dls.mean(0) - dls.std(0)), 100 * (dls.mean(0) + dls.std(0)), color=BLUE, alpha=0.18, lw=0)
        ax.plot(steps, 100 * dls.mean(0), **STYLE['dls'])
        with eval_h5(c['hybrid'], c['name']) as f:
            ax.plot(steps, 100 * f['ms_err'][:].mean(0), **STYLE['hybrid'])
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Forecast step')
        ax.set_xlim(1, 30)
        ax.grid(True, which='major', ls='--', lw=0.4, color='0.75')
    axs[0].set_ylabel(r'Relative field error (\%)')
    h = [plt.Line2D([], [], **STYLE[k]) for k in ('dls', 'hybrid', 'pod', 'extrap', 'persist')]
    fig.legend(handles=h, loc='outside lower center', ncols=3, frameon=False, fontsize=8)
    fig.savefig(OUT + 'forecast_error.png', dpi=250, bbox_inches='tight')
    plt.close(fig)


def fig_rms(plt, pl, ex, key, columns, fname):
    c = CASES[key]
    with h5py.File(c['data'], 'r') as f:
        gt = f[c['rms_key']][:].transpose(1, 0, 2)      # source is [y, x]
        xg, yg = f['x_grid'][:].T, f['y_grid'][:].T
    n = len(columns) + 1
    fig, axs = plt.subplots(2, n, figsize=(pl.width, (2 / n + 0.11) * pl.width), sharex=True, sharey=True)
    fields = [('Ground truth', gt, None)]
    for label, field, err in columns:
        fields.append((label, field, err))
    for j, (label, q, err) in enumerate(fields):
        for i, comp in enumerate(('u', 'v')):
            ax = axs[i, j]
            im = ax.pcolormesh(xg, yg, q[..., i] / gt[..., i].max(), cmap='Blues', vmin=0, vmax=1,
                               shading='auto', rasterized=True)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(rf'RMS ${comp}$')
        axs[0, j].set_title(label, fontsize=8)
        axs[1, j].set_xlabel(f'L2 error = {err:.1f}\\%' if err is not None else ' ', fontsize=8)
    cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.015)
    cb.set_label('Normalized RMS', fontsize=8)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=8)
    fig.savefig(OUT + fname, dpi=250, bbox_inches='tight')
    plt.close(fig)


def fig_tke(plt, pl, ex, key, fname):
    c = CASES[key]
    with eval_h5(c['dls'][0], c['name']) as f:
        gt, dls = f['tke_gt_t'][:], f['tke_rec_t'][:]
    with eval_h5(c['hybrid'], c['name']) as f:
        hyb = f['tke_rec_t'][:]
    t = np.arange(len(gt)) / FS
    series = [('dls', t, dls), ('hybrid', t, hyb)]
    if 'pod' in c:
        with eval_h5(c['pod'], c['name']) as f:
            pod, gtp = f['tke_rec_t'][:], f['tke_gt_t'][:]
        series.append(('pod', (np.arange(len(pod)) + len(gt) - len(gtp)) / FS, pod))
    series.append(('extrap', t, ex[f'{key}_tke']))
    rows = 2 if len(series) < 4 else 3
    fig, axs = plt.subplots(rows, 2, figsize=(pl.width, (0.30 * rows + 0.02) * pl.width))
    axs = axs.ravel()
    ymax = max(gt.max(), *[s[2].max() for s in series]) * 1.05
    for ax, (k, ts, q) in zip(axs, series):
        ax.plot(t, gt, **STYLE['truth'])
        ax.plot(ts, q, **STYLE[k])
        ax.set_title(STYLE[k]['label'], fontsize=9)
        ax.set_ylim(0, ymax)
        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel('Nondimensional time')
        ax.set_ylabel(r'TKE / mean true TKE')
        ax.grid(True, ls='--', lw=0.4, color='0.75')
    ax = axs[len(series)]
    fs = {}
    for k, s in [('truth', gt)] + [(a, b) for a, _, b in series]:
        fs[k], p = welch(s, fs=FS, nperseg=min(256, len(s)))
        ax.loglog(fs[k][1:], p[1:], **STYLE[k])
    ax.set_title('Power spectral density of TKE', fontsize=9)
    ax.set_xlabel('Nondimensional frequency')
    ax.set_ylabel('PSD(TKE)')
    ax.grid(True, which='major', ls='--', lw=0.4, color='0.75')
    for a in axs[len(series) + 1:]:
        a.set_axis_off()
    h = [plt.Line2D([], [], **STYLE[k]) for k in ['truth'] + [s[0] for s in series]]
    fig.legend(handles=h, loc='outside lower center', ncols=len(h), frameon=False, fontsize=7)
    fig.savefig(OUT + fname, dpi=250, bbox_inches='tight')
    plt.close(fig)


def fig_challenge(plt, pl):
    """Open cavity, in the same measure as fig_forecast_error (results/cavity_baselines.json holds DMD)."""
    run = 'results/cavity_dls_p13m5_fca0_1odd/'
    dls = []
    for sfx in ('', '_s2', '_s3'):
        with eval_h5(run + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1' + sfx, 'cavity') as f:
            dls.append(f['ms_err'][:])
            pers = f['ms_persist'][:]
    dls = 100 * np.concatenate(dls)
    with eval_h5(run + 'spatial_diff_res_fc_all_d128_ca0_1_ms8', 'cavity') as f:
        hybrid = 100 * f['ms_err'][:].mean(0)
    dmd = 100 * np.array(json.load(open('results/cavity_baselines.json'))['DMD-25']['in_basis']['by_step'])
    steps = np.arange(1, 31)
    fig, ax = plt.subplots(figsize=(0.72 * pl.width, 0.42 * pl.width))
    ax.plot(steps, 100 * pers.mean(0), **STYLE['persist'])
    ax.plot(steps, dmd, **STYLE['dmd'])
    ax.axhline(100, **STYLE['mean'])
    ax.fill_between(steps, dls.mean(0) - dls.std(0), dls.mean(0) + dls.std(0), color=BLUE, alpha=0.18, lw=0)
    ax.plot(steps, dls.mean(0), **STYLE['dls'])
    ax.plot(steps, hybrid, **STYLE['hybrid'])
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 160)
    ax.set_xlabel('Forecast step')
    ax.set_ylabel(r'Relative field error (\%)')
    ax.grid(True, which='major', ls='--', lw=0.4, color='0.75')
    ax.legend(frameon=False, fontsize=7, ncols=3, loc='lower right')
    fig.savefig(OUT + 'challenge_error.png', dpi=250, bbox_inches='tight')
    plt.close(fig)


def fig_airwake(plt, pl):
    run = 'results/0aoa_dls_p21m25/distributed_refix/pred/0aoa_eval_data_pred_stats.h5'
    tl, slice_y = 20, 40
    with h5py.File(run, 'r') as f:
        tke_rec, rms = f['TKE_rec'][:], f['RMS_rec'][:]
    with h5py.File('data/gencarr_transpose.h5', 'r') as f:
        tke_gt = f['TKE_gt_0aoa_dls_p21m25_0aoaeval_data'][540:600]
        nx, ny, nz = rms.shape[:3]
        gt = f['RMS_gt_0aoa_560-599_bc9be8d3'][:nx, :ny, :nz, :]
        xg, zg = f['x_grid'][:nx, slice_y, :nz], f['z_grid'][:nx, slice_y, :nz]
    ref = tke_gt[tl:].mean()
    fig = plt.figure(figsize=(pl.width, 0.30 * pl.width))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.25])
    ax = fig.add_subplot(gs[:, 0])
    dt = np.arange(60) - tl + 1
    ax.plot(dt, tke_gt / ref, **STYLE['truth'])
    ax.plot(dt[tl - 1:], tke_rec[tl - 1:] / ref, **dict(STYLE['dls'], label='DLS transformer'))
    ax.set_xlabel('Forecast step')
    ax.set_ylabel('TKE / mean true TKE')
    ax.set_ylim(0, None)
    ax.grid(True, ls='--', lw=0.4, color='0.75')
    ax.legend(frameon=False, fontsize=7, loc='lower left')
    vmax = gt[:, slice_y, :, 0].max()
    for r, (q, label) in enumerate(((gt, 'Ground truth'), (rms, 'DLS transformer'))):
        ax = fig.add_subplot(gs[r, 1])
        im = ax.pcolormesh(xg, zg, q[:, slice_y, :, 0] / vmax, cmap='Blues', vmin=0, vmax=1,
                           shading='auto', rasterized=True)
        ax.set_title(rf'RMS $u$, {label}', fontsize=8, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=fig.axes[1:], shrink=0.85, pad=0.02)
    cb.set_label('Normalized RMS', fontsize=8)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=8)
    fig.savefig(OUT + 'airwake.png', dpi=250, bbox_inches='tight')
    plt.close(fig)


def main():
    import matplotlib.pyplot as plt
    import lib.plotting as pl          # repo figure style (paper size, LaTeX text)
    os.makedirs(OUT, exist_ok=True)
    ex = np.load('results/extrap_baseline.npz')
    ej = json.load(open('results/extrap_baseline.json'))
    fig_forecast_error(plt, pl, ex)

    c = CASES['re15k']
    fig_rms(plt, pl, ex, 're15k', [
        ('DLS transformer', rms_rec(c['dls'][0], c['name']), metrics(c['dls'][0], c['name'])['rms_error_pct']),
        ('DLS hybrid', rms_rec(c['hybrid'], c['name']), metrics(c['hybrid'], c['name'])['rms_error_pct']),
        ('Frequency extrapolation', ex['re15k_rms'], ej['re15k']['rms_error_pct']),
    ], 'rms_15k.png')
    fig_tke(plt, pl, ex, 're15k', 'tke_15k.png')

    c = CASES['re30k']
    fig_rms(plt, pl, ex, 're30k', [
        ('DLS transformer', rms_rec(c['dls'][0], c['name']), metrics(c['dls'][0], c['name'])['rms_error_pct']),
        ('DLS hybrid', rms_rec(c['hybrid'], c['name']), metrics(c['hybrid'], c['name'])['rms_error_pct']),
        ('POD, 400 modes', rms_rec(c['pod'], c['name']), metrics(c['pod'], c['name'])['rms_error_pct']),
        ('Extrapolation', ex['re30k_rms'], ej['re30k']['rms_error_pct']),
    ], 'rms_30k.png')
    fig_tke(plt, pl, ex, 're30k', 'tke_30k.png')
    fig_challenge(plt, pl)
    fig_airwake(plt, pl)


if __name__ == '__main__':
    main()
