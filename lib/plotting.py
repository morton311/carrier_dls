from __future__ import annotations

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import os
import numpy as np
import h5py
import torch
from scipy.signal import welch, correlate, coherence, correlation_lags, csd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

from lib.metrics import l2_err_norm
import lib.metrics as l_metrics

import logging
logger = logging.getLogger(__name__)

_PAPER_WIDTH_PT = 470
_PAPER_WIDTH_IN = _PAPER_WIDTH_PT / 72.27
_PAPER_HEIGHT_IN = _PAPER_WIDTH_IN / 1.618

Q_ISO_VALUE = 0.01
Q_ISO_VALUE_ANIM = 0.001
DOMAIN_X_MAX = 500

width = _PAPER_WIDTH_IN
height = _PAPER_HEIGHT_IN

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.constrained_layout.use'] = True


def plot_loss(runner) -> None:
    # losses.pkl is only written when training finishes; fall back to the
    # checkpoint so an interrupted run can still be evaluated
    loss_path = runner.paths_bib.model_dir + 'losses.pkl'
    if not os.path.exists(loss_path):
        ckpt = runner.paths_bib.checkpoint_path
        if not os.path.exists(ckpt):
            logger.warning("No losses.pkl or checkpoint found; skipping loss plot.")
            return
        rec = torch.load(ckpt, map_location='cpu', weights_only=False)
        with open(loss_path, 'wb') as f:
            pickle.dump({'train_losses': rec['losses'],
                         'test_losses': rec['test_losses']}, f)
        logger.info("Recovered losses.pkl from the training checkpoint.")
    with open(loss_path, 'rb') as f:
        results = pickle.load(f)
    size = 0.6
    plt.figure(figsize=(size*width,size*height))
    plt.plot(results['train_losses'], label='Training Loss', color='k', linestyle='-')
    plt.plot(results['test_losses'], label='Test Loss', color='r', linestyle='-.')
    plt.yscale('log')
    plt.title('Losses During Training', pad=16)
    plt.legend(
        loc='lower right',
        bbox_to_anchor=(1.025, 0.95),  # Adjust position to the right of the plot
        ncol=2,  # Spread horizontally
        frameon=False,  # Removes legend border,
        fontsize=8  # Adjust font size
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.savefig(runner.paths_bib.fig_dir + 'losses.png', dpi=300)
    plt.close()


def plot_TKE(runner, rec_path: str, gt_path: str, name: str, source: str, ids) -> None:
    with h5py.File(rec_path, 'r') as f_rec, h5py.File(gt_path, 'r') as f_gt:
        time_lag = runner.config['model_params']['time_lag']
        val_id = ids
        TKE_rec = f_rec['TKE_rec'][:]
        TKE_gt = f_gt['TKE_gt_'+runner.latent_id + '_' + name + source][:]

    dt = np.arange(len(TKE_gt)) - time_lag - val_id[0] + 1
    dt_rec = np.arange(len(TKE_rec)) - time_lag + 1

    size = 0.8
    plt.figure(figsize=(size*width,size*height))
    plt.plot(dt, 
             TKE_gt, 
             label='True', 
             color='k', 
             linestyle='-')
    plt.plot(dt_rec[time_lag-1:], 
             TKE_rec[time_lag-1:], 
             label='Predicted', 
             color='r', 
             linestyle='-.')
    
    plt.title('Comparison of True and Predicted TKE', pad=16)
    plt.ylabel(r'$\mathrm{TKE} = \frac{1}{2} \sum \mathbf{u}^2$')
    plt.xlabel(r'$\Delta t$ (time steps)')
    plt.legend(
        ['True', 'Predicted'],
        loc='lower right',
        bbox_to_anchor=(1.025, 0.975),  # Adjust position to the right of the plot
        ncol=2,  # Spread horizontally
        frameon=False,  # Removes legend border,
        fontsize=8  # Adjust font size
    )
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'past_tke_comparison.png'), dpi=300)
    plt.close()


    size = 0.8
    plt.figure(figsize=(size*width,size*height))
    plt.plot(dt[val_id], 
             TKE_gt[val_id], 
             label='True', 
             color='k', 
             linestyle='-')
    plt.plot(dt_rec[time_lag-1:], 
             TKE_rec[time_lag-1:], 
             label='Predicted', 
             color='r', 
             linestyle='-.')
    
    plt.title('Comparison of True and Predicted TKE', pad=16)
    plt.ylabel(r'$\mathrm{TKE} = \frac{1}{2} \sum \mathbf{u}^2$')
    plt.xlabel(r'$\Delta t$ (time steps)')
    plt.legend(
        ['True', 'Predicted'],
        loc='lower right',
        bbox_to_anchor=(1.025, 0.975),  # Adjust position to the right of the plot
        ncol=2,  # Spread horizontally
        frameon=False,  # Removes legend border,
        fontsize=8  # Adjust font size
    )
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'future_tke_comparison.png'), dpi=300)
    plt.close()


    # psd of TKE
    frequency = runner.frequency
    f, Pxx_true = welch(TKE_gt[val_id[time_lag:]], fs=frequency)
    f, Pxx_pred = welch(TKE_rec[time_lag:], fs=frequency)
    plt.figure(figsize=(size*width,size*height))
    plt.loglog(f, Pxx_true, label='True TKE', color='k', linestyle='-')
    plt.loglog(f, Pxx_pred, label='Predicted TKE', color='r', linestyle='-.')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD(TKE)')
    plt.title('Power Spectral Density of TKE', pad=16)
    plt.legend(
        ['True', 'Predicted'],
        loc='lower right',
        bbox_to_anchor=(1.025, 0.95),  # Adjust position to the right of the plot
        ncol=2,  # Spread horizontally
        frameon=False,  # Removes legend border,
        fontsize=8  # Adjust font size
    )
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'tke_psd_comparison.png'), dpi=300)
    plt.close()


def _plot_rms_component(
    x_grid, y_grid, x_grid_t, y_grid_t,
    rms_true, rms_pred, component_idx: int, component_name: str,
    fig_width, fig_height, ticks, domain_x_limits, domain_y_limits,
    x_label: str, y_label: str, slice_label: str, slice_val, rms_dir: str,
) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    c1 = axs[0].contourf(x_grid, y_grid, rms_true[..., component_idx],
                         levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    axs[0].set_title(f'True {component_name.upper()} RMS')
    axs[0].set_ylabel(y_label)
    axs[0].set_aspect('equal')
    axs[0].set_xlim(domain_x_limits)
    axs[0].set_ylim(domain_y_limits)
    axs[1].contourf(x_grid_t, y_grid_t, rms_pred[..., component_idx],
                    levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    axs[1].set_title(f'Predicted {component_name.upper()} RMS')
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)
    axs[1].set_aspect('equal')
    axs[1].set_xlim(domain_x_limits)
    axs[1].set_ylim(domain_y_limits)
    fig.colorbar(c1, ax=axs, shrink=0.8, ticks=ticks, format='%.2f', pad=0.03)
    plt.savefig(os.path.join(rms_dir, f'{component_name}_{slice_label}{slice_val}.png'), dpi=300)
    plt.close()


def plot_RMS(runner, rec_path: str, gt_path: str, name: str, source: str,
             x_slice: Optional[int] = None, y_slice: Optional[int] = None,
             z_slice: Optional[int] = None) -> None:
    rms_dir = os.path.join(runner.paths_bib.pred_fig_dir, 'rms_compare/')
    os.makedirs(rms_dir, exist_ok=True)
    nx_t, ny_t = runner.l_config.nx_t, runner.l_config.ny_t
    x_grid, y_grid = runner.x_grid, runner.y_grid
    x_grid_t, y_grid_t = x_grid[:nx_t, :ny_t], y_grid[:nx_t, :ny_t]

    if runner.dim == 3:
        nz_t = runner.l_config.nz_t
        x_grid, y_grid, z_grid = runner.x_grid, runner.y_grid, runner.z_grid
        x_grid_t, y_grid_t, z_grid_t = (
            x_grid[:nx_t, :ny_t, :nz_t],
            y_grid[:nx_t, :ny_t, :nz_t],
            z_grid[:nx_t, :ny_t, :nz_t],
        )

    slice_axis = None
    slice_val = None
    if z_slice is not None:
        slice_axis, slice_val = 'z', z_slice
    elif y_slice is not None:
        slice_axis, slice_val = 'y', y_slice
    elif x_slice is not None:
        slice_axis, slice_val = 'x', x_slice

    with h5py.File(rec_path, 'r') as f_rec, h5py.File(gt_path, 'r') as f_gt:
        RMS_rec = f_rec['RMS_rec'][:]
        RMS_gt = runner._to_xy(f_gt['RMS_gt_'+runner.latent_id + '_' + name + source][:], 0)


    
    if z_slice is not None:
        RMS_rec = RMS_rec[:, :, z_slice]
        RMS_gt = RMS_gt[:, :, z_slice]
        x_grid = x_grid[:, :, z_slice]
        y_grid = y_grid[:, :, z_slice]
        x_grid_t = x_grid_t[:, :, z_slice]
        y_grid_t = y_grid_t[:, :, z_slice]
        print(f"RMS for z-slice {z_slice} extracted.")
    if y_slice is not None:
        RMS_rec = RMS_rec[:, y_slice, :]
        RMS_gt = RMS_gt[:, y_slice, :]
        x_grid = x_grid[:, y_slice, :]
        z_grid = z_grid[:, y_slice, :]
        x_grid_t = x_grid_t[:, y_slice, :]
        z_grid_t = z_grid_t[:, y_slice, :]
        print(f"RMS for y-slice {y_slice} extracted.")
    if x_slice is not None:
        RMS_rec = RMS_rec[x_slice, :, :]
        RMS_gt = RMS_gt[x_slice, :, :]
        y_grid = y_grid[x_slice, :, :]
        z_grid = z_grid[x_slice, :, :]
        y_grid_t = y_grid_t[x_slice, :, :]
        z_grid_t = z_grid_t[x_slice, :, :]
        print(f"RMS for x-slice {x_slice} extracted.")
        RMS_gt = f_gt['RMS_gt_' + runner.latent_id + '_' + name + source][:]

    if slice_axis is not None:
        data_slice_map = {
            'z': (slice(None), slice(None), slice_val),
            'y': (slice(None), slice_val, slice(None)),
            'x': (slice_val, slice(None), slice(None)),
        }
        RMS_rec = RMS_rec[data_slice_map[slice_axis]]
        RMS_gt = RMS_gt[data_slice_map[slice_axis]]

        if slice_axis == 'z':
            x_grid = x_grid[:, :, slice_val]
            y_grid = y_grid[:, :, slice_val]
            x_grid_t = x_grid_t[:, :, slice_val]
            y_grid_t = y_grid_t[:, :, slice_val]
        elif slice_axis == 'y':
            x_grid = x_grid[:, slice_val, :]
            y_grid = z_grid[:, slice_val, :]
            x_grid_t = x_grid_t[:, slice_val, :]
            y_grid_t = z_grid_t[:, slice_val, :]
        else:
            x_grid = y_grid[slice_val, :, :]
            y_grid = z_grid[slice_val, :, :]
            x_grid_t = y_grid_t[slice_val, :, :]
            y_grid_t = z_grid_t[slice_val, :, :]

    RMS_max = np.max(RMS_gt, axis=(0, 1), keepdims=True)
    rms_true = RMS_gt / RMS_max
    rms_pred = RMS_rec / RMS_max

    slice_props = {
        'z':    ('z', 2,   'x', 'y'),
        'y':    ('y', 3.5, 'x', 'z'),
        'x':    ('x', 2,   'y', 'z'),
        None:   ('full', 2, 'x', 'y'),
    }
    slice_label, dim_mult, x_label, y_label = slice_props[slice_axis]

    domain_x_limits = (x_grid.min(), x_grid.max())
    domain_y_limits = (y_grid.min(), y_grid.max())
    aspect = (domain_y_limits[1] - domain_y_limits[0]) / (domain_x_limits[1] - domain_x_limits[0])
    fig_width = width
    fig_height = width * aspect * dim_mult
    ticks = np.linspace(0, 1, 6)

    print(f"x_grid shape: {x_grid.shape}, y_grid shape: {y_grid.shape}")
    print(f"x_grid_t shape: {x_grid_t.shape}, y_grid_t shape: {y_grid_t.shape}")
    print(f"rms_true shape: {rms_true.shape}, rms_pred shape: {rms_pred.shape}")

    components = ['u', 'v', 'w'] if runner.dim == 3 else ['u', 'v']
    for i, comp in enumerate(components):
        _plot_rms_component(
            x_grid, y_grid, x_grid_t, y_grid_t,
            rms_true, rms_pred,
            component_idx=i, component_name=comp,
            fig_width=fig_width, fig_height=fig_height,
            ticks=ticks,
            domain_x_limits=domain_x_limits,
            domain_y_limits=domain_y_limits,
            x_label=x_label, y_label=y_label,
            slice_label=slice_label, slice_val=slice_val,
            rms_dir=rms_dir,
        )



def plot_slice_compare(runner, rec_path: str, gt_path: str, name: str, ids, indx: int,
                       x_slice: Optional[int] = None, y_slice: Optional[int] = None,
                       z_slice: Optional[int] = None) -> None:
    slice_dir = os.path.join(runner.paths_bib.pred_fig_dir, 'slice_compare/')
    os.makedirs(slice_dir, exist_ok=True)
    nx, ny = runner.l_config.nx, runner.l_config.ny
    nx_t, ny_t = runner.l_config.nx_t, runner.l_config.ny_t
    x_grid, y_grid = runner.x_grid, runner.y_grid

    x_grid_t, y_grid_t = x_grid[:nx_t, :ny_t], y_grid[:nx_t, :ny_t]



    if runner.dim == 3:
        nz = runner.l_config.nz
        nz_t = runner.l_config.nz_t
        x_grid, y_grid, z_grid = runner.x_grid, runner.y_grid, runner.z_grid
        x_grid_t, y_grid_t, z_grid_t = x_grid[:nx_t, :ny_t, :nz_t], y_grid[:nx_t, :ny_t, :nz_t], z_grid[:nx_t, :ny_t, :nz_t]

    time_lag = runner.config['model_params']['time_lag']
    val_id = ids
    indx = indx + 1

    with h5py.File(rec_path, 'r') as f_rec, h5py.File(gt_path, 'r') as f_gt:
        mean = f_gt['mean'][:]
        Q_gt = runner._to_xy(f_gt['UV'][val_id[time_lag + indx]] - mean, 0)
        Q_rec = f_rec['Q_rec'][time_lag + indx]
    slice_val = None

    if z_slice is not None:
        slice_dim = 'z'
        slice_val = z_slice
        x_label = 'x'
        y_label = 'y'
        Q_gt = Q_gt[:, :, slice_val]
        Q_rec = Q_rec[:, :, slice_val]
        x_grid = x_grid[:, :, slice_val]
        y_grid = y_grid[:, :, slice_val]
        x_grid_t = x_grid_t[:, :, slice_val]
        y_grid_t = y_grid_t[:, :, slice_val]
        dim_mult = 2
    elif y_slice is not None:
        slice_dim = 'y'
        slice_val = y_slice
        x_label = 'x'
        y_label = 'z'
        Q_gt = Q_gt[:, slice_val, :]
        Q_rec = Q_rec[:, slice_val, :]
        x_grid = x_grid[:, slice_val, :]
        y_grid = z_grid[:, slice_val, :]
        x_grid_t = x_grid_t[:, slice_val, :]
        y_grid_t = z_grid_t[:, slice_val, :]
        dim_mult = 3.5
    elif x_slice is not None:
        slice_dim = 'x'
        x_label = 'y'
        y_label = 'z'
        slice_val = x_slice
        Q_gt = Q_gt[slice_val, :, :]
        Q_rec = Q_rec[slice_val, :, :]
        x_grid = y_grid[slice_val, :, :]
        y_grid = z_grid[slice_val, :, :]
        x_grid_t = y_grid_t[slice_val, :, :]
        y_grid_t = z_grid_t[slice_val, :, :]
        dim_mult = 2

    if slice_val is None:
        dim_mult = 2
        x_label = 'x'
        y_label = 'y'
        slice_dim = ''
        slice_val = 'full'

    # normalize Q_gt between -1 and 1 and apply same normalization to Q_rec
    Q_gt_max = np.max(np.abs(Q_gt), axis=(0, 1), keepdims=True)
    Q_gt = Q_gt / Q_gt_max
    Q_rec = Q_rec / Q_gt_max

    size = 1
    domain_x_limits = (x_grid_t.min(), x_grid_t.max())
    domain_y_limits = (y_grid_t.min(), y_grid_t.max())
    domain_aspect_ratio = (domain_y_limits[1] - domain_y_limits[0]) / (domain_x_limits[1] - domain_x_limits[0])
    fig_width = size * width
    fig_height = size * width * domain_aspect_ratio*dim_mult
    ticks = np.linspace(-1, 1, 6)
    fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True, sharey=True)

    c1 = axs[0].contourf(x_grid, y_grid, Q_gt[..., 0], levels=200, cmap='RdBu_r', vmin=-1, vmax=1)
    axs[0].set_title('True U')
    axs[0].set_ylabel(y_label)
    axs[0].set_aspect('equal')
    axs[0].set_xlim(domain_x_limits)
    axs[0].set_ylim(domain_y_limits)
    axs[1].contourf(x_grid_t, y_grid_t, Q_rec[..., 0], levels=200, cmap='RdBu_r', vmin=-1, vmax=1)
    axs[1].set_title('Predicted U')
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)
    axs[1].set_aspect('equal')
    axs[1].set_xlim(domain_x_limits)
    axs[1].set_ylim(domain_y_limits)
    fig.colorbar(c1, ax=axs, shrink=0.8, ticks=ticks, format='%.2f', pad=0.03)
    plt.savefig(os.path.join(slice_dir, f'u_{slice_dim}{slice_val}_t{indx}.png'), dpi=300)
    plt.close()

    fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    c1 = axs[0].contourf(x_grid, y_grid, Q_gt[..., 1], levels=200, cmap='RdBu_r', vmin=-1, vmax=1)
    axs[0].set_title('True V')
    axs[0].set_ylabel(y_label)
    axs[0].set_aspect('equal')
    axs[0].set_xlim(domain_x_limits)
    axs[0].set_ylim(domain_y_limits)
    axs[1].contourf(x_grid_t, y_grid_t, Q_rec[..., 1], levels=200, cmap='RdBu_r', vmin=-1, vmax=1)
    axs[1].set_title('Predicted V')
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)
    axs[1].set_aspect('equal')
    axs[1].set_xlim(domain_x_limits)
    axs[1].set_ylim(domain_y_limits)
    fig.colorbar(c1, ax=axs, shrink=0.8, ticks=ticks, format='%.2f', pad=0.03)
    plt.savefig(os.path.join(slice_dir, f'v_{slice_dim}{slice_val}_t{indx}.png'), dpi=300)
    plt.close()

    if runner.dim == 3:
        fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
        c1 = axs[0].contourf(x_grid, y_grid, Q_gt[..., 2], levels=200, cmap='RdBu_r', vmin=-1, vmax=1)
        axs[0].set_title('True W')
        axs[0].set_ylabel(y_label)
        axs[0].set_aspect('equal')
        axs[0].set_xlim(domain_x_limits)
        axs[0].set_ylim(domain_y_limits)
        axs[1].contourf(x_grid_t, y_grid_t, Q_rec[..., 2], levels=200, cmap='RdBu_r', vmin=-1, vmax=1)
        axs[1].set_title('Predicted W')
        axs[1].set_xlabel(x_label)
        axs[1].set_ylabel(y_label)
        axs[1].set_aspect('equal')
        axs[1].set_xlim(domain_x_limits)
        axs[1].set_ylim(domain_y_limits)
        fig.colorbar(c1, ax=axs, shrink=0.8, ticks=ticks, format='%.2f', pad=0.03)
        plt.savefig(os.path.join(slice_dir, f'w_{slice_dim}{slice_val}_t{indx}.png'), dpi=300)
        plt.close()

    

def q_criterion(runner, rec_path: str, gt_path: str, name: str, ids, indx: int) -> None:
    import pyvista as pv
    q_dir = os.path.join(runner.paths_bib.pred_fig_dir, 'q_criterion/')
    os.makedirs(q_dir, exist_ok=True)
    val_id = ids
    time_lag = runner.config['model_params']['time_lag']
    indx = indx + 1
    with h5py.File(rec_path, 'r') as f_rec, h5py.File(gt_path, 'r') as f_gt:
        mean = f_gt['mean'][:]
        Q_gt = f_gt['UV'][val_id[time_lag + indx]] - mean
        Q_rec = f_rec['Q_rec'][time_lag + indx]

    x_grid, y_grid, z_grid = runner.x_grid, runner.y_grid, runner.z_grid
    nx_t, ny_t, nz_t = runner.l_config.nx_t, runner.l_config.ny_t, runner.l_config.nz_t
    x_grid_t, y_grid_t, z_grid_t = x_grid[:nx_t, :ny_t, :nz_t], y_grid[:nx_t, :ny_t, :nz_t], z_grid[:nx_t, :ny_t, :nz_t]

    iso_value = Q_ISO_VALUE
    x_grid, y_grid, z_grid = x_grid[:DOMAIN_X_MAX], y_grid[:DOMAIN_X_MAX], z_grid[:DOMAIN_X_MAX]
    x_grid_t, y_grid_t, z_grid_t = x_grid_t[:DOMAIN_X_MAX], y_grid_t[:DOMAIN_X_MAX], z_grid_t[:DOMAIN_X_MAX]
    Q_gt = Q_gt[:DOMAIN_X_MAX]
    Q_rec = Q_rec[:DOMAIN_X_MAX]

    x_grid = -x_grid
    x_grid_t = -x_grid_t
    
    q_gt_grid = compute_q_criterion(Q_gt, x_grid, y_grid, z_grid)
    q_rec_grid = compute_q_criterion(Q_rec, x_grid_t, y_grid_t, z_grid_t)

    contours_gt = q_gt_grid.contour(isosurfaces=[iso_value], scalars="Q-criterion")
    contours_rec = q_rec_grid.contour(isosurfaces=[iso_value], scalars="Q-criterion")

    pl = pv.Plotter(shape=(1, 2), window_size=(3200, 1200), off_screen=True)
    pl.subplot(0, 0)
    pl.add_mesh(contours_gt, opacity=0.7, color='dodger_blue')
    pl.add_mesh(q_gt_grid.outline(), color="black")
    pl.add_text("Original", font_size=12)

    pl.subplot(0, 1)
    pl.add_mesh(contours_rec, opacity=0.7, color='dodger_blue')
    pl.add_mesh(q_rec_grid.outline(), color="black")
    pl.add_text("Predicted", font_size=12)

    pl.link_views()
    pl.view_isometric()
    
    out_path = os.path.join(q_dir, 'q_criterion_t' + str(indx) + '.png')
    pl.screenshot(out_path, transparent_background=True)
    pl.close()



def compute_q_criterion(vel_grid, x, y, z):
    import pyvista as pv
    grid = pv.StructuredGrid(x, y, z)
    grid["velocity"] = vel_grid.reshape(-1, 3, order="F")
    grad = grid.compute_derivative(scalars="velocity", gradient=True)
    g = grad["gradient"].reshape(-1, 3, 3)
    S = 0.5 * (g + g.transpose(0, 2, 1))
    Omega = 0.5 * (g - g.transpose(0, 2, 1))
    q_val = 0.5 * (np.sum(Omega**2, axis=(1, 2)) - np.sum(S**2, axis=(1, 2)))
    grid["Q-criterion"] = q_val
    return grid
    

def anim_q_criterion(runner, rec_path: str, gt_path: str, name: str, ids) -> None:
    import pyvista as pv
    q_dir = os.path.join(runner.paths_bib.pred_fig_dir, 'q_criterion/')
    os.makedirs(q_dir, exist_ok=True)
    val_id = ids
    time_lag = runner.config['model_params']['time_lag']
    with h5py.File(rec_path, 'r') as f_rec, h5py.File(gt_path, 'r') as f_gt:
        mean = f_gt['mean'][:]
        Q_rec = f_rec['Q_rec'][:]
        Q_gt = f_gt['UV'][val_id[time_lag:Q_rec.shape[0]]] - mean
        Q_rec = Q_rec[time_lag:]

    x_grid, y_grid, z_grid = runner.x_grid, runner.y_grid, runner.z_grid
    nx_t, ny_t, nz_t = runner.l_config.nx_t, runner.l_config.ny_t, runner.l_config.nz_t
    x_grid_t, y_grid_t, z_grid_t = x_grid[:nx_t, :ny_t, :nz_t], y_grid[:nx_t, :ny_t, :nz_t], z_grid[:nx_t, :ny_t, :nz_t]

    iso_value = Q_ISO_VALUE_ANIM
    x_grid, y_grid, z_grid = x_grid[:DOMAIN_X_MAX], y_grid[:DOMAIN_X_MAX], z_grid[:DOMAIN_X_MAX]
    x_grid_t, y_grid_t, z_grid_t = x_grid_t[:DOMAIN_X_MAX], y_grid_t[:DOMAIN_X_MAX], z_grid_t[:DOMAIN_X_MAX]
    Q_gt = Q_gt[:, :DOMAIN_X_MAX]
    Q_rec = Q_rec[:, :DOMAIN_X_MAX]

    x_grid = -x_grid
    x_grid_t = -x_grid_t
    
    q_gt_grids = [compute_q_criterion(Q_gt[i], x_grid, y_grid, z_grid) for i in range(Q_gt.shape[0])]
    q_rec_grids = [compute_q_criterion(Q_rec[i], x_grid_t, y_grid_t, z_grid_t) for i in range(Q_rec.shape[0])]

    
    
    for i in range(len(q_gt_grids)):
        pl = pv.Plotter(shape=(1, 2), window_size=(1600, 600), off_screen=True)
        pl.subplot(0, 0)
        contours_gt = q_gt_grids[i].contour(isosurfaces=[iso_value], scalars="Q-criterion")
        pl.add_mesh(contours_gt, opacity=0.7, color='dodger_blue')
        pl.add_mesh(q_gt_grids[i].outline(), color="black")
        pl.add_text("Original", font_size=12)

        pl.subplot(0, 1)
        contours_rec = q_rec_grids[i].contour(isosurfaces=[iso_value], scalars="Q-criterion")
        pl.add_mesh(contours_rec, opacity=0.7, color='dodger_blue')
        pl.add_mesh(q_rec_grids[i].outline(), color="black")
        pl.add_text("Predicted", font_size=12)

        pl.link_views()
        pl.view_isometric()
        
        out_path = os.path.join(q_dir, 'q_criterion_t' + str(i) + '.png')
        pl.show(screenshot=out_path, interactive=False)
        pl.close()

    
    # create .mp4 from pngs
    import imageio
    images = []
    for i in range(len(q_gt_grids)):
        img_path = os.path.join(q_dir, 'q_criterion_t' + str(i) + '.png')
        images.append(imageio.imread(img_path))
    imageio.mimwrite(os.path.join(q_dir, 'mov_q_crit.mp4'), images, fps=5)
    

def plot_horizon_errors(runner, rec_path: str, gt_path: str, name: str, source: str) -> None:
    pred_path = rec_path.replace('_rec.h5', '.h5')
    with h5py.File(pred_path, 'r') as f_pred, h5py.File(gt_path, 'r') as f_gt:
        horizon_errors = f_pred[f'horizon_errors_{name}_{source}'][:] # shape (num_horizons, length)
        t = np.arange(1, horizon_errors.shape[1] + 1)
    std_horizon_errors = np.std(horizon_errors, axis=0)
    mean_horizon_errors = np.mean(horizon_errors, axis=0)

    size = 0.8
    plt.figure(figsize=(size*width,size*height))
    plt.plot(t, horizon_errors.T, color=(0.8, 0.8, 0.8))
    rgb = plt.get_cmap("tab10").colors
    h_band = plt.fill_between(
        t,
        mean_horizon_errors - std_horizon_errors,
        mean_horizon_errors + std_horizon_errors,
        color=rgb[1],
        alpha=0.1,
        edgecolor="none",
        label="std",
    )

    h_mean, = plt.plot(t, mean_horizon_errors, "-", linewidth=2, color=rgb[1], label="mean")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel('L2 Error')
    plt.title('Prediction Horizon Errors', pad=16)
    plt.legend(handles=[h_band, h_mean], loc="best")
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'horizon_errors.png'), dpi=300)
    plt.close()


def plot_multistart_errors(runner, model_err: np.ndarray, persist_err: np.ndarray) -> None:
    """Field error of many short forecasts launched across the split (lib/multistart).

    Faint lines are individual forecasts; the band is one standard deviation
    across forecasts at each step. Persistence (holding the last seed frame) is
    the floor a useful forecast has to beat.
    """
    S, H = model_err.shape
    t = np.arange(1, H + 1)
    m_mean, m_std = model_err.mean(axis=0), model_err.std(axis=0)
    p_mean = persist_err.mean(axis=0)
    c_model = plt.get_cmap("tab10").colors[1]

    size = 0.8
    plt.figure(figsize=(size * width, size * height))
    plt.plot(t, model_err.T, color=c_model, alpha=0.12, linewidth=0.6)
    plt.fill_between(t, m_mean - m_std, m_mean + m_std, color=c_model, alpha=0.15,
                     edgecolor='none', label=r'model $\pm 1\sigma$')
    plt.plot(t, m_mean, color=c_model, linewidth=2, label='model mean')
    plt.plot(t, p_mean, color='0.45', linewidth=1.5, linestyle='--', label='persistence')
    # direct labels at the line ends, in text ink so identity is not color-alone
    plt.text(t[-1], m_mean[-1], ' model', va='center', fontsize=8, color='0.15')
    plt.text(t[-1], p_mean[-1], ' persistence', va='center', fontsize=8, color='0.15')
    plt.xlim(1, H * 1.22)
    plt.ylim(bottom=0)
    # the margin past H only holds the end labels; keep ticks to the forecast range
    ax = plt.gca()
    ax.set_xticks([x for x in ax.get_xticks() if 1 <= x <= H])
    plt.xlabel('forecast step')
    plt.ylabel(r'$\|u-\hat u\|/\|u\|$')
    plt.title(f'Multi-start forecast error ({S} starts)', pad=10)
    plt.legend(frameon=False, fontsize=8, loc='upper left')
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    _save(runner, 'multistart_errors.png')

## ============================ Rollout statistics diagnostics ===============================
# Pointwise error says how far the prediction drifted from the reference
# trajectory; these say whether it still behaves like the right flow after it
# has. That distinction is the whole point of scoring a chaotic rollout.

_STAT_MAX_SNAPS = 128

# Rollout steps at which per-horizon field/energy errors are reported, so a run
# is scored inside the 30-snapshot challenge window and not only on the full
# 500-step aggregates. The per-step series are dumped in full alongside these.
_HORIZON_STEPS = (10, 30)


def _pred_index(runner, rec_path, ids):
    """Aligned (rec_idx, gt_idx) over the predicted part of a rollout.

    The first `time_lag` reconstructed snapshots are the seeded initial
    condition copied from ground truth, so they are excluded; thereafter
    rec[i] corresponds to gt[ids[i]] (the alignment plot_TKE uses).
    """
    time_lag = runner.config['model_params']['time_lag']
    with h5py.File(rec_path, 'r') as f:
        n_rec = f['Q_rec'].shape[0]
    n = min(n_rec, len(ids))
    return np.arange(time_lag, n), np.asarray(ids[time_lag:n])


def _load_pair(runner, rec_path, gt_path, ids, frac=None, max_snaps=_STAT_MAX_SNAPS,
               rec_is_reference=False):
    """Load aligned ground-truth and predicted fluctuation fields.

    `frac` selects a (lo, hi) fraction of the prediction window, so the same
    statistic can be measured early and late in a rollout. Snapshots are
    strided down to `max_snaps` to bound the FFT allocations.

    Returns (Q_gt, Q_rec), each (T, *spatial, ncomp).
    """
    if rec_is_reference:
        # the codec reference is reconstructed over `ids` directly, so its row i
        # is ids[i]; drop the same leading time_lag rows the prediction drops
        time_lag = runner.config['model_params']['time_lag']
        with h5py.File(rec_path, 'r') as f:
            n_ref = f['Q_rec'].shape[0]
        n = min(n_ref, len(ids))
        rec_idx = np.arange(time_lag, n)
        gt_idx = np.asarray(ids[time_lag:n])
    else:
        rec_idx, gt_idx = _pred_index(runner, rec_path, ids)
    if frac is not None:
        lo = int(len(rec_idx) * frac[0])
        hi = max(lo + 1, int(len(rec_idx) * frac[1]))
        rec_idx, gt_idx = rec_idx[lo:hi], gt_idx[lo:hi]
    if len(rec_idx) > max_snaps:
        sel = np.linspace(0, len(rec_idx) - 1, max_snaps).astype(int)
        rec_idx, gt_idx = rec_idx[sel], gt_idx[sel]

    nx, ny = runner.l_config.nx_t, runner.l_config.ny_t
    nz = runner.l_config.nz_t if runner.dim == 3 else None

    with h5py.File(rec_path, 'r') as f:
        Q_rec = f['Q_rec'][rec_idx]

    with h5py.File(gt_path, 'r') as f:
        if runner.dim == 3:
            mean_gt = f['mean'][:nx, :ny, :nz, :]
            Q_gt = f['UV'][gt_idx][:, :nx, :ny, :nz, :] - mean_gt[np.newaxis]
        else:
            crop = runner._xy_crop(nx, ny)
            mean_gt = runner._to_xy(f['mean'][crop], 0)
            Q_gt = runner._to_xy(f['UV'][gt_idx][(slice(None),) + crop], 1) - mean_gt[np.newaxis]
    return Q_gt, Q_rec


def _save(runner, fname):
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, fname), dpi=300)
    plt.close()


def plot_energy_spectrum(runner, rec_path, gt_path, name, source, ids, n_windows=3,
                         ref_path=None):
    """Energy spectrum E(k), ground truth vs prediction, in rollout time windows.

    Mean regression shows up as the predicted spectrum sagging below the true
    one at high k, with the gap widening from the early window to the late one.

    Returns scalars for the metrics record.
    """
    labels = ['early', 'mid', 'late'][:n_windows]
    bounds = [(i / n_windows, (i + 1) / n_windows) for i in range(n_windows)]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_windows))

    out = {}
    size = 0.9
    plt.figure(figsize=(size * width, size * height))
    for label, frac, color in zip(labels, bounds, colors):
        Q_gt, Q_rec = _load_pair(runner, rec_path, gt_path, ids, frac=frac)
        k, E_gt = l_metrics.energy_spectrum(Q_gt)
        _, E_rec = l_metrics.energy_spectrum(Q_rec)
        m = k > 0
        hi = k >= 0.5 * k.max()
        plt.loglog(k[m], E_gt[m], color=color, linestyle='-', label=f'True ({label})')
        plt.loglog(k[m], E_rec[m], color=color, linestyle='--', label=f'Pred ({label})')

        # vs raw truth: reports the codec floor as much as the model
        out[f'spectrum_kl_vs_gt_{label}'] = l_metrics.spectrum_kl(E_gt, E_rec)
        out[f'spectrum_hik_vs_gt_{label}'] = float(E_rec[hi].sum() / E_gt[hi].sum())

        # vs the codec round trip: the model's own contribution, and the
        # number to compare across runs
        if ref_path is not None:
            _, Q_ref = _load_pair(runner, ref_path, gt_path, ids, frac=frac,
                                  rec_is_reference=True)
            _, E_ref = l_metrics.energy_spectrum(Q_ref)
            if label == labels[-1]:
                plt.loglog(k[m], E_ref[m], color='0.5', linestyle=':',
                           label='Codec round trip')
            out[f'spectrum_logratio_{label}'] = l_metrics.spectrum_error(E_ref, E_rec)
            out[f'spectrum_kl_{label}'] = l_metrics.spectrum_kl(E_ref, E_rec)
            out[f'spectrum_hik_ratio_{label}'] = float(E_rec[hi].sum() / E_ref[hi].sum())

    plt.xlabel(r'$k$ (cycles per domain)')
    plt.ylabel(r'$E(k)$')
    plt.title('Energy spectrum: true (solid) vs predicted (dashed)', pad=10)
    plt.legend(frameon=False, fontsize=7, ncol=n_windows)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.4)
    _save(runner, 'energy_spectrum.png')
    return out


def plot_energy_retention(runner, rec_path, gt_path, name, source, ids):
    """Fluctuation energy over the rollout, normalized by the true mean.

    This is the direct measurement of "the rollout is losing energy": the
    headline number is the ratio over the final tenth of the rollout.
    """
    time_lag = runner.config['model_params']['time_lag']
    with h5py.File(rec_path, 'r') as f_rec, h5py.File(gt_path, 'r') as f_gt:
        TKE_rec = f_rec['TKE_rec'][:]
        TKE_gt = f_gt['TKE_gt_' + runner.latent_id + '_' + name + source][:]

    n = min(len(TKE_rec), len(ids))
    rec = TKE_rec[time_lag:n]
    gt = TKE_gt[np.asarray(ids[time_lag:n])]
    ref = float(gt.mean())
    dt = np.arange(len(rec))

    size = 0.8
    plt.figure(figsize=(size * width, size * height))
    plt.plot(dt, gt / ref, color='k', linestyle='-', label='True')
    plt.plot(dt, rec / ref, color='r', linestyle='-.', label='Predicted')
    plt.axhline(1.0, color='0.6', linewidth=0.6, linestyle=':')
    plt.xlabel(r'$\Delta t$ (time steps)')
    plt.ylabel(r'$\mathrm{TKE}/\langle \mathrm{TKE}_{\mathrm{true}}\rangle$')
    plt.title('Fluctuation energy retention over the rollout', pad=10)
    plt.legend(frameon=False, fontsize=8)
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    _save(runner, 'energy_retention.png')

    tail = max(1, len(rec) // 10)
    scalars = {
        'energy_retention_final': float(rec[-tail:].mean() / max(gt[-tail:].mean(), 1e-30)),
        'energy_retention_mean': float(rec.mean() / max(ref, 1e-30)),
    }
    # instantaneous energy ratio at the challenge horizons (ideal 1.0)
    for h in _HORIZON_STEPS:
        if h < len(rec):
            scalars[f'energy_retention_at{h}'] = float(rec[h] / max(gt[h], 1e-30))
    # the two plotted curves, both normalized by the true mean energy, so the
    # retention trace can be replotted or compared across runs after the fact
    arrays = {'tke_rec_t': rec / ref, 'tke_gt_t': gt / ref}
    return scalars, arrays


def plot_field_error_vs_time(runner, rec_path, gt_path, name, source, ids):
    """Relative L2 error of the reconstructed field over the whole rollout.

    The existing horizon study covers 30 restarts x 10 steps in *latent* space;
    this is the long rollout in physical space. Full decorrelation saturates
    near sqrt(2); settling below that means energy has been lost.
    """
    Q_gt, Q_rec = _load_pair(runner, rec_path, gt_path, ids, max_snaps=512)
    err = l_metrics.field_l2_vs_time(Q_gt, Q_rec)
    dt = np.linspace(0, 1, len(err)) * len(err)

    size = 0.8
    plt.figure(figsize=(size * width, size * height))
    plt.plot(dt, err, color='r')
    plt.axhline(np.sqrt(2), color='0.6', linewidth=0.7, linestyle=':',
                label=r'$\sqrt{2}$ (decorrelated)')
    plt.xlabel(r'$\Delta t$ (time steps)')
    plt.ylabel(r'$\|u-\hat u\|/\|u\|$')
    plt.title('Field error over the rollout', pad=10)
    plt.legend(frameon=False, fontsize=8)
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    _save(runner, 'field_error_vs_time.png')

    tail = max(1, len(err) // 10)
    scalars = {'field_l2_mean': float(err.mean()),
               'field_l2_final': float(err[-tail:].mean()),
               'field_l2_first': float(err[0])}
    # per-horizon values at the challenge window, so a run is scored where it
    # is actually used rather than only over the full 500-step rollout, whose
    # aggregates are dominated by late-time blow-up
    for h in _HORIZON_STEPS:
        if h < len(err):
            scalars[f'field_l2_at{h}'] = float(err[h])
    # first step the prediction fully decorrelates (crosses sqrt(2)); -1 if it
    # never does over the measured rollout
    crossed = np.argmax(err > np.sqrt(2))
    scalars['field_l2_decorr_step'] = int(crossed) if err[crossed] > np.sqrt(2) else -1
    return scalars, {'field_l2_t': err}


def plot_velocity_pdf(runner, rec_path, gt_path, name, source, ids):
    """PDF of fluctuation velocity values, true vs predicted.

    Mean regression narrows the distribution and grows a spike at zero.
    """
    Q_gt, Q_rec = _load_pair(runner, rec_path, gt_path, ids)
    lim = float(np.abs(Q_gt).max())
    centers, pdf_gt = l_metrics.velocity_pdf(Q_gt, rng=(-lim, lim))
    _, pdf_rec = l_metrics.velocity_pdf(Q_rec, rng=(-lim, lim))

    comps = ['u', 'v', 'w'][:pdf_gt.shape[0]]
    size = 0.9
    fig, axes = plt.subplots(1, len(comps), figsize=(size * width, size * height * 0.8),
                             squeeze=False)
    for i, comp in enumerate(comps):
        ax = axes[0][i]
        ax.semilogy(centers, pdf_gt[i], color='k', linestyle='-', label='True')
        ax.semilogy(centers, pdf_rec[i], color='r', linestyle='-.', label='Predicted')
        ax.set_xlabel(rf"${comp}'$")
        ax.grid(visible=True, linestyle='--', linewidth=0.4)
        if i == 0:
            ax.set_ylabel('PDF')
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Fluctuation velocity PDF', y=1.02)
    _save(runner, 'velocity_pdf.png')

    return {'pdf_std_ratio': float(Q_rec.std() / max(Q_gt.std(), 1e-30)),
            'pdf_flatness_ratio': float(
                (np.mean(Q_rec ** 4) / max(np.mean(Q_rec ** 2) ** 2, 1e-30)) /
                max(np.mean(Q_gt ** 4) / max(np.mean(Q_gt ** 2) ** 2, 1e-30), 1e-30))}


def plot_autocorrelation(runner, rec_path, gt_path, name, source, ids, n_probes=64):
    """Probe-averaged temporal autocorrelation, true vs predicted.

    Separates the two failure modes a variance-matching loss can confuse:
    drifting to a steady state keeps rho near 1 at all lags, while emitting
    uncorrelated noise collapses rho at lag 1.
    """
    Q_gt, Q_rec = _load_pair(runner, rec_path, gt_path, ids, max_snaps=512)
    n_time = Q_gt.shape[0]
    flat_gt = Q_gt.reshape(n_time, -1)
    flat_rec = Q_rec.reshape(n_time, -1)
    # fixed strided probes so both fields are sampled at the same points
    probes = np.linspace(0, flat_gt.shape[1] - 1, n_probes).astype(int)
    lags, rho_gt = l_metrics.temporal_autocorr(flat_gt[:, probes])
    _, rho_rec = l_metrics.temporal_autocorr(flat_rec[:, probes])

    size = 0.8
    plt.figure(figsize=(size * width, size * height))
    plt.plot(lags, rho_gt, color='k', linestyle='-', label='True')
    plt.plot(lags, rho_rec, color='r', linestyle='-.', label='Predicted')
    plt.axhline(0.0, color='0.6', linewidth=0.6, linestyle=':')
    plt.xlabel(r'lag $\tau$ (time steps)')
    plt.ylabel(r'$\rho(\tau)$')
    plt.title('Temporal autocorrelation', pad=10)
    plt.legend(frameon=False, fontsize=8)
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    _save(runner, 'autocorrelation.png')

    return {'autocorr_l2': float(np.linalg.norm(rho_gt - rho_rec) / np.sqrt(len(lags)))}
