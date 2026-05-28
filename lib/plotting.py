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


paper_width = 470 # pt
width = paper_width / 72.27 # inches
height = width / 1.618 # inches
plt.rcParams['text.usetex'] = True
# set default font size
plt.rcParams['font.size'] = 10 # Change default font size to 12
plt.rcParams['axes.titlesize'] = 12 # Change axes title font size
plt.rcParams['axes.labelsize'] = 10 # Change axes labels font size
plt.rcParams['xtick.labelsize'] = 10 # Change x-axis tick labels font size
plt.rcParams['ytick.labelsize'] = 10 # Change y-axis tick labels font size
plt.rcParams['legend.fontsize'] = 10 # Change legend font size
plt.rcParams['figure.constrained_layout.use'] = True


def l2_err_norm(true, pred, axis=None):
    """
    Compute the L2 norm between two arrays.
    """
    return np.linalg.norm(true - pred, axis=axis) / np.linalg.norm(true, axis=axis)


def plot_loss(runner):
    with open(runner.paths_bib.model_dir + 'losses.pkl', 'rb') as f:
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
    # plt.tight_layout()
    plt.savefig(runner.paths_bib.fig_dir + 'losses.png', dpi=300)
    plt.close()


def plot_TKE(runner, rec_path, gt_path, name, source, ids):
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
    # plt.tight_layout()
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
    # plt.tight_layout()
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
    # plt.tight_layout() #rect=[0, 0, 1, 0.95]
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'tke_psd_comparison.png'), dpi=300)
    plt.close()


def plot_RMS(runner, rec_path, gt_path, name, source, x_slice=None, y_slice=None, z_slice=None):
    rms_dir = os.path.join(runner.paths_bib.pred_fig_dir, 'rms_compare/')
    os.makedirs(rms_dir, exist_ok=True)
    nx, ny = runner.l_config.nx, runner.l_config.ny
    nx_t, ny_t = runner.l_config.nx_t, runner.l_config.ny_t
    x_grid, y_grid = runner.x_grid, runner.y_grid
    x_grid_t, y_grid_t = x_grid[:nx_t, :ny_t], y_grid[:nx_t, :ny_t]
    if runner.dim == 3:
        nz = runner.l_config.nz
        nz_t = runner.l_config.nz_t
        x_grid, y_grid, z_grid = runner.x_grid, runner.y_grid, runner.z_grid
        x_grid_t, y_grid_t, z_grid_t = x_grid[:nx_t, :ny_t, :nz_t], y_grid[:nx_t, :ny_t, :nz_t], z_grid[:nx_t, :ny_t, :nz_t]
    if z_slice is not None:
        print(f"Plotting RMS for z-slice {z_slice}...")
    if y_slice is not None:
        print(f"Plotting RMS for y-slice {y_slice}...")
    if x_slice is not None:
        print(f"Plotting RMS for x-slice {x_slice}...")
    with h5py.File(rec_path, 'r') as f_rec, h5py.File(gt_path, 'r') as f_gt:
        RMS_rec = f_rec['RMS_rec'][:]
        RMS_gt = f_gt['RMS_gt_'+runner.latent_id + '_' + name + source][:]
    
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

    RMS_max = np.max(RMS_gt, axis=(0, 1), keepdims=True)
    rms_true = RMS_gt / RMS_max
    rms_pred = RMS_rec / RMS_max
    slice_val = None
    if z_slice is not None:
        slice = 'z'
        slice_val = z_slice
        dim_mult = 2
        x_label = 'x'
        y_label = 'y'

    if y_slice is not None:
        y_grid = z_grid
        y_grid_t = z_grid_t
        slice = 'y'
        slice_val = y_slice
        dim_mult = 3.5
        x_label = 'x'
        y_label = 'z'
    
    if x_slice is not None:
        x_grid = y_grid
        x_grid_t = y_grid_t
        y_grid = z_grid
        y_grid_t = z_grid_t
        slice = 'x'
        slice_val = x_slice
        dim_mult = 2
        x_label = 'y'
        y_label = 'z'

    if slice_val is None:
        dim_mult = 2
        x_label = 'x'
        y_label = 'y'
        slice = 'full'



    size = 1
    
    domain_x_limits = (x_grid_t.min(), x_grid_t.max())
    domain_y_limits = (y_grid_t.min(), y_grid_t.max())
    domain_aspect_ratio = (domain_y_limits[1] - domain_y_limits[0]) / (domain_x_limits[1] - domain_x_limits[0])

    fig_width = size * width
    fig_height = size * width * domain_aspect_ratio*dim_mult
    ticks = np.linspace(0, 1, 6)

    # u RMS plot
    fig, axs = plt.subplots(2,1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    c1 = axs[0].contourf(x_grid, y_grid, rms_true[..., 0], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    axs[0].set_title('True U RMS')
    # axs[0].set_xticks([])
    # axs[0].set_yticks([])
    # axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(y_label)
    axs[0].set_aspect('equal')
    axs[0].set_xlim(domain_x_limits)
    axs[0].set_ylim(domain_y_limits)

    c2 = axs[1].contourf(x_grid_t, y_grid_t, rms_pred[..., 0], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    axs[1].set_title('Predicted U RMS')
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)
    axs[1].set_aspect('equal')
    axs[1].set_xlim(domain_x_limits)
    axs[1].set_ylim(domain_y_limits)

    fig.colorbar(c1, ax=axs, shrink=0.8, ticks=ticks, format='%.2f', pad=0.03)
    plt.savefig(os.path.join(rms_dir, 'u_' +slice+str(slice_val)+ '.png'), dpi=300)
    plt.close() 


    # v RMS plot
    fig, axs = plt.subplots(2,1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    c1 = axs[0].contourf(x_grid, y_grid, rms_true[..., 1], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    axs[0].set_title('True V RMS')
    # axs[0].set_xticks([])
    # axs[0].set_yticks([])
    # axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(y_label)
    axs[0].set_aspect('equal')
    axs[0].set_xlim(domain_x_limits)
    axs[0].set_ylim(domain_y_limits)    

    c2 = axs[1].contourf(x_grid_t, y_grid_t, rms_pred[..., 1], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    axs[1].set_title('Predicted V RMS')
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)
    axs[1].set_aspect('equal')
    axs[1].set_xlim(domain_x_limits)
    axs[1].set_ylim(domain_y_limits)
    fig.colorbar(c1, ax=axs, shrink=0.8, ticks=ticks, format='%.2f', pad=0.03)
    plt.savefig(os.path.join(rms_dir, 'v_' +slice+str(slice_val)+ '.png'), dpi=300)
    plt.close()

    if runner.dim == 3:
        # w RMS plot
        fig, axs = plt.subplots(2,1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
        c1 = axs[0].contourf(x_grid, y_grid, rms_true[..., 2], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
        axs[0].set_title('True W RMS')
        # axs[0].set_xticks([])
        # axs[0].set_yticks([])
        # axs[0].set_xlabel(x_label)
        axs[0].set_ylabel(y_label)
        axs[0].set_aspect('equal')
        axs[0].set_xlim(domain_x_limits)
        axs[0].set_ylim(domain_y_limits)    

        c2 = axs[1].contourf(x_grid_t, y_grid_t, rms_pred[..., 2], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
        axs[1].set_title('Predicted W RMS')
        # axs[1].set_xticks([])
        # axs[1].set_yticks([])
        axs[1].set_xlabel(x_label)
        axs[1].set_ylabel(y_label)
        axs[1].set_aspect('equal')
        axs[1].set_xlim(domain_x_limits)
        axs[1].set_ylim(domain_y_limits)
        fig.colorbar(c1, ax=axs, shrink=0.8, ticks=ticks, format='%.2f', pad=0.03)
        plt.savefig(os.path.join(rms_dir, 'w_' +slice+str(slice_val)+ '.png'), dpi=300)
        plt.close()


    

def plot_slice_compare(runner, rec_path, gt_path, name, ids, indx, x_slice=None, y_slice=None, z_slice=None):
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

    if runner.dim == 3:
        nz = runner.l_config.nz
        nz_t = runner.l_config.nz_t
        x_grid, y_grid, z_grid = runner.x_grid, runner.y_grid, runner.z_grid
        x_grid_t, y_grid_t, z_grid_t = x_grid[:nx_t, :ny_t, :nz_t], y_grid[:nx_t, :ny_t, :nz_t], z_grid[:nx_t, :ny_t, :nz_t]
    with h5py.File(rec_path, 'r') as f_rec, h5py.File(gt_path, 'r') as f_gt:
        mean = f_gt['mean'][:]
        Q_gt = f_gt['UV'][val_id[time_lag + indx]] - mean
        Q_rec = f_rec['Q_rec'][indx]


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
    fig, axs = plt.subplots(2,1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    c1 = axs[0].contourf(x_grid, y_grid, Q_gt[..., 0], levels=200, cmap='RdBu_r', vmin=-1, vmax=1)
    axs[0].set_title('True U')
    # axs[0].set_xticks([])
    # axs[0].set_yticks([])
    # axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(y_label)
    axs[0].set_aspect('equal')
    axs[0].set_xlim(domain_x_limits)
    axs[0].set_ylim(domain_y_limits)    
    c2 = axs[1].contourf(x_grid_t, y_grid_t, Q_rec[..., 0], levels=200, cmap='RdBu_r', vmin=-1, vmax=1)
    axs[1].set_title('Predicted U')
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)
    axs[1].set_aspect('equal')
    axs[1].set_xlim(domain_x_limits)
    axs[1].set_ylim(domain_y_limits)
    fig.colorbar(c1, ax=axs, shrink=0.8, ticks=ticks, format='%.2f', pad=0.03)
    plt.savefig(os.path.join(slice_dir, 'u_' +slice_dim+str(slice_val)+ '_t' + str(indx) + '.png'), dpi=300)
    plt.close() 

    fig, axs = plt.subplots(2,1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    c1 = axs[0].contourf(x_grid, y_grid, Q_gt[..., 1], levels=200, cmap='RdBu_r', vmin=-1, vmax=1)
    axs[0].set_title('True V')
    # axs[0].set_xticks([])
    # axs[0].set_yticks([])   
    # axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(y_label)
    axs[0].set_aspect('equal')
    axs[0].set_xlim(domain_x_limits)
    axs[0].set_ylim(domain_y_limits)    
    c2 = axs[1].contourf(x_grid_t, y_grid_t, Q_rec[..., 1], levels=200, cmap='RdBu_r', vmin=-1, vmax=1)
    axs[1].set_title('Predicted V')
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)
    axs[1].set_aspect('equal')
    axs[1].set_xlim(domain_x_limits)
    axs[1].set_ylim(domain_y_limits)
    fig.colorbar(c1, ax=axs, shrink=0.8, ticks=ticks, format='%.2f', pad=0.03)
    plt.savefig(os.path.join(slice_dir, 'v_' +slice_dim+str(slice_val)+ '_t' + str(indx) + '.png'), dpi=300)
    plt.close() 

    if runner.dim == 3:
        fig, axs = plt.subplots(2,1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
        c1 = axs[0].contourf(x_grid, y_grid, Q_gt[..., 2], levels=200, cmap='RdBu_r', vmin=-1, vmax=1)
        axs[0].set_title('True W')
        # axs[0].set_xticks([])
        # axs[0].set_yticks([])   
        # axs[0].set_xlabel(x_label)
        axs[0].set_ylabel(y_label)
        axs[0].set_aspect('equal')
        axs[0].set_xlim(domain_x_limits)
        axs[0].set_ylim(domain_y_limits)    
        c2 = axs[1].contourf(x_grid_t, y_grid_t, Q_rec[..., 2], levels=200, cmap='RdBu_r', vmin=-1, vmax=1)
        axs[1].set_title('Predicted W')
        # axs[1].set_xticks([])
        # axs[1].set_yticks([])
        axs[1].set_xlabel(x_label)
        axs[1].set_ylabel(y_label)
        axs[1].set_aspect('equal')
        axs[1].set_xlim(domain_x_limits)
        axs[1].set_ylim(domain_y_limits)
        fig.colorbar(c1, ax=axs, shrink=0.8, ticks=ticks, format='%.2f', pad=0.03)
        plt.savefig(os.path.join(slice_dir, 'w_' +slice_dim+str(slice_val)+ '_t' + str(indx) + '.png'), dpi=300)
        plt.close()

    

def q_criterion(runner, rec_path, gt_path, name, ids, indx):
    import pyvista as pv
    # pv.set_plot_theme('paraview')
    q_dir = os.path.join(runner.paths_bib.pred_fig_dir, 'q_criterion/')
    os.makedirs(q_dir, exist_ok=True)
    val_id = ids
    time_lag = runner.config['model_params']['time_lag']
    with h5py.File(rec_path, 'r') as f_rec, h5py.File(gt_path, 'r') as f_gt:
        mean = f_gt['mean'][:]
        Q_gt = f_gt['UV'][val_id[time_lag + indx]] - mean
        Q_rec = f_rec['Q_rec'][time_lag + indx]

    x_grid, y_grid, z_grid = runner.x_grid, runner.y_grid, runner.z_grid
    nx_t, ny_t, nz_t = runner.l_config.nx_t, runner.l_config.ny_t, runner.l_config.nz_t
    x_grid_t, y_grid_t, z_grid_t = x_grid[:nx_t, :ny_t, :nz_t], y_grid[:nx_t, :ny_t, :nz_t], z_grid[:nx_t, :ny_t, :nz_t]

    iso_value = 0.01
    xmax = 500
    x_grid, y_grid, z_grid = x_grid[:xmax], y_grid[:xmax], z_grid[:xmax]
    x_grid_t, y_grid_t, z_grid_t = x_grid_t[:xmax], y_grid_t[:xmax], z_grid_t[:xmax]
    Q_gt = Q_gt[:xmax]
    Q_rec = Q_rec[:xmax]

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
    

def anim_q_criterion(runner, rec_path, gt_path, name, ids):
    import pyvista as pv
    # pv.set_plot_theme('paraview')
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

    iso_value = 0.001
    xmax = 500
    x_grid, y_grid, z_grid = x_grid[:xmax], y_grid[:xmax], z_grid[:xmax]
    x_grid_t, y_grid_t, z_grid_t = x_grid_t[:xmax], y_grid_t[:xmax], z_grid_t[:xmax]
    Q_gt = Q_gt[:, :xmax]
    Q_rec = Q_rec[:, :xmax]

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
    

def plot_horizon_errors(runner, rec_path, gt_path, name, source):
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
    # plt.tight_layout()
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'horizon_errors.png'), dpi=300)
    plt.close()