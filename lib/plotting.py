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


