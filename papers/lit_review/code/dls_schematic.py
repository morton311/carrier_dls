"""One-dimensional illustration of the DLS / GFEM construction on a line cut of the
Re = 15,000 lid-driven cavity.

  (a) linear FE shape functions N_i on a coarse mesh (a partition of unity)
  (b) local POD modes psi_k from randomly sampled, mean-removed patches
  (c) enrichment functions N_i(x) psi_k(x - x_i), compactly supported on the
      two elements adjacent to node i and vanishing at the neighbouring nodes
  (d) L2 projection of the signal onto span{N_i, N_i psi_k}: C0-continuous

Produces figs/dls_schematic.pdf.
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

DATA = Path("/home/HDD/cmorton/carrier_dls/data/ldc_15k.h5")
NSKIP = 48                    # element size in grid points; patch = 2*NSKIP + 1


def local_modes(signal, p, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(signal) - p, n)
    P = np.stack([signal[s:s + p] for s in starts], 1)
    P = P - P.mean(0, keepdims=True)
    U, sv, _ = np.linalg.svd(P, full_matrices=False)
    return U, sv


def basis(nx, nodes, psi, m):
    """Columns: N_i and N_i * psi_k (k < m) for every node i."""
    x = np.arange(nx)
    cols = []
    for i, xi in enumerate(nodes):
        h_l = xi - nodes[i - 1] if i > 0 else 1
        h_r = nodes[i + 1] - xi if i < len(nodes) - 1 else 1
        N = np.where(x <= xi, 1 - (xi - x) / h_l, 1 - (x - xi) / h_r)
        N = np.clip(N, 0, 1)
        if i == 0:
            N[x < xi] = 0
        if i == len(nodes) - 1:
            N[x > xi] = 0
        cols.append(N)
        for k in range(m):
            e = np.zeros(nx)
            lo, hi = max(0, xi - NSKIP), min(nx, xi + NSKIP + 1)
            e[lo:hi] = psi[lo - xi + NSKIP:hi - xi + NSKIP, k]
            cols.append(N * e)
    return np.stack(cols, 1)


def main():
    with h5py.File(DATA, "r") as f:
        uv = f["UV"][100]
        mean = f["mean"][:]
    # horizontal line cut through the upper part of the cavity, y ~ 0.8
    row = int(0.8 * 511)
    sig = (uv - mean)[row, :, 0]
    nx = len(sig)
    nodes = np.arange(0, nx, NSKIP)
    if nodes[-1] != nx - 1:
        nodes = np.append(nodes[nodes < nx - NSKIP // 2], nx - 1)
    psi, _ = local_modes(sig, 2 * NSKIP + 1)
    x = np.linspace(0, 1, nx)

    S.setup()
    fig, axs = plt.subplots(2, 2, figsize=(S.TEXTWIDTH, 3.9),
                            gridspec_kw=dict(hspace=0.62, wspace=0.28))
    ax = axs[0, 0]
    B0 = basis(nx, nodes, psi, 0)
    for j in range(B0.shape[1]):
        ax.plot(x, B0[:, j], color=S.BLUE_RAMP[2 + (j % 2) * 2], lw=1.0)
    ax.plot(x[nodes], np.zeros(len(nodes)), "o", color=S.INK, ms=3, clip_on=False, zorder=3)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("$x$"); ax.set_ylabel("$N_i(x)$")
    ax.set_title(r"FE shape functions, $\sum_i N_i = 1$", fontsize=8)
    S.panel_label(ax, "a", x=-0.12)

    ax = axs[0, 1]
    xp = np.arange(-NSKIP, NSKIP + 1) / (nx - 1)
    for k, c in zip(range(3), [S.BLUE, S.ORANGE, S.AQUA]):
        ax.plot(xp, psi[:, k], color=c, label=rf"$\psi_{k + 1}$")
    ax.set_xlabel(r"$x - x_i$"); ax.set_ylabel(r"$\psi_k$")
    ax.set_title(r"local POD modes from random patches", fontsize=8)
    ax.legend(loc="lower right", fontsize=7, ncol=3)
    S.panel_label(ax, "b", x=-0.12)

    ax = axs[1, 0]
    i0 = len(nodes) // 2
    xi = nodes[i0]
    Bm = basis(nx, nodes, psi, 3)
    base = i0 * 4
    ax.plot(x, Bm[:, base], color=S.INK, lw=1.0, ls=":", label=r"$N_i$")
    for k, c in zip(range(3), [S.BLUE, S.ORANGE, S.AQUA]):
        ax.plot(x, Bm[:, base + 1 + k] / np.abs(Bm[:, base + 1 + k]).max(), color=c,
                label=rf"$N_i\psi_{k + 1}$")
    for xn in nodes[i0 - 1:i0 + 2]:
        ax.axvline(x[xn], color=S.INK_2, lw=0.6, ls="--")
    ax.set_xlim(x[nodes[i0 - 2]], x[nodes[i0 + 2]])
    ax.set_xlabel("$x$"); ax.set_ylabel("normalized")
    ax.set_ylim(-1.75, 1.2)
    ax.set_title(r"enrichments at node $x_i$ (zero at $x_{i\pm1}$)", fontsize=8)
    ax.legend(loc="lower center", fontsize=6.5, ncol=4, handlelength=1.4, columnspacing=0.8)
    S.panel_label(ax, "c", x=-0.12)

    ax = axs[1, 1]
    ax.plot(x, sig, color=S.INK, lw=2.4, alpha=0.22, label="truth")
    for m, c, ls in [(0, S.ORANGE, "-"), (2, S.BLUE, "--")]:
        B = basis(nx, nodes, psi, m)
        coef, *_ = np.linalg.lstsq(B, sig, rcond=None)
        rec = B @ coef
        err = np.linalg.norm(rec - sig) / np.linalg.norm(sig)
        lab = "FE only" if m == 0 else rf"DLS, $m={m}$"
        ax.plot(x, rec, color=c, lw=1.1, ls=ls, label=rf"{lab} ({100 * err:.1f}\%)")
    ax.plot(x[nodes], np.zeros(len(nodes)) + ax.get_ylim()[0], "|", color=S.INK_2, ms=5)
    ax.set_xlabel("$x$"); ax.set_ylabel(r"$u'(x, y{=}0.8)$")
    ax.set_title(r"projection onto $\mathrm{span}\{N_i,\; N_i\psi_k\}$", fontsize=8)
    ax.legend(loc="upper right", fontsize=6.5)
    S.panel_label(ax, "d", x=-0.12)
    S.save(fig, "dls_schematic")


if __name__ == "__main__":
    main()
