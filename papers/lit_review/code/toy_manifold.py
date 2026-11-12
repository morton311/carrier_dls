"""One-dimensional examples of the linear-subspace (Kolmogorov n-width) barrier.

Produces
  figs/nwidth.pdf      advected vs diffused pulse and their singular-value decay
  figs/ae_vs_pod.pdf   held-out error of POD vs a small autoencoder, learned latent
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

N = 256
X = np.linspace(0, 1, N)
SIG = 0.025
CACHE = S.FIG_DIR / "ae_vs_pod.npz"


def advected(s):
    """Gaussian pulse centred at s (one snapshot per entry of s)."""
    return np.exp(-0.5 * ((X[None, :] - np.asarray(s)[:, None]) / SIG) ** 2)


def diffused(t, nu=1.0):
    """Heat-equation solution from the same pulse, fixed at x = 0.5."""
    sig = np.sqrt(SIG ** 2 + 2 * nu * np.asarray(t)[:, None])
    return (SIG / sig) * np.exp(-0.5 * ((X[None, :] - 0.5) / sig) ** 2)


def svals(A):
    s = np.linalg.svd(A, compute_uv=False)
    return s / s[0]


def front(s):
    """Advected discontinuity (top-hat of width 0.1) centred at s."""
    return (np.abs(X[None, :] - np.asarray(s)[:, None]) < 0.05).astype(float)


def fig_nwidth():
    S.setup()
    T = 200
    s = np.linspace(0.15, 0.85, T)
    t = np.linspace(0, 0.01, T)
    Uf, Ua, Ud = front(s), advected(s), diffused(t)
    fig, axs = plt.subplots(1, 3, figsize=(S.TEXTWIDTH, 2.15),
                            gridspec_kw=dict(width_ratios=[1, 1, 1.45], wspace=0.42))
    for ax, U, title, lab in [(axs[0], Uf, r"advected front", "a"),
                              (axs[1], Ud, r"diffusing pulse", "b")]:
        ax.imshow(U, origin="lower", aspect="auto", cmap=S.SEQUENTIAL, extent=[0, 1, 0, 1],
                  vmin=0, vmax=1, rasterized=True)
        ax.set_xlabel("$x$"); ax.set_ylabel("$t/t_f$")
        ax.set_title(title, fontsize=8); ax.grid(False)
        S.panel_label(ax, lab, x=-0.22)
    ax = axs[2]
    j = np.arange(1, T + 1)
    for U, c, name in [(Uf, S.BLUE, "advected front"),
                       (Ua, S.ORANGE, "advected Gaussian"),
                       (Ud, S.AQUA, "diffusing Gaussian")]:
        ax.semilogy(j, svals(U), color=c, label=name)
    ax.legend(loc="center right", fontsize=7)
    ax.set_ylim(1e-10, 2)
    ax.set_xlim(0, 150)
    ax.set_xlabel("index $j$"); ax.set_ylabel(r"$\sigma_j/\sigma_1$")
    ax.set_title("singular values of the snapshot matrix", fontsize=8)
    S.panel_label(ax, "c", x=-0.17)
    S.save(fig, "nwidth")


class AE(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(N, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, r))
        self.dec = nn.Sequential(nn.Linear(r, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU(), nn.Linear(128, N))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def train_ae(Utr, r, seed=0, epochs=6000):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AE(r).to(dev)
    x = torch.tensor(Utr, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        idx = torch.randint(0, len(x), (256,), device=dev)
        loss = ((m(x[idx])[0] - x[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.cpu().eval()


def rel_err(U, R):
    return np.linalg.norm(U - R) / np.linalg.norm(U)


def run_ae_vs_pod():
    rng = np.random.default_rng(0)
    Utr = advected(rng.uniform(0.15, 0.85, 800))
    s_te = np.linspace(0.16, 0.84, 150)
    Ute = advected(s_te)
    mean = Utr.mean(0)
    _, _, Vt = np.linalg.svd(Utr - mean, full_matrices=False)
    ranks = [1, 2, 4, 8, 16, 32, 64]
    pod = [rel_err(Ute, mean + (Ute - mean) @ Vt[:r].T @ Vt[:r]) for r in ranks]
    ae_ranks = [1, 2, 4, 8]
    ae, best2 = [], None
    for r in ae_ranks:
        errs, ms = [], []
        for seed in range(3):
            m = train_ae(Utr, r, seed)
            with torch.no_grad():
                R = m(torch.tensor(Ute, dtype=torch.float32))[0].numpy()
            errs.append(rel_err(Ute, R)); ms.append(m)
        k = int(np.argmin(errs))
        ae.append(errs[k])
        if r == 2:
            best2 = ms[k]
        print(f"r={r}: AE errors {np.round(errs, 4)}, POD {pod[ranks.index(r)]:.4f}", flush=True)
    k = 40
    with torch.no_grad():
        xt = torch.tensor(Ute, dtype=torch.float32)
        rec_ae = best2(xt[k:k + 1])[0].numpy()[0]
        z2 = best2.enc(xt).numpy()
    rec_pod = mean + (Ute[k] - mean) @ Vt[:2].T @ Vt[:2]
    np.savez(CACHE, ranks=ranks, pod=pod, ae_ranks=ae_ranks, ae=ae, s_te=s_te,
             truth=Ute[k], rec_ae=rec_ae, rec_pod=rec_pod, z2=z2)


def fig_ae_vs_pod():
    S.setup()
    d = np.load(CACHE)
    fig, axs = plt.subplots(1, 3, figsize=(S.TEXTWIDTH, 2.15),
                            gridspec_kw=dict(width_ratios=[1.15, 1.1, 1], wspace=0.45))
    ax = axs[0]
    ax.loglog(d["ranks"], d["pod"], "o-", color=S.ORANGE, mec="white", mew=0.8, ms=5, label="POD (linear)")
    ax.loglog(d["ae_ranks"], d["ae"], "s-", color=S.BLUE, mec="white", mew=0.8, ms=5, label="autoencoder")
    ax.set_xlabel("latent dimension $r$"); ax.set_ylabel(r"relative $L^2$ error, held-out")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_xticks(d["ranks"]); ax.set_xticklabels([str(r) for r in d["ranks"]])
    ax.minorticks_off()
    S.panel_label(ax, "a", x=-0.2)

    ax = axs[1]
    ax.plot(X, d["truth"], color=S.INK, lw=2.6, alpha=0.22, label="truth")
    ax.plot(X, d["rec_pod"], color=S.ORANGE, lw=1.3, label="POD, $r=2$")
    ax.plot(X, d["rec_ae"], color=S.BLUE, lw=1.3, ls="--", label="AE, $r=2$")
    ax.set_xlabel("$x$"); ax.set_ylabel("$u$")
    ax.set_ylim(-0.25, 1.25)
    ax.legend(loc="upper right", fontsize=7)
    S.panel_label(ax, "b", x=-0.2)

    ax = axs[2]
    z = d["z2"]
    ax.plot(z[:, 0], z[:, 1], color=S.GRID, lw=1.0, zorder=1)
    sc = ax.scatter(z[:, 0], z[:, 1], c=d["s_te"], cmap=S.SEQUENTIAL, s=9, zorder=2,
                    edgecolors="none", vmin=0.05, vmax=0.84)
    cb = fig.colorbar(sc, ax=ax, fraction=0.06, pad=0.03)
    cb.set_label("pulse position $s$", fontsize=8); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    ax.set_title("AE latent space, $r=2$", fontsize=8)
    S.panel_label(ax, "c", x=-0.22)
    S.save(fig, "ae_vs_pod")


if __name__ == "__main__":
    fig_nwidth()
    if not CACHE.exists() or "--recompute" in sys.argv:
        run_ae_vs_pod()
    fig_ae_vs_pod()
