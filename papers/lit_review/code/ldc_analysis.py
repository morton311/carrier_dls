"""Global POD vs. Data-informed Local Subspaces (DLS) on the Re = 15,000 lid-driven cavity.

DLS is evaluated with the project implementation (lib/dls_2d.py): local POD modes
from randomly sampled patches are multiplied by bilinear FE shape functions and
used as GFEM enrichments, giving a C0-continuous basis whose DOFs are obtained
by a global L2 projection.

Produces
  figs/ldc_pod.pdf           held-out vorticity, leading POD modes, spectrum
  figs/ldc_local_global.pdf  held-out error vs latent dimension, local spectra
  figs/ldc_local_recon.pdf   reconstructions at an equal latent budget
and caches the numbers in figs/ldc_results.npz.
"""
import logging
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path("/home/HDD/cmorton/carrier_dls")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO))
import style as S  # noqa: E402
import lib.dls_2d as dls  # noqa: E402

logging.getLogger("lib.dls_2d").setLevel(logging.ERROR)

DATA = REPO / "data" / "ldc_15k.h5"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_IDX = np.arange(0, 2000)             # 2000 training snapshots
TEST_IDX = np.arange(2050, 2499, 5)        # 90 held-out snapshots, separated in time
TRAIN_EVAL_IDX = np.linspace(0, 1999, 90).astype(int)
DLS_RUNS = {19: [0, 1, 2, 3], 49: [0, 1, 2, 5, 10], 99: [0, 1, 2, 5, 7, 10, 20]}
RECON = [(49, 1), (99, 7)]                 # both 1936 coefficients per snapshot
REGION = 491                               # grid points covered by every DLS mesh
POD_R = np.unique(np.round(np.logspace(0, np.log10(2000), 40)).astype(int))
CACHE = S.FIG_DIR / "ldc_results.npz"


def load(idx, f):
    d = f["UV"]
    out = np.empty((len(idx), *d.shape[1:]), dtype=np.float32)
    for k, i in enumerate(idx):
        out[k] = d[i]
    return out


def vorticity(uv, dx):
    """uv indexed [..., y, x, (u, v)]."""
    dvdx = np.gradient(uv[..., 1], dx, axis=-1)
    dudy = np.gradient(uv[..., 0], dx, axis=-2)
    return dvdx - dudy


def rel_err(pred, true):
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def crop(a):
    """Crop [..., y, x, c] arrays to the common DLS region."""
    return a[..., :REGION, :REGION, :]


def patch_spectrum(field, p, n=10000, seed=0):
    """Singular values of randomly sampled, mean-removed p x p patches (as in DLS)."""
    rng = np.random.default_rng(seed)
    ny, nx = field.shape
    i = rng.integers(0, ny - p, n)
    j = rng.integers(0, nx - p, n)
    P = np.stack([field[a:a + p, b:b + p].ravel() for a, b in zip(i, j)], 1)
    P -= P.mean(0, keepdims=True)
    return np.linalg.svd(P, compute_uv=False)


def compute():
    t0 = time.time()
    with h5py.File(DATA, "r") as f:
        x = f["x"][:]
        w = f["weights"][:].astype(np.float32)
        xg, yg = f["x_grid"][:], f["y_grid"][:]
        Xtr = load(TRAIN_IDX, f)
        Xte = load(TEST_IDX, f)
    print(f"loaded {Xtr.shape} / {Xte.shape} in {time.time() - t0:.0f}s", flush=True)
    dx = x[1] - x[0]
    mean = Xtr.mean(0)
    sw = np.sqrt(w)[..., None]
    Fte = Xte - mean                                  # held-out fluctuations [t, y, x, c]
    Ftr_eval = Xtr[TRAIN_EVAL_IDX] - mean
    Xtr0 = Xtr[:1].copy()                             # DLS local modes come from here
    res = {}

    # ---- global POD (weighted method of snapshots) -----------------------
    # (modifies Xtr in place; only Xtr0 / Ftr_eval are used afterwards)
    A = torch.from_numpy(Xtr).sub_(torch.from_numpy(mean)).mul_(torch.from_numpy(sw))
    A = A.reshape(len(TRAIN_IDX), -1).to(DEV)
    del Xtr
    C = torch.zeros(len(A), len(A), dtype=torch.float64, device=DEV)
    for s in range(0, A.shape[1], 65536):
        blk = A[:, s:s + 65536].double()
        C += blk @ blk.T
    lam, V = torch.linalg.eigh(C)
    lam, V = lam.flip(0).clamp_min(0), V.flip(1)
    keep = int((lam > lam[0] * 1e-12).sum())
    Phi = (A.T @ V[:, :keep].float()) / lam[:keep].sqrt().float()       # (M, keep)
    del A, C
    res["lam"] = lam.cpu().numpy()
    swt = torch.from_numpy(sw).to(DEV)

    def pod_errors(F):
        Fw = torch.from_numpy(F).to(DEV) * swt
        coef = Fw.reshape(len(F), -1) @ Phi
        truth = crop(F)
        out = []
        for r in POD_R:
            rec = (coef[:, :r] @ Phi[:, :r].T).reshape(Fw.shape) / swt
            out.append(rel_err(crop(rec.cpu().numpy()), truth))
        return np.array(out), coef

    res["pod_r"] = POD_R
    res["pod_err_test"], coef_te = pod_errors(Fte)
    res["pod_err_train"], _ = pod_errors(Ftr_eval)
    print(f"global POD done ({keep} modes) in {time.time() - t0:.0f}s", flush=True)

    modes = (Phi[:, :4].T.cpu().numpy().reshape(4, *mean.shape)) / sw
    res["mode_vort"] = vorticity(modes, dx)
    res["snap_vort"] = vorticity(Fte[0], dx)
    r_eq = 2 * 121 * 8                                 # = 1936, DLS budgets in RECON
    rec = (coef_te[:1, :r_eq] @ Phi[:, :r_eq].T).reshape(1, *mean.shape) / swt
    res["rec_truth"] = vorticity(crop(Fte[0]), dx)
    res["rec_pod"] = vorticity(crop(rec[0].cpu().numpy()), dx)
    del Phi, coef_te
    torch.cuda.empty_cache()

    # ---- DLS (project implementation) ------------------------------------
    # Local modes come from the first *training* snapshot; held-out snapshots
    # are then projected onto that fixed basis via dls_config.
    base = {"mean": mean, "x_grid": xg, "y_grid": yg}
    truth_xy = crop(Fte).transpose(0, 2, 1, 3)        # [t, x, y, c]
    for p, ms in DLS_RUNS.items():
        # Local POD modes once per patch size (library routine, same RNG use),
        # then the enriched element matrices for each number of modes m.
        mode_data, gx, gy = dls._read_static({**base, "UV": Xtr0}, "UV")
        np.random.seed(0)
        lm_u, sv_u = dls.Modal_decomp(mode_data[..., 0], p)
        lm_v, _ = dls.Modal_decomp(mode_data[..., 1], p)
        nskip = (p - 1) // 2
        dims, errs = [], []
        for m in ms:
            mu, _ = dls.local_modemat_over_elem(gx, gy, nskip, lm_u, m, p)
            mv, _ = dls.local_modemat_over_elem(gx, gy, nskip, lm_v, m, p)
            cfg = dls.dls_long_Config_Flexible(
                num_snaps=len(Xte), nx=gx.shape[0], ny=gx.shape[1], num_vars=2,
                patch_size=p, num_modes=m, modemat_local_u=mu, modemat_local_v=mv)
            _, du, dv = dls.gfem_compress_flexible({**base, "UV": Xte}, "UV", p, m,
                                                   dls_config=cfg, batch_size=len(Xte))
            q = dls.gfem_recon_flexible(None, cfg, du, dv, batch_size=len(Xte))
            q = q[:, :REGION, :REGION, :]
            dims.append(du.shape[1] + dv.shape[1])
            errs.append(rel_err(q, truth_xy))
            if (p, m) in RECON:
                res[f"rec_dls_p{p}m{m}"] = vorticity(q[0].transpose(1, 0, 2), dx)
                res[f"cr_dls_p{p}m{m}"] = cfg.compression_ratio
            print(f"DLS p={p} m={m}: latent={dims[-1]}, err={errs[-1]:.4f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        res[f"dls{p}_dim"] = np.array(dims)
        res[f"dls{p}_err"] = np.array(errs)
        res[f"dls{p}_m"] = np.array(ms)
        res[f"dls{p}_sv"] = sv_u

    np.savez_compressed(CACHE, **res)
    print(f"done in {time.time() - t0:.0f}s")
    return dict(np.load(CACHE))


EFF_T = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000]


def compute_efficiency(res):
    """Held-out error of global POD built from the first T snapshots (all modes kept).

    The long-time mean (2000 snapshots) is removed for every method, so only the
    amount of data used to build the basis varies.
    """
    with h5py.File(DATA, "r") as f:
        w = f["weights"][:].astype(np.float32)
        Xtr = load(TRAIN_IDX, f)
        Xte = load(TEST_IDX, f)
    mean = Xtr.mean(0)
    sw = np.sqrt(w)[..., None]
    swt = torch.from_numpy(sw).to(DEV)
    Fte = Xte - mean
    Bw = (torch.from_numpy(Fte).to(DEV) * swt).reshape(len(Fte), -1)
    truth = crop(Fte)
    errs, ranks = [], []
    for T in EFF_T:
        A = torch.from_numpy((Xtr[:T] - mean) * sw).to(DEV).reshape(T, -1)
        lam, V = torch.linalg.eigh((A.double() @ A.double().T))
        lam, V = lam.flip(0).clamp_min(0), V.flip(1)
        keep = int((lam > lam[0] * 1e-12).sum())
        Phi = (A.T @ V[:, :keep].float()) / lam[:keep].sqrt().float()
        rec = ((Bw @ Phi) @ Phi.T).reshape(Fte.shape) / swt
        errs.append(rel_err(crop(rec.cpu().numpy()), truth))
        ranks.append(keep)
        print(f"POD from T={T}: rank {keep}, held-out err {errs[-1]:.4f}", flush=True)
        del A, Phi, rec
    res["eff_T"] = np.array(EFF_T)
    res["eff_err"] = np.array(errs)
    res["eff_rank"] = np.array(ranks)
    np.savez_compressed(CACHE, **res)
    return res


def _img(ax, a, L, title):
    ax.imshow(a, origin="lower", cmap=S.DIVERGING, vmin=-L, vmax=L,
              extent=[0, 1, 0, 1], interpolation="bilinear", rasterized=True)
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(S.INK_2); s.set_linewidth(0.4)


def fig_pod(r):
    S.setup()
    fig = plt.figure(figsize=(S.TEXTWIDTH, 3.55))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.05], hspace=0.42, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    _img(ax, r["snap_vort"], np.percentile(np.abs(r["snap_vort"]), 99), r"held-out $\omega'$")
    S.panel_label(ax, "a", x=-0.05)
    for k in range(4):
        m = r["mode_vort"][k]
        m = m * np.sign(m.flat[np.abs(m).argmax()])
        _img(fig.add_subplot(gs[0, k + 1]), m, np.percentile(np.abs(m), 99), rf"POD mode {k + 1}")

    lam = r["lam"]
    lam = lam[lam > lam[0] * 1e-12]
    n = np.arange(1, len(lam) + 1)
    ax1 = fig.add_subplot(gs[1, :2])
    ax1.loglog(n, lam / lam.sum(), color=S.BLUE)
    ax1.set_xlabel(r"mode index $j$")
    ax1.set_ylabel(r"$\lambda_j / \sum_k \lambda_k$")
    ax1.set_title("POD eigenvalue spectrum", fontsize=8)
    S.panel_label(ax1, "b", x=-0.12)

    ax2 = fig.add_subplot(gs[1, 3:])
    cum = np.cumsum(lam) / lam.sum()
    ax2.semilogx(n, cum, color=S.BLUE)
    for lev, dy in ((0.9, -14), (0.99, -12)):
        j = int(np.searchsorted(cum, lev)) + 1
        ax2.plot([j], [cum[j - 1]], "o", color=S.BLUE, mec="white", mew=1.0, ms=5)
        ax2.annotate(rf"{int(lev * 100)}\% at $r={j}$", (j, cum[j - 1]), xytext=(6, dy),
                     textcoords="offset points", ha="left", fontsize=7.5, color=S.INK_2)
    ax2.set_ylim(0, 1.03)
    ax2.set_xlabel(r"number of modes $r$")
    ax2.set_ylabel("cumulative energy")
    ax2.set_title(r"energy captured by the leading $r$ modes", fontsize=8)
    S.panel_label(ax2, "c", x=-0.12)
    S.save(fig, "ldc_pod")


def fig_local(r):
    S.setup()
    fig, axs = plt.subplots(1, 3, figsize=(S.TEXTWIDTH, 2.35),
                            gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.5))
    colors = dict(zip(DLS_RUNS, [S.BLUE_RAMP[1], S.BLUE_RAMP[3], S.BLUE_RAMP[5]]))
    ax = axs[0]
    pr = r["pod_r"]
    ax.loglog(pr, r["pod_err_test"], color=S.ORANGE, label="global POD")
    for p in DLS_RUNS:
        d, e = r[f"dls{p}_dim"], r[f"dls{p}_err"]
        ax.loglog(d, e, "o-", color=colors[p], mec="white", mew=0.7, ms=4,
                  label=rf"DLS ${p}{{\times}}{p}$")
    ax.set_xlabel(r"latent dimension $r$")
    ax.set_ylabel(r"relative $L^2$ error, held-out")
    ax.set_title("error vs.\ latent size", fontsize=8)
    ax.legend(loc="lower left", fontsize=6.5)
    S.panel_label(ax, "a", x=-0.2)

    ax = axs[1]
    ax.loglog(r["eff_T"], r["eff_err"], "o-", color=S.ORANGE, mec="white", mew=0.7, ms=4,
              label="global POD, all modes")
    for p, m in [(19, 3), (49, 10), (99, 20)]:
        ms = list(r[f"dls{p}_m"])
        if m in ms:
            e = r[f"dls{p}_err"][ms.index(m)]
            ax.axhline(e, color=colors[p], lw=1.1, ls="--")
            ax.text(1.2, e * 1.12, rf"DLS ${p}^2$, $m{{=}}{m}$", fontsize=6.3,
                    color=S.INK_2, va="bottom")
    ax.set_xlabel(r"training snapshots $T$")
    ax.set_title("error vs.\ training data", fontsize=8)
    ax.text(12, 0.62, "global POD, all modes", fontsize=6.5, color=S.INK_2, va="bottom")
    S.panel_label(ax, "b", x=-0.2)

    ax = axs[2]
    lam = r["lam"]
    lam = lam[lam > lam[0] * 1e-12]
    ax.loglog(np.arange(1, len(lam) + 1), np.sqrt(lam / lam[0]), color=S.ORANGE, label="global")
    for p in DLS_RUNS:
        sv = r[f"dls{p}_sv"]
        ax.loglog(np.arange(1, len(sv) + 1), sv / sv[0], color=colors[p], label=rf"${p}^2$ patch")
    ax.set_xlabel(r"mode index $j$")
    ax.set_ylabel(r"$\sigma_j / \sigma_1$")
    ax.set_ylim(1e-6, 1.5)
    ax.set_title("singular values", fontsize=8)
    ax.legend(loc="upper right", fontsize=6.3, handlelength=1.2)
    S.panel_label(ax, "c", x=-0.22)
    S.save(fig, "ldc_local_global")

    fig, axs = plt.subplots(1, 5, figsize=(S.TEXTWIDTH, 1.62), gridspec_kw=dict(wspace=0.05))
    L = np.percentile(np.abs(r["rec_truth"]), 99)
    items = [("rec_truth", "held-out truth"),
             ("rec_pod", r"POD, $r{=}1936$"),
             ("rec_dls_p49m1", r"DLS $49^2$, $m{=}1$"),
             ("rec_dls_p99m7", r"DLS $99^2$, $m{=}7$"),
             ("rec_dls_p19m3", r"DLS $19^2$, $m{=}3$")]
    for ax, (k, t) in zip(axs, items):
        e = "" if k == "rec_truth" else rf" ({100 * rel_err(r[k], r['rec_truth']):.0f}\%)"
        _img(ax, r[k], L, t)
        if e:
            ax.set_xlabel(e.strip(" ()"), fontsize=7, labelpad=2)
    for ax, rr in zip(axs[1:], ["1936", "1936", "1936", "25{,}992"]):
        ax.set_xlabel(rf"$r={rr}$" + "\n" + ax.get_xlabel() + " error", fontsize=7,
                      labelpad=2, linespacing=1.1)
    S.save(fig, "ldc_local_recon")


def add_fine_dls(r):
    """Reconstruction of held-out snapshot 0 with DLS 19x19, m = 3."""
    with h5py.File(DATA, "r") as f:
        x = f["x"][:]
        xg, yg = f["x_grid"][:], f["y_grid"][:]
        Xtr = load(TRAIN_IDX, f)
        X0 = f["UV"][TEST_IDX[0]][None].astype(np.float32)
    mean = Xtr.mean(0)
    Xtr0 = Xtr[:1].copy()
    del Xtr
    base = {"mean": mean, "x_grid": xg, "y_grid": yg}
    p, m = 19, 3
    mode_data, gx, gy = dls._read_static({**base, "UV": Xtr0}, "UV")
    np.random.seed(0)
    lm_u, _ = dls.Modal_decomp(mode_data[..., 0], p)
    lm_v, _ = dls.Modal_decomp(mode_data[..., 1], p)
    mu, _ = dls.local_modemat_over_elem(gx, gy, (p - 1) // 2, lm_u, m, p)
    mv, _ = dls.local_modemat_over_elem(gx, gy, (p - 1) // 2, lm_v, m, p)
    cfg = dls.dls_long_Config_Flexible(num_snaps=1, nx=gx.shape[0], ny=gx.shape[1], num_vars=2,
                                       patch_size=p, num_modes=m, modemat_local_u=mu, modemat_local_v=mv)
    _, du, dv = dls.gfem_compress_flexible({**base, "UV": X0}, "UV", p, m, dls_config=cfg, batch_size=1)
    q = dls.gfem_recon_flexible(None, cfg, du, dv, batch_size=1)[:, :REGION, :REGION, :]
    r["rec_dls_p19m3"] = vorticity(q[0].transpose(1, 0, 2), x[1] - x[0])
    np.savez_compressed(CACHE, **r)
    return r


if __name__ == "__main__":
    r = dict(np.load(CACHE)) if CACHE.exists() and "--recompute" not in sys.argv else compute()
    if "eff_err" not in r:
        r = compute_efficiency(r)
    if "rec_dls_p19m3" not in r:
        r = add_fine_dls(r)
    fig_pod(r)
    fig_local(r)
