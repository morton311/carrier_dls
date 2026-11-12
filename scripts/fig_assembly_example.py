"""Real element predictions before and after the global solve (mass-weighted projection).

Rolls a trained element-token model forward from the start of its held-out block,
exactly as runner._predict_dofs_forward_spatial does, and at chosen steps keeps
both the network's raw element outputs (each element decoded on its own: a
patchwork field with seams at element edges) and the assembled output after
runner._reassemble_tokens (one value per node, continuous field).

    python scripts/fig_assembly_example.py compute   # GPU; caches results/<latent>/<run>/assembly_example.npz
    python scripts/fig_assembly_example.py plot      # writes figures/assembly_example.pdf/.png
"""
import json
import logging
import os
import pickle
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

NAME = 'ldc_15k'
LATENT = 'ldc_15k_dls_p99m10'
RUNS = {'Galerkin in the loop': 'spatial_gram_rollout_galerkin_heldout_s1',
        'trained raw, Galerkin at inference': 'spatial_gram_rollout_heldout_s1'}
STEPS = (1, 30, 200)
CACHE = 'results/{latent}/{run}/assembly_example.npz'


def patchwork(cfg, dl, B, e_ij):
    """Decode element DOFs (E, n) one element at a time onto the mesh-extent grid.

    Shared edges are written by each element in turn, so a discontinuity between
    neighbours shows as a jump between adjacent grid lines.
    """
    import lib.dls_2d as dls
    nkx, nky = dls.config_nskip(cfg)
    nx_e, ny_e = getattr(cfg, 'nx_e', cfg.nx_t), getattr(cfg, 'ny_e', cfg.ny_t)
    out = np.zeros((nx_e, ny_e))
    for e, (i, j) in enumerate(e_ij):
        patch = (B @ dl[e]).reshape(nkx + 1, nky + 1, order='F')
        out[i * nkx:i * nkx + nkx + 1, j * nky:j * nky + nky + 1] = patch
    return out[:cfg.nx_t, :cfg.ny_t]


def edge_jumps(cfg, dl, B, e_ij):
    """RMS mismatch between neighbouring elements along their shared edges."""
    import lib.dls_2d as dls
    nkx, nky = dls.config_nskip(cfg)
    idx = {ij: e for e, ij in enumerate(e_ij)}
    sq, n = 0.0, 0
    for (i, j), e in idx.items():
        p = (B @ dl[e]).reshape(nkx + 1, nky + 1, order='F')
        if (i + 1, j) in idx:
            q = (B @ dl[idx[(i + 1, j)]]).reshape(nkx + 1, nky + 1, order='F')
            sq += ((p[-1, :] - q[0, :]) ** 2).sum(); n += p.shape[1]
        if (i, j + 1) in idx:
            q = (B @ dl[idx[(i, j + 1)]]).reshape(nkx + 1, nky + 1, order='F')
            sq += ((p[:, -1] - q[:, 0]) ** 2).sum(); n += p.shape[0]
    return np.sqrt(sq / max(n, 1))


def compute():
    import torch
    from lib.runner import runner
    logging.basicConfig(level=logging.WARNING)
    for label, run in RUNS.items():
        c = json.load(open(f'configs/ldc/{run}.json'))
        c['model_params'].update(reassemble=True, reassemble_lambda=0.01)
        c.update(overwrite='x', mode='eval', name=run, log='terminal', distributed=False, device='cuda')
        r = runner(c)
        cfg = r.l_config
        tl = c['model_params']['time_lag']
        ids = np.asarray(pickle.load(open(f'results/{LATENT}/{run}/split_ids.pkl', 'rb'))
                         ['eval_data'][NAME]['val_indices'])
        geom = r._elem_geometry(use_skip=False)
        L = geom['lltogl_np']
        e_ij = [(i, j) for i in range(cfg.nx_g - 1) for j in range(cfg.ny_g - 1)]
        n = cfg.dof_elem
        with h5py.File(r.paths_bib.latent_path, 'r') as f:
            last = ids[0] + tl + max(STEPS)
            tu, tv = f[NAME]['dof_u'][ids[0]:last], f[NAME]['dof_v'][ids[0]:last]
        mean = r.dof_mean[NAME].to(r.device)
        std = r.dof_std[NAME].to(r.device)
        x = np.concatenate([tu[:tl][:, L], tv[:tl][:, L]], axis=2)
        x = ((torch.from_numpy(x).float().to(r.device) - mean) / std).permute(1, 0, 2).unsqueeze(0)
        model = r.model.module if hasattr(r.model, 'module') else r.model
        model.eval()
        keep = {}
        with torch.no_grad():
            for k in range(1, max(STEPS) + 1):
                out = model(x, geom['coords'])
                asm = r._reassemble_tokens(out, NAME, geom)
                if k in STEPS:
                    keep[k] = ((out[0].double() * std + mean).cpu().numpy(),
                               (asm[0].double() * std + mean).cpu().numpy())
                x = torch.cat((x[:, :, 1:, :], asm.unsqueeze(2)), dim=2)
        res = {}
        Bu, Bv = cfg.modemat_local_u, cfg.modemat_local_v
        for k, (raw, asm) in keep.items():
            truth = (tu[tl - 1 + k][L], tv[tl - 1 + k][L])
            for tag, d in (('raw', raw), ('asm', asm), ('true', None)):
                du = truth[0] if d is None else d[:, :n]
                dv = truth[1] if d is None else d[:, n:]
                res[f'{tag}_u_{k}'] = patchwork(cfg, du, Bu, e_ij)
                res[f'{tag}_v_{k}'] = patchwork(cfg, dv, Bv, e_ij)
                res[f'{tag}_jump_{k}'] = np.hypot(edge_jumps(cfg, du, Bu, e_ij), edge_jumps(cfg, dv, Bv, e_ij))
        res['nskip'] = np.array(r.dls.config_nskip(cfg))
        out_path = CACHE.format(latent=LATENT, run=run)
        np.savez_compressed(out_path, **res)
        for k in STEPS:
            t = np.stack([res[f'true_u_{k}'], res[f'true_v_{k}']])
            rms = np.sqrt((t ** 2).mean())
            def err(tag):
                p = np.stack([res[f'{tag}_u_{k}'], res[f'{tag}_v_{k}']])
                return np.linalg.norm(p - t) / np.linalg.norm(t)
            print(f"{label:36} step {k:3d}: edge jump / field rms  raw {res[f'raw_jump_{k}'] / rms:.4f}  "
                  f"assembled {res[f'asm_jump_{k}'] / rms:.2e} | error vs projected truth  raw {err('raw'):.4f}  "
                  f"assembled {err('asm'):.4f}")
        print(f"wrote {out_path}")
        del r
        torch.cuda.empty_cache()


def plot(k=30):
    import matplotlib.pyplot as plt
    import lib.plotting as pl  # noqa: F401  (repo figure style: LaTeX text)
    plt.rcParams['figure.constrained_layout.use'] = False
    h = 1.0 / 511                                    # uniform 512x512 unit-square grid
    # (model, quantity, colorbar label, row label, annotate velocity errors?)
    rows = [('Galerkin in the loop', 'u', r"$u'$", r"trained through the solve, $u'$", True),
            ('Galerkin in the loop', 'w', r"$\omega'$", r"trained through the solve, $\omega'$", False),
            ('trained raw, Galerkin at inference', 'w', r"$\omega'$", r"trained raw, $\omega'$", True)]
    cols = [r'before the global solve' '\n' r'(raw element outputs)',
            r'after the global solve',
            r'correction' '\n' r'(after $-$ before)',
            r'truth' '\n' r'(DLS projection)']
    W = pl.width
    fig = plt.figure(figsize=(W, 0.80 * W))
    left, right, top, bottom = 0.075, 0.905, 0.925, 0.02
    pw = (right - left) / 4
    ph = (top - bottom) / 3
    data = {label: np.load(CACHE.format(latent=LATENT, run=run)) for label, run in RUNS.items()}
    for r_i, (label, q, qlab, rlab, annotate) in enumerate(rows):
        d = data[label]
        def field(tag):
            u, v = d[f'{tag}_u_{k}'], d[f'{tag}_v_{k}']
            return u if q == 'u' else np.gradient(v, h, axis=0) - np.gradient(u, h, axis=1)
        raw, asm, tru = field('raw'), field('asm'), field('true')
        lim = np.percentile(np.abs(tru), 99.5)
        corr = asm - raw
        clim = np.percentile(np.abs(corr), 99.5)
        # field-level numbers (both velocity components) for the annotations
        t = np.stack([d[f'true_u_{k}'], d[f'true_v_{k}']])
        rms = np.sqrt((t ** 2).mean())
        def err(tag):
            p = np.stack([d[f'{tag}_u_{k}'], d[f'{tag}_v_{k}']])
            return np.linalg.norm(p - t) / np.linalg.norm(t)
        # velocity-field numbers: RMS jump along shared edges / field RMS, and error vs the projected truth
        notes = [rf"edge jump {100 * d[f'raw_jump_{k}'] / rms:.3g}\%" '\n' rf"error {100 * err('raw'):.2g}\%",
                 r"edge jump $\approx 0$" '\n' rf"error {100 * err('asm'):.2g}\%",
                 None, None] if annotate else [None] * 4
        y0 = top - (r_i + 1) * ph
        for c_i, (a, l) in enumerate([(raw, lim), (asm, lim), (corr, clim), (tru, lim)]):
            ax = fig.add_axes([left + c_i * pw + 0.004, y0 + 0.012, pw - 0.008, ph - 0.024])
            ax.imshow(a.T, origin='lower', cmap='RdBu_r', vmin=-l, vmax=l, extent=(0, 1, 0, 1),
                           interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if notes[c_i]:
                ax.text(0.03, 0.03, notes[c_i], transform=ax.transAxes, fontsize=6.3, va='bottom', ha='left',
                        color='0.1', bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.2))
            if c_i == 2:
                ax.text(0.97, 0.03, rf'$\pm{clim:.2g}$', transform=ax.transAxes, fontsize=6.3, ha='right',
                        va='bottom', color='0.1', bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.2))
            if r_i == 0:
                ax.set_title(cols[c_i], fontsize=7.6, pad=3)
        cax = fig.add_axes([right + 0.012, y0 + 0.05, 0.012, ph - 0.10])
        # shared by the before / after / truth panels; the correction has its own range (annotated)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-lim, lim), cmap='RdBu_r'), cax=cax)
        cb.ax.tick_params(labelsize=6, length=2)
        cb.set_label(qlab, fontsize=7.5, labelpad=2)
        fig.text(left - 0.012, y0 + ph / 2, rlab, rotation=90, ha='right', va='center', fontsize=7.4)
    fig.text(left, 0.975, rf'Lid-driven cavity, $\mathrm{{Re}}=15{{,}}000$, held-out rollout, forecast step {k}; '
             r'p99m10 basis ($10\times10$ elements)', fontsize=7.6, va='top')
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/assembly_example.pdf')
    fig.savefig('figures/assembly_example.png', dpi=220)
    print('wrote figures/assembly_example.pdf/.png')


if __name__ == '__main__':
    if sys.argv[1:] == ['compute']:
        compute()
    elif sys.argv[1:] == ['plot']:
        plot()
    else:
        print(__doc__)
