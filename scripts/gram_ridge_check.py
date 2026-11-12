"""How much of the Gram loss is ridge, on the truth and on a trained run's forecast errors.

The training loss scores element errors in M_e + lambda (tr M_e / n) I (lib/decode.py). For each run this prints, for
the true coefficients and for the run's 1- and 8-step forecast errors on the last 8% of the record: the share of the
metric norm that is ridge (field-invisible), and the angle between the loss gradient and the field-energy gradient
with respect to the standardized outputs the optimizer works in, for the Gram loss and for a plain MSE on those outputs.

    python scripts/gram_ridge_check.py results/<latent>/<run> ...
"""
import json, logging, os, sys
import numpy as np, torch, h5py
sys.path.insert(0, '.')
logging.basicConfig(level=logging.ERROR)
from lib.runner import runner
import lib.decode as decode
for d in sys.argv[1:]:
    d = d.rstrip('/')
    cfg, = [f for f in os.listdir(d) if f.endswith('.json')]
    c = json.load(open(f'{d}/{cfg}'))
    c.update(overwrite='x', mode='eval', name=cfg[:-5], log='terminal', distributed=False, device='cuda')
    r = runner(c)
    lt = r._compute_lltogl_mat()
    mats = [r.l_config.modemat_local_u, r.l_config.modemat_local_v]
    full = float(getattr(r.l_config, 'cut_weight', 0.0)) > 0
    masks = decode.element_masks_for(r, strict=True)
    lam = float(c['model_params'].get('gram_lambda', decode.GRAM_LAMBDA))
    def gram(l):   # (2, E, n, n) element metrics M_e (+ ridge)
        out = []
        for m in mats:
            Wr = (np.broadcast_to(decode.element_gram_sqrt(m, l), (lt.shape[0],) + (m.shape[1],) * 2) if masks is None
                  else decode.element_gram_roots(m, masks, l, full))
            out.append(torch.from_numpy(np.einsum('eij,ejk->eik', Wr, Wr)).double())
        return out
    M0, Ml = gram(0.0), gram(lam)
    name = next(iter(r.dof_std))
    sd = r.dof_std[name].double().reshape(2, -1)            # per slot: (2, n)
    tl = c['model_params']['time_lag']
    with h5py.File(r.paths_bib.latent_path, 'r') as f:
        g = f[name]
        n = g['dof_u'].shape[0]
        starts = np.linspace(int(0.92 * n), n - 20, 12).astype(int) - tl
        E1, E8, T = [], [], []
        for s in starts:
            pu, pv, _ = r._predict_dofs_forward(name=name, ids=np.arange(s, s + tl + 8), dof_u=g['dof_u'][s:s + tl], dof_v=g['dof_v'][s:s + tl], total_steps=tl + 8)
            tu, tv = g['dof_u'][s + tl:s + tl + 8], g['dof_v'][s + tl:s + tl + 8]
            E1.append(np.stack([pu[tl] - tu[0], pv[tl] - tv[0]])); E8.append(np.stack([pu[tl + 7] - tu[7], pv[tl + 7] - tv[7]]))
            T.append(np.stack([tu[0], tv[0]]))
    for tag, X in (('truth', T), ('error step 1', E1), ('error step 8', E8)):
        X = torch.from_numpy(np.stack(X)).double()             # (B, 2, N) physical global coefficients
        e = torch.stack([X[:, k][:, lt] for k in range(2)], 1)   # (B, 2, E, n) element tokens
        q0 = sum(torch.einsum('eij,bej,bei->b', M0[k], e[:, k], e[:, k]) for k in range(2))
        ql = sum(torch.einsum('eij,bej,bei->b', Ml[k], e[:, k], e[:, k]) for k in range(2))
        # gradient w.r.t. the standardized outputs z (e = sd * z): sd * M e
        g0 = torch.stack([sd[k] * torch.einsum('eij,bej->bei', M0[k], e[:, k]) for k in range(2)], 1).flatten(1)
        gl = torch.stack([sd[k] * torch.einsum('eij,bej->bei', Ml[k], e[:, k]) for k in range(2)], 1).flatten(1)
        gr = torch.stack([e[:, k] / sd[k] for k in range(2)], 1).flatten(1)   # MSE on z: the error in z itself
        ang = lambda a: float(torch.rad2deg(torch.acos(torch.clamp((g0 * a).sum(1) / (g0.norm(dim=1) * a.norm(dim=1)), -1, 1))).mean())
        print(f"{d.split('/')[1]:34} {tag:13} ridge share of metric {float(1 - q0.sum() / ql.sum()):6.1%}   "
              f"gradient angle to field energy: Gram loss {ang(gl):5.1f} deg, MSE on coefficients {ang(gr):5.1f} deg", flush=True)
