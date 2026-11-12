"""Reassembly study for the element-token spatial transformer (LDC, inference + training variants).

Four settings of one recipe (configs/ldc/spatial_gram_rollout*.json):

    none                 trained on raw element outputs; no reassembly in the rollout
    average              trained raw; each node's copies averaged after every step
    Galerkin (inference) trained raw; mass-weighted projection after every step
    Galerkin (trained)   the same projection inside the training rollout and at inference

For every trained run of a setting it records the inter-element disagreement at
each rollout step -- the fraction of the predicted frame's field energy that is
mismatch between elements over the nodes they share, E(raw - assembled)/E(assembled),
with the exact element mass matrix -- plus multi-start 30-step skill and the
long-run RMS error against truth averaged over the same snapshots.

    python scripts/reassembly_study.py compute   # GPU; writes results/<latent>/reassembly_study.json
    python scripts/reassembly_study.py plot      # writes figures/reassembly_study.pdf/.png
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
# runs on the held-out split (split_version 2); the version-1 study is kept in
# reassembly_study.json, from spatial_gram_rollout{,_s1,_s2} and their Galerkin twins
OUT = f'results/{LATENT}/reassembly_study_heldout.json'
LAM = 0.01
RAW_RUNS = [f'spatial_gram_rollout_heldout_s{s}' for s in (1, 2, 3)]
GAL_RUNS = [f'spatial_gram_rollout_galerkin_heldout_s{s}' for s in (1, 2, 3)]
# setting -> (runs, model_params.reassemble, model_params.reassemble_lambda)
SETTINGS = {
    'none': (RAW_RUNS, False, None),
    'average': (RAW_RUNS, True, None),
    'Galerkin (inference)': (RAW_RUNS, True, LAM),
    'Galerkin (trained)': (GAL_RUNS, True, LAM),
}


def compute():
    import torch
    from lib.runner import runner
    import lib.multistart as multistart
    logging.basicConfig(level=logging.WARNING)

    # every run shares the latent's eval split
    ids = np.asarray(pickle.load(open(f'results/{LATENT}/{RAW_RUNS[0]}/split_ids.pkl', 'rb'))['eval_data'][NAME]['val_indices'])
    with h5py.File('data/ldc_15k.h5', 'r') as f:
        mean_f = f['mean'][:].astype(np.float64)
        UV = f['UV'][ids[0]:ids[-1] + 1]
    truth = {}
    def rms_truth(tl):           # true RMS over the predicted snapshots, val_id[tl:]
        if tl not in truth:
            truth[tl] = np.sqrt(((UV[tl:].astype(np.float64) - mean_f) ** 2).mean(0)).transpose(1, 0, 2)
        return truth[tl]
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for setting, (runs, reassemble, lam) in SETTINGS.items():
        for run in runs:
            if not os.path.exists(f'results/{LATENT}/{run}/model.pth'):
                print(f"{setting:22} {run}: not trained yet, skipped")
                continue
            if run in results.get(setting, {}):
                continue
            c = json.load(open(f'configs/ldc/{run}.json'))
            c['model_params'].update(reassemble=reassemble, reassemble_lambda=lam)
            c.update(overwrite='x', mode='eval', name=run, log='terminal', distributed=False, device='cuda')
            r = runner(c)
            tl = c['model_params']['time_lag']
            with h5py.File(r.paths_bib.latent_path, 'r') as f:
                tu, tv = f[NAME]['dof_u'][ids[0]:ids[-1] + 1], f[NAME]['dof_v'][ids[0]:ids[-1] + 1]

            s, _ = multistart.evaluate(r, NAME, ids)
            pu, pv, _ = r._predict_dofs_forward(name=NAME, ids=ids, dof_u=tu[:tl], dof_v=tv[:tl], total_steps=len(ids))
            Q = r.dls.gfem_recon_flexible(config=r.l_config, dof_u=pu.astype(np.float32),
                                          dof_v=pv.astype(np.float32), rec_target=None)[tl:]
            rms = np.sqrt((Q.astype(np.float64) ** 2).mean(0))
            g = rms_truth(tl)
            k = tuple(slice(0, min(a, b)) for a, b in zip(rms.shape, g.shape))
            rms_err = 100 * np.linalg.norm(rms[k] - g[k]) / np.linalg.norm(g[k])

            # disagreement along the same rollout, replicated so the raw output is visible
            model = r.model.module if hasattr(r.model, 'module') else r.model
            geom = r._elem_geometry(use_skip=False)
            n = r.l_config.dof_elem
            mean = r.dof_mean[NAME].to(r.device).double()
            std = r.dof_std[NAME].to(r.device).double()
            Wt = r.dls._build_wt_vec(*r.dls.config_nskip(r.l_config))
            Ms = [torch.from_numpy((B * Wt[:, None]).T @ B).double().to(r.device)
                  for B in (r.l_config.modemat_local_u, r.l_config.modemat_local_v)]
            def energy(d):           # physical element tokens (1, E, 2n) -> patchwork field energy
                return sum(float(torch.einsum('gei,ij,gej->', d[..., c_ * n:(c_ + 1) * n], M, d[..., c_ * n:(c_ + 1) * n]))
                           for c_, M in enumerate(Ms))
            x = np.concatenate([tu[:tl][:, geom['lltogl_np']], tv[:tl][:, geom['lltogl_np']]], axis=2)
            x = ((torch.from_numpy(x).double().to(r.device) - mean) / std).float().permute(1, 0, 2).unsqueeze(0)
            dis = []
            with torch.no_grad():
                for _ in range(len(ids) - tl):
                    out = model(x, geom['coords'])
                    asm = r._reassemble_tokens(out, NAME, geom)
                    dis.append(energy((out.double() - asm.double()) * std) / energy(asm.double() * std + mean))
                    x = torch.cat((x[:, :, 1:, :], (asm if reassemble else out).unsqueeze(2)), dim=2)

            results.setdefault(setting, {})[run] = dict(
                skill=s['ms_skill'], err30=s['ms_err_at30_mean'], err_mean=s['ms_err_mean'], rms=float(rms_err),
                disagreement=[float(v) for v in dis])
            print(f"{setting:22} {run:34} skill {s['ms_skill']:.3f}  err@30 {s['ms_err_at30_mean']:.3f}  RMS {rms_err:.2f}%", flush=True)
            json.dump(results, open(OUT, 'w'))
            del r
            torch.cuda.empty_cache()

    # compression floor: the codec reconstruction of the true coefficients, same window,
    # decoded from the latent (decoded fields are no longer stored)
    import lib.dls_2d as dls
    tl = json.load(open(f'configs/ldc/{RAW_RUNS[0]}.json'))['model_params']['time_lag']
    cfg = pickle.load(open(f'results/{LATENT}/latent_config.pkl', 'rb'))
    lo, hi = int(ids[tl]), int(ids[-1]) + 1
    with h5py.File(f'results/{LATENT}/latent_coeff.h5', 'r') as f:
        Q = dls.decode_fields(cfg, f[NAME]['dof_u'][lo:hi], f[NAME]['dof_v'][lo:hi])
    rms = np.sqrt((Q.astype(np.float64) ** 2).mean(0)); g = rms_truth(tl)
    k = tuple(slice(0, min(a, b)) for a, b in zip(rms.shape, g.shape))
    results['_floor_rms'] = float(100 * np.linalg.norm(rms[k] - g[k]) / np.linalg.norm(g[k]))
    json.dump(results, open(OUT, 'w'))
    print(f"compression floor RMS {results['_floor_rms']:.2f}%  -> {OUT}")


def plot():
    import matplotlib.pyplot as plt
    import lib.plotting as pl          # repo figure style (paper size, LaTeX text)

    res = json.load(open(OUT))
    floor = res.get('_floor_rms')
    # fixed order; 'none' is the reference, drawn neutral; the three projections use
    # the validated categorical slots 1-3 and distinct dashes (aqua needs the relief)
    style = {
        'none': dict(color='0.55', ls='-', label='no reassembly'),
        'average': dict(color='#2a78d6', ls='-', label='nodal average'),
        'Galerkin (inference)': dict(color='#eb6834', ls='--', label='mass-weighted, inference only'),
        'Galerkin (trained)': dict(color='#1baf7a', ls='-.', label='mass-weighted, trained in the loop'),
    }
    settings = [s for s in style if s in res]
    # display names; the 'Galerkin (...)' keys are the setting names stored in the results JSON
    short = {'none': 'none', 'average': 'average', 'Galerkin (inference)': 'mass-weighted',
             'Galerkin (trained)': 'mass-weighted, trained'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(pl.width, 0.55 * pl.width),
                                   gridspec_kw=dict(width_ratios=[1.6, 1]))

    # (a) the mechanism, on the model trained WITHOUT assembly. A model trained with
    # the projection in the loop leaves consistency to it, so its raw-output
    # mismatch is large by design; it is reported in the table, not drawn here.
    for s in [s for s in ('none', 'average', 'Galerkin (inference)') if s in res]:
        curves = np.array([v['disagreement'] for v in res[s].values()])
        t = np.arange(1, curves.shape[1] + 1)
        med = np.median(curves, axis=0)
        st = style[s]
        ax1.plot(t, med, color=st['color'], ls=st['ls'], lw=1.6, label=st['label'])
        if len(curves) > 1:
            ax1.fill_between(t, curves.min(0), curves.max(0), color=st['color'], alpha=0.15, lw=0)
        ax1.text(t[-1] * 1.02, med[-1], short[s], fontsize=7, va='center', color='0.15')
    T = len(t)
    ax1.set_yscale('log')
    ax1.set_xlim(1, T * 1.18)
    ax1.set_xticks([x for x in (100, 200, 300, 400, 500) if x <= T])
    ax1.set_xlabel('rollout step')
    ax1.set_ylabel('inter-element disagreement')
    ax1.set_title('(a) mismatch at shared nodes', fontsize=9)
    ax1.grid(visible=True, which='major', linestyle='--', linewidth=0.4)
    ax1.legend(frameon=False, fontsize=7, loc='center right', bbox_to_anchor=(1.0, 0.62))

    for i, s in enumerate(settings):
        vals = [v['rms'] for v in res[s].values()]
        st = style[s]
        ax2.scatter(np.full(len(vals), i), vals, s=14, color=st['color'], zorder=3, alpha=0.8)
        ax2.plot([i - 0.25, i + 0.25], [np.mean(vals)] * 2, color=st['color'], lw=2, zorder=4)
        ax2.text(i + 0.3, np.mean(vals), f"{np.mean(vals):.1f}", fontsize=7, va='center', color='0.15')
    if floor is not None:
        ax2.axhline(floor, color='0.3', lw=0.8, ls=':')
        ax2.text(-0.4, floor, f'compression floor {floor:.1f}', fontsize=7, va='bottom', ha='left', color='0.15')
    ax2.set_ylim(0, None)
    ax2.set_xlim(-0.5, len(settings) - 0.2)
    ax2.set_xticks(range(len(settings)))
    ax2.set_xticklabels(['none', 'average', 'mass-\nweighted,\ninference', 'mass-\nweighted,\ntrained'][:len(settings)],
                        fontsize=7)
    ax2.set_ylabel(r'long-run RMS error (\%)')
    ax2.set_title('(b) statistics over 500 steps', fontsize=9)
    ax2.grid(visible=True, axis='y', linestyle='--', linewidth=0.4)

    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/reassembly_study.pdf', bbox_inches='tight')
    fig.savefig('figures/reassembly_study.png', dpi=200, bbox_inches='tight')
    print("wrote figures/reassembly_study.pdf/.png")
    print(f"\n{'setting':24}{'runs':>5}{'skill':>16}{'err@30':>16}{'RMS %':>16}{'raw mismatch 400-500':>22}")
    for s in settings:
        v = list(res[s].values())
        f = lambda key: f"{np.mean([x[key] for x in v]):.3f} ± {np.std([x[key] for x in v]):.3f}"
        mis = np.median([np.mean(x['disagreement'][-100:]) for x in v])
        print(f"{s:24}{len(v):>5}{f('skill'):>16}{f('err30'):>16}{f('rms'):>16}{mis:>22.1e}")
    print(f"compression floor RMS {floor:.2f}%")


if __name__ == '__main__':
    {'compute': compute, 'plot': plot}[sys.argv[1] if len(sys.argv) > 1 else 'plot']()
