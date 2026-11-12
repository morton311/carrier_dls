"""Multi-start short-horizon forecast evaluation.

The long rollout scored elsewhere in eval() starts from ONE initial condition,
so any spread it reports is over time steps, not over forecasts. This launches
many `horizon`-step forecasts from start points spread evenly across a split
and scores every step in field space, against the truth and against
persistence (holding the last seed frame). That is the statistic a
short-horizon forecasting task is judged on: the mean error over the window,
how much it varies from one forecast to the next, and whether the model beats
doing nothing.

Field error comes straight from the DOFs. The DLS reconstruction is linear and
block-diagonal per element, so with the unregularized per-element Gram root W,
||W d||^2 is the field energy of d (see lib/decode.py) and no field has to be
decoded. Checked against the full 491x491 reconstructions on the LDC p99m10
encdec rollout: median deviation 0.36% over 500 steps. POD modes are
orthonormal, so there the coefficient norm already is the field norm.

A diffusion model (model_type 'tr_spatial_diff') is scored as an ensemble of
diffusion.members sampled forecasts per start: 'ms_err' is the error of the
ensemble mean, which is what a point forecast is compared on, and the members
are scored on their own as well ('ms_member_*'), together with the ensemble
spread and the spread-skill ratio, sqrt((M+1)/M) * spread / error of the mean,
which is 1 for an ensemble whose spread matches its error (Fortin et al. 2014).

model_params keys:
    horizon       forecast length in steps (default 30, shared with pred())
    eval_starts   number of forecasts per split (default 50)
"""
import logging

import h5py
import numpy as np

import lib.decode as decode
import lib.diffusion as diffusion

logger = logging.getLogger(__name__)

DEFAULT_STARTS = 50
# per-step statistics are reported at these steps plus the final one
_REPORT_STEPS = (1, 10)


def start_positions(ids, window: int, num_starts: int) -> np.ndarray:
    """Positions i, spread evenly over the split, where ids[i:i+window] is contiguous.

    A forecast needs `window` consecutive snapshots (seed + horizon); a split
    whose indices skip snapshots would otherwise pair a seed with the wrong
    future.
    """
    ids = np.asarray(ids)
    n = len(ids) - window + 1
    if n <= 0:
        return np.array([], dtype=int)
    valid = np.flatnonzero(ids[window - 1:] - ids[:n] == window - 1)
    if len(valid) <= num_starts:
        return valid
    pick = np.unique(np.linspace(0, len(valid) - 1, num_starts).round().astype(int))
    return valid[pick]


def field_energy_fn(runner):
    """Return energy(comps) -> per-snapshot field energy of a list of DOF arrays.

    `comps` holds one (T, num_dofs) array per velocity component, as returned by
    runner._predict_dofs_forward.
    """
    if runner.config['latent_params']['type'] == 'pod':
        return lambda comps: sum((c ** 2).sum(axis=-1) for c in comps)

    lc = runner.l_config
    mats = [lc.modemat_local_u, lc.modemat_local_v]
    if runner.dim == 3:
        mats.append(lc.modemat_local_w)
    # lam = 0: exact field energy. The regularized metric used for training
    # deliberately departs from it, which is right for a loss but not a score.
    lltogl = runner._compute_lltogl_mat()
    # scores count in-domain points only, whatever weight cut elements carry outside
    masks = decode.element_masks_for(runner, strict=True)
    if masks is None:
        W = [decode.element_gram_sqrt(m, 0.0) for m in mats]

        def energy(comps):
            return sum(((c[..., lltogl] @ Wc.T) ** 2).sum(axis=(-1, -2))
                       for c, Wc in zip(comps, W))
        return energy
    # the mesh overhangs the grid: cut elements count only their points inside it
    W = [decode.element_gram_roots(m, masks, 0.0) for m in mats]

    def energy(comps):
        return sum((np.einsum('eij,...ej->...ei', We, c[..., lltogl]) ** 2).sum(axis=(-1, -2))
                   for c, We in zip(comps, W))
    return energy


def summarize(model_err: np.ndarray, persist_err: np.ndarray) -> dict:
    """Scalar statistics of (starts, horizon) relative field errors."""
    S, H = model_err.shape
    per_start = model_err.mean(axis=1)
    out = {
        'ms_starts': S,
        'ms_horizon': H,
        'ms_err_mean': float(model_err.mean()),
        # over every (forecast, step) in the window
        'ms_err_std': float(model_err.std()),
        # forecast-to-forecast consistency: spread of each forecast's window mean
        'ms_err_start_std': float(per_start.std()),
        'ms_err_worst_start': float(per_start.max()),
        'ms_persist_mean': float(persist_err.mean()),
        'ms_skill': float(1.0 - model_err.mean() / max(persist_err.mean(), 1e-30)),
    }
    for h in sorted({*_REPORT_STEPS, H}):
        if h <= H:
            out[f'ms_err_at{h}_mean'] = float(model_err[:, h - 1].mean())
            out[f'ms_err_at{h}_std'] = float(model_err[:, h - 1].std())
    # first step at which the mean forecast error drops below persistence's
    better = np.flatnonzero(model_err.mean(axis=0) < persist_err.mean(axis=0))
    out['ms_beats_persist_step'] = int(better[0]) + 1 if len(better) else -1
    return out


def evaluate(runner, name: str, ids, horizon: int = None, num_starts: int = None):
    """Run the multi-start forecasts for one dataset split.

    Returns (scalars, arrays). Arrays are 'ms_err' and 'ms_persist', each
    (starts, horizon), and 'ms_start_ids', the snapshot id of each forecast's
    first seed frame. Both are empty if the split cannot hold one forecast.
    """
    mp = runner.config['model_params']
    tl = mp['time_lag']
    horizon = horizon or mp.get('horizon', 30)
    num_starts = num_starts or mp.get('eval_starts', DEFAULT_STARTS)
    window = tl + horizon

    ids = np.asarray(ids)
    starts = start_positions(ids, window, num_starts)
    if len(starts) == 0:
        logger.warning("Multi-start: no contiguous %d-snapshot window in '%s'; skipped",
                       window, name)
        return {}, {}
    if runner.config['latent_params'].get('global_pod'):
        logger.warning("Multi-start scores the local (residual) field only; the global "
                       "POD part added back in reconstruction is not included.")

    energy = field_energy_fn(runner)
    keys = ['dof_u', 'dof_v'] + (['dof_w'] if runner.dim == 3 else [])
    if diffusion.is_diffusion(runner):
        return _evaluate_ensemble(runner, name, ids, starts, horizon, energy, keys)
    model_err = np.zeros((len(starts), horizon))
    persist_err = np.zeros((len(starts), horizon))

    with h5py.File(runner.paths_bib.latent_path, 'r') as f:
        group = f[name]
        for k, i in enumerate(starts):
            win = ids[i:i + window]
            true = [group[key][win[0]:win[-1] + 1, :] for key in keys]
            seed = true + [None] * (3 - len(true))
            pred = runner._predict_dofs_forward(
                name=name, ids=win, dof_u=seed[0][:tl], dof_v=seed[1][:tl],
                dof_w=None if seed[2] is None else seed[2][:tl], total_steps=window)
            pred = [p[tl:] for p in pred if p is not None]
            future = [t[tl:] for t in true]
            last = [t[tl - 1:tl] for t in true]

            denom = np.maximum(energy(future), 1e-30)
            model_err[k] = np.sqrt(energy([p - t for p, t in zip(pred, future)]) / denom)
            persist_err[k] = np.sqrt(energy([l - t for l, t in zip(last, future)]) / denom)

    scalars = summarize(model_err, persist_err)
    logger.info("Multi-start %s: %d forecasts x %d steps | error %.4f +/- %.4f "
                "(forecast-to-forecast %.4f) | persistence %.4f | skill %.3f",
                name, len(starts), horizon, scalars['ms_err_mean'], scalars['ms_err_std'],
                scalars['ms_err_start_std'], scalars['ms_persist_mean'], scalars['ms_skill'])
    arrays = {'ms_err': model_err, 'ms_persist': persist_err,
              'ms_start_ids': ids[starts]}
    return scalars, arrays


def _evaluate_ensemble(runner, name, ids, starts, horizon, energy, keys):
    """evaluate() for a diffusion model: diffusion.members sampled forecasts per start."""
    tl = runner.config['model_params']['time_lag']
    window = tl + horizon
    M = int(diffusion.settings(runner)['members'])
    with h5py.File(runner.paths_bib.latent_path, 'r') as f:
        group = f[name]
        trues = [[group[key][ids[i]:ids[i + window - 1] + 1, :] for key in keys] for i in starts]
    seeds = [tuple([t[:tl] for t in tr] + [None] * (3 - len(tr))) for tr in trues]
    pred = diffusion.forecast(runner, name, seeds, horizon, members=M)   # (S, M, window, dim, N)

    S = len(starts)
    model_err = np.zeros((S, horizon))
    persist_err = np.zeros((S, horizon))
    member_err = np.zeros((S, M, horizon))
    spread = np.zeros((S, horizon))
    for k, tr in enumerate(trues):
        future = [t[tl:] for t in tr]
        last = [t[tl - 1:tl] for t in tr]
        members = [pred[k, :, tl:, c].astype(np.float64) for c in range(len(keys))]   # (M, H, N)
        mean = [m.mean(axis=0) for m in members]
        denom = np.maximum(energy(future), 1e-30)
        model_err[k] = np.sqrt(energy([m - t for m, t in zip(mean, future)]) / denom)
        persist_err[k] = np.sqrt(energy([l - t for l, t in zip(last, future)]) / denom)
        member_err[k] = np.sqrt(energy([m - t[None] for m, t in zip(members, future)]) / denom)
        if M > 1:
            var = energy([m - mu[None] for m, mu in zip(members, mean)]).sum(axis=0) / (M - 1)
            spread[k] = np.sqrt(var / denom)

    scalars = summarize(model_err, persist_err)
    persist = max(persist_err.mean(), 1e-30)
    member_mean = member_err.mean(axis=1)
    scalars.update({
        'ms_members': M,
        'ms_member_err_mean': float(member_err.mean()),
        'ms_member_err_start_std': float(member_mean.mean(axis=1).std()),
        'ms_member_skill': float(1.0 - member_err.mean() / persist),
    })
    if M > 1:
        # per step over the starts, then averaged over the window
        ssr = np.sqrt((M + 1) / M * (spread ** 2).mean(axis=0)) / np.sqrt(
            np.maximum((model_err ** 2).mean(axis=0), 1e-30))
        scalars['ms_spread_mean'] = float(spread.mean())
        scalars['ms_ssr'] = float(ssr.mean())
        for h in sorted({*_REPORT_STEPS, horizon}):
            if h <= horizon:
                scalars[f'ms_member_err_at{h}_mean'] = float(member_err[:, :, h - 1].mean())
                scalars[f'ms_spread_at{h}_mean'] = float(spread[:, h - 1].mean())
                scalars[f'ms_ssr_at{h}'] = float(ssr[h - 1])
    logger.info("Multi-start %s: %d starts x %d members x %d steps | ensemble-mean error %.4f "
                "(forecast-to-forecast %.4f) | member error %.4f | spread %.4f | spread-skill %.2f "
                "| persistence %.4f | skill %.3f (members %.3f)",
                name, S, M, horizon, scalars['ms_err_mean'], scalars['ms_err_start_std'],
                scalars['ms_member_err_mean'], scalars.get('ms_spread_mean', 0.0),
                scalars.get('ms_ssr', float('nan')), scalars['ms_persist_mean'],
                scalars['ms_skill'], scalars['ms_member_skill'])
    arrays = {'ms_err': model_err, 'ms_persist': persist_err, 'ms_start_ids': ids[starts],
              'ms_err_member': member_err, 'ms_spread': spread}
    return scalars, arrays
