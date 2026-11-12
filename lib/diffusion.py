"""Conditional diffusion over the next-frame increment (model_type 'tr_spatial_diff').

The deterministic spatial transformer minimizes the expected field error, so it
predicts the conditional mean of the next frame. Over a rollout the mean is
smoother than any one realization, and the cavity and Re 30k models keep only
40 % and 65 % of the fluctuating energy. A conditional diffusion model samples
the next frame instead of averaging over the possible ones.

What is diffused is the increment of the global nodal DLS coefficients,
dd = d(t+1) - d(t), one vector per velocity component. Noise is drawn in the
global coefficient space, so every noisy state, every intermediate estimate and
every sample is one consistent field; element tokens are only ever gathered
from a global vector.

Diffusion coordinates y (diffusion.space):
    'dof'   y = dd / D, with D the per-DOF standard deviation of the increment
            on the training split (default)
    'gram'  y = A^{1/2} dd / s, with A the assembled matrix of the mass-weighted
            projection and s a global scale, so white noise in y is white in
            the (regularized) field energy

The denoiser (models.DiffusionSpatialTransformer) reads the conditioning frames
and the noisy increment as element tokens, plus the diffusion time, and emits
a per-element v. The tokens go through the mass-weighted projection, the same
operator the deterministic model applies between rollout steps, before they
are read as a global v. So the x0 estimate at each denoising step is a
consistent field.

Schedule: variance preserving and cosine, alpha = cos(pi t / 2) and
sigma = sin(pi t / 2) for t in [0, 1], with the v-parameterization (Salimans &
Ho 2022): y0_hat = alpha y_t - sigma v_hat. The loss is the x0 error in the
field-energy metric of the projection, (dd_hat - dd)^T A (dd_hat - dd),
normalized by the mean increment energy on the training split and weighted by
w(t):
    'snr1_trunc'  w = min(SNR + 1, gamma): the plain v loss, truncated (default)
    'min_snr'     w = min(SNR, gamma): min-SNR-gamma (Hang et al. 2023)
    'v'           w = SNR + 1: the plain v loss
Any positive weighting has the same minimizer (the posterior mean of y0); the
weighting only decides where the capacity goes. Sampling is DDIM (Song et al.
2021) on a uniform t grid. With eta 0 a sample is deterministic given its
initial noise; eta 1 is ancestral sampling.

model_params.diffusion keys (defaults in DEFAULTS):
    space, weighting, gamma, t_min   training objective (see above)
    sample_steps, eta                sampler
    members                          ensemble size in the multi-start evaluation
    ema_decay                        weight EMA; 0 disables it
    cond_noise_std                   training-time perturbation of the conditioning
                                     window (below), as a fraction of each DOF's
                                     state standard deviation; 0 disables it
    pushforward                      k_max of the self-unrolled training (below); 0
                                     is plain one-step training
    pushforward_steps                DDIM steps of the unrolled samples
    val_unroll                       validation scores the unrolled steps 0..val_unroll
                                     (default: pushforward), so model selection sees
                                     multi-step quality
    mean_model                       results/<latent>/<run> of a trained tr_spatial
                                     model: diffuse the residual about its forecast
                                     instead of the increment (below); None (default)
    mean_model_folds                 runs of the same model trained on disjoint blocks
                                     of the training period (train_data[].train_fold):
                                     training residuals come from the fold model that
                                     did not see the window (below)
    ms_steps                         K of the multi-step energy-score training (below);
                                     0 (default) is one-step denoising training only
    ms_members                       sampled trajectories per window in that loss (>= 2)
    ms_sampler_steps                 DDIM steps per forecast step inside it
    ms_dsm_weight                    weight of the one-step denoising loss kept beside it
    ms_checkpoint                    gradient-checkpoint each denoiser call of the unroll
    target                           'increment' (default): forecast the next frame;
                                     'state': an unconditional model of single snapshots,
                                     the thermalizer below
    thermalizer                      {run, t, steps, every, after, trigger, detect_t}: in
                                     sampled rollouts, every `every` steps past step
                                     `after`, noise the state to diffusion time t and
                                     denoise it in `steps` DDIM steps with the 'state'
                                     model of results run `run`. With trigger q (e.g.
                                     0.99) only states that the detector below scores
                                     above the q-quantile of real snapshots are treated;
                                     None (default) treats every state. thermalizer None
                                     disables it
    stat_pairs                       training pairs used for the increment statistics
    seed                             sampler seed in pred/eval

Residual mode (mean_model). A frozen deterministic model f, trained on multi-step
rollouts, gives the forecast mu = f(window), assembled by the mass-weighted
projection. The diffusion model then samples r = d(t+1) - mu, conditioned on the
window and on mu as an extra frame, and the next state is mu + r: a regression
mean plus a generative residual, as in CorrDiff (Mardani et al. 2023). The aim
is to keep the rollout-trained model's short-horizon skill while restoring the
variance it averages away. "Increment" below then means this residual.

Multi-step training (ms_steps). One-step training never scores a rollout, and
the first models carry an unstable mode in their learned dynamics (Re 15k and
30k pure diffusion blow up, every sample at the same step). The deterministic
models avoid it only through their 16-step rollout loss. The probabilistic
counterpart used here: from each training window, ms_members trajectories of K
steps are sampled with the ms_sampler_steps DDIM sampler, with gradients through
every denoiser call (the sampler is deterministic given its noise, so the noise
is reparameterized), and scored at every step with the fair energy score in the
field-energy norm of the mass-weighted projection,

    ES = mean_m ||X_m - y|| - 1 / (2 M (M - 1)) sum_{m != m'} ||X_m - X_m'||,
    ||e|| = sqrt(e^T A e / increment energy),

a proper score: it is minimized by sampling the true conditional distribution,
so unlike a rollout MSE it does not reward collapsing the members onto their
mean (Gneiting & Raftery 2007; multi-step CRPS training of ensemble weather
models, Lang et al. 2024). The one-step denoising loss is kept beside it, with
weight ms_dsm_weight, so the model stays a denoiser at every noise level.
Validation is the same energy score on a fixed draw. Set sample_steps to
ms_sampler_steps so inference runs the sampler that was trained (on Re 15k, 4
DDIM steps already score as well as 20 before this training).

Thermalization (thermalizer). Multi-step training improves the forecasts and
delays the Re 15k blow-up, but from every start point tried the sampled rollout
still leaves the attractor after 330-420 steps: a slow drift of the learned
dynamics that no affordable training horizon sees. Following Pedersen et al.
(2025), a second, unconditional diffusion model of single snapshots (target
'state', trained on the same split) is applied during long rollouts: partially
noising the state and denoising it pulls it back toward the distribution of real
snapshots, removing off-attractor growth, while the forecaster keeps doing the
dynamics. The state model diffuses (d - m) / S per DOF, m and S the training
mean and standard deviation of each DOF.

Applied to every state, it costs the forecast its accuracy: the unconditional
model cannot restore a state's detail, so each application replaces it with a
different plausible snapshot (Re 15k: t = 0.1 every step keeps 1,000-step
rollouts at 0.7-1.1x the true energy, but the 30-step error goes from 1.6 % to
31 %). So, as in the Thermalizer paper, it can be triggered: the detector
noises a state to detect_t (default 0.2) with a fixed draw, denoises it in one
step and measures how far it moved, sqrt(A-energy of the change / A-energy of
the state's fluctuation). Real snapshots move little; a state that has left the
attractor moves a lot. The threshold is the trigger-quantile of the score on
real training snapshots.

Out-of-fold residuals (mean_model_folds). On its own training windows the mean
model is more accurate than on new data (Re 15k: one-step error 4.4 % of the
increment in-sample, 5.8-6.1 % on the test and validation blocks), so residuals
taken in-sample are too small and too regular, and the residual model overfits
them. With fold models, each training window's residual is taken about the
forecast of the fold model whose training block overlaps it least, as in
stacking; validation and inference use the full mean model.

One-step training on true windows never shows the model a state it produced
itself, and its own errors then grow unchecked in a long rollout (the first Re
15k model diverges from one corner after ~150 steps). Two remedies, both aimed
at teaching the model to steer a perturbed state back to the truth:

    cond_noise_std  the whole conditioning window is offset by one consistent
                    global perturbation (the same at every element copy of a DOF
                    and in every frame, like an accumulated rollout error).
                    cond_noise_target 'increment' (default) keeps the true
                    increment as the target, so the model learns to ignore the
                    offset; 'truth' keeps the true next frame, so it must remove
                    it. 'truth' fails for a sampler: an offset it cannot detect
                    is learned as extra randomness and injected at every step
                    (Re 15k, 0.02: the unrolled validation loss grew 35x in 20
                    epochs)
    pushforward     the window is advanced k ~ U{0..k_max} steps by the model's
                    own samples (no gradient through the unroll), and the target
                    is the true frame k + 1 steps ahead: the pushforward trick of
                    Brandstetter et al. (2022), applied to a sampler
"""
import logging
import math

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULTS = {
    'space': 'dof',
    'weighting': 'snr1_trunc',
    'gamma': 5.0,
    't_min': 1e-3,
    'sample_steps': 20,
    'eta': 0.0,
    'members': 8,
    'ema_decay': 0.999,
    'cond_noise_std': 0.0,
    'cond_noise_target': 'increment',
    'pushforward': 0,
    'pushforward_steps': 8,
    'val_unroll': None,
    'mean_model': None,
    'mean_model_folds': None,
    'ms_steps': 0,
    'ms_members': 2,
    'ms_sampler_steps': 4,
    'ms_dsm_weight': 1.0,
    'ms_checkpoint': True,
    'target': 'increment',
    'thermalizer': None,
    'stat_pairs': 4000,
    'seed': 0,
}
_SPACES = ('dof', 'gram')
_WEIGHTINGS = ('snr1_trunc', 'min_snr', 'v')
# the fixed draw of diffusion times and noise that scores every validation epoch
_VAL_SEED = 20261019


def is_diffusion(runner) -> bool:
    return runner.config['model_params']['model_type'] == 'tr_spatial_diff'


def settings(runner) -> dict:
    """model_params.diffusion merged over DEFAULTS, validated."""
    cfg = dict(DEFAULTS)
    cfg.update(runner.config['model_params'].get('diffusion') or {})
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"model_params.diffusion: unknown keys {sorted(unknown)}")
    if cfg['space'] not in _SPACES:
        raise ValueError(f"diffusion.space '{cfg['space']}' not recognized. Use {_SPACES}.")
    if cfg['weighting'] not in _WEIGHTINGS:
        raise ValueError(f"diffusion.weighting '{cfg['weighting']}' not recognized. Use {_WEIGHTINGS}.")
    if cfg['cond_noise_target'] not in ('increment', 'truth'):
        raise ValueError("diffusion.cond_noise_target must be 'increment' or 'truth'")
    if cfg['target'] not in ('increment', 'state'):
        raise ValueError("diffusion.target must be 'increment' or 'state'")
    if cfg['target'] == 'state' and (cfg['mean_model'] or cfg['ms_steps'] or cfg['pushforward']):
        raise ValueError("a 'state' model takes no mean_model, ms_steps or pushforward")
    if cfg['val_unroll'] is None:
        cfg['val_unroll'] = cfg['pushforward']
    if cfg['ms_steps'] and cfg['ms_members'] < 2:
        raise ValueError("diffusion.ms_members must be >= 2: the energy score needs member pairs")
    need = max(cfg['pushforward'] + 1, cfg['val_unroll'] + 1, cfg['ms_steps'])
    if runner.config['model_params'].get('train_ahead', 1) < need:
        raise ValueError(f"diffusion.pushforward / val_unroll / ms_steps need train_ahead >= {need}")
    return cfg


def alpha_sigma(t: torch.Tensor):
    """Cosine VP schedule: signal and noise scales at diffusion time t in [0, 1]."""
    return torch.cos(0.5 * math.pi * t), torch.sin(0.5 * math.pi * t)


def loss_weight(alpha, sigma, kind: str, gamma: float):
    snr = alpha ** 2 / sigma.clamp_min(1e-12) ** 2
    if kind == 'snr1_trunc':
        return torch.clamp(snr + 1.0, max=gamma)
    if kind == 'min_snr':
        return torch.clamp(snr, max=gamma)
    return snr + 1.0


def run_config(run_dir: str) -> dict:
    """The config saved in a run directory. The directory turns dots into underscores
    (..._fca0_1); the config keeps them (..._fca0.1.json), so take the one json there."""
    import json
    import os
    cfg, = [f for f in os.listdir(run_dir) if f.endswith('.json')]
    return json.load(open(os.path.join(run_dir, cfg)))


def plan(runner, name: str) -> 'Plan':
    """The Plan for data source `name`, built on first use and cached on the runner."""
    cache = runner.__dict__.setdefault('_diffusion_plans', {})
    if name not in cache:
        cache[name] = Plan(runner, name)
    return cache[name]


class Plan:
    """Operators and scales for diffusing one source's increments.

    Global vectors are (G, dim, N): G samples, one row of N nodal DOFs per
    velocity component. Element tokens are (G, E, dim * dof_elem), the
    component blocks concatenated, as the data loaders build them.
    """

    def __init__(self, runner, name: str, overrides=None):
        self.cfg = settings(runner)
        self.cfg.update(overrides or {})
        self.state_target = self.cfg['target'] == 'state'
        lp = runner.config['latent_params']
        if any(lp.get(k, 1) != 1 for k in ('skipx', 'skipy', 'skipz')):
            raise ValueError("tr_spatial_diff needs every element (skipx = skipy = skipz = 1): "
                             "the diffused state is the global coefficient vector")
        if runner.config['model_params'].get('reassemble_lambda') is None:
            raise ValueError("tr_spatial_diff needs model_params.reassemble_lambda: the "
                             "denoiser output goes through the mass-weighted projection")
        self.device = torch.device(runner.device)
        self.dim = runner.dim
        self.ndof = runner.l_config.dof_elem
        geom = runner._elem_geometry(use_skip=False)
        self.geom = geom
        self.N = geom['num_dofs']
        self.lltogl = geom['lltogl']
        self.scatter = geom['scatter_index']
        self.counts = geom['counts']
        ops = runner._assembly_ops(geom)
        self.Mr = [o['Mr_t'] for o in ops]
        self.Ainv = [o['Ainv_t'] for o in ops]
        A64 = [o['A'] for o in ops]
        self.A = [torch.from_numpy(a).float().to(self.device) for a in A64]
        self.mean = runner.dof_mean[name].float().to(self.device)
        self.std = runner.dof_std[name].float().to(self.device)
        self.tl = runner.config['model_params']['time_lag']
        self.mean_net = None
        if self.cfg['mean_model']:
            self._load_mean_model(runner, name)

        dd, S, m = self._train_stats(runner, name)                 # (P, dim, N), (dim, N), (dim, N)
        self.S = torch.from_numpy(S).float().to(self.device)       # state std per DOF
        self.m = torch.from_numpy(m).float().to(self.device)       # state mean per DOF
        D = dd.std(axis=0)
        D = np.maximum(D, 1e-3 * D.mean(axis=1, keepdims=True))    # no DOF is left unscaled
        self.D = torch.from_numpy(D).float().to(self.device)       # (dim, N)
        # mean A-energy of the increment: the loss is relative to it
        energy = sum(np.einsum('pn,nm,pm->p', dd[:, c], A64[c], dd[:, c]) for c in range(self.dim))
        self.energy0 = float(energy.mean())
        if self.cfg['space'] == 'gram':
            halves, inv_halves, diag_inv = [], [], []
            for a in A64:
                w, Q = np.linalg.eigh(0.5 * (a + a.T))
                w = np.clip(w, w.max() * 1e-12, None)
                halves.append((Q * np.sqrt(w)) @ Q.T)
                inv_halves.append((Q / np.sqrt(w)) @ Q.T)
                diag_inv.append(((Q ** 2) / w).sum(axis=1))
            self.s = math.sqrt(self.energy0 / (self.dim * self.N))
            self.Ah = torch.from_numpy(np.stack(halves)).float().to(self.device)
            self.Aih = torch.from_numpy(np.stack(inv_halves)).float().to(self.device)
            # noise variance per DOF in the network's input units (dd / D)
            noise_var = self.s ** 2 * np.stack(diag_inv) / D ** 2
        else:
            noise_var = np.ones_like(D)
        self.noise_var = torch.from_numpy(noise_var).float().to(self.device)   # (dim, N)
        self.D_tok = self.gather(self.D.unsqueeze(0))[0]                       # (E, F)
        self.noise_var_tok = self.gather(self.noise_var.unsqueeze(0))[0]       # (E, F)
        y = self.to_y(torch.from_numpy(dd).float().to(self.device))
        logger.info(
            "Diffusion plan '%s': space %s, %d increment pairs, increment energy %.3e, "
            "y variance %.3f (per DOF: median %.3f, max %.3f), input noise variance "
            "median %.3g / max %.3g, weighting %s (gamma %g), %d DDIM steps (eta %g)",
            name, self.cfg['space'], dd.shape[0], self.energy0, float(y.var(dim=0).mean()),
            float(y.var(dim=0).median()), float(y.var(dim=0).max()),
            float(np.median(noise_var)), float(noise_var.max()), self.cfg['weighting'], self.cfg['gamma'],
            self.cfg['sample_steps'], self.cfg['eta'])
        self.therm = None
        if self.cfg['thermalizer'] and not self.state_target:
            self._load_thermalizer(runner, name, dict(self.cfg['thermalizer']))

    # ------------------------------------------------------------------ data
    def _train_stats(self, runner, name):
        """Training increments (or residuals, with a mean model) and the per-DOF state std.

        Increments are d(i+1) - d(i) over consecutive training pairs; residuals are
        d(i+1) - mu over training windows of time_lag + 1 consecutive snapshots.
        """
        ids = np.sort(np.asarray(runner.indices['train_data'][name]['train_indices']))
        w = self.tl if self.mean_net is not None else 1
        pos = np.flatnonzero(ids[w:] - ids[:-w] == w)
        n = self.cfg['stat_pairs']
        if len(pos) > n:
            pos = pos[np.unique(np.linspace(0, len(pos) - 1, n).round().astype(int))]
        keys = ['dof_u', 'dof_v'] + (['dof_w'] if self.dim == 3 else [])
        with h5py.File(runner.paths_bib.latent_path, 'r') as f:
            comps = [np.asarray(f[name][k][:], dtype=np.float64) for k in keys]
        S = np.stack([c[ids].std(axis=0) for c in comps])
        S = np.maximum(S, 1e-6 * S.mean(axis=1, keepdims=True))
        m = np.stack([c[ids].mean(axis=0) for c in comps])
        if self.state_target:
            return np.stack([c[ids[pos]] for c in comps], axis=1) - m, S, m
        if self.mean_net is None:
            return np.stack([c[ids[pos] + 1] - c[ids[pos]] for c in comps], axis=1), S, m
        out = []
        for b in range(0, len(pos), 256):
            p0 = ids[pos[b:b + 256]]
            win = np.stack([np.stack([c[p0 + j] for c in comps], axis=1) for j in range(self.tl)], axis=1)
            X = self.normalize(torch.from_numpy(win).float().to(self.device)).permute(0, 2, 1, 3)
            mu = self.mean_forecast(X, p0).double().cpu().numpy()
            nxt = np.stack([c[p0 + self.tl] for c in comps], axis=1)
            out.append(nxt - mu)
        return np.concatenate(out), S, m

    def _load_mean_model(self, runner, name):
        """The frozen deterministic forecaster(s) of residual mode."""
        self.mean_net, self.m_mean, self.m_std, _ = self._load_spatial(runner, name, self.cfg['mean_model'])
        self.folds = []
        for run_dir in self.cfg['mean_model_folds'] or []:
            net, m_mean, m_std, span = self._load_spatial(runner, name, run_dir)
            self.folds.append({'net': net, 'mean': m_mean, 'std': m_std, 'span': span})
            logger.info("  fold model %s: trained on snapshots [%d, %d)", run_dir, *span)

    def _load_spatial(self, runner, name, run_dir):
        """A frozen tr_spatial run: (net, its dof mean, its dof std, [lo, hi) of its training windows)."""
        import os
        import pickle
        import lib.models as models
        run_dir = run_dir.rstrip('/')
        mcfg = run_config(run_dir)['model_params']
        if mcfg['model_type'] != 'tr_spatial':
            raise ValueError("diffusion.mean_model must be a tr_spatial run")
        if mcfg['time_lag'] != self.tl:
            raise ValueError(f"mean_model time_lag {mcfg['time_lag']} != {self.tl}")
        mine = runner.config['model_params'].get('reassemble_lambda')
        if mcfg.get('reassemble_lambda') != mine:
            raise ValueError(f"mean_model reassemble_lambda {mcfg.get('reassemble_lambda')} != {mine}")
        net = models.SpatialTransformer(
            time_lag=mcfg['time_lag'], input_dim=runner.config['model_params']['input_dim'],
            d_model=mcfg['d_model'], ff_dim=mcfg.get('ff_dim', 4 * mcfg['d_model']),
            nhead=mcfg['nhead'], num_layers=mcfg['num_layers'],
            activation=mcfg.get('activation', 'relu'), pre_norm=mcfg.get('prenorm', False),
            spatial_dim=self.dim, num_freqs=mcfg.get('coord_freqs', 6),
            residual=mcfg.get('residual', False))
        state = torch.load(os.path.join(run_dir, 'model.pth'), weights_only=True, map_location=self.device)
        net.load_state_dict({k.replace('module.', '', 1): v for k, v in state.items()})
        net = net.to(self.device).eval()
        for prm in net.parameters():
            prm.requires_grad_(False)
        with open(os.path.join(run_dir, 'dof_scaler.pkl'), 'rb') as f:
            m_mean, m_std = pickle.load(f)
        with open(os.path.join(run_dir, 'split_ids.pkl'), 'rb') as f:
            tr = np.asarray(pickle.load(f)['train_data'][name]['train_indices'])
        span = (int(tr.min()), int(tr.max()) + mcfg['time_lag'] + mcfg['train_ahead'])
        logger.info("Residual mode: mean forecast from %s (%d parameters, frozen)", run_dir,
                    sum(prm.numel() for prm in net.parameters()))
        return net, m_mean[name].float().to(self.device), m_std[name].float().to(self.device), span

    @torch.no_grad()
    def mean_forecast(self, X, t0=None):
        """Assembled forecast of the frozen mean model: window (G, E, tl, F) -> (G, dim, N).

        With fold models and the windows' first snapshot ids t0 (training), each
        window is forecast by the fold model whose training block overlaps it least.
        """
        raw = X.float() * self.std + self.mean
        coords = self.geom['coords']

        def run(net, m_mean, m_std, x):
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                out = net((x - m_mean) / m_std, coords)
            with torch.autocast(device_type=self.device.type, enabled=False):
                return self.project(out.float() * m_std + m_mean)

        if t0 is None or not self.folds:
            return run(self.mean_net, self.m_mean, self.m_std, raw)
        t0 = np.asarray(t0 if not torch.is_tensor(t0) else t0.cpu().numpy())
        w = self.tl + 1
        overlap = np.stack([np.clip(np.minimum(f['span'][1], t0 + w) - np.maximum(f['span'][0], t0), 0, None)
                            for f in self.folds])                      # (n_folds, G)
        pick = overlap.argmin(axis=0)
        out = torch.empty(X.shape[0], self.dim, self.N, device=X.device)
        for k, f in enumerate(self.folds):
            sel = np.flatnonzero(pick == k)
            if len(sel):
                idx = torch.from_numpy(sel).to(X.device)
                out[idx] = run(f['net'], f['mean'], f['std'], raw[idx])
        return out

    def condition(self, X, t0=None):
        """Denoiser conditioning and the state the sample is added to.

        Increment mode: (window, its last frame). Residual mode: (window plus the
        mean forecast as one more frame, the mean forecast).
        """
        if self.state_target:
            return X[:, :, :0], self.m.expand(X.shape[0], -1, -1)
        if self.mean_net is None:
            return X, self.last_state(X)
        mu = self.mean_forecast(X, t0)
        return torch.cat((X, self.normalize(mu).unsqueeze(2)), dim=2), mu

    # ------------------------------------------------------------- operators
    def gather(self, g):
        """(G, dim, N) global -> (G, E, F) element tokens."""
        return torch.cat([g[:, c][:, self.lltogl] for c in range(self.dim)], dim=2)

    def scatter_mean(self, tok):
        """(G, E, F) consistent element tokens -> (G, dim, N), averaging the shared copies."""
        G = tok.shape[0]
        out = []
        for c in range(self.dim):
            local = tok[:, :, c * self.ndof:(c + 1) * self.ndof].reshape(G, -1)
            flat = torch.zeros(G, self.N, device=tok.device, dtype=tok.dtype)
            out.append(flat.index_add(1, self.scatter, local) / self.counts)
        return torch.stack(out, dim=1)

    def project(self, tok):
        """(G, E, F) element tokens -> (G, dim, N) by the mass-weighted projection."""
        G = tok.shape[0]
        out = []
        for c in range(self.dim):
            local = tok[:, :, c * self.ndof:(c + 1) * self.ndof]
            Mr = self.Mr[c]
            weighted = torch.einsum('gei,eij->gej', local, Mr) if Mr.dim() == 3 else local @ Mr
            flat = torch.zeros(G, self.N, device=tok.device, dtype=tok.dtype)
            flat = flat.index_add(1, self.scatter, weighted.reshape(G, -1))
            out.append(flat @ self.Ainv[c])
        return torch.stack(out, dim=1)

    def to_y(self, dd):
        if self.cfg['space'] == 'gram':
            return torch.einsum('gcn,cnm->gcm', dd, self.Ah) / self.s
        return dd / self.D

    def from_y(self, y):
        if self.cfg['space'] == 'gram':
            return torch.einsum('gcn,cnm->gcm', y, self.Aih) * self.s
        return y * self.D

    def energy(self, dd):
        """A-metric energy of global increments, (G, dim, N) -> (G,)."""
        return sum(((dd[:, c] @ self.A[c]) * dd[:, c]).sum(-1) for c in range(self.dim))

    def normalize(self, g):
        """(G, ..., dim, N) raw global states -> standardized element tokens (G, ..., E, F)."""
        lead = g.shape[:-2]
        tok = self.gather(g.reshape(-1, self.dim, self.N))
        return ((tok - self.mean) / self.std).reshape(*lead, *tok.shape[1:])

    # ------------------------------------------------------------- denoiser
    def denoise(self, model, cond, coords, y_t, t):
        """x0 estimate in y coordinates from the noisy y_t at times t (G,)."""
        a, s = alpha_sigma(t)
        a3, s3 = a.view(-1, 1, 1), s.view(-1, 1, 1)
        # the noisy increment in per-DOF units, rescaled to unit variance at every t
        z = self.gather(self.from_y(y_t) / self.D)
        z = z / torch.sqrt(a3 ** 2 + s3 ** 2 * self.noise_var_tok)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            out = model(cond, z, t, coords)
        with torch.autocast(device_type=self.device.type, enabled=False):
            v = self.to_y(self.project(out.float() * self.D_tok))
            return a3 * y_t - s3 * v

    def perturb(self, X, scale):
        """Offset a window (G, E, tl, F) by one consistent global perturbation per sample."""
        g = scale * self.S * torch.randn(X.shape[0], self.dim, self.N, device=X.device)
        return X + (self.gather(g) / self.std).unsqueeze(2)

    def last_state(self, X):
        """Raw global state of a window's last frame: (G, E, tl, F) -> (G, dim, N)."""
        return self.scatter_mean(X[:, :, -1].float() * self.std + self.mean)

    def _load_thermalizer(self, runner, name, th):
        """The frozen 'state' model of run th['run'] and its plan, for rollout thermalization."""
        import os
        import lib.models as models
        run_dir = th['run'].rstrip('/')
        mcfg = run_config(run_dir)['model_params']
        if (mcfg.get('diffusion') or {}).get('target') != 'state':
            raise ValueError(f"thermalizer run {run_dir} is not a diffusion.target 'state' model")
        net = models.DiffusionSpatialTransformer(
            time_lag=0, input_dim=runner.config['model_params']['input_dim'], d_model=mcfg['d_model'],
            ff_dim=mcfg.get('ff_dim', 3 * mcfg['d_model']), nhead=mcfg['nhead'], num_layers=mcfg['num_layers'],
            activation=mcfg.get('activation', 'swiglu'), spatial_dim=self.dim,
            num_freqs=mcfg.get('coord_freqs', 6), dropout=mcfg.get('dropout', 0.1),
            ema=(mcfg.get('diffusion') or {}).get('ema_decay', DEFAULTS['ema_decay']) > 0)
        state = torch.load(os.path.join(run_dir, 'model.pth'), weights_only=True, map_location=self.device)
        net.load_state_dict({k.replace('module.', '', 1): v for k, v in state.items()})
        net = net.to(self.device).eval()
        for prm in net.parameters():
            prm.requires_grad_(False)
        sp = Plan(runner, name, overrides={'target': 'state', 'thermalizer': None, 'mean_model': None,
                                           'mean_model_folds': None, 'ms_steps': 0, 'pushforward': 0,
                                           'space': 'dof'})
        self.therm = {'plan': sp, 'model': net, 't': float(th.get('t', 0.2)),
                      'steps': int(th.get('steps', 2)), 'every': int(th.get('every', 1)),
                      'after': int(th.get('after', 0)), 'trigger': th.get('trigger'),
                      'detect_t': float(th.get('detect_t', 0.2)), 'threshold': None, 'fired': 0, 'checked': 0}
        if self.therm['trigger'] is not None:
            self.therm['threshold'] = self._calibrate_detector(runner, name, float(self.therm['trigger']))
        logger.info("Thermalizer: %s, t=%g in %d DDIM steps, every %d steps after step %d%s", run_dir,
                    self.therm['t'], self.therm['steps'], self.therm['every'], self.therm['after'],
                    '' if self.therm['threshold'] is None else
                    f", triggered above detector score {self.therm['threshold']:.4f} "
                    f"(q{self.therm['trigger']:g} of real snapshots)")

    @torch.no_grad()
    def detector_score(self, d):
        """How far one noise-and-denoise of the state model moves each raw global state (G, dim, N)."""
        th = self.therm
        sp = th['plan']
        G = d.shape[0]
        gen = torch.Generator(device=d.device).manual_seed(_VAL_SEED)
        t = torch.tensor(th['detect_t'], device=d.device)
        a, s = alpha_sigma(t)
        y = sp.to_y(d - sp.m)
        y_t = a * y + s * torch.randn(y.shape, device=d.device, generator=gen)
        cond = torch.zeros(G, sp.lltogl.shape[0], 0, self.dim * self.ndof, device=d.device)
        y0 = sp.denoise(th['model'], cond, sp.geom['coords'], y_t, t.expand(G))
        moved = sp.energy(sp.from_y(y0 - y))
        return torch.sqrt(moved / sp.energy(d - sp.m).clamp_min(1e-12))

    def _calibrate_detector(self, runner, name, q):
        """The q-quantile of the detector score over real training snapshots."""
        ids = np.sort(np.asarray(runner.indices['train_data'][name]['train_indices']))
        ids = ids[np.unique(np.linspace(0, len(ids) - 1, min(512, len(ids))).round().astype(int))]
        keys = ['dof_u', 'dof_v'] + (['dof_w'] if self.dim == 3 else [])
        with h5py.File(runner.paths_bib.latent_path, 'r') as f:
            d = torch.from_numpy(np.stack([np.asarray(f[name][k][:])[ids] for k in keys], 1)).float().to(self.device)
        scores = torch.cat([self.detector_score(d[i:i + 128]) for i in range(0, len(d), 128)]).cpu().numpy()
        logger.info("Thermalizer detector on %d real snapshots: median %.4f, q%g %.4f, max %.4f",
                    len(scores), np.median(scores), q, np.quantile(scores, q), scores.max())
        return float(np.quantile(scores, q))

    @torch.no_grad()
    def thermalize(self, d, generator=None):
        """Pull raw global states (G, dim, N) toward the snapshot distribution of the state model."""
        th = self.therm
        sp = th['plan']
        G = d.shape[0]
        t = torch.tensor(th['t'], device=d.device)
        a, s = alpha_sigma(t)
        y = sp.to_y(d - sp.m)
        y_t = a * y + s * torch.randn(y.shape, device=d.device, generator=generator)
        cond = torch.zeros(G, sp.lltogl.shape[0], 0, self.dim * self.ndof, device=d.device)
        return sp.m + sp.ddim(th['model'], cond, sp.geom['coords'], generator, th['steps'], eta=0.0,
                              t_start=th['t'], y_start=y_t)

    def energy_score(self, model, X, Y, coords, generator=None):
        """Fair energy score of sampled K-step trajectories, averaged over the steps.

        X (G, E, tl, F) and Y (G, E, >= K, F) normalized. Each window is rolled
        out ms_members times with the ms_sampler_steps sampler, through the same
        conditioning as inference (residual mode included). Differentiable when
        gradients are enabled. Norm: sqrt(A-energy / increment energy).
        """
        K, M = self.cfg['ms_steps'], self.cfg['ms_members']
        steps, ckpt = self.cfg['ms_sampler_steps'], self.cfg['ms_checkpoint']
        G = X.shape[0]
        Xm = X.repeat_interleave(M, dim=0)
        pairs = [(a, b) for a in range(M) for b in range(a + 1, M)]
        total = 0.0
        for k in range(K):
            cond, base = self.condition(Xm)
            last = base + self.ddim(model, cond, coords, generator, steps, eta=0.0, checkpoint=ckpt)
            with torch.autocast(device_type=self.device.type, enabled=False):
                truth = self.scatter_mean(Y[:, :, k].float() * self.std + self.mean)            # (G, dim, N)
                d = last.float().view(G, M, self.dim, self.N)

                def norm(e):
                    return torch.sqrt(self.energy(e) / self.energy0 + 1e-8)

                skill = norm((d - truth[:, None]).reshape(G * M, self.dim, self.N)).view(G, M).mean(1)
                # each unordered pair once: sum over m != m' is twice this
                spread = sum(norm(d[:, a] - d[:, b]) for a, b in pairs) / len(pairs)
                total = total + (skill - 0.5 * spread).mean()
            Xm = torch.cat((Xm[:, :, 1:], self.normalize(last).unsqueeze(2)), dim=2)
        return total / K

    @torch.no_grad()
    def unroll(self, model, X, coords, k, generator=None, steps=None):
        """Advance a normalized window k steps with the model's own samples."""
        steps = steps or self.cfg['pushforward_steps']
        for _ in range(k):
            cond, base = self.condition(X)
            last = base + self.sample(model, cond, coords, generator, steps=steps)
            X = torch.cat((X[:, :, 1:], self.normalize(last).unsqueeze(2)), dim=2)
        return X

    def loss(self, model, X, Y0, coords, generator=None, cond_noise=0.0, t0=None):
        """Weighted x0 loss for one batch: X (G, E, tl, F) and Y0 (G, E, F), normalized.

        The target increment is taken from the window's last frame as given, so
        a self-generated window (or a perturbed one, with cond_noise_target
        'truth') is trained back toward the truth.
        """
        G = X.shape[0]
        truth = self.cfg['cond_noise_target'] == 'truth'
        if cond_noise > 0 and truth:
            X = self.perturb(X, cond_noise)
        cond, base = self.condition(X, t0)
        with torch.autocast(device_type=self.device.type, enabled=False):
            dd = self.scatter_mean(Y0.float() * self.std + self.mean) - base
        if cond_noise > 0 and not truth:
            cond, _ = self.condition(self.perturb(X, cond_noise), t0)
        with torch.autocast(device_type=self.device.type, enabled=False):
            y0 = self.to_y(dd)
            tmin = self.cfg['t_min']
            if generator is None:
                t = tmin + (1.0 - tmin) * torch.rand(G, device=X.device)
                eps = torch.randn_like(y0)
            else:
                # stratified: the fixed validation draw covers the whole schedule
                u = (torch.arange(G, device=X.device) + torch.rand(G, device=X.device, generator=generator)) / G
                t = tmin + (1.0 - tmin) * u[torch.randperm(G, device=X.device, generator=generator)]
                eps = torch.randn(y0.shape, device=X.device, generator=generator)
            a, s = alpha_sigma(t)
            y_t = a.view(-1, 1, 1) * y0 + s.view(-1, 1, 1) * eps
        y0_hat = self.denoise(model, cond, coords, y_t, t)
        with torch.autocast(device_type=self.device.type, enabled=False):
            err = self.energy(self.from_y(y0_hat - y0)) / self.energy0
            w = loss_weight(a, s, self.cfg['weighting'], self.cfg['gamma'])
            return (w * err).mean()

    # --------------------------------------------------------------- sampler
    @torch.no_grad()
    def sample(self, model, cond, coords, generator=None, steps=None, eta=None):
        """Draw one increment per conditioning window: cond (G, E, tl, F) -> dd (G, dim, N)."""
        return self.ddim(model, cond, coords, generator, steps, eta)

    def ddim(self, model, cond, coords, generator=None, steps=None, eta=None, checkpoint=False,
             t_start=1.0, y_start=None):
        """The DDIM sampler; differentiable when called with gradients enabled.

        checkpoint recomputes each denoiser call in the backward pass instead of
        storing its activations, which is what makes a K-step unroll fit.
        t_start < 1 with y_start (already noised to t_start) runs the tail of the
        schedule from there, as thermalization does.
        """
        steps = steps or self.cfg['sample_steps']
        eta = self.cfg['eta'] if eta is None else eta
        G = cond.shape[0]
        y = (torch.randn(G, self.dim, self.N, device=cond.device, generator=generator)
             if y_start is None else y_start)
        ts = torch.linspace(float(t_start), 0.0, steps + 1, device=cond.device)
        for i in range(steps):
            t, t_next = ts[i], ts[i + 1]
            if checkpoint and torch.is_grad_enabled():
                y0 = torch.utils.checkpoint.checkpoint(self.denoise, model, cond, coords, y, t.expand(G),
                                                       use_reentrant=False)
            else:
                y0 = self.denoise(model, cond, coords, y, t.expand(G))
            if i == steps - 1:
                y = y0
                break
            a, s = alpha_sigma(t)
            a_n, s_n = alpha_sigma(t_next)
            eps = (y - a * y0) / s
            c = eta * (s_n / s) * torch.sqrt((1.0 - (a / a_n) ** 2).clamp_min(0.0))
            y = a_n * y0 + torch.sqrt((s_n ** 2 - c ** 2).clamp_min(0.0)) * eps
            if eta > 0:
                y = y + c * torch.randn(y.shape, device=y.device, generator=generator)
        return self.from_y(y)

    @torch.no_grad()
    def rollout(self, model, coords, seeds, steps, members=1, generator=None):
        """Autoregressive sampled forecasts.

        seeds (S, tl, dim, N) raw global states -> (S, members, tl + steps, dim, N),
        seed frames included. Each member draws its own increments.
        """
        S, tl = seeds.shape[:2]
        state = seeds.to(self.device).float().repeat_interleave(members, dim=0)  # (S*M, tl, dim, N)
        out = torch.empty(S * members, tl + steps, self.dim, self.N)
        out[:, :tl] = state.cpu()
        cond = self.normalize(state)                                             # (S*M, tl, E, F)
        cond = cond.permute(0, 2, 1, 3).contiguous()                             # (S*M, E, tl, F)
        th = self.therm
        for k in range(steps):
            c, base = self.condition(cond)
            last = base + self.sample(model, c, coords, generator)
            if th is not None and k + 1 > th['after'] and (k + 1 - th['after']) % th['every'] == 0:
                if th['threshold'] is None:
                    last = self.thermalize(last, generator)
                else:
                    hot = torch.nonzero(self.detector_score(last) > th['threshold']).flatten()
                    th['checked'] += last.shape[0]
                    th['fired'] += len(hot)
                    if len(hot):
                        last = last.clone()
                        last[hot] = self.thermalize(last[hot], generator)
            out[:, tl + k] = last.cpu()
            cond = torch.cat((cond[:, :, 1:], self.normalize(last).unsqueeze(2)), dim=2)
        return out.reshape(S, members, tl + steps, self.dim, self.N)


# ------------------------------------------------------------------- training
def run_epoch(runner, epoch: int, train: bool, max_norm: float = None) -> float:
    """One epoch of one-step diffusion training or of the fixed-draw validation loss."""
    model = runner.model
    model.train(train)
    loaders = runner.train_loader if train else runner.test_loader
    cfg = settings(runner)
    coords = runner._elem_geometry(use_skip=False)['coords']
    gen = None if train else torch.Generator(device=runner.device).manual_seed(_VAL_SEED)
    core = model.module if hasattr(model, 'module') else model

    epoch_loss, norms, n_skipped, n_steps = 0.0, [], 0, 0
    term_sum = {}
    with torch.set_grad_enabled(train):
        for key in loaders:
            p = plan(runner, key)
            if train and runner.config['distributed']:
                runner.sampler[key].set_epoch(epoch)
            n_batches = len(loaders[key])
            for X, Y, _, t0 in loaders[key]:
                X, Y = X.to(runner.device), Y.to(runner.device)
                if cfg['ms_steps']:
                    loss, terms = _ms_step(runner, p, model, core, X, Y, coords, cfg, train, gen, max_norm, t0)
                    epoch_loss += loss / n_batches
                    for name_, v in terms.items():
                        if not name_.startswith('_'):
                            term_sum[name_] = term_sum.get(name_, 0.0) + v / n_batches
                    if train:
                        n_steps += 1
                        if terms['_gn'] is not None:
                            norms.append(terms['_gn'])
                        else:
                            n_skipped += 1
                    continue
                if not train:
                    # the one-step loss, then the loss after each self-unrolled step
                    vals = []
                    for k in range(cfg['val_unroll'] + 1):
                        if k:
                            X = p.unroll(model, X, coords, 1, generator=gen)
                        vals.append(float(p.loss(model, X, Y[:, :, k], coords, generator=gen)))
                    epoch_loss += float(np.mean(vals)) / n_batches
                    continue
                k = int(np.random.randint(cfg['pushforward'] + 1)) if cfg['pushforward'] else 0
                if k:
                    # unroll with the current weights, dropout off
                    core.net.eval()
                    X = p.unroll(core.net, X, coords, k)
                    core.net.train()
                runner.optimizer.zero_grad(set_to_none=True)
                # an unrolled window is no longer a training window: the full mean model then
                loss = p.loss(model, X, Y[:, :, k], coords, cond_noise=cfg['cond_noise_std'],
                              t0=None if k else t0)
                epoch_loss += float(loss.detach()) / n_batches
                runner.scaler.scale(loss).backward()
                runner.scaler.unscale_(runner.optimizer)
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
                runner.scaler.step(runner.optimizer)
                runner.scaler.update()
                n_steps += 1
                if np.isfinite(gn):
                    norms.append(gn)
                    core.update_ema(cfg['ema_decay'])
                else:
                    n_skipped += 1
    runner.last_terms = (term_sum or None) if train else None
    if train:
        runner.last_grad_norm = float(np.mean(norms)) if norms else float('nan')
        runner.last_skipped = n_skipped
        runner.last_steps = n_steps
    return epoch_loss


def _ms_step(runner, p, model, core, X, Y, coords, cfg, train, gen, max_norm, t0):
    """One batch of multi-step training (energy score + weighted one-step loss) or its validation.

    Returns (loss, terms); terms['_gn'] is the pre-clip gradient norm, None when
    the step was skipped for fp16 overflow. Validation scores the energy score alone.
    """
    if not train:
        es = float(p.energy_score(model, X, Y, coords, generator=gen))
        return es, {'es': es, '_gn': None}
    runner.optimizer.zero_grad(set_to_none=True)
    es = p.energy_score(model, X, Y, coords)
    dsm = p.loss(model, X, Y[:, :, 0], coords, t0=t0) if cfg['ms_dsm_weight'] > 0 else torch.zeros((), device=X.device)
    loss = es + cfg['ms_dsm_weight'] * dsm
    runner.scaler.scale(loss).backward()
    runner.scaler.unscale_(runner.optimizer)
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
    runner.scaler.step(runner.optimizer)
    runner.scaler.update()
    if np.isfinite(gn):
        core.update_ema(cfg['ema_decay'])
    return float(loss.detach()), {'es': float(es.detach()), 'dsm': float(dsm.detach()),
                                  '_gn': gn if np.isfinite(gn) else None}


# ----------------------------------------------------------------- prediction
def _seed_tensor(dof_u, dof_v, dof_w, dim):
    comps = [dof_u, dof_v] + ([dof_w] if dim == 3 else [])
    return torch.from_numpy(np.stack([np.asarray(c, dtype=np.float32) for c in comps], axis=1))


def forecast(runner, name, seeds, steps, members=1, seed_offset=0):
    """Sampled forecasts from seed windows.

    seeds: list of (dof_u, dof_v, dof_w) windows, each component (tl, N).
    Returns (S, members, tl + steps, dim, N) as float32 numpy, seed frames included.
    Starts are batched together up to about 512 trajectories per call.
    """
    p = plan(runner, name)
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    model.eval()
    coords = p.geom['coords']
    seed = int(p.cfg['seed']) + int(seed_offset)
    gen = torch.Generator(device=runner.device).manual_seed(seed)
    S = len(seeds)
    per_call = max(1, 512 // max(members, 1))
    outs = []
    for s0 in range(0, S, per_call):
        batch = torch.stack([_seed_tensor(*w, runner.dim) for w in seeds[s0:s0 + per_call]])
        outs.append(p.rollout(model, coords, batch, steps, members, gen).numpy())
    return np.concatenate(outs, axis=0)
