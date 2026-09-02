"""Training loss functions, selected from the config.

``model_params.loss`` is either a name or a dict carrying the name plus its
keyword arguments::

    "loss": "huber"
    "loss": {"type": "huber", "delta": 0.5}
    "loss": {"type": "combo", "terms": [{"type": "mse"},
                                        {"type": "traj_energy", "weight": 0.1}]}

Omitting the key keeps the historical ``mse`` behavior. Every loss here takes
(pred, target) with a leading batch axis and reduces to a scalar, so the
training loops can call it exactly like ``nn.MSELoss``.
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class RelativeL2Loss(nn.Module):
    """Mean relative L2 error ``||pred - target|| / ||target||`` per sample.

    Mirrors the l2_err_norm metric the predictions are scored with, so every
    snapshot contributes equally regardless of its energy. Norms are taken over
    all non-batch axes and averaged over the batch.
    """

    def __init__(self, eps: float = 1e-8):
        super(RelativeL2Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        # the training loops run under fp16 autocast; the norm ratio needs fp32
        pred = pred.float()
        target = target.float()
        dims = tuple(range(1, pred.dim()))
        num = torch.linalg.vector_norm(pred - target, dim=dims)
        den = torch.linalg.vector_norm(target, dim=dims).clamp_min(self.eps)
        return (num / den).mean()

    def extra_repr(self):
        return f"eps={self.eps}"


class LogCoshLoss(nn.Module):
    """Mean log(cosh(pred - target)): MSE-like near zero, L1-like in the tails."""

    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, pred, target):
        diff = (pred - target).float()
        # log(cosh(d)) written stably as |d| + log1p(exp(-2|d|)) - log(2)
        abs_diff = diff.abs()
        return (abs_diff + torch.log1p(torch.exp(-2.0 * abs_diff)) - 0.6931471805599453).mean()


class EnergyMatchLoss(nn.Module):
    """Relative error in fluctuation energy, E(x) = sum_f x_f^2.

    A quadratic loss is minimized, for an unpredictable component, by
    predicting its conditional mean -- which has *less* energy than any real
    realization. That under-dispersion is invisible to MSE (it is the optimum)
    but shows up directly here. Measured on the cavity PIV model:
    energy_retention_final 0.767, i.e. 23% of the fluctuation energy missing.

    Normalized by the batch-mean true energy rather than per sample, so a
    single low-energy snapshot cannot dominate the gradient.

    Reduces the feature axis and averages whatever axes precede it, so it works
    unchanged on a single step (B, F), on element tokens (B, E, F), or on a
    full rollout (B, E, ta, F). Wrapped in a MetricLoss it measures *field*
    energy rather than raw latent energy.
    """

    def __init__(self, eps: float = 1e-8):
        super(EnergyMatchLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        e_pred = pred.pow(2).sum(dim=-1)
        e_true = target.pow(2).sum(dim=-1)
        return (e_pred - e_true).abs().mean() / (e_true.mean() + self.eps)

    def extra_repr(self):
        return f"eps={self.eps}"


class VarianceMatchLoss(nn.Module):
    """Relative error in per-feature standard deviation.

    Complements EnergyMatchLoss: energy can be matched in aggregate while the
    variance is redistributed across features. This scores the spread of each
    feature over the sample (and rollout) axes, so it penalizes a narrowed
    predictive distribution feature by feature -- the pdf_std_ratio of 0.861
    measured on the cavity model.

    std is used rather than variance: it is scale-linear, so the term stays
    comparable in magnitude to a pointwise loss instead of quadratic in it.

    The estimator needs several samples to be meaningful; with a batch of 32
    the per-feature std carries roughly 1/sqrt(2*32) ~ 12% relative noise, so
    keep the weight modest and prefer larger batches.
    """

    def __init__(self, eps: float = 1e-8, unbiased: bool = False):
        super(VarianceMatchLoss, self).__init__()
        self.eps = eps
        self.unbiased = unbiased

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        dims = tuple(range(pred.dim() - 1))
        s_pred = pred.var(dim=dims, unbiased=self.unbiased).clamp_min(0).sqrt()
        s_true = target.var(dim=dims, unbiased=self.unbiased).clamp_min(0).sqrt()
        return (s_pred - s_true).abs().mean() / (s_true.mean() + self.eps)

    def extra_repr(self):
        return f"eps={self.eps}, unbiased={self.unbiased}"


class CombinedLoss(nn.Module):
    """Weighted sum of several losses, e.g. MSE regularized by a relative L2 term.

    After each forward, ``last_terms`` holds each term's *weighted* contribution
    as a detached tensor, so a training loop can report the split without
    forcing a host sync inside the hot loop.
    """

    def __init__(self, terms, weights, names=None):
        super(CombinedLoss, self).__init__()
        self.terms = nn.ModuleList(terms)
        self.weights = list(weights)
        self.names = list(names) if names else [type(t).__name__ for t in terms]
        self.last_terms = None

    def forward(self, pred, target):
        values = [w * term(pred, target) for w, term in zip(self.weights, self.terms)]
        self.last_terms = [v.detach() for v in values]
        total = values[0]
        for value in values[1:]:
            total = total + value
        return total

    def extra_repr(self):
        return ", ".join(f"{n}*{w}" for n, w in zip(self.names, self.weights))


class MetricLoss(nn.Module):
    """Apply a linear metric transform to both arguments, then a base loss.

    Used to score latent errors in field-energy coordinates instead of raw
    latent ones (see lib.decode.ElementGramMetric). The transform is linear, so
    metric(pred) - metric(target) == metric(pred - target) and any pointwise
    base loss keeps its meaning; only the geometry it measures changes.
    """

    def __init__(self, base, metric):
        super(MetricLoss, self).__init__()
        self.base = base
        self.metric = metric

    def forward(self, pred, target):
        return self.base(self.metric(pred), self.metric(target))

    @property
    def last_terms(self):
        return getattr(self.base, 'last_terms', None)

    @property
    def names(self):
        return getattr(self.base, 'names', None)

    def extra_repr(self):
        return 'metric=field_energy'


# Short labels for the per-term columns of the training table, so a combo loss
# does not push the epoch line past a terminal width.
TERM_ABBREV = {
    'mse': 'mse', 'l2': 'mse', 'mae': 'mae', 'l1': 'mae', 'huber': 'hub',
    'smooth_l1': 'sl1', 'logcosh': 'lch', 'rel_l2': 'rl2',
    'traj_energy': 'nrg', 'traj_var': 'var',
}


def abbrev(name: str) -> str:
    """Three-letter column label for a loss term name."""
    return TERM_ABBREV.get(name, name[:3])


def _build_simple(loss_type: str, params: dict) -> nn.Module:
    """Build a single (non-combo) loss from its type and keyword arguments."""
    if loss_type in ('mse', 'l2'):
        return nn.MSELoss()
    elif loss_type in ('mae', 'l1'):
        return nn.L1Loss()
    elif loss_type == 'huber':
        return nn.HuberLoss(delta=params.get('delta', 1.0))
    elif loss_type == 'smooth_l1':
        return nn.SmoothL1Loss(beta=params.get('beta', 1.0))
    elif loss_type == 'logcosh':
        return LogCoshLoss()
    elif loss_type == 'rel_l2':
        return RelativeL2Loss(eps=params.get('eps', 1e-8))
    elif loss_type == 'traj_energy':
        return EnergyMatchLoss(eps=params.get('eps', 1e-8))
    elif loss_type == 'traj_var':
        return VarianceMatchLoss(eps=params.get('eps', 1e-8),
                                 unbiased=params.get('unbiased', False))
    else:
        raise ValueError(
            f"Loss '{loss_type}' not recognized. Use 'mse', 'mae', 'huber', "
            "'smooth_l1', 'logcosh', 'rel_l2', 'traj_energy', 'traj_var', "
            "or 'combo'."
        )


def build_loss(loss_config=None) -> nn.Module:
    """Build the training criterion from ``model_params.loss``.

    Args:
        loss_config: A loss name, a dict of {'type': name, **kwargs}, or None
            for the default MSE.

    Returns:
        An nn.Module callable as criterion(pred, target).
    """
    if loss_config is None:
        loss_config = 'mse'

    if isinstance(loss_config, str):
        params = {}
        loss_type = loss_config.lower()
    elif isinstance(loss_config, dict):
        params = dict(loss_config)
        loss_type = params.pop('type', 'mse').lower()
    else:
        raise ValueError(f"model_params.loss must be a string or a dict, got {type(loss_config)}")

    if loss_type == 'combo':
        term_configs = params.get('terms', None)
        if not term_configs:
            raise ValueError("Loss 'combo' requires a non-empty 'terms' list")
        terms = []
        weights = []
        names = []
        for term_config in term_configs:
            term_config = dict(term_config) if isinstance(term_config, dict) else {'type': term_config}
            weights.append(term_config.pop('weight', 1.0))
            names.append(str(term_config.get('type', 'mse')).lower())
            terms.append(build_loss(term_config))
        return CombinedLoss(terms, weights, names)

    return _build_simple(loss_type, params)