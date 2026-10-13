"""Torch-side view of the latent codec.

The DLS reconstruction is linear and block-diagonal per element:
``field_patch = modemat_local_c @ dof[lltogl_e]``. That makes the field energy
of a latent error exactly

    ||B d||^2 = d^T (B^T B) d = d^T M d

so a Euclidean loss on raw latent DOFs is measuring a *different* metric than
field energy. On the LDC p99m10 config `B` has condition number 3.2e5, and the
gradient of a latent-space loss sits ~89 degrees from the field-energy gradient
for real model errors -- it still descends (M is PSD), just ~70x less per step.

`ElementGramMetric` re-expresses latent vectors in coordinates where the
Euclidean distance IS the field energy, by applying the matrix square root of
the small per-element Gram matrix. It costs one (dof_elem, dof_elem) matmul,
not a decode: 44x44 here rather than the 2500x44 reconstruction.

POD does not need this -- its modes are orthonormal, so M is the identity.
"""
import logging

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

# Identity blend for the Gram metric. 0 is pure field energy but leaves the
# codec's null space unconstrained; 1e-2 keeps the gradient within ~11 degrees
# of the field-energy gradient (vs ~89 for the raw latent metric) while giving
# the weakest direction ~3.5e-4 relative weight instead of ~4e-11.
GRAM_LAMBDA = 1e-2


def element_gram_roots(modemat: np.ndarray, masks: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """(E, n, n) Gram roots, one per element, from that element's in-domain points.

    masks is dls_2d.element_masks: 1 where an element's grid point lies inside the
    domain. Interior elements share one root; elements that overhang the grid
    (full_cover) count only their points inside it.
    """
    cache = {}
    out = np.empty((masks.shape[0], modemat.shape[1], modemat.shape[1]))
    for e, m in enumerate(masks):
        key = m.tobytes()
        if key not in cache:
            cache[key] = element_gram_sqrt(modemat[m > 0], lam)
        out[e] = cache[key]
    return out


def element_masks_for(runner):
    """dls_2d.element_masks for the runner's latent when its mesh overhangs the grid, else None."""
    dls = runner.dls
    if runner.dim != 2 or not hasattr(dls, 'is_cut') or not dls.is_cut(runner.l_config):
        return None
    return dls.element_masks(runner.l_config)


def element_gram_sqrt(modemat: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """Symmetric square root W of the regularized Gram matrix.

        M_reg = B^T B + lam * (trace(B^T B) / n) * I

    With ``lam = 0`` this gives ||W d|| == ||B d|| exactly, but the codec's near
    null space then carries essentially no weight -- on the p13m20 cavity basis
    43 of 84 local directions hold under 0.0002% of the field energy. A loss
    built on that metric does not constrain them at all, and in an
    autoregressive rollout nothing bounds their growth: they drift until the
    model's own forward overflows fp16. ``lam`` puts a floor under them.

    M is positive semi-definite and numerically singular for these bases, so
    the root is taken by eigendecomposition with negatives clamped, not by
    Cholesky.
    """
    B = np.asarray(modemat, dtype=np.float64)
    M = B.T @ B
    if lam:
        M = M + lam * (np.trace(M) / M.shape[0]) * np.eye(M.shape[0])
    evals, evecs = np.linalg.eigh(0.5 * (M + M.T))
    evals = np.clip(evals, 0.0, None)
    return (evecs * np.sqrt(evals)) @ evecs.T


class ElementGramMetric(nn.Module):
    """Map latent DOF vectors into field-energy coordinates.

    Input is whatever the model emits:
      localized     (..., E, dim*dof_elem)
      non-localized (..., dim*num_dofs)   -- gathered to elements via lltogl
    Output is always (..., E, dim*dof_elem), with Euclidean norm equal to the
    DLS field energy of the input (summed over elements, so the small overlap
    between neighbouring patches is counted once per element).

    The de-normalization is folded into W, so the module takes the *normalized*
    DOFs the training loop works in and still reports physical energy.
    """

    def __init__(self, weights, std, lltogl=None, dof_elem=None, elem_weights=None):
        super(ElementGramMetric, self).__init__()
        # (dim, dof_elem, dof_elem), scaled by the normalization factor only
        self.register_buffer('weights', weights)
        # (dim, E, dof_elem, dof_elem) when the mesh overhangs the grid: cut
        # elements count only their points inside the domain. Applied whenever the
        # input carries the full element axis; inputs without one (element-local
        # models that batch elements independently) fall back to `weights`.
        self.per_element = elem_weights is not None
        if self.per_element:
            self.register_buffer('elem_weights', elem_weights)
        # de-normalization is applied to the vector BEFORE gathering, so it works
        # whether dof_std is a scalar or one value per global DOF; folding it
        # into `weights` would only be valid when std indexes the element slot
        self.register_buffer('std', std)
        self.localized = lltogl is None
        if lltogl is not None:
            self.register_buffer('lltogl', lltogl)
        self.dim = weights.shape[0]
        self.dof_elem = dof_elem if dof_elem is not None else weights.shape[1]

    def forward(self, z):
        # fp32 only: the contraction sums dof_elem terms whose magnitudes span
        # several orders of magnitude and largely cancel, so in fp16 the small
        # outputs lose all precision (measured: gradient direction cos 0.968
        # against the fp32 result, and per-entry relative errors up to 4e4).
        with torch.autocast(device_type='cuda', enabled=False):
            return self._transform(z.float())

    def _transform(self, z):
        z = z * self.std
        if self.localized:
            comps = [z[..., c * self.dof_elem:(c + 1) * self.dof_elem]
                     for c in range(self.dim)]
        else:
            num_dofs = z.shape[-1] // self.dim
            comps = [z[..., c * num_dofs:(c + 1) * num_dofs][..., self.lltogl]
                     for c in range(self.dim)]
        if self.per_element and comps[0].dim() >= 2 and comps[0].shape[-2] == self.elem_weights.shape[1]:
            out = [torch.einsum('eij,...ej->...ei', self.elem_weights[c], comp)
                   for c, comp in enumerate(comps)]
        else:
            out = [torch.einsum('ij,...j->...i', self.weights[c], comp)
                   for c, comp in enumerate(comps)]
        return torch.cat(out, dim=-1)

    def extra_repr(self):
        return (f"dim={self.dim}, dof_elem={self.dof_elem}, "
                f"localized={self.localized}, per_element={self.per_element}")


def build_gram_metric(runner, name: str):
    """Build the field-energy metric for `runner`'s latent config, or None.

    Returns None for POD (already orthonormal) and for any config where the
    per-element mode matrices are unavailable.
    """
    lp = runner.config['latent_params']
    if lp['type'] != 'dls':
        logger.info("Gram metric skipped: latent type '%s' is already orthonormal "
                    "or unsupported", lp['type'])
        return None

    l_config = runner.l_config
    mats = [l_config.modemat_local_u, l_config.modemat_local_v]
    if runner.dim == 3:
        mats.append(l_config.modemat_local_w)

    lam = float(runner.config['model_params'].get('gram_lambda', GRAM_LAMBDA))
    W = np.stack([element_gram_sqrt(m, lam) for m in mats])  # (dim, dof_elem, dof_elem)
    masks = element_masks_for(runner)
    We = None if masks is None else np.stack([element_gram_roots(m, masks, lam) for m in mats])

    std = runner.dof_std[name].double()
    dof_elem = l_config.dof_elem
    localized = runner.config['latent_params'].get('localized', False)

    lltogl = None
    if not localized:
        lltogl = torch.from_numpy(runner._compute_lltogl_mat()).long()

    # Normalize so that a unit-variance latent error keeps unit expected energy
    # per feature, which keeps loss magnitudes (and so lr / grad_clip) in the
    # same range as the unweighted metric. Computed from the actual std, since
    # per-DOF scaling changes it. Uniform scaling never moves the gradient
    # direction, only the magnitude.
    if localized or std.ndim == 0:
        s_elem = (std.expand(dof_elem * W.shape[0]) if std.ndim == 0
                  else std).numpy().reshape(W.shape[0], dof_elem)
        energy = sum(float(((W[c] ** 2) * (s_elem[c] ** 2)[None, :]).sum())
                     for c in range(W.shape[0]))
        n_terms = W.shape[0] * dof_elem
    else:
        # non-localized with per-DOF std: gather each element's slots
        num_dofs = std.shape[0] // W.shape[0]
        energy = 0.0
        for c in range(W.shape[0]):
            sc = std[c * num_dofs:(c + 1) * num_dofs].numpy()[lltogl.numpy()]
            energy += float(((W[c] ** 2)[None] * (sc ** 2)[:, None, :]).sum())
        n_terms = W.shape[0] * lltogl.shape[0] * dof_elem
    scale = np.sqrt(n_terms / max(energy, 1e-30))
    W = W * scale
    # the interior scale for every element: cut elements then weigh exactly their
    # share of the field energy relative to full ones
    We = None if We is None else torch.from_numpy(We * scale).float()

    metric = ElementGramMetric(torch.from_numpy(W).float(), std.float(),
                               lltogl, dof_elem, elem_weights=We)
    logger.info("Field-energy (Gram) loss metric built: %s, lambda %g, "
                "normalization scale %.4g", metric, lam, scale)
    return metric.to(runner.device)
