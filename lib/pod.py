"""Global Proper Orthogonal Decomposition (POD) latent space.

This is a drop-in alternative to the localized GFEM/DLS latent space in
``lib/dls.py`` / ``lib/dls_2d.py``. Instead of patch-local modal coefficients,
the latent representation here is the set of *temporal coefficients* of a set of
global POD modes computed over the whole spatial domain.

POD is performed independently per velocity component (u, v, and w in 3D). For a
mean-subtracted snapshot matrix ``X`` of shape ``(P, T)`` (``P`` spatial points,
``T`` snapshots) the truncated SVD ``X = U S V^T`` gives orthonormal spatial
modes ``U`` and temporal coefficients ``a = S V^T`` (shape ``(num_modes, T)``).
The neural model learns to advance these temporal coefficients in time; the
field is reconstructed as ``X_rec = U @ a``.

The coefficients are stored in the latent HDF5 file under ``dof_u`` / ``dof_v``
/ ``dof_w`` with shape ``(num_snaps, num_modes)`` so that the rest of the runner
(splitting, scaling, training, autoregressive rollout) treats them exactly like
the per-node DLS DOFs. ``pod_Config`` exposes compatibility attributes
(``num_gfem_nodes = num_modes``, ``dof_node = 1``) that make the runner's
per-component bookkeeping work without special-casing.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.sparse.linalg import svds

import logging
logger = logging.getLogger(__name__)


ArrayDict = Dict[str, np.ndarray]
DataInput = Union[str, ArrayDict]


@dataclass
class pod_Config:
    """Configuration / saved state for a global POD latent space.

    ``modes`` holds one ``(P, num_modes)`` orthonormal spatial-mode matrix per
    velocity component, where ``P`` is the number of flattened spatial points
    (C-order flatten of the ``spatial`` grid shape).
    """

    num_snaps: int
    spatial: Tuple[int, ...]
    num_vars: int
    num_modes: int
    modes: List[np.ndarray]

    def __post_init__(self) -> None:
        self.dim = len(self.spatial)
        self.nx = int(self.spatial[0])
        self.ny = int(self.spatial[1])
        self.nz = int(self.spatial[2]) if self.dim == 3 else 1

        # POD reconstructs on the full grid, so the "truncated" grid used by the
        # runner/plotting for reconstructed fields is just the full grid.
        self.nx_t = self.nx
        self.ny_t = self.ny
        self.nz_t = self.nz

        # Compatibility shims with the DLS config interface. Each velocity
        # component is described by ``num_modes`` temporal coefficients, so we
        # present them as a single GFEM "node" carrying ``num_modes`` DOFs. This
        # lets the runner compute input_dim and split per-component DOF blocks
        # (``num_gfem_nodes * dof_node`` per component) without any branching.
        self.dof_node = self.num_modes
        self.num_gfem_nodes = 1
        self.dof_elem = self.num_modes

        n_points = self.nx * self.ny * self.nz
        self.compression_ratio = (
            self.num_vars * self.num_snaps * n_points
            / (
                self.num_vars * self.num_snaps * self.num_modes
                + self.num_vars * self.num_modes * n_points
            )
        )


def _field_layout(data_source: DataInput, field_name: str) -> Tuple[int, Tuple[int, ...], int]:
    """Return (num_snaps, spatial_shape, num_vars) for the source field."""
    if isinstance(data_source, str):
        with h5py.File(data_source, "r") as f:
            shape = f[field_name].shape
    else:
        shape = data_source[field_name].shape
    return shape[0], tuple(int(s) for s in shape[1:-1]), int(shape[-1])


def _load_fluctuations(
    data_source: DataInput,
    field_name: str,
    batch_size: int,
) -> Tuple[List[np.ndarray], Tuple[int, ...], int]:
    """Load mean-subtracted snapshot matrices, one per velocity component.

    Returns ``(X_list, spatial, num_vars)`` where each ``X_list[c]`` has shape
    ``(P, T)`` (spatial points by snapshots) in float64. The spatial axes are
    flattened in C order; reconstruction must use the same ordering.
    """
    num_snaps, spatial, num_vars = _field_layout(data_source, field_name)
    n_points = int(np.prod(spatial))

    X_list = [np.zeros((n_points, num_snaps), dtype=np.float64) for _ in range(num_vars)]

    if isinstance(data_source, str):
        f = h5py.File(data_source, "r")
        close = True
    else:
        f = data_source
        close = False
    try:
        mean = f["mean"][:]  # (*spatial, num_vars)
        dset = f[field_name]
        loops = num_snaps // batch_size + (1 if num_snaps % batch_size else 0)
        for ib in range(loops):
            s0 = ib * batch_size
            s1 = min((ib + 1) * batch_size, num_snaps)
            batch = np.asarray(dset[s0:s1])  # (b, *spatial, num_vars)
            fluc = batch - mean[np.newaxis, ...]
            b = s1 - s0
            for c in range(num_vars):
                # (b, *spatial) -> (b, P) -> (P, b)
                X_list[c][:, s0:s1] = fluc[..., c].reshape(b, n_points).T
    finally:
        if close:
            f.close()

    return X_list, spatial, num_vars


def _component_pod(X: np.ndarray, num_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Truncated POD of a single component snapshot matrix.

    ``X`` has shape ``(P, T)``. Returns ``(modes, coeffs)`` with
    ``modes`` shape ``(P, k)`` (orthonormal spatial modes) and ``coeffs`` shape
    ``(T, k)`` (temporal coefficients), ``k = min(num_modes, min(P, T) - 1)``,
    ordered by descending singular value.
    """
    max_k = min(X.shape) - 1
    k = min(num_modes, max_k)
    if k < num_modes:
        logger.warning(
            "Requested %d POD modes but only %d are available (P=%d, T=%d); "
            "truncating to %d.", num_modes, k, X.shape[0], X.shape[1], k)

    # Top-k singular triplets via Lanczos; far cheaper than a full SVD when
    # num_modes << min(P, T).
    U, s, Vt = svds(X, k=k)
    order = np.argsort(s)[::-1]
    U = U[:, order]
    s = s[order]
    Vt = Vt[order, :]

    modes = U.astype(np.float32)                  # (P, k)
    coeffs = (s[:, np.newaxis] * Vt).T.astype(np.float32)  # (T, k)
    return modes, coeffs


def pod_compress(
    data_source: DataInput,
    field_name: str,
    num_modes: int,
    group_name: Optional[str] = None,
    latent_target: Optional[Union[str, Dict[str, np.ndarray]]] = None,
    batch_size: int = 2500,
    pod_config: Optional[pod_Config] = None,
):
    """Compute global POD temporal coefficients for ``data_source``.

    When ``pod_config`` is ``None`` the POD basis is computed from this data
    (used for the "source" dataset that defines the latent space). When a
    ``pod_config`` is provided, its modes are reused and the data is simply
    projected onto them (used for additional train/eval datasets), guaranteeing
    every dataset shares the same global basis.

    latent_target:
      - None: returns ``(config, dof_u, dof_v[, dof_w])``
      - str path: appends ``dof_u``/``dof_v``[/``dof_w``] under ``group_name``
      - dict: stores the dof arrays in the provided dict
    """
    t0 = time.time()
    X_list, spatial, num_vars = _load_fluctuations(data_source, field_name, batch_size)
    num_snaps = X_list[0].shape[1]
    logger.info(f"POD: loaded fluctuations, spatial={spatial}, num_vars={num_vars}, "
                f"num_snaps={num_snaps}")

    if pod_config is None:
        logger.info(f"Computing global POD basis ({num_modes} modes per component)")
        modes: List[np.ndarray] = []
        coeffs: List[np.ndarray] = []
        for c in range(num_vars):
            m, a = _component_pod(X_list[c], num_modes)
            modes.append(m)
            coeffs.append(a)
            logger.info(f"  component {c}: modes {m.shape}, coeffs {a.shape}")
        config = pod_Config(
            num_snaps=num_snaps,
            spatial=spatial,
            num_vars=num_vars,
            num_modes=modes[0].shape[1],
            modes=modes,
        )
    else:
        logger.info("Projecting data onto provided global POD basis")
        config = pod_config
        if config.num_vars != num_vars:
            raise ValueError(
                f"POD basis has {config.num_vars} components but data has {num_vars}")
        coeffs = []
        for c in range(num_vars):
            m = config.modes[c]  # (P, num_modes), orthonormal
            # a = modes^T x  ->  (T, num_modes)
            coeffs.append((X_list[c].T @ m).astype(np.float32))
            logger.info(f"  component {c}: coeffs {coeffs[-1].shape}")

    logger.info(f"POD coefficients computed in {time.time() - t0:.2f}s "
                f"(compression ratio {config.compression_ratio:.1f}x)")

    dof_names = ["dof_u", "dof_v", "dof_w"][:num_vars]

    if latent_target is None:
        return (config, *coeffs)
    elif isinstance(latent_target, str):
        with h5py.File(latent_target, "a") as dof_file:
            grp = dof_file.create_group(group_name if group_name else "dofs")
            for name, a in zip(dof_names, coeffs):
                grp.create_dataset(name, data=a, dtype="float32")
        return config
    else:
        for name, a in zip(dof_names, coeffs):
            latent_target[name] = a
        return config


def pod_recon(
    rec_target: Optional[Union[str, Dict[str, np.ndarray]]],
    config: pod_Config,
    dof_u: Union[str, np.ndarray],
    dof_v: Optional[np.ndarray] = None,
    dof_w: Optional[np.ndarray] = None,
    batch_size: int = 100,
):
    """Reconstruct fluctuation fields from predicted POD temporal coefficients.

    ``dof_*`` are ``(num_snaps, num_modes)`` temporal-coefficient arrays (or an
    HDF5 path containing them). Returns / writes ``Q_rec`` with shape
    ``(num_snaps, *spatial, num_vars)`` matching ``gfem_recon_flexible``.
    """
    if isinstance(dof_u, str):
        with h5py.File(dof_u, "r") as f:
            dof_u_arr = f["dof_u"][:]
            dof_v_arr = f["dof_v"][:]
            dof_w_arr = f["dof_w"][:] if "dof_w" in f else None
    else:
        dof_u_arr = dof_u
        dof_v_arr = dof_v
        dof_w_arr = dof_w

    comps = [dof_u_arr, dof_v_arr]
    if config.dim == 3:
        if dof_w_arr is None:
            raise ValueError("3D POD reconstruction requires dof_w")
        comps.append(dof_w_arr)

    num_snaps = dof_u_arr.shape[0]
    spatial_t = (config.nx_t, config.ny_t, config.nz_t) if config.dim == 3 else (config.nx_t, config.ny_t)
    q_rec = np.zeros((num_snaps, *spatial_t, config.num_vars), dtype=np.float32)

    num_batches = num_snaps // batch_size + (1 if num_snaps % batch_size else 0)
    t_time = 0.0
    for bid in range(num_batches):
        s0 = bid * batch_size
        s1 = min((bid + 1) * batch_size, num_snaps)
        start_time = time.time()
        for c, dof in enumerate(comps):
            modes = config.modes[c]                 # (P, num_modes)
            rec = modes @ dof[s0:s1].T              # (P, b)
            rec = rec.T.reshape(s1 - s0, *spatial_t)  # C-order back to grid
            q_rec[s0:s1, ..., c] = rec.astype(np.float32)
        t_time += time.time() - start_time

    logger.info(f"POD reconstruction time (excluding disk): {t_time:.2f}s")

    if rec_target is None:
        return q_rec

    if isinstance(rec_target, str):
        with h5py.File(rec_target, "a") as rec_file:
            rec_file.create_dataset("Q_rec", data=q_rec, dtype="float32")
    else:
        rec_target["Q_rec"] = q_rec

    return q_rec
