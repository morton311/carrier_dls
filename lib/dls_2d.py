"""2D local-DLS compression on [x, y] ordered fields.

Every spatial array handled here is indexed ``[x_index, y_index]`` -- x varies
along axis 0, y along axis 1 (numpy meshgrid ``indexing='ij'``), matching
``lib/dls.py``.  Snapshot fields are therefore ``(num_snaps, nx, ny, num_vars)``
and grids are ``(nx, ny)``.

Sources written the other way round ([y, x], i.e. meshgrid ``indexing='xy'``)
are detected from x_grid/y_grid and transposed on read, so they still work; the
arrays this module returns are always [x, y].
"""
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)


ArrayDict = Dict[str, np.ndarray]
DataInput = Union[str, ArrayDict]


@dataclass
class dls_long_Config_Flexible:
    num_snaps: int
    nx: int
    ny: int
    num_vars: int
    patch_size: int
    num_modes: int
    modemat_local_u: np.ndarray
    modemat_local_v: np.ndarray
    # Extend the element mesh past the last full element so it covers the whole
    # grid; grid points outside the domain carry zero weight (see element_masks).
    # False keeps the historical mesh, which stops at the last full element and
    # truncates the grid to (nx_t, ny_t).
    full_cover: bool = False
    # Weight of the grid points of cut elements that lie outside the domain (see
    # element_masks). 0 leaves the coefficients of nodes past the edge nearly
    # free (held only by a 1e-8 ridge); alpha > 0 fits them, with that small
    # weight, to an edge-replicated extension of the field, as the finite cell
    # method gives its fictitious domain a small stiffness alpha.
    cut_weight: float = 0.0
    # The extension those points are fit to: 'edge' repeats the edge value,
    # 'odd' reflects the field oddly about it (2 q_edge - q_mirror), which keeps
    # the slope across the edge and suits boundary layers.
    cut_extension: str = 'edge'

    def __post_init__(self) -> None:
        self.patch_x, self.patch_y = patch_dims(self.patch_size)
        self.nskip_x = (self.patch_x - 1) // 2
        self.nskip_y = (self.patch_y - 1) // 2
        self.nskip = self.nskip_x            # legacy alias (square patches)
        self.nskip_sample = self.patch_x - 1
        self.mid_pt = 1 + self.nskip_sample // 2
        self.mid_pt_x = self.nskip_x + 1
        self.mid_pt_y = self.nskip_y + 1
        # nx_e, ny_e: extent of the element mesh. nx_t, ny_t: extent of the
        # reconstruction, which is the whole grid when the mesh covers it.
        self.nx_e, self.ny_e = mesh_extent(self.nx, self.ny, self.nskip_x, self.nskip_y, self.full_cover)
        self.sample_x = range(0, self.nx_e, self.nskip_x)
        self.sample_y = range(0, self.ny_e, self.nskip_y)
        self.nx_t = self.nx if self.full_cover else self.nx_e
        self.ny_t = self.ny if self.full_cover else self.ny_e
        self.nx_g = len(self.sample_x)
        self.ny_g = len(self.sample_y)
        self.num_gfem_nodes = self.nx_g * self.ny_g 
        self.num_gfem_elems = (self.nx_g - 1) * (self.ny_g - 1) 
        self.dof_node = self.num_modes + 1
        self.dof_elem = 4 * self.dof_node
        self.compression_ratio = (
            self.num_vars * self.num_snaps * self.nx * self.ny
            / (
                self.num_vars * self.num_snaps * self.dof_node
                + self.num_vars * self.num_modes * self.patch_x * self.patch_y
            )
        )


def node_map(opp=False) -> np.ndarray:
    # shape: (2, 4)
    IJK = np.array([[0, 1, 1, 0], [1, 1, 0, 0]])
    return IJK


def patch_dims(patch_size):
    """Normalize ``patch_size`` to (nx, ny).

    An int keeps the historical square patch; a pair gives a rectangular one,
    which lets the element aspect follow the structures being resolved (an
    elongated shear layer wants patches elongated the same way) instead of
    spending local modes resolving a homogeneous direction.
    """
    if np.isscalar(patch_size):
        return int(patch_size), int(patch_size)
    px, py = patch_size
    return int(px), int(py)


def config_nskip(config):
    """(nskip_x, nskip_y) for a config, tolerating pickles from before
    rectangular patches existed."""
    return (getattr(config, 'nskip_x', config.nskip),
            getattr(config, 'nskip_y', config.nskip))


def mesh_extent(nx, ny, nskip_x, nskip_y, full_cover=False):
    """Grid extent (nx_e, ny_e) spanned by the element mesh.

    Nodes sit every nskip points from 0. Without full_cover the mesh stops at the
    last node inside the grid, dropping up to nskip-1 trailing rows and columns;
    with it, one more node is added past the edge wherever the grid would be
    truncated, so the last elements overhang the domain.
    """
    def ext(n, k):
        last = ((n - 1) // k) * k
        return n if full_cover and last == n - 1 else (last + k + 1 if full_cover else last + 1)
    return ext(nx, nskip_x), ext(ny, nskip_y)


def element_masks(config, skipx=1, skipy=1, strict=False):
    """(E, P) weights of each element's grid points: 1 inside the domain.

    Rows follow the element loop of compression, reconstruction and
    runner._compute_lltogl_mat (x index outer, y inner); columns follow the
    element's local point order (x fastest, as _build_wt_vec). All ones unless
    the mesh overhangs the grid (full_cover). Cut elements get their own mass
    matrix B^T diag(Wt * mask) B. Points outside the domain weigh
    config.cut_weight (0 by default): they hold the edge-replicated extension
    of the field, not data, so with a small weight they fix the coefficients
    of nodes past the edge without pulling on the fit inside. strict=True
    gives the 0/1 indicator of the domain, for scoring in-domain error.
    """
    nkx, nky = config_nskip(config)
    nx, ny = config.nx, config.ny
    ix = np.arange(nkx + 1)
    iy = np.arange(nky + 1)
    rows = []
    for i in range(0, config.nx_g - 1, skipx):
        for j in range(0, config.ny_g - 1, skipy):
            inside = ((i * nkx + ix)[:, None] < nx) & ((j * nky + iy)[None, :] < ny)
            rows.append(inside.reshape(-1, order='F'))
    rows = np.asarray(rows, dtype=np.float64)
    alpha = 0.0 if strict else float(getattr(config, 'cut_weight', 0.0))
    return rows + alpha * (1.0 - rows) if alpha else rows


def is_cut(config) -> bool:
    """True when some element overhangs the grid (full_cover with a remainder)."""
    return bool(getattr(config, 'full_cover', False)) and (
        getattr(config, 'nx_e', config.nx_t) > config.nx or getattr(config, 'ny_e', config.ny_t) > config.ny)


# element-to-DOF index tables for decode_fields, keyed by id(config) and checked against
# a weak reference (the dataclass is unhashable, and the table must not be pickled with it)
_DECODE_PLANS = {}


def _decode_lltogl(config):
    """(E, dof_elem) global DOF index of every element slot, in the loop order of gfem_recon_flexible."""
    import weakref
    hit = _DECODE_PLANS.get(id(config))
    if hit is not None and hit[0]() is config:
        return hit[1]
    IJK = node_map()
    L = np.array([build_lltogl(i, j, config.ny_g, config.dof_node, IJK)
                  for i in range(config.nx_g - 1) for j in range(config.ny_g - 1)])
    _DECODE_PLANS[id(config)] = (weakref.ref(config), L)
    return L


def decode_fields(config, dof_u, dof_v, chunk: int = 64) -> np.ndarray:
    """Vectorized reconstruction: (T, num_dofs) per component -> (T, nx_t, ny_t, 2) float32.

    The same field as the per-element loop it replaces in gfem_recon_flexible:
    one 2-D product of all element DOFs with the local basis, then a reshape that
    lays the element patches out on the grid. Where elements share an edge the
    loop kept the value of the element written last, i.e. the one for which the
    edge is local index 0; the bulk copy below does the same, and the far row and
    column of the mesh come from the last elements.
    """
    nkx, nky = config_nskip(config)
    Ex, Ey = config.nx_g - 1, config.ny_g - 1
    nxt, nyt = config.nx_t, config.ny_t
    L = _decode_lltogl(config)
    T = dof_u.shape[0]
    out = np.empty((T, nxt, nyt, 2), dtype=np.float32)
    for c, (d, B) in enumerate(((dof_u, config.modemat_local_u), (dof_v, config.modemat_local_v))):
        Bt = np.asarray(B, dtype=np.float64).T                      # (dof_elem, P)
        for s in range(0, T, chunk):
            D = np.asarray(d[s:s + chunk], dtype=np.float64)[:, L]   # (t, E, dof_elem)
            t = D.shape[0]
            # local point index is iy * (nkx + 1) + ix, so the patch axis splits as (iy, ix)
            pt = (D.reshape(-1, D.shape[-1]) @ Bt).reshape(t, Ex, Ey, nky + 1, nkx + 1)
            full = np.empty((t, Ex * nkx + 1, Ey * nky + 1))
            full[:, :-1, :-1] = pt[..., :nky, :nkx].transpose(0, 1, 4, 2, 3).reshape(t, Ex * nkx, Ey * nky)
            full[:, -1, :-1] = pt[:, -1, :, :nky, nkx].reshape(t, Ey * nky)
            full[:, :-1, -1] = pt[:, :, -1, nky, :nkx].reshape(t, Ex * nkx)
            full[:, -1, -1] = pt[:, -1, -1, nky, nkx]
            out[s:s + t, :, :, c] = full[:, :nxt, :nyt]
    return out


def _build_wt_vec(nskip, nskip_y=None) -> np.ndarray:
    nskip_x = nskip
    if nskip_y is None:
        nskip_y = nskip_x
    Wt = np.ones((nskip_x+1, nskip_y+1))
    Wt[1:-1,0] = 1/2
    Wt[1:-1,-1] = 1/2
    Wt[0,1:-1] = 1/2
    Wt[-1,1:-1] = 1/2
    Wt[0,0] = 1/4
    Wt[0,-1] = 1/4
    Wt[-1,0] = 1/4
    Wt[-1,-1] = 1/4

    Wt_vec = Wt.reshape((nskip_x+1)*(nskip_y+1), order='F')
    return Wt_vec

def build_lltogl(
    i: int,
    j: int,
    ny_g: int,
    dof_node: int,
    IJK: np.ndarray,
) -> np.ndarray:
    lltogl = np.zeros(4 * dof_node, dtype=int)
    for node in range(4):
        indx_dof_start = (
            (i + IJK[0, node]) * ny_g
            + (j + IJK[1, node])
        ) * dof_node
        idx_dof_end = indx_dof_start + dof_node
        lltogl[node * dof_node : (node + 1) * dof_node] = np.arange(indx_dof_start, idx_dof_end)
    return lltogl


def _validate_data_dict(data_dict: ArrayDict, field_name: str) -> None:
    required = [field_name, "mean", "x_grid", "y_grid"]
    missing = [k for k in required if k not in data_dict]
    if missing:
        raise KeyError(f"Missing required keys for in-memory input: {missing}")


def _is_yx_ordered(x_grid: np.ndarray, y_grid: np.ndarray) -> bool:
    """True when spatial arrays are stored [y, x] instead of the required [x, y].

    This module indexes every spatial array as [x_index, y_index] (see
    local_modemat_over_elem: x_val = x_grid[ix, iy]), i.e. x must vary along
    axis 0 and y along axis 1 (numpy meshgrid 'ij' indexing).  Sources built
    with meshgrid 'xy' indexing store the two spatial axes the other way round;
    left uncorrected the element spacing along one axis is identically zero and
    the GFEM mass matrix is exactly singular.

    The grids are self-describing, so infer the layout from them rather than
    trusting the caller.
    """
    x_axis0, x_axis1 = np.ptp(x_grid[:, 0]), np.ptp(x_grid[0, :])
    y_axis0, y_axis1 = np.ptp(y_grid[:, 0]), np.ptp(y_grid[0, :])
    if x_axis0 > 0 and y_axis1 > 0:
        return False
    if x_axis1 > 0 and y_axis0 > 0:
        return True
    raise ValueError(
        "Cannot infer grid orientation: expected each of x_grid/y_grid to vary "
        "along exactly one axis, got x_grid variation "
        f"(axis0={x_axis0:.6g}, axis1={x_axis1:.6g}) and y_grid variation "
        f"(axis0={y_axis0:.6g}, axis1={y_axis1:.6g})."
    )


def _needs_transpose(data_source: DataInput) -> bool:
    if isinstance(data_source, str):
        with h5py.File(data_source, "r") as f:
            return _is_yx_ordered(f["x_grid"][:], f["y_grid"][:])
    return _is_yx_ordered(data_source["x_grid"], data_source["y_grid"])


def normalize_orientation(data_dict: ArrayDict, field_name: str) -> ArrayDict:
    """Return data_dict with its spatial axes swapped if it is stored [y, x].

    Use this to bring ground-truth arrays and grids into the same layout the
    solver returns, so comparisons and contour plots line up:

        input_dict = dls.normalize_orientation(input_dict, 'UV')

    Note that x_grid/y_grid are transposed but *not* renamed: x_grid.T still
    holds x-coordinates, and after the transpose they vary along axis 0.
    Swapping the names instead also yields a non-singular system (the element
    geometry is merely reflected across its diagonal) but silently mislabels the
    axes, so plots come out with x and y interchanged.
    """
    _validate_data_dict(data_dict, field_name)
    if not _is_yx_ordered(data_dict["x_grid"], data_dict["y_grid"]):
        return data_dict

    out = dict(data_dict)
    out[field_name] = data_dict[field_name].transpose(0, 2, 1, 3)
    out["mean"] = data_dict["mean"].transpose(1, 0, 2)
    out["x_grid"] = data_dict["x_grid"].T
    out["y_grid"] = data_dict["y_grid"].T
    return out


def _source_metadata(data_source: DataInput, field_name: str) -> Tuple[int, int, int, int, int]:
    """(num_snaps, nx, ny, num_vars) of the source once brought into [x, y]."""
    if isinstance(data_source, str):
        with h5py.File(data_source, "r") as f:
            shape = f[field_name].shape
    else:
        _validate_data_dict(data_source, field_name)
        shape = data_source[field_name].shape

    if _needs_transpose(data_source):
        return shape[0], shape[2], shape[1], shape[3]
    return shape[0], shape[1], shape[2], shape[3]


def _read_static(data_source: DataInput, field_name: str):
    if isinstance(data_source, str):
        with h5py.File(data_source, "r") as f:
            mode_data = f[field_name][0, :, :, :] - f["mean"][:]
            grid_x = f["x_grid"][:]
            grid_y = f["y_grid"][:]
    else:
        mode_data = data_source[field_name][0, :, :, :] - data_source["mean"]
        grid_x = data_source["x_grid"]
        grid_y = data_source["y_grid"]

    if _is_yx_ordered(grid_x, grid_y):
        logger.info(
            "Source is stored [y, x] (meshgrid 'xy' indexing); transposing "
            "spatial axes on read to the required [x, y] layout."
        )
        mode_data = mode_data.transpose(1, 0, 2)
        grid_x = grid_x.T
        grid_y = grid_y.T
    return mode_data, grid_x, grid_y


def _read_batch(data_source: DataInput, field_name: str, snap_start: int, snap_end: int):
    if isinstance(data_source, str):
        with h5py.File(data_source, "r") as f:
            mean = f["mean"][:]
            q_u = f[field_name][snap_start:snap_end, :, :, 0]
            q_v = f[field_name][snap_start:snap_end, :, :, 1]

    else:
        mean = data_source["mean"]
        q_u = data_source[field_name][snap_start:snap_end, :, :, 0]
        q_v = data_source[field_name][snap_start:snap_end, :, :, 1]

    if _needs_transpose(data_source):
        mean = mean.transpose(1, 0, 2)
        q_u = q_u.transpose(0, 2, 1)
        q_v = q_v.transpose(0, 2, 1)

    u_mean = mean[:, :, 0]
    v_mean = mean[:, :, 1]

    q_u = q_u.transpose(1, 2, 0) - u_mean[:, :, np.newaxis]
    q_v = q_v.transpose(1, 2, 0) - v_mean[:, :, np.newaxis]
    return q_u, q_v

def random_patch_sampling(data, patch_size):
    num_patches = 10000
    num_images = 1
    nx = data.shape[0]
    ny = data.shape[1]
    sz_x, sz_y = patch_dims(patch_size)
    BUFF = 0
    totalsamples = 0
    X = np.zeros((sz_x * sz_y, num_patches))
    
    for i in range(num_images):
        this_image = data

        # Determine how many patches to take
        getsample = num_patches // num_images
        if i == num_images - 1:
            getsample = num_patches - totalsamples

        # Extract patches at random from this image to make data vector X
        for j in range(getsample):
            d1 = BUFF + np.random.randint(0, nx - sz_x - 2 * BUFF)
            d2 = BUFF + np.random.randint(0, ny - sz_y - 2 * BUFF)

            totalsamples += 1
            temp = this_image[d1:d1 + sz_x, d2:d2 + sz_y].reshape(sz_x * sz_y, order='F')
            X[:, totalsamples - 1] = temp - np.mean(temp)

    
    return X


def Modal_decomp(data, patch_size):
    data_shape = 'fat' if data.shape[0] < data.shape[1] else 'tall' # not yet implemented
    P = random_patch_sampling(data, patch_size)
    local_modes, eigVal, _ = np.linalg.svd(P, full_matrices=False)
    return local_modes, eigVal


def FEM_shape_calculator_ortho_gfemlr(zeta_coord, eta_coord, zeta_pts, eta_pts):
    """Bilinear shape functions of a parallelogram element at one grid point.

    The element's zeta direction runs along array axis 1 and its eta direction
    along axis 0, so under this module's [x, y] layout the ``zeta_*`` arguments
    carry y-coordinates and the ``eta_*`` arguments x-coordinates.  ``*_pts``
    are the four corner values in the adopted nodal order (see
    local_modemat_over_elem).
    """
    sum_zeta_pt = np.sum(zeta_pts) / 4
    sum_eta_pt = np.sum(eta_pts) / 4

    dzeta_pt = (-zeta_pts[0] + zeta_pts[1] + zeta_pts[2] - zeta_pts[3]) / 2
    deta_pt = (eta_pts[0] + eta_pts[1] - eta_pts[2] - eta_pts[3]) / 2

    zeta_i = [-1, 1, 1, -1]
    eta_i = [1, 1, -1, -1]

    # Inverse transform for parallelogram elements, bilinear shape functions
    zeta = 2 * (zeta_coord - sum_zeta_pt) / dzeta_pt
    eta = 2 * (eta_coord - sum_eta_pt) / deta_pt

    N = np.zeros((4,1))
    # shape function values
    for i in range(4):
        N[i] = (1 / 4) * (1 + zeta_i[i] * zeta) * (1 + eta_i[i] * eta)
    return N


def local_modemat_over_elem(x_grid, y_grid, nskip, modes_vec, num_modes, patch_size):

    patch_x, patch_y = patch_dims(patch_size)
    nskip_x = (patch_x - 1) // 2
    nskip_y = (patch_y - 1) // 2
    mid_pt_x = nskip_x + 1
    mid_pt_y = nskip_y + 1

    dof_node = num_modes + 1
    dof_elem = 4 * dof_node

    modes_grid = np.zeros((patch_x, patch_y, num_modes))
    for i in range(num_modes):
        modes_grid[:,:, i] = modes_vec[:, i].reshape(patch_x, patch_y, order='F')

    # adopted nodal order
    gfem_nodal_order = [1, 2, 3, 4]
    diag_opp_indx =    [3, 4, 1, 2]

    # Mode grid components for the four quadrants
    F_indx = [1, 2, 3, 4]  # faces order, F1:x-ve, F2:x+ve, F3:y-ve, F4:y+ve
    F1 = list(range(0, mid_pt_x))
    F2 = list(range(mid_pt_x - 1, patch_x))
    F3 = list(range(0, mid_pt_y))
    F4 = list(range(mid_pt_y - 1, patch_y))

    indices = [
        (F1, F4), # 1
        (F2, F4), # 2
        (F2, F3), # 3
        (F1, F3), # 4
    ]

    modes_grid_1_comp = np.zeros((mid_pt_x, mid_pt_y, num_modes))
    modes_grid_2_comp = np.zeros((mid_pt_x, mid_pt_y, num_modes))
    modes_grid_3_comp = np.zeros((mid_pt_x, mid_pt_y, num_modes))
    modes_grid_4_comp = np.zeros((mid_pt_x, mid_pt_y, num_modes))

    comp1_x, comp1_y = np.meshgrid(F1, F4, indexing='ij')
    comp2_x, comp2_y = np.meshgrid(F2, F4, indexing='ij')
    comp3_x, comp3_y = np.meshgrid(F2, F3, indexing='ij')
    comp4_x, comp4_y = np.meshgrid(F1, F3, indexing='ij')

    for i in range(num_modes):
        modes_grid_1_comp[:,:, i] = modes_grid[comp1_x, comp1_y, i]
        modes_grid_2_comp[:,:, i] = modes_grid[comp2_x, comp2_y, i]
        modes_grid_3_comp[:,:, i] = modes_grid[comp3_x, comp3_y, i]
        modes_grid_4_comp[:,:, i] = modes_grid[comp4_x, comp4_y, i]


    npatch = mid_pt_x * mid_pt_y
    modes_vec_comp1 = modes_grid_1_comp.reshape(npatch, num_modes, order='F')
    modes_vec_comp2 = modes_grid_2_comp.reshape(npatch, num_modes, order='F')
    modes_vec_comp3 = modes_grid_3_comp.reshape(npatch, num_modes, order='F')
    modes_vec_comp4 = modes_grid_4_comp.reshape(npatch, num_modes, order='F')


    # local modal matrix for single element
    i, j = 1, 1

    # Corners of the reference element, in the adopted nodal order.  The index
    # pairs are (axis 0, axis 1) = (x, y) under this module's [x, y] layout, so
    # the element's zeta direction (axis 1) carries y and eta (axis 0) carries x.
    x1 = x_grid[ i   *nskip_x,  (j-1)*nskip_y]
    x2 = x_grid[ i   *nskip_x,   j   *nskip_y]
    x3 = x_grid[(i-1)*nskip_x,   j   *nskip_y]
    x4 = x_grid[(i-1)*nskip_x,  (j-1)*nskip_y]

    y1 = y_grid[ i   *nskip_x,  (j-1)*nskip_y]
    y2 = y_grid[ i   *nskip_x,   j   *nskip_y]
    y3 = y_grid[(i-1)*nskip_x,   j   *nskip_y]
    y4 = y_grid[(i-1)*nskip_x,  (j-1)*nskip_y]

    xpt = [x1, x2, x3, x4]
    ypt = [y1, y2, y3, y4]

    N1 = np.zeros((mid_pt_x * mid_pt_y))
    N2 = np.zeros((mid_pt_x * mid_pt_y))
    N3 = np.zeros((mid_pt_x * mid_pt_y))
    N4 = np.zeros((mid_pt_x * mid_pt_y))

    for iy in range(mid_pt_y):

        for ix in range(mid_pt_x):

            x_val = x_grid[ix, iy]
            y_val = y_grid[ix, iy]
            iind = ix*mid_pt_y + iy
            N = FEM_shape_calculator_ortho_gfemlr(y_val, x_val, ypt, xpt)
            N1[iind] = N[0][0]
            N2[iind] = N[1][0]
            N3[iind] = N[2][0]
            N4[iind] = N[3][0]

    modemat_local = np.hstack([
        N1[:, np.newaxis],
        N1[:, np.newaxis] * modes_vec_comp3,
        N2[:, np.newaxis],
        N2[:, np.newaxis] * modes_vec_comp4,
        N3[:, np.newaxis],
        N3[:, np.newaxis] * modes_vec_comp1,
        N4[:, np.newaxis],
        N4[:, np.newaxis] * modes_vec_comp2
    ])

    Wt_vec = _build_wt_vec(nskip_x, nskip_y)

    modemat_local_wt = modemat_local.copy()
    for kk in range(dof_elem):
        modemat_local_wt[:, kk] *= Wt_vec

    return modemat_local, modemat_local_wt


def gfem_compress_flexible(
    data_source: DataInput,
    field_name: str,
    patch_size: int,
    num_modes: int,
    group_name: Optional[str] = None,
    latent_target: Optional[Union[str, Dict[str, np.ndarray]]] = None,
    batch_size: int = 2500,
    dls_config: Optional[dls_long_Config_Flexible] = None,
    max_snaps: Optional[int] = None,
    full_cover: bool = False,
    cut_weight: float = 0.0,
    cut_extension: str = 'edge',
):
    """
    Flexible variant of gfem_3d_long.

    full_cover: extend the mesh past the grid edge so the whole grid is
      represented (see dls_long_Config_Flexible.full_cover). A provided
      dls_config keeps its own setting.

    cut_weight: weight of the points of cut elements outside the domain, fit
      to an extension of the field (see element_masks).
      A provided dls_config keeps its own value.

    cut_extension: 'edge' (repeat the edge value) or 'odd' (odd reflection about
      it, keeping the slope); with cut_weight > 0 only.

    max_snaps: compress only the leading max_snaps snapshots (None: all). The
      local modes come from snapshot 0 either way, so the basis is unchanged.

    data_source:
      - str path to an HDF5 file
      - dict with in-memory arrays keyed by: field_name, mean, x_grid, y_grid

    Spatial arrays are [x, y]: field_name is (num_snaps, nx, ny, num_vars),
    mean is (nx, ny, num_vars) and the grids are (nx, ny).  A [y, x] source is
    detected from its grids and transposed on read.

    latent_target:
      - None: returns dof arrays in-memory
      - str path: writes dof_u/dof_v to an HDF5 file
      - dict: stores dof arrays in provided dict keys dof_u/dof_v
    """
    ndim = 2
    num_snaps, nx, ny, _ = _source_metadata(data_source, field_name)
    if max_snaps is not None and max_snaps < num_snaps:
        logger.info(f"Compressing the leading {max_snaps} of {num_snaps} snapshots")
        num_snaps = int(max_snaps)
    mode_data, grid_x, grid_y = _read_static(data_source, field_name)

    patch_x, patch_y = patch_dims(patch_size)
    nskip_x = (patch_x - 1) // 2
    nskip_y = (patch_y - 1) // 2
    if dls_config is not None:
        full_cover = bool(getattr(dls_config, 'full_cover', False))
        cut_weight = float(getattr(dls_config, 'cut_weight', 0.0))
        cut_extension = getattr(dls_config, 'cut_extension', 'edge')
    if cut_extension not in ('edge', 'odd'):
        raise ValueError(f"cut_extension '{cut_extension}' not recognized; use 'edge' or 'odd'")
    nx_e, ny_e = mesh_extent(nx, ny, nskip_x, nskip_y, full_cover)
    sample_x = range(0, nx_e, nskip_x)
    sample_y = range(0, ny_e, nskip_y)
    if full_cover and (nx_e > nx or ny_e > ny):
        logger.info(f"Mesh extended to {nx_e}x{ny_e} to cover the {nx}x{ny} grid; points outside "
                    + ("the domain carry zero weight" if not cut_weight else
                       f"the domain fit an {cut_extension} extension with weight {cut_weight:g}"))

    nx_g = len(sample_x)
    ny_g = len(sample_y)

    num_gfem_nodes = nx_g * ny_g
    dof_node = num_modes + 1
    dof_elem = 4 * dof_node

    patch_points = (nskip_x + 1) * (nskip_y + 1)
    if dof_elem > patch_points:
        logger.warning(
            "Over-complete element: %d DOFs describe only %d sample points "
            "(patch %dx%d, num_modes=%d). The local basis is rank deficient, so "
            "%d coefficient directions produce no field change; they are fixed "
            "only by node sharing in the global assembly, carry the largest "
            "coefficients, and are temporally incoherent. Reduce num_modes to "
            "<= %d or enlarge the patch.",
            dof_elem, patch_points, patch_x, patch_y, num_modes,
            dof_elem - patch_points, patch_points // 4 - 1)
    elif dof_elem > patch_points // 2:
        logger.info("Element uses %d DOFs for %d sample points (patch %dx%d); "
                    "the local basis is getting close to over-complete.",
                    dof_elem, patch_points, patch_x, patch_y)

    logger.info(f"shape of mode data: {mode_data.shape}")
    logger.info(f"number of snapshots: {num_snaps}")
    logger.info(f"number of batches: {num_snaps // batch_size}")
    logger.info(f"nx: {nx}")
    logger.info(f"ny: {ny}")
    logger.info(f"num_vars: {ndim}")

    if dls_config is None:
        logger.info("Performing modal decomposition to get local modes")
        local_modes_u, _ = Modal_decomp(mode_data[..., 0], patch_size)
        local_modes_v, _ = Modal_decomp(mode_data[..., 1], patch_size)
        
        logger.info("Modal decomposition done")

        logger.info("Constructing local modal matrices")
        modemat_local_u, modemat_local_wt_u = local_modemat_over_elem(
            grid_x, grid_y, nskip_x, local_modes_u, num_modes, patch_size
        )
        modemat_local_v, modemat_local_wt_v = local_modemat_over_elem(
            grid_x, grid_y, nskip_x, local_modes_v, num_modes, patch_size
        )

        logger.info("Local modal matrices constructed")
    else:
        modemat_local_u = dls_config.modemat_local_u
        modemat_local_v = dls_config.modemat_local_v

        logger.info("Using provided local modal matrices from dls_config")

        Wt_vec = _build_wt_vec(nskip_x, nskip_y)
        modemat_local_wt_u = modemat_local_u.copy()
        modemat_local_wt_v = modemat_local_v.copy()

        for kk in range(dof_elem):
            modemat_local_wt_u[:, kk] *= Wt_vec
            modemat_local_wt_v[:, kk] *= Wt_vec

        logger.info("Constructed weighted local modal matrices from provided local modal matrices")


    # With a uniform/structured mesh in this formulation, one local mass matrix is
    # reused; elements that overhang the grid (full_cover) weight only their points
    # inside it, so they get their own (one per distinct mask).
    config = dls_long_Config_Flexible(
        num_snaps=num_snaps,
        nx=nx,
        ny=ny,
        num_vars=ndim,
        patch_size=patch_size,
        num_modes=num_modes,
        modemat_local_u=modemat_local_u,
        modemat_local_v=modemat_local_v,
        full_cover=full_cover,
        cut_weight=cut_weight,
        cut_extension=cut_extension,
    )
    masks = element_masks(config)
    Wt_vec = _build_wt_vec(nskip_x, nskip_y)
    M_cache = {}

    def local_mass(mask):
        key = mask.tobytes()
        if key not in M_cache:
            w = Wt_vec * mask
            M_cache[key] = ((modemat_local_u * w[:, None]).T @ modemat_local_u,
                            (modemat_local_v * w[:, None]).T @ modemat_local_v)
        return M_cache[key]

    M_u = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_v = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))

    IJK = node_map()

    logger.info("Constructing global M GFEM matrices")
    e = 0
    for i in range(nx_g - 1):
        for j in range(ny_g - 1):
            lltogl = build_lltogl(i, j, ny_g, dof_node, IJK)
            M_local_u, M_local_v = local_mass(masks[e])
            M_u[np.ix_(lltogl, lltogl)] += M_local_u
            M_v[np.ix_(lltogl, lltogl)] += M_local_v
            e += 1
    if is_cut(config):
        # Nodes past the edge see only the few grid points their support shares
        # with the domain, so some of their DOFs are nearly unconstrained. A ridge
        # far below every resolved direction keeps the solve well posed.
        ridge = 1e-8 * np.mean(M_u.diagonal()), 1e-8 * np.mean(M_v.diagonal())
        M_u.setdiag(M_u.diagonal() + ridge[0])
        M_v.setdiag(M_v.diagonal() + ridge[1])
        logger.info(f"{len(M_cache)} distinct element mass matrices; ridge 1e-8 x mean diagonal")

    logger.info("Prefactorizing M")
    solve_M_u = factorized(M_u.tocsc())
    solve_M_v = factorized(M_v.tocsc())

    logger.info("M prefactorized")

    dof_u_all = np.zeros((num_snaps, num_gfem_nodes * dof_node), dtype=np.float32)
    dof_v_all = np.zeros((num_snaps, num_gfem_nodes * dof_node), dtype=np.float32)

    logger.info("Looping through snapshots, solving for dofs")
    loops = num_snaps // batch_size + (1 if num_snaps % batch_size else 0)
    total_time = 0.0

    for ib in tqdm(range(loops)):
        t0 = time.time()
        snap_start = ib * batch_size
        snap_end = min((ib + 1) * batch_size, num_snaps)

        Q_grid_u, Q_grid_v = _read_batch(data_source, field_name, snap_start, snap_end)
        if nx_e > nx or ny_e > ny:
            # outside the domain: zeros, ignored by zero weights, or (cut_weight > 0)
            # the edge-replicated extension the small weights fit
            pad = ((0, nx_e - nx), (0, ny_e - ny), (0, 0))
            if not cut_weight:
                kw = {'mode': 'constant'}
            elif cut_extension == 'odd':
                kw = {'mode': 'reflect', 'reflect_type': 'odd'}
            else:
                kw = {'mode': 'edge'}
            Q_grid_u, Q_grid_v = np.pad(Q_grid_u, pad, **kw), np.pad(Q_grid_v, pad, **kw)

        L_u = np.zeros((num_gfem_nodes * dof_node, snap_end - snap_start))
        L_v = np.zeros((num_gfem_nodes * dof_node, snap_end - snap_start))

        e = 0
        for i in range(nx_g - 1):
            for j in range(ny_g - 1):
                    lltogl = build_lltogl(i, j, ny_g, dof_node, IJK)
                    # the data take the element's own weights, as its mass matrix does
                    # (all ones except on cut elements)
                    mask_e = masks[e][:, None]
                    e += 1

                    indx_cell = i * nskip_x
                    indy_cell = j * nskip_y

                    Q_local_u = Q_grid_u[
                        indx_cell : indx_cell + nskip_x + 1,
                        indy_cell : indy_cell + nskip_y + 1,
                        :,
                    ]
                    Q_local_v = Q_grid_v[
                        indx_cell : indx_cell + nskip_x + 1,
                        indy_cell : indy_cell + nskip_y + 1,
                        :,
                    ]

                    q_rows = (nskip_x + 1) * (nskip_y + 1)
                    q_cols = snap_end - snap_start
                    Q_local_u_vec = np.zeros((q_rows, q_cols))
                    Q_local_v_vec = np.zeros((q_rows, q_cols))

                    for ix in range(nskip_x + 1):
                        for iy in range(nskip_y + 1):
                            iind = iy * (nskip_x + 1) + ix
                            Q_local_u_vec[iind, :] = Q_local_u[ix, iy, :]
                            Q_local_v_vec[iind, :] = Q_local_v[ix, iy, :]

                    L_u[lltogl, :] += modemat_local_wt_u.T @ (Q_local_u_vec * mask_e)
                    L_v[lltogl, :] += modemat_local_wt_v.T @ (Q_local_v_vec * mask_e)

        dof_u_all[snap_start:snap_end, :] = solve_M_u(L_u).T.astype(np.float32)
        dof_v_all[snap_start:snap_end, :] = solve_M_v(L_v).T.astype(np.float32)

        total_time += time.time() - t0

    logger.info(f"Solved for dofs in {total_time:.2f} seconds")

    if latent_target is None:
        return config, dof_u_all, dof_v_all
    elif isinstance(latent_target, str):
        with h5py.File(latent_target, "w") as dof_file:
            grp = dof_file.create_group(group_name if group_name else "dofs")
            grp.create_dataset("dof_u", data=dof_u_all, dtype="float32")
            grp.create_dataset("dof_v", data=dof_v_all, dtype="float32")
        return config
    else:
        latent_target["dof_u"] = dof_u_all
        latent_target["dof_v"] = dof_v_all

        return config

    

    


def gfem_recon_flexible(
    rec_target: Optional[Union[str, Dict[str, np.ndarray]]],
    config: dls_long_Config_Flexible,
    dof_u: Union[str, np.ndarray],
    dof_v: Optional[np.ndarray] = None,
    batch_size: int = 100,
):
    """Rebuild the field from its dofs as (num_snaps, nx_t, ny_t, num_vars).

    The reconstruction is [x, y] regardless of how the source was stored.
    """
    if isinstance(dof_u, str):
        with h5py.File(dof_u, "r") as f:
            dof_u_arr = f["dof_u"][:]
            dof_v_arr = f["dof_v"][:]

    else:
        if dof_v is None:
            raise ValueError("When dof_u is an array, dof_v and dof_w must also be provided.")
        dof_u_arr = dof_u
        dof_v_arr = dof_v

    num_snaps = dof_u_arr.shape[0]
    num_batches = num_snaps // batch_size + (1 if num_snaps % batch_size else 0)

    q_rec = np.zeros(
        (num_snaps, config.nx_t, config.ny_t, config.num_vars),
        dtype=np.float32,
    )

    t_time = 0.0

    for bid in range(num_batches):
        snap_start = bid * batch_size
        snap_end = min((bid + 1) * batch_size, num_snaps)

        start_time = time.time()
        logger.info(f"Processing batch {bid+1}/{num_batches}, batch size: {batch_size}")

        # vectorized over elements; bit-identical to the former per-element loop
        q_rec[snap_start:snap_end] = decode_fields(config, dof_u_arr[snap_start:snap_end],
                                                   dof_v_arr[snap_start:snap_end])
        t_time += time.time() - start_time

        end_time = time.time()
        batch_time = end_time - start_time
        msg = f"Batch {bid+1}/{num_batches} processed in {batch_time:.2f}s"
        if bid + 1 != num_batches:
            proj_time = (num_batches - (bid + 1)) * batch_time / 60
            proj_time_str = f"{int(proj_time)}m {int((proj_time - int(proj_time)) * 60)}s"
            msg += f" -> Proj. time: {proj_time_str}"
        logger.info(msg)

    logger.info(f"Total reconstruction time not with saving to disk: {t_time:.2f}s")

    if rec_target is None:
        return q_rec

    if isinstance(rec_target, str):
        with h5py.File(rec_target, "w") as rec_file:
            rec_file.create_dataset("Q_rec", data=q_rec, dtype="float32")
    else:
        rec_target["Q_rec"] = q_rec

    return q_rec
