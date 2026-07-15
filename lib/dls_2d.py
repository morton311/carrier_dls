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

    def __post_init__(self) -> None:
        self.nskip = (self.patch_size - 1) // 2
        self.nskip_sample = self.patch_size - 1
        self.mid_pt = 1 + self.nskip_sample // 2
        self.sample_x = range(0, self.nx, self.nskip)
        self.sample_y = range(0, self.ny, self.nskip)
        self.nx_t = max(self.sample_x) + 1
        self.ny_t = max(self.sample_y) + 1
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
                + self.num_vars * self.num_modes * self.patch_size**2
            )
        )


def node_map(opp=False) -> np.ndarray:
    # shape: (2, 4)
    IJK = np.array([[0, 1, 1, 0], [1, 1, 0, 0]])
    return IJK


def _build_wt_vec(nskip: int) -> np.ndarray:
    Wt = np.ones((nskip+1, nskip+1))
    Wt[1:-1,0] = 1/2
    Wt[1:-1,-1] = 1/2
    Wt[0,1:-1] = 1/2
    Wt[-1,1:-1] = 1/2
    Wt[0,0] = 1/4
    Wt[0,-1] = 1/4
    Wt[-1,0] = 1/4
    Wt[-1,-1] = 1/4

    Wt_vec = Wt.reshape((nskip+1)**2, order='F')
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


def _source_metadata(data_source: DataInput, field_name: str) -> Tuple[int, int, int, int, int]:
    if isinstance(data_source, str):
        with h5py.File(data_source, "r") as f:
            shape = f[field_name].shape
        return shape[0], shape[1], shape[2], shape[3]

    _validate_data_dict(data_source, field_name)
    shape = data_source[field_name].shape
    return shape[0], shape[1], shape[2], shape[3]


def _read_static(data_source: DataInput, field_name: str):
    if isinstance(data_source, str):
        with h5py.File(data_source, "r") as f:
            mode_data = f[field_name][0, :, :, :] - f["mean"][:]
            grid_x = f["x_grid"][:]
            grid_y = f["y_grid"][:]
        return mode_data, grid_x, grid_y
    mode_data = data_source[field_name][0, :, :, :] - data_source["mean"]
    return mode_data, data_source["x_grid"], data_source["y_grid"]


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

    u_mean = mean[:, :, 0]
    v_mean = mean[:, :, 1]

    q_u = q_u.transpose(1, 2, 0) - u_mean[:, :, np.newaxis]
    q_v = q_v.transpose(1, 2, 0) - v_mean[:, :, np.newaxis]
    return q_u, q_v

def random_patch_sampling(data, patch_size):
    num_patches = 10000
    num_images = 1
    ndim = data.ndim 
    nx = data.shape[0]
    ny = data.shape[1]
    sz = patch_size
    BUFF = 0
    totalsamples = 0
    X = np.zeros((sz ** ndim, num_patches))
    
    for i in range(num_images):
        this_image = data

        # Determine how many patches to take
        getsample = num_patches // num_images
        if i == num_images - 1:
            getsample = num_patches - totalsamples

        # Extract patches at random from this image to make data vector X
        for j in range(getsample):
            d1 = BUFF + np.random.randint(0, nx - sz - 2 * BUFF)
            d2 = BUFF + np.random.randint(0, ny - sz - 2 * BUFF)
            
            totalsamples += 1
            temp = this_image[d1:d1 + sz, d2:d2 + sz].reshape(sz ** ndim, order='F')
            X[:, totalsamples - 1] = temp - np.mean(temp)

    
    return X


def Modal_decomp(data, patch_size):
    data_shape = 'fat' if data.shape[0] < data.shape[1] else 'tall' # not yet implemented
    P = random_patch_sampling(data, patch_size)
    local_modes, eigVal, _ = np.linalg.svd(P, full_matrices=False)
    return local_modes, eigVal


def FEM_shape_calculator_ortho_gfemlr(x, y, xpt, ypt):
    sumxpt = np.sum(xpt) / 4
    sumypt = np.sum(ypt) / 4

    dxpt = (-xpt[0] + xpt[1] + xpt[2] - xpt[3]) / 2
    dypt = (ypt[0] + ypt[1] - ypt[2] - ypt[3]) / 2

    zeta_i = [-1, 1, 1, -1]
    eta_i = [1, 1, -1, -1]

    # Inverse transform for parallelogram elements, bilinear shape functions
    zeta = 2 * (x - sumxpt) / dxpt
    eta = 2 * (y - sumypt) / dypt

    N = np.zeros((4,1))
    # shape function values
    for i in range(4):
        N[i] = (1 / 4) * (1 + zeta_i[i] * zeta) * (1 + eta_i[i] * eta)
    return N


def local_modemat_over_elem(x_grid, y_grid, nskip, modes_vec, num_modes, patch_size):

    nskip = (patch_size - 1) // 2
    nskip_sample = patch_size - 1
    mid_pt = 1 + nskip_sample // 2

    dof_node = num_modes + 1
    dof_elem = 4 * dof_node

    modes_grid = np.zeros((patch_size, patch_size, num_modes))
    for i in range(num_modes):
        modes_grid[:,:, i] = modes_vec[:, i].reshape(patch_size, patch_size, order='F')

    # adopted nodal order
    gfem_nodal_order = [1, 2, 3, 4]
    diag_opp_indx =    [3, 4, 1, 2]

    # Mode grid components for the four quadrants
    F_indx = [1, 2, 3, 4]  # faces order, F1:x-ve, F2:x+ve, F3:y-ve, F4:y+ve
    F1 = list(range(0, mid_pt))
    F2 = list(range(mid_pt - 1, patch_size))
    F3 = list(range(0, mid_pt))
    F4 = list(range(mid_pt - 1, patch_size))

    indices = [
        (F1, F4), # 1
        (F2, F4), # 2
        (F2, F3), # 3
        (F1, F3), # 4
    ]

    modes_grid_1_comp = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_2_comp = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_3_comp = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_4_comp = np.zeros((mid_pt, mid_pt, num_modes))

    comp1_x, comp1_y = np.meshgrid(F1, F4, indexing='ij')
    comp2_x, comp2_y = np.meshgrid(F2, F4, indexing='ij')
    comp3_x, comp3_y = np.meshgrid(F2, F3, indexing='ij')
    comp4_x, comp4_y = np.meshgrid(F1, F3, indexing='ij')

    for i in range(num_modes):
        modes_grid_1_comp[:,:, i] = modes_grid[comp1_x, comp1_y, i]
        modes_grid_2_comp[:,:, i] = modes_grid[comp2_x, comp2_y, i]
        modes_grid_3_comp[:,:, i] = modes_grid[comp3_x, comp3_y, i]
        modes_grid_4_comp[:,:, i] = modes_grid[comp4_x, comp4_y, i]


    modes_vec_comp1 = modes_grid_1_comp.reshape(mid_pt**2, num_modes, order='F')
    modes_vec_comp2 = modes_grid_2_comp.reshape(mid_pt**2, num_modes, order='F')
    modes_vec_comp3 = modes_grid_3_comp.reshape(mid_pt**2, num_modes, order='F')
    modes_vec_comp4 = modes_grid_4_comp.reshape(mid_pt**2, num_modes, order='F')


    # local modal matrix for single element
    i, j = 1, 1

    x1 = x_grid[ i   *nskip,  (j-1)*nskip]
    x2 = x_grid[ i   *nskip,   j   *nskip]
    x3 = x_grid[(i-1)*nskip,   j   *nskip]
    x4 = x_grid[(i-1)*nskip,  (j-1)*nskip]

    y1 = y_grid[ i   *nskip,  (j-1)*nskip]
    y2 = y_grid[ i   *nskip,   j   *nskip]
    y3 = y_grid[(i-1)*nskip,   j   *nskip]
    y4 = y_grid[(i-1)*nskip,  (j-1)*nskip]

    xpt = [x1, x2, x3, x4]
    ypt = [y1, y2, y3, y4]

    N1 = np.zeros(((nskip+1)**2))
    N2 = np.zeros(((nskip+1)**2))
    N3 = np.zeros(((nskip+1)**2))
    N4 = np.zeros(((nskip+1)**2))

    for kx in range((nskip+1)):
        
        for ky in range((nskip+1)):
            
            x_val = x_grid[ky, kx]
            y_val = y_grid[ky, kx]
            iind = ky*(nskip+1) + kx
            N = FEM_shape_calculator_ortho_gfemlr(x_val, y_val, xpt, ypt)
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

    Wt_vec = _build_wt_vec(nskip)

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
    dls_config: Optional[dls_long_Config_Flexible] = None
):
    """
    Flexible variant of gfem_3d_long.

    data_source:
      - str path to an HDF5 file
      - dict with in-memory arrays keyed by: field_name, mean, x_grid, y_grid, z_grid

    latent_target:
      - None: returns dof arrays in-memory
      - str path: writes dof_u/dof_v/dof_w to an HDF5 file
      - dict: stores dof arrays in provided dict keys dof_u/dof_v/dof_w
    """
    ndim = 2
    num_snaps, nx, ny, _ = _source_metadata(data_source, field_name)
    mode_data, grid_x, grid_y = _read_static(data_source, field_name)

    nskip = (patch_size - 1) // 2
    sample_x = range(0, nx, nskip)
    sample_y = range(0, ny, nskip)

    nx_g = len(sample_x)
    ny_g = len(sample_y)

    num_gfem_nodes = nx_g * ny_g
    dof_node = num_modes + 1
    dof_elem = 4 * dof_node

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
            grid_x, grid_y, nskip, local_modes_u, num_modes, patch_size
        )
        modemat_local_v, modemat_local_wt_v = local_modemat_over_elem(
            grid_x, grid_y, nskip, local_modes_v, num_modes, patch_size
        )

        logger.info("Local modal matrices constructed")
    else:
        modemat_local_u = dls_config.modemat_local_u
        modemat_local_v = dls_config.modemat_local_v

        logger.info("Using provided local modal matrices from dls_config")

        Wt_vec = _build_wt_vec(nskip)
        modemat_local_wt_u = modemat_local_u.copy()
        modemat_local_wt_v = modemat_local_v.copy()

        for kk in range(dof_elem):
            modemat_local_wt_u[:, kk] *= Wt_vec
            modemat_local_wt_v[:, kk] *= Wt_vec

        logger.info("Constructed weighted local modal matrices from provided local modal matrices")


    # With a uniform/structured mesh in this formulation, one local mass matrix is reused.
    M_local_u = modemat_local_wt_u.T @ modemat_local_u
    M_local_v = modemat_local_wt_v.T @ modemat_local_v


    M_u = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_v = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))

    IJK = node_map()

    logger.info("Constructing global M GFEM matrices")
    for i in range(nx_g - 1):
        for j in range(ny_g - 1):
            lltogl = build_lltogl(i, j, ny_g, dof_node, IJK)
            M_u[np.ix_(lltogl, lltogl)] += M_local_u
            M_v[np.ix_(lltogl, lltogl)] += M_local_v

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

        L_u = np.zeros((num_gfem_nodes * dof_node, snap_end - snap_start))
        L_v = np.zeros((num_gfem_nodes * dof_node, snap_end - snap_start))

        for i in range(nx_g - 1):
            for j in range(ny_g - 1):
                    lltogl = build_lltogl(i, j, ny_g, dof_node, IJK)

                    indx_cell = i * nskip
                    indy_cell = j * nskip

                    Q_local_u = Q_grid_u[
                        indx_cell : indx_cell + nskip + 1,
                        indy_cell : indy_cell + nskip + 1,
                        :,
                    ]
                    Q_local_v = Q_grid_v[
                        indx_cell : indx_cell + nskip + 1,
                        indy_cell : indy_cell + nskip + 1,
                        :,
                    ]

                    q_rows = (nskip + 1) ** 2
                    q_cols = snap_end - snap_start
                    Q_local_u_vec = np.zeros((q_rows, q_cols))
                    Q_local_v_vec = np.zeros((q_rows, q_cols))

                    for kx in range(nskip + 1):
                        for ky in range(nskip + 1):
                            iind = ky * (nskip + 1) + kx
                            Q_local_u_vec[iind, :] = Q_local_u[kx, ky, :]
                            Q_local_v_vec[iind, :] = Q_local_v[kx, ky, :]

                    L_u[lltogl, :] += modemat_local_wt_u.T @ Q_local_u_vec
                    L_v[lltogl, :] += modemat_local_wt_v.T @ Q_local_v_vec

        dof_u_all[snap_start:snap_end, :] = solve_M_u(L_u).T.astype(np.float32)
        dof_v_all[snap_start:snap_end, :] = solve_M_v(L_v).T.astype(np.float32)

        total_time += time.time() - t0

    logger.info(f"Solved for dofs in {total_time:.2f} seconds")

    config = dls_long_Config_Flexible(
        num_snaps=num_snaps,
        nx=nx,
        ny=ny,
        num_vars=ndim,
        patch_size=patch_size,
        num_modes=num_modes,
        modemat_local_u=modemat_local_u,
        modemat_local_v=modemat_local_v
    )

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

    IJK = node_map()
    nskip = config.nskip
    dof_node = config.dof_node
    dof_elem = config.dof_elem

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

        Q_rec_u = np.zeros((config.nx_t, config.ny_t, snap_end - snap_start))
        Q_rec_v = np.zeros((config.nx_t, config.ny_t, snap_end - snap_start))

        for i in range(config.nx_g - 1):
            for j in range(config.ny_g - 1):
                lltogl = build_lltogl(i, j, config.ny_g, dof_node, IJK)
                indx_cell = i * nskip
                indy_cell = j * nskip

                dof_local_u = dof_u_arr[snap_start:snap_end, :][:, lltogl].T
                dof_local_v = dof_v_arr[snap_start:snap_end, :][:, lltogl].T

                Q_local_u_vec = config.modemat_local_u @ dof_local_u
                Q_local_v_vec = config.modemat_local_v @ dof_local_v

                Q_local_u = Q_local_u_vec.reshape((nskip + 1, nskip + 1, snap_end - snap_start), order="F")
                Q_local_v = Q_local_v_vec.reshape((nskip + 1, nskip + 1, snap_end - snap_start), order="F")

                Q_rec_u[indx_cell:indx_cell + nskip + 1, indy_cell:indy_cell + nskip + 1, :] = Q_local_u
                Q_rec_v[indx_cell:indx_cell + nskip + 1, indy_cell:indy_cell + nskip + 1, :] = Q_local_v
                compute_time = time.time() 
                t_time += compute_time - start_time

        q_rec[snap_start:snap_end, :, :, 0] = Q_rec_u.transpose(2, 0, 1)
        q_rec[snap_start:snap_end, :, :, 1] = Q_rec_v.transpose(2, 0, 1)

        end_time = time.time()
        batch_time = end_time - start_time
        msg = f"Batch {bid+1}/{num_batches} processed in {batch_time:.2f}s"
        if bid + 1 != num_batches:
            proj_time = (num_batches - (bid + 1)) * batch_time / 60
            proj_time_str = f"{int(proj_time)}m {int((proj_time - int(proj_time)) * 60)}s"
            msg += f" -> Proj. time: {proj_time_str}"
        logger.info(msg)

    if batch_time < 1:
        t_time = t_time/1000  # convert from ms if under 1s
    logger.info(f"Total reconstruction time not with saving to disk: {t_time:.2f}s")

    if rec_target is None:
        return q_rec

    if isinstance(rec_target, str):
        with h5py.File(rec_target, "w") as rec_file:
            rec_file.create_dataset("Q_rec", data=q_rec, dtype="float32")
    else:
        rec_target["Q_rec"] = q_rec

    return q_rec
