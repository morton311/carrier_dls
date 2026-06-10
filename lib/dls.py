import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized
from tqdm import tqdm


ArrayDict = Dict[str, np.ndarray]
DataInput = Union[str, ArrayDict]


@dataclass
class dls_long_Config_Flexible:
    num_snaps: int
    nx: int
    ny: int
    nz: int
    num_vars: int
    patch_size: int
    num_modes: int
    modemat_local_u: np.ndarray
    modemat_local_v: np.ndarray
    modemat_local_w: np.ndarray

    def __post_init__(self) -> None:
        self.nskip = (self.patch_size - 1) // 2
        self.nskip_sample = self.patch_size - 1
        self.mid_pt = 1 + self.nskip_sample // 2
        self.sample_x = range(0, self.nx, self.nskip)
        self.sample_y = range(0, self.ny, self.nskip)
        self.sample_z = range(0, self.nz, self.nskip)
        self.nx_t = max(self.sample_x) + 1
        self.ny_t = max(self.sample_y) + 1
        self.nz_t = max(self.sample_z) + 1
        self.nx_g = len(self.sample_x)
        self.ny_g = len(self.sample_y)
        self.nz_g = len(self.sample_z)
        self.num_gfem_nodes = self.nx_g * self.ny_g * self.nz_g
        self.num_gfem_elems = (self.nx_g - 1) * (self.ny_g - 1) * (self.nz_g - 1)
        self.dof_node = self.num_modes + 1
        self.dof_elem = 8 * self.dof_node
        self.compression_ratio = (
            self.num_vars * self.num_snaps * self.nx * self.ny * self.nz
            / (
                self.num_vars * self.num_snaps * self.dof_node
                + self.num_vars * self.num_modes * self.patch_size**3
            )
        )


def node_map(opp=False) -> np.ndarray:
    # shape: (3, 8)
    IJK = np.array([
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [0, 0, 0]
        ]).T
    if opp:
        diag_opp_indx = [6, 7, 4, 5, 2, 3, 0, 1]
        IJK = IJK[:, diag_opp_indx]
    return IJK


def _build_wt_vec(nskip: int) -> np.ndarray:
    Wt = np.ones((nskip+1, nskip+1, nskip+1))
    
    # Face centers: 1/2
    for i in range(nskip+1):
        for j in range(nskip+1):
            Wt[i, j, 0] = Wt[i, j, nskip] = 1/2
            Wt[0, i, j] = Wt[nskip, i, j] = 1/2
            Wt[i, 0, j] = Wt[i, nskip, j] = 1/2
    
    # Edge centers: 1/4
    edges = [(0, 0), 
             (0, nskip), 
             (nskip, 0), 
             (nskip, nskip)]
    for i in range(nskip+1):
        for a, b in edges:
            Wt[i, a, b] = Wt[a, i, b] = Wt[a, b, i] = 1/4
    
    # Corner vertices: 1/8
    corners = [(0, 0, 0), 
               (nskip, 0, 0), 
               (0, nskip, 0), 
               (0, 0, nskip),
               (nskip, nskip, 0), 
               (nskip, 0, nskip), 
               (0, nskip, nskip), 
               (nskip, nskip, nskip)]
    for corner in corners:
        Wt[corner] = 1/8
    
    Wt_vec = Wt.reshape((nskip+1)**3, order='F')
    return Wt_vec

def build_lltogl(
    i: int,
    j: int,
    k: int,
    ny_g: int,
    nz_g: int,
    dof_node: int,
    IJK: np.ndarray,
) -> np.ndarray:
    lltogl = np.zeros(8 * dof_node, dtype=int)
    for node in range(8):
        indx_dof_start = (
            (i + IJK[0, node]) * ny_g * nz_g
            + (j + IJK[1, node]) * nz_g
            + (k + IJK[2, node]) 
        ) * dof_node
        idx_dof_end = indx_dof_start + dof_node
        lltogl[node * dof_node : (node + 1) * dof_node] = np.arange(indx_dof_start, idx_dof_end)
    return lltogl


def _validate_data_dict(data_dict: ArrayDict, field_name: str) -> None:
    required = [field_name, "mean", "x_grid", "y_grid", "z_grid"]
    missing = [k for k in required if k not in data_dict]
    if missing:
        raise KeyError(f"Missing required keys for in-memory input: {missing}")


def _source_metadata(data_source: DataInput, field_name: str) -> Tuple[int, int, int, int, int]:
    if isinstance(data_source, str):
        with h5py.File(data_source, "r") as f:
            shape = f[field_name].shape
        return shape[0], shape[1], shape[2], shape[3], shape[4]

    _validate_data_dict(data_source, field_name)
    shape = data_source[field_name].shape
    return shape[0], shape[1], shape[2], shape[3], shape[4]


def _read_static(data_source: DataInput, field_name: str):
    if isinstance(data_source, str):
        with h5py.File(data_source, "r") as f:
            mode_data = f[field_name][0, :, :, :, :] - f["mean"][:]
            grid_x = f["x_grid"][:]
            grid_y = f["y_grid"][:]
            grid_z = f["z_grid"][:]
        return mode_data, grid_x, grid_y, grid_z

    mode_data = data_source[field_name][0, :, :, :, :] - data_source["mean"]
    return mode_data, data_source["x_grid"], data_source["y_grid"], data_source["z_grid"]


def _read_batch(data_source: DataInput, field_name: str, snap_start: int, snap_end: int):
    if isinstance(data_source, str):
        with h5py.File(data_source, "r") as f:
            mean = f["mean"][:]
            q_u = f[field_name][snap_start:snap_end, :, :, :, 0]
            q_v = f[field_name][snap_start:snap_end, :, :, :, 1]
            q_w = f[field_name][snap_start:snap_end, :, :, :, 2]
    else:
        mean = data_source["mean"]
        q_u = data_source[field_name][snap_start:snap_end, :, :, :, 0]
        q_v = data_source[field_name][snap_start:snap_end, :, :, :, 1]
        q_w = data_source[field_name][snap_start:snap_end, :, :, :, 2]

    u_mean = mean[:, :, :, 0]
    v_mean = mean[:, :, :, 1]
    w_mean = mean[:, :, :, 2]

    q_u = q_u.transpose(1, 2, 3, 0) - u_mean[:, :, :, np.newaxis]
    q_v = q_v.transpose(1, 2, 3, 0) - v_mean[:, :, :, np.newaxis]
    q_w = q_w.transpose(1, 2, 3, 0) - w_mean[:, :, :, np.newaxis]
    return q_u, q_v, q_w

def random_patch_sampling(data, patch_size):
    num_patches = 1000
    num_images = 1
    ndim = data.ndim 
    nx = data.shape[0]
    ny = data.shape[1]
    nz = data.shape[2] if ndim == 3 else None
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
            d3 = BUFF + np.random.randint(0, nz - sz - 2 * BUFF) if ndim == 3 else None
            
            totalsamples += 1
            if ndim == 3:
                temp = this_image[d1:d1 + sz, d2:d2 + sz, d3:d3 + sz].reshape(sz ** ndim, order='F')
            else:
                temp = this_image[d1:d1 + sz, d2:d2 + sz].reshape(sz ** ndim, order='F')
            X[:, totalsamples - 1] = temp - np.mean(temp)

    
    return X


def Modal_decomp(data, patch_size):
    data_shape = 'fat' if data.shape[0] < data.shape[1] else 'tall' # not yet implemented
    P = random_patch_sampling(data, patch_size)
    local_modes, eigVal, _ = np.linalg.svd(P, full_matrices=False)
    return local_modes, eigVal



def FEM_shape_calculator_ortho_gfemlr(x, y, z, xpt, ypt, zpt):

    sumxpt = np.sum(xpt) / 8
    sumypt = np.sum(ypt) / 8
    sumzpt = np.sum(zpt) / 8

    dxpt = (xpt[0] + xpt[1] + xpt[2] + xpt[3] - xpt[4] - xpt[5] - xpt[6] - xpt[7]) / 4
    dypt = (-ypt[0] + ypt[1] + ypt[2] - ypt[3] - ypt[4] + ypt[5] + ypt[6] - ypt[7]) / 4
    dzpt = (zpt[0] + zpt[1] - zpt[2] - zpt[3] + zpt[4] + zpt[5] - zpt[6] - zpt[7]) / 4

    zeta_i = [ 1, 1, 1, 1, -1, -1, -1, -1]
    eta_i = [-1, 1, 1, -1, -1, 1, 1, -1]
    phi_i = [ 1, 1, -1, -1, 1, 1, -1, -1]

    # Inverse transform for parallelepiped elements, trilinear shape functions
    zeta = 2 * (x - sumxpt) / dxpt
    eta = 2 * (y - sumypt) / dypt
    phi = 2 * (z - sumzpt) / dzpt

    N = np.zeros((8,1))
    # shape function values
    for i in range(8):
        N[i] = (1 / 8) * (1 + zeta_i[i] * zeta) * (1 + eta_i[i] * eta) * (1 + phi_i[i] * phi)
    return N

def local_modemat_over_elem(x_grid, y_grid, z_grid, nskip, modes_vec, num_modes, patch_size):

    nskip = (patch_size - 1) // 2
    nskip_sample = patch_size - 1
    mid_pt = 1 + nskip_sample // 2

    dof_node = num_modes + 1
    dof_elem = 8 * dof_node

    modes_grid = np.zeros((patch_size, patch_size, patch_size, num_modes))
    for i in range(num_modes):
        modes_grid[:,:,:, i] = modes_vec[:, i].reshape(patch_size, patch_size, patch_size, order='F')

    # adopted nodal order
    gfem_nodal_order = [1, 2, 3, 4, 5, 6, 7, 8]
    diag_opp_indx =    [7, 8, 5, 6, 3, 4, 1, 2]

    # Mode grid components for the four quadrants
    F_indx = [1, 2, 3, 4, 5, 6]  # faces order, F1:x-ve, F2:x+ve, F3:y-ve, F4:y+ve, F5:z-ve, F6:z+ve
    F1 = list(range(0, mid_pt))
    F2 = list(range(mid_pt - 1, patch_size))
    F3 = list(range(0, mid_pt))
    F4 = list(range(mid_pt - 1, patch_size))
    F5 = list(range(0, mid_pt))
    F6 = list(range(mid_pt - 1, patch_size))

    indices = [
        (F2, F3, F6), # 1
        (F2, F4, F6), # 2
        (F2, F4, F5), # 3
        (F2, F3, F5), # 4
        (F1, F3, F6), # 5
        (F1, F4, F6), # 6
        (F1, F4, F5), # 7
        (F1, F3, F5)  # 8
    ]

    modes_grid_1_comp = np.zeros((mid_pt, mid_pt, mid_pt, num_modes))
    modes_grid_2_comp = np.zeros((mid_pt, mid_pt, mid_pt, num_modes))
    modes_grid_3_comp = np.zeros((mid_pt, mid_pt, mid_pt, num_modes))
    modes_grid_4_comp = np.zeros((mid_pt, mid_pt, mid_pt, num_modes))
    modes_grid_5_comp = np.zeros((mid_pt, mid_pt, mid_pt, num_modes))
    modes_grid_6_comp = np.zeros((mid_pt, mid_pt, mid_pt, num_modes))
    modes_grid_7_comp = np.zeros((mid_pt, mid_pt, mid_pt, num_modes))
    modes_grid_8_comp = np.zeros((mid_pt, mid_pt, mid_pt, num_modes))

    comp1_x, comp1_y, comp1_z = np.meshgrid(F2, F3, F6, indexing='ij')
    comp2_x, comp2_y, comp2_z = np.meshgrid(F2, F4, F6, indexing='ij')
    comp3_x, comp3_y, comp3_z = np.meshgrid(F2, F4, F5, indexing='ij')
    comp4_x, comp4_y, comp4_z = np.meshgrid(F2, F3, F5, indexing='ij')
    comp5_x, comp5_y, comp5_z = np.meshgrid(F1, F3, F6, indexing='ij')
    comp6_x, comp6_y, comp6_z = np.meshgrid(F1, F4, F6, indexing='ij')
    comp7_x, comp7_y, comp7_z = np.meshgrid(F1, F4, F5, indexing='ij')
    comp8_x, comp8_y, comp8_z = np.meshgrid(F1, F3, F5, indexing='ij')

    for i in range(num_modes):
        modes_grid_1_comp[:,:,:, i] = modes_grid[comp1_x, comp1_y, comp1_z, i]
        modes_grid_2_comp[:,:,:, i] = modes_grid[comp2_x, comp2_y, comp2_z, i]
        modes_grid_3_comp[:,:,:, i] = modes_grid[comp3_x, comp3_y, comp3_z, i]
        modes_grid_4_comp[:,:,:, i] = modes_grid[comp4_x, comp4_y, comp4_z, i]
        modes_grid_5_comp[:,:,:, i] = modes_grid[comp5_x, comp5_y, comp5_z, i]
        modes_grid_6_comp[:,:,:, i] = modes_grid[comp6_x, comp6_y, comp6_z, i]
        modes_grid_7_comp[:,:,:, i] = modes_grid[comp7_x, comp7_y, comp7_z, i]
        modes_grid_8_comp[:,:,:, i] = modes_grid[comp8_x, comp8_y, comp8_z, i]


    modes_vec_comp1 = modes_grid_1_comp.reshape(mid_pt**3, num_modes, order='F')
    modes_vec_comp2 = modes_grid_2_comp.reshape(mid_pt**3, num_modes, order='F')
    modes_vec_comp3 = modes_grid_3_comp.reshape(mid_pt**3, num_modes, order='F')
    modes_vec_comp4 = modes_grid_4_comp.reshape(mid_pt**3, num_modes, order='F')
    modes_vec_comp5 = modes_grid_5_comp.reshape(mid_pt**3, num_modes, order='F')
    modes_vec_comp6 = modes_grid_6_comp.reshape(mid_pt**3, num_modes, order='F')
    modes_vec_comp7 = modes_grid_7_comp.reshape(mid_pt**3, num_modes, order='F')
    modes_vec_comp8 = modes_grid_8_comp.reshape(mid_pt**3, num_modes, order='F')


    # local modal matrix for single element
    i, j, k = 1, 1, 1

    x1 = x_grid[ i   *nskip,  (j-1)*nskip,  k   *nskip]
    x2 = x_grid[ i   *nskip,   j   *nskip,  k   *nskip]
    x3 = x_grid[ i   *nskip,   j   *nskip, (k-1)*nskip]
    x4 = x_grid[ i   *nskip,  (j-1)*nskip, (k-1)*nskip]
    x5 = x_grid[(i-1)*nskip,  (j-1)*nskip,  k   *nskip]
    x6 = x_grid[(i-1)*nskip,   j   *nskip,  k   *nskip]
    x7 = x_grid[(i-1)*nskip,   j   *nskip, (k-1)*nskip]
    x8 = x_grid[(i-1)*nskip,  (j-1)*nskip, (k-1)*nskip]

    y1 = y_grid[ i   *nskip,  (j-1)*nskip,  k   *nskip]
    y2 = y_grid[ i   *nskip,   j   *nskip,  k   *nskip]
    y3 = y_grid[ i   *nskip,   j   *nskip, (k-1)*nskip]
    y4 = y_grid[ i   *nskip,  (j-1)*nskip, (k-1)*nskip]
    y5 = y_grid[(i-1)*nskip,  (j-1)*nskip,  k   *nskip]
    y6 = y_grid[(i-1)*nskip,   j   *nskip,  k   *nskip]
    y7 = y_grid[(i-1)*nskip,   j   *nskip, (k-1)*nskip]
    y8 = y_grid[(i-1)*nskip,  (j-1)*nskip, (k-1)*nskip]

    z1 = z_grid[ i   *nskip,  (j-1)*nskip,  k   *nskip]
    z2 = z_grid[ i   *nskip,   j   *nskip,  k   *nskip]
    z3 = z_grid[ i   *nskip,   j   *nskip, (k-1)*nskip]
    z4 = z_grid[ i   *nskip,  (j-1)*nskip, (k-1)*nskip]
    z5 = z_grid[(i-1)*nskip,  (j-1)*nskip,  k   *nskip]
    z6 = z_grid[(i-1)*nskip,   j   *nskip,  k   *nskip]
    z7 = z_grid[(i-1)*nskip,   j   *nskip, (k-1)*nskip]
    z8 = z_grid[(i-1)*nskip,  (j-1)*nskip, (k-1)*nskip]

    xpt = [x1, x2, x3, x4, x5, x6, x7, x8]
    ypt = [y1, y2, y3, y4, y5, y6, y7, y8]
    zpt = [z1, z2, z3, z4, z5, z6, z7, z8]

    N1 = np.zeros(((nskip+1)**3))
    N2 = np.zeros(((nskip+1)**3))
    N3 = np.zeros(((nskip+1)**3))
    N4 = np.zeros(((nskip+1)**3))
    N5 = np.zeros(((nskip+1)**3))
    N6 = np.zeros(((nskip+1)**3))
    N7 = np.zeros(((nskip+1)**3))
    N8 = np.zeros(((nskip+1)**3))

    for kx in range((nskip+1)):
        
        for ky in range((nskip+1)):

            for kz in range((nskip+1)):
                
                x_val = x_grid[kx, ky, kz]
                y_val = y_grid[kx, ky, kz]
                z_val = z_grid[kx, ky, kz]
                iind = kz*(nskip+1)**2 + ky*(nskip+1) + kx
                N = FEM_shape_calculator_ortho_gfemlr(x_val, y_val, z_val, xpt, ypt, zpt)
                N1[iind] = N[0][0]
                N2[iind] = N[1][0]
                N3[iind] = N[2][0]
                N4[iind] = N[3][0]
                N5[iind] = N[4][0]
                N6[iind] = N[5][0]
                N7[iind] = N[6][0]
                N8[iind] = N[7][0]

    modemat_local = np.hstack([
        N1[:, np.newaxis], N1[:, np.newaxis]*modes_vec_comp7,
        N2[:, np.newaxis], N2[:, np.newaxis]*modes_vec_comp8,
        N3[:, np.newaxis], N3[:, np.newaxis]*modes_vec_comp5,
        N4[:, np.newaxis], N4[:, np.newaxis]*modes_vec_comp6,
        N5[:, np.newaxis], N5[:, np.newaxis]*modes_vec_comp3,
        N6[:, np.newaxis], N6[:, np.newaxis]*modes_vec_comp4,
        N7[:, np.newaxis], N7[:, np.newaxis]*modes_vec_comp1,
        N8[:, np.newaxis], N8[:, np.newaxis]*modes_vec_comp2
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
    ndim = 3
    num_snaps, nx, ny, nz, _ = _source_metadata(data_source, field_name)
    mode_data, grid_x, grid_y, grid_z = _read_static(data_source, field_name)

    nskip = (patch_size - 1) // 2
    sample_x = range(0, nx, nskip)
    sample_y = range(0, ny, nskip)
    sample_z = range(0, nz, nskip)

    nx_g = len(sample_x)
    ny_g = len(sample_y)
    nz_g = len(sample_z)

    num_gfem_nodes = nx_g * ny_g * nz_g
    dof_node = num_modes + 1
    dof_elem = 8 * dof_node

    print("shape of mode data:", mode_data.shape)
    print("number of snapshots:", num_snaps)
    print("number of batches:", num_snaps // batch_size)
    print("nx:", nx)
    print("ny:", ny)
    print("nz:", nz)
    print("num_vars:", ndim)

    if dls_config is None:
        print("Performing modal decomposition to get local modes")
        local_modes_u, _ = Modal_decomp(mode_data[..., 0], patch_size)
        local_modes_v, _ = Modal_decomp(mode_data[..., 1], patch_size)
        local_modes_w, _ = Modal_decomp(mode_data[..., 2], patch_size)
        print("Modal decomposition done")

        print("Constructing local modal matrices")
        modemat_local_u, modemat_local_wt_u = local_modemat_over_elem(
            grid_x, grid_y, grid_z, nskip, local_modes_u, num_modes, patch_size
        )
        modemat_local_v, modemat_local_wt_v = local_modemat_over_elem(
            grid_x, grid_y, grid_z, nskip, local_modes_v, num_modes, patch_size
        )
        modemat_local_w, modemat_local_wt_w = local_modemat_over_elem(
            grid_x, grid_y, grid_z, nskip, local_modes_w, num_modes, patch_size
        )
        print("Local modal matrices constructed")
    else:
        modemat_local_u = dls_config.modemat_local_u
        modemat_local_v = dls_config.modemat_local_v
        modemat_local_w = dls_config.modemat_local_w
        print("Using provided local modal matrices from dls_config")

        Wt_vec = _build_wt_vec(nskip)
        modemat_local_wt_u = modemat_local_u.copy()
        modemat_local_wt_v = modemat_local_v.copy()
        modemat_local_wt_w = modemat_local_w.copy()
        for kk in range(dof_elem):
            modemat_local_wt_u[:, kk] *= Wt_vec
            modemat_local_wt_v[:, kk] *= Wt_vec
            modemat_local_wt_w[:, kk] *= Wt_vec
        print("Constructed weighted local modal matrices from provided local modal matrices")


    # With a uniform/structured mesh in this formulation, one local mass matrix is reused.
    M_local_u = modemat_local_wt_u.T @ modemat_local_u
    M_local_v = modemat_local_wt_v.T @ modemat_local_v
    M_local_w = modemat_local_wt_w.T @ modemat_local_w

    M_u = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_v = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_w = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))

    IJK = node_map()

    print("Constructing global M GFEM matrices")
    for i in range(nx_g - 1):
        for j in range(ny_g - 1):
            for k in range(nz_g - 1):
                lltogl = build_lltogl(i, j, k, ny_g, nz_g, dof_node, IJK)
                M_u[np.ix_(lltogl, lltogl)] += M_local_u
                M_v[np.ix_(lltogl, lltogl)] += M_local_v
                M_w[np.ix_(lltogl, lltogl)] += M_local_w

    print("Prefactorizing M")
    solve_M_u = factorized(M_u.tocsc())
    solve_M_v = factorized(M_v.tocsc())
    solve_M_w = factorized(M_w.tocsc())
    print("M prefactorized")

    dof_u_all = np.zeros((num_snaps, num_gfem_nodes * dof_node), dtype=np.float32)
    dof_v_all = np.zeros((num_snaps, num_gfem_nodes * dof_node), dtype=np.float32)
    dof_w_all = np.zeros((num_snaps, num_gfem_nodes * dof_node), dtype=np.float32)

    print("Looping through snapshots, solving for dofs")
    loops = num_snaps // batch_size + (1 if num_snaps % batch_size else 0)
    total_time = 0.0

    for ib in tqdm(range(loops)):
        t0 = time.time()
        snap_start = ib * batch_size
        snap_end = min((ib + 1) * batch_size, num_snaps)

        Q_grid_u, Q_grid_v, Q_grid_w = _read_batch(data_source, field_name, snap_start, snap_end)

        L_u = np.zeros((num_gfem_nodes * dof_node, snap_end - snap_start))
        L_v = np.zeros((num_gfem_nodes * dof_node, snap_end - snap_start))
        L_w = np.zeros((num_gfem_nodes * dof_node, snap_end - snap_start))

        for i in range(nx_g - 1):
            for j in range(ny_g - 1):
                for k in range(nz_g - 1):
                    lltogl = build_lltogl(i, j, k, ny_g, nz_g, dof_node, IJK)

                    indx_cell = i * nskip
                    indy_cell = j * nskip
                    indz_cell = k * nskip

                    Q_local_u = Q_grid_u[
                        indx_cell : indx_cell + nskip + 1,
                        indy_cell : indy_cell + nskip + 1,
                        indz_cell : indz_cell + nskip + 1,
                        :,
                    ]
                    Q_local_v = Q_grid_v[
                        indx_cell : indx_cell + nskip + 1,
                        indy_cell : indy_cell + nskip + 1,
                        indz_cell : indz_cell + nskip + 1,
                        :,
                    ]
                    Q_local_w = Q_grid_w[
                        indx_cell : indx_cell + nskip + 1,
                        indy_cell : indy_cell + nskip + 1,
                        indz_cell : indz_cell + nskip + 1,
                        :,
                    ]

                    q_rows = (nskip + 1) ** 3
                    q_cols = snap_end - snap_start
                    Q_local_u_vec = np.zeros((q_rows, q_cols))
                    Q_local_v_vec = np.zeros((q_rows, q_cols))
                    Q_local_w_vec = np.zeros((q_rows, q_cols))

                    for kx in range(nskip + 1):
                        for ky in range(nskip + 1):
                            for kz in range(nskip + 1):
                                iind = kz * (nskip + 1) ** 2 + ky * (nskip + 1) + kx
                                Q_local_u_vec[iind, :] = Q_local_u[kx, ky, kz, :]
                                Q_local_v_vec[iind, :] = Q_local_v[kx, ky, kz, :]
                                Q_local_w_vec[iind, :] = Q_local_w[kx, ky, kz, :]

                    L_u[lltogl, :] += modemat_local_wt_u.T @ Q_local_u_vec
                    L_v[lltogl, :] += modemat_local_wt_v.T @ Q_local_v_vec
                    L_w[lltogl, :] += modemat_local_wt_w.T @ Q_local_w_vec

        dof_u_all[snap_start:snap_end, :] = solve_M_u(L_u).T.astype(np.float32)
        dof_v_all[snap_start:snap_end, :] = solve_M_v(L_v).T.astype(np.float32)
        dof_w_all[snap_start:snap_end, :] = solve_M_w(L_w).T.astype(np.float32)

        total_time += time.time() - t0

    print(f"Solved for dofs in {total_time:.2f} seconds")

    config = dls_long_Config_Flexible(
        num_snaps=num_snaps,
        nx=nx,
        ny=ny,
        nz=nz,
        num_vars=ndim,
        patch_size=patch_size,
        num_modes=num_modes,
        modemat_local_u=modemat_local_u,
        modemat_local_v=modemat_local_v,
        modemat_local_w=modemat_local_w,
    )

    if latent_target is None:
        return config, dof_u_all, dof_v_all, dof_w_all
    elif isinstance(latent_target, str):
        with h5py.File(latent_target, "a") as dof_file:
            grp = dof_file.create_group(group_name if group_name else "dofs")
            grp.create_dataset("dof_u", data=dof_u_all, dtype="float32")
            grp.create_dataset("dof_v", data=dof_v_all, dtype="float32")
            grp.create_dataset("dof_w", data=dof_w_all, dtype="float32")
        return config
    else:
        latent_target["dof_u"] = dof_u_all
        latent_target["dof_v"] = dof_v_all
        latent_target["dof_w"] = dof_w_all
        return config

    

    


def gfem_recon_flexible(
    rec_target: Optional[Union[str, Dict[str, np.ndarray]]],
    config: dls_long_Config_Flexible,
    dof_u: Union[str, np.ndarray],
    dof_v: Optional[np.ndarray] = None,
    dof_w: Optional[np.ndarray] = None,
    batch_size: int = 100,
):
    if isinstance(dof_u, str):
        with h5py.File(dof_u, "r") as f:
            dof_u_arr = f["dof_u"][:]
            dof_v_arr = f["dof_v"][:]
            dof_w_arr = f["dof_w"][:]
    else:
        if dof_v is None or dof_w is None:
            raise ValueError("When dof_u is an array, dof_v and dof_w must also be provided.")
        dof_u_arr = dof_u
        dof_v_arr = dof_v
        dof_w_arr = dof_w

    num_snaps = dof_u_arr.shape[0]
    num_batches = num_snaps // batch_size + (1 if num_snaps % batch_size else 0)

    IJK = node_map()
    nskip = config.nskip
    dof_node = config.dof_node
    dof_elem = config.dof_elem

    q_rec = np.zeros(
        (num_snaps, config.nx_t, config.ny_t, config.nz_t, config.num_vars),
        dtype=np.float32,
    )

    t_time = 0.0

    for bid in range(num_batches):
        snap_start = bid * batch_size
        snap_end = min((bid + 1) * batch_size, num_snaps)

        start_time = time.time()
        sys.stdout.write(f"Processing batch {bid+1}/{num_batches}, batch size: {batch_size}")
        sys.stdout.flush()

        Q_rec_u = np.zeros((config.nx_t, config.ny_t, config.nz_t, snap_end - snap_start))
        Q_rec_v = np.zeros((config.nx_t, config.ny_t, config.nz_t, snap_end - snap_start))
        Q_rec_w = np.zeros((config.nx_t, config.ny_t, config.nz_t, snap_end - snap_start))

        for i in range(config.nx_g - 1):
            for j in range(config.ny_g - 1):
                for k in range(config.nz_g - 1):
                    lltogl = build_lltogl(i, j, k, config.ny_g, config.nz_g, dof_node, IJK)

                    dof_local_u = dof_u_arr[snap_start:snap_end, :][:, lltogl].T
                    dof_local_v = dof_v_arr[snap_start:snap_end, :][:, lltogl].T
                    dof_local_w = dof_w_arr[snap_start:snap_end, :][:, lltogl].T

                    Q_local_u_vec = config.modemat_local_u @ dof_local_u
                    Q_local_v_vec = config.modemat_local_v @ dof_local_v
                    Q_local_w_vec = config.modemat_local_w @ dof_local_w

                    Q_local_u = Q_local_u_vec.reshape((nskip + 1, nskip + 1, nskip + 1, snap_end - snap_start), order="F")
                    Q_local_v = Q_local_v_vec.reshape((nskip + 1, nskip + 1, nskip + 1, snap_end - snap_start), order="F")
                    Q_local_w = Q_local_w_vec.reshape((nskip + 1, nskip + 1, nskip + 1, snap_end - snap_start), order="F")

                    indx_cell = i * nskip
                    indy_cell = j * nskip
                    indz_cell = k * nskip

                    Q_rec_u[indx_cell:indx_cell + nskip + 1, indy_cell:indy_cell + nskip + 1, indz_cell:indz_cell + nskip + 1, :] = Q_local_u
                    Q_rec_v[indx_cell:indx_cell + nskip + 1, indy_cell:indy_cell + nskip + 1, indz_cell:indz_cell + nskip + 1, :] = Q_local_v
                    Q_rec_w[indx_cell:indx_cell + nskip + 1, indy_cell:indy_cell + nskip + 1, indz_cell:indz_cell + nskip + 1, :] = Q_local_w
                    compute_time = time.time() 
                    t_time += compute_time - start_time

        q_rec[snap_start:snap_end, :, :, :, 0] = Q_rec_u.transpose(3, 0, 1, 2)
        q_rec[snap_start:snap_end, :, :, :, 1] = Q_rec_v.transpose(3, 0, 1, 2)
        q_rec[snap_start:snap_end, :, :, :, 2] = Q_rec_w.transpose(3, 0, 1, 2)

        end_time = time.time()
        batch_time = end_time - start_time
        sys.stdout.write(f", processed in {batch_time:.2f}s")
        if bid + 1 != num_batches:
            proj_time = (num_batches - (bid + 1)) * batch_time / 60
            proj_time_str = f"{int(proj_time)}m {int((proj_time - int(proj_time)) * 60)}s"
            sys.stdout.write(f" -> Proj. time: {proj_time_str}")
        sys.stdout.write("\n")
        sys.stdout.flush()

    if batch_time < 1:
        t_time = t_time/1000  # convert from ms if under 1s
    sys.stdout.write(f"Total reconstruction time not with saving to disk: {t_time:.2f}s\n\n")

    if rec_target is None:
        return q_rec

    if isinstance(rec_target, str):
        with h5py.File(rec_target, "a") as rec_file:
            rec_file.create_dataset("Q_rec", data=q_rec, dtype="float32")
    else:
        rec_target["Q_rec"] = q_rec

    return q_rec
