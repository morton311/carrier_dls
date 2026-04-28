import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized
from tqdm import tqdm

from .dls import Modal_decomp, local_modemat_over_elem


ArrayDict = Dict[str, np.ndarray]
DataInput = Union[str, ArrayDict]


@dataclass
class dls_long_Config_3D_Flexible:
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
        self.dof_node = self.num_modes + 1
        self.dof_elem = 8 * self.dof_node
        self.compression_ratio = (
            self.num_vars * self.num_snaps * self.nx * self.ny * self.nz
            / (
                self.num_vars * self.num_snaps * self.dof_node
                + self.num_vars * self.num_modes * self.patch_size**3
            )
        )


def _node_map(opp=False) -> np.ndarray:
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


def _build_lltogl(
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


def gfem_3d_long_flexible(
    data_source: DataInput,
    field_name: str,
    patch_size: int,
    num_modes: int,
    latent_target: Optional[Union[str, Dict[str, np.ndarray]]] = None,
    batch_size: int = 2500,
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

    # With a uniform/structured mesh in this formulation, one local mass matrix is reused.
    M_local_u = modemat_local_wt_u.T @ modemat_local_u
    M_local_v = modemat_local_wt_v.T @ modemat_local_v
    M_local_w = modemat_local_wt_w.T @ modemat_local_w

    M_u = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_v = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_w = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))

    IJK = _node_map()

    print("Constructing global M GFEM matrices")
    for i in range(nx_g - 1):
        for j in range(ny_g - 1):
            for k in range(nz_g - 1):
                lltogl = _build_lltogl(i, j, k, ny_g, nz_g, dof_node, IJK)
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
                    lltogl = _build_lltogl(i, j, k, ny_g, nz_g, dof_node, IJK)

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

    if latent_target is None:
        pass
    elif isinstance(latent_target, str):
        with h5py.File(latent_target, "w") as dof_file:
            dof_file.create_dataset("dof_u", data=dof_u_all, dtype="float32")
            dof_file.create_dataset("dof_v", data=dof_v_all, dtype="float32")
            dof_file.create_dataset("dof_w", data=dof_w_all, dtype="float32")
    else:
        latent_target["dof_u"] = dof_u_all
        latent_target["dof_v"] = dof_v_all
        latent_target["dof_w"] = dof_w_all

    config = dls_long_Config_3D_Flexible(
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

    return config, dof_u_all, dof_v_all, dof_w_all


def gfem_recon_long_3D_flexible(
    rec_target: Optional[Union[str, Dict[str, np.ndarray]]],
    config: dls_long_Config_3D_Flexible,
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

    if dof_u_arr.shape[0] != config.num_snaps:
        # Accept transposed input format [n_dof, n_snap] as in legacy callers.
        if dof_u_arr.shape[1] == config.num_snaps:
            dof_u_arr = dof_u_arr.T
            dof_v_arr = dof_v_arr.T
            dof_w_arr = dof_w_arr.T
        else:
            raise ValueError("DOF array shape does not match config.num_snaps.")

    num_snaps = dof_u_arr.shape[0]
    num_batches = num_snaps // batch_size + (1 if num_snaps % batch_size else 0)

    IJK = _node_map()
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
                    lltogl = _build_lltogl(i, j, k, config.ny_g, config.nz_g, dof_node, IJK)

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
        with h5py.File(rec_target, "w") as rec_file:
            rec_file.create_dataset("Q_rec", data=q_rec, dtype="float32")
    else:
        rec_target["Q_rec"] = q_rec

    return q_rec
