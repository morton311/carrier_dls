import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import time
import sys
import h5py
from tqdm import tqdm
from scipy.sparse.linalg import factorized

class dls_Config:
    def __init__(self, data, patch_size, num_modes, modemat_local):
        self.ndim = data.ndim - 2  # Subtract 2 for channel and snapshot dimensions
        self.nx = data.shape[1]
        self.ny = data.shape[2]
        self.nz = data.shape[3] if self.ndim == 3 else None

        self.num_snaps = data.shape[-1]
        self.patch_size = patch_size
        self.num_modes = num_modes
        self.nskip = (patch_size - 1) // 2
        self.nskip_sample = patch_size - 1
        self.mid_pt = 1 + self.nskip_sample // 2
        self.sample_x = range(0, self.nx, self.nskip)
        self.sample_y = range(0, self.ny, self.nskip)
        if self.ndim == 3:
            self.sample_z = range(0, self.nz, self.nskip)

        self.nx_t = max(self.sample_x) + 1
        self.ny_t = max(self.sample_y) + 1
        self.nz_t = max(self.sample_z) + 1 if self.ndim == 3 else None

        self.nx_g = len(self.sample_x)
        self.ny_g = len(self.sample_y)
        self.nz_g = len(self.sample_z) if self.ndim == 3 else None

        self.num_gfem_nodes = self.nx_g * self.ny_g * (self.nz_g if self.ndim == 3 else 1)
        self.dof_node = num_modes + 1
        self.dof_elem = (4 if self.ndim == 2 else 8) * self.dof_node

        self.modemat_local = modemat_local

        spatial_size = self.nx * self.ny * (self.nz if self.ndim == 3 else 1)
        self.compression_ratio = data.shape[0] * self.num_snaps * spatial_size / (data.shape[0] * self.num_snaps * self.dof_node + data.shape[0] * num_modes * self.patch_size**self.ndim)


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
    data_shape = 'fat' if data.shape[0] < data.shape[1] else 'tall'
    print(f"NYI ----- Data shape: {data.shape}, treating as {data_shape} for SVD")
    P = random_patch_sampling(data, patch_size)
    local_modes, eigVal, _ = np.linalg.svd(P, full_matrices=False)
    return local_modes, eigVal


def FEM_shape_calculator_2D_ortho_gfemlr(x, y, xpt, ypt):
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


def FEM_shape_calculator_3D_ortho_gfemlr(x, y, z, xpt, ypt, zpt):

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
                N = FEM_shape_calculator_3D_ortho_gfemlr(x_val, y_val, z_val, xpt, ypt, zpt)
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

    modemat_local_wt = modemat_local.copy()
    for kk in range(dof_elem):
        modemat_local_wt[:, kk] *= Wt_vec

    return modemat_local, modemat_local_wt
        

def gfem_3d_long(data_path: str, field_name: str, latent_file: str, patch_size: int, num_modes: int, batch_size: int = 2500):
    with h5py.File(data_path, 'r') as f:
        ndim = 3
        num_snaps = f[field_name].shape[0]
        nx = f[field_name].shape[1]
        ny = f[field_name].shape[2]
        nz = f[field_name].shape[3] 
        
        mode_data = f[field_name][0,:,:,:,:] - f['mean'][:]

        grid_x = f['x_grid'][:]
        grid_y = f['y_grid'][:]
        grid_z = f['z_grid'][:]


        print('shape of mode data: ', mode_data.shape)
        print('number of snapshots: ', num_snaps)
        print('number of batches: ', num_snaps // batch_size)
        print('nx: ', nx)
        print('ny: ', ny)
        print('num_vars: ', ndim)



    start_time = time.time()

    nskip = (patch_size - 1) // 2
    nskip_sample = patch_size - 1
    mid_pt = 1 + nskip_sample // 2

    sample_x = range(0, nx, nskip)
    sample_y = range(0, ny, nskip)
    sample_z = range(0, nz, nskip)

    nx_t = max(sample_x) + 1
    ny_t = max(sample_y) + 1
    nz_t = max(sample_z) + 1 

    nx_g = len(sample_x)
    ny_g = len(sample_y)
    nz_g = len(sample_z)

    num_gfem_nodes = nx_g * ny_g * nz_g
    dof_node = num_modes + 1
    dof_elem = 8 * dof_node

    print('Performing modal decomposition to get local modes')
    local_modes_u, eigVal_u = Modal_decomp(mode_data[..., 0], patch_size)
    local_modes_v, eigVal_v = Modal_decomp(mode_data[..., 1], patch_size)
    local_modes_w, eigVal_w = Modal_decomp(mode_data[..., 2], patch_size)
    print('Modal decomposition done')

    print('Constructing local modal matrices')
    modemat_local_u, modemat_local_wt_u = local_modemat_over_elem(grid_x, grid_y, grid_z, nskip, local_modes_u, num_modes, patch_size)
    modemat_local_v, modemat_local_wt_v = local_modemat_over_elem(grid_x, grid_y, grid_z, nskip, local_modes_v, num_modes, patch_size)
    modemat_local_w, modemat_local_wt_w = local_modemat_over_elem(grid_x, grid_y, grid_z, nskip, local_modes_w, num_modes, patch_size)
    print('Local modal matrices constructed')

    

    for i in range(1): # Loop over GFEM elements, x direction
        for j in range(1): # Loop over GFEM elements, y direction
            for k in range(1): # Loop over GFEM elements, z direction
                # Local mass matrix
                M_local_u = modemat_local_wt_u.T @ modemat_local_u
                M_local_v = modemat_local_wt_v.T @ modemat_local_v
                M_local_w = modemat_local_wt_w.T @ modemat_local_w

    M_u = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_v = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_w = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    IJK = np.array([
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [0, 0, 0]
        ])
    
    print('Constructing global M GFEM matrices')

    for i in range(nx_g - 1): # Loop over GFEM elements, x direction
        for j in range(ny_g - 1): # Loop over GFEM elements, y direction
            for k in range(nz_g - 1): # Loop over GFEM elements, z direction
                lltogl = np.zeros(dof_elem, dtype=int)
                for lindx in range(8):
                    indx_dof_start = ((i+IJK[lindx, 0])*ny_g*nz_g + (j+IJK[lindx, 1])*nz_g + (k+IJK[lindx, 2]))*dof_node
                    idx_dof_end = indx_dof_start + dof_node

                    lltogl[lindx*dof_node:(lindx+1)*dof_node] = np.arange(indx_dof_start, idx_dof_end)
                
                M_u[np.ix_(lltogl, lltogl)] += M_local_u
                M_v[np.ix_(lltogl, lltogl)] += M_local_v
                M_w[np.ix_(lltogl, lltogl)] += M_local_w


    end_time = time.time()
    print(f'Computed DLS latent space in {end_time - start_time:.2f} seconds')
    print('M constructed')

    print('Prefactorizing M')
    # Convert lilmatrix to csr matrix
    M_u = M_u.tocsc()
    M_v = M_v.tocsc()
    M_w = M_w.tocsc()

    start_time = time.time()

    # Pre-factorize the matrices for efficiency
    solve_M_u = factorized(M_u)
    solve_M_v = factorized(M_v)
    solve_M_w = factorized(M_w)
    print('M prefactorized')

    # create h5 dataset for dof of shape (num_gfem_nodes*dof_node, num_snaps)
    dof_file = h5py.File(latent_file, 'w')
    dof_file.create_dataset('dof_u', (num_snaps, num_gfem_nodes * dof_node), dtype='float32')
    dof_file.create_dataset('dof_v', (num_snaps, num_gfem_nodes * dof_node), dtype='float32')
    dof_file.create_dataset('dof_w', (num_snaps, num_gfem_nodes * dof_node), dtype='float32')

    dof_u = np.zeros((num_gfem_nodes * dof_node, batch_size))
    dof_v = np.zeros((num_gfem_nodes * dof_node, batch_size))
    dof_w = np.zeros((num_gfem_nodes * dof_node, batch_size))

    end_time = time.time()

    t_time = end_time - start_time

    print('Looping through snapshots, solving for dofs')
    with h5py.File(data_path, 'r') as f:
        loops = num_snaps // batch_size
        if num_snaps % batch_size != 0:
            loops += 1
        for i in tqdm(range(loops)):
            start_time = time.time()
            snap_start = i * batch_size
            snap_end = min((i + 1) * batch_size, num_snaps)
            u_mean = f['mean'][:, :, :, 0]
            v_mean = f['mean'][:, :, :, 1]
            w_mean = f['mean'][:, :, :, 2]
            Q_grid_u = f[field_name][snap_start:snap_end, :, :, :, 0]
            Q_grid_v = f[field_name][snap_start:snap_end, :, :, :, 1]
            Q_grid_w = f[field_name][snap_start:snap_end, :, :, :, 2]
            Q_grid_u = Q_grid_u.transpose(1,2,3,0) - u_mean[:, :, :, np.newaxis]
            Q_grid_v = Q_grid_v.transpose(1,2,3,0) - v_mean[:, :, :, np.newaxis]
            Q_grid_w = Q_grid_w.transpose(1,2,3,0) - w_mean[:, :, :, np.newaxis]

            L_u = np.zeros((num_gfem_nodes * dof_node, snap_end - snap_start))
            L_v = np.zeros((num_gfem_nodes * dof_node, snap_end - snap_start))
            L_w = np.zeros((num_gfem_nodes * dof_node, snap_end - snap_start))

            for i in range(nx_g - 1): # Loop over GFEM elements, x direction
                for j in range(ny_g - 1): # Loop over GFEM elements, y direction
                    for k in range(nz_g - 1): # Loop over GFEM elements, z direction
                        lltogl = np.zeros(dof_elem, dtype=int)
                        for lindx in range(8):
                            indx_dof_start = ((i+IJK[lindx, 0])*ny_g*nz_g + (j+IJK[lindx, 1])*nz_g + (k+IJK[lindx, 2]))*dof_node
                            idx_dof_end = indx_dof_start + dof_node

                            lltogl[lindx*dof_node:(lindx+1)*dof_node] = np.arange(indx_dof_start, idx_dof_end)

                        indx_cell = i*nskip 
                        indy_cell = j*nskip
                        indz_cell = k*nskip

                        Q_local_u = Q_grid_u[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, indz_cell:indz_cell+nskip+1, :]
                        Q_local_v = Q_grid_v[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, indz_cell:indz_cell+nskip+1, :]
                        Q_local_w = Q_grid_w[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, indz_cell:indz_cell+nskip+1, :]

                        Q_local_u_vec = np.zeros(((nskip+1)**3, snap_end - snap_start))
                        Q_local_v_vec = np.zeros(((nskip+1)**3, snap_end - snap_start))
                        Q_local_w_vec = np.zeros(((nskip+1)**3, snap_end - snap_start))

                        for kx in range((nskip+1)):
                            for ky in range((nskip+1)):
                                for kz in range((nskip+1)):
                                    iind = kz*(nskip+1)**2 + ky*(nskip+1) + kx
                                    Q_local_u_vec[iind, :] = Q_local_u[kx, ky, kz, :]
                                    Q_local_v_vec[iind, :] = Q_local_v[kx, ky, kz, :]
                                    Q_local_w_vec[iind, :] = Q_local_w[kx, ky, kz, :]
                        
                        L_local_u = modemat_local_wt_u.T @ Q_local_u_vec
                        L_local_v = modemat_local_wt_v.T @ Q_local_v_vec
                        L_local_w = modemat_local_wt_w.T @ Q_local_w_vec

                        L_u[lltogl, :] += L_local_u
                        L_v[lltogl, :] += L_local_v
                        L_w[lltogl, :] += L_local_w

            dof_u = solve_M_u(L_u)
            dof_v = solve_M_v(L_v)
            dof_w = solve_M_w(L_w)

            end_time = time.time()
            t_time += end_time - start_time

            dof_file['dof_u'][snap_start:snap_end, :] = dof_u.T
            dof_file['dof_v'][snap_start:snap_end, :] = dof_v.T
            dof_file['dof_w'][snap_start:snap_end, :] = dof_w.T

    print(f'Solved for dofs in {t_time:.2f} seconds')

    config = dls_long_Config_3D(data_path, field_name, patch_size, num_modes, modemat_local_u, modemat_local_v, modemat_local_w)

    return config


class dls_long_Config_3D:
    def __init__(self, data_path, field_name, patch_size, num_modes, modemat_local_u, modemat_local_v, modemat_local_w):
        with h5py.File(data_path, 'r') as f:
            self.num_snaps = f[field_name].shape[0]
            self.nx = f[field_name].shape[1]
            self.ny = f[field_name].shape[2]
            self.nz = f[field_name].shape[3]
            self.num_vars = f[field_name].shape[4]
        self.patch_size = patch_size
        self.num_modes = num_modes
        self.nskip = (patch_size - 1) // 2
        self.nskip_sample = patch_size - 1
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
        self.dof_node = num_modes + 1
        self.dof_elem = 8 * self.dof_node
        self.modemat_local_u = modemat_local_u
        self.modemat_local_v = modemat_local_v
        self.modemat_local_w = modemat_local_w
        self.compression_ratio = self.num_vars*self.num_snaps*self.nx*self.ny*self.nz / (self.num_vars*self.num_snaps*self.dof_node + self.num_vars * num_modes * self.patch_size**3 )


def gfem_recon_long_3D(rec_path, config, dof_u=None, dof_v=None, dof_w=None, batch_size=100):
    if dof_u.dtype==str:
        dof_path = dof_u
        with h5py.File(dof_path, 'r') as f:
            dof_u = f['dof_u'][:].T
            dof_v = f['dof_v'][:].T
            dof_w = f['dof_w'][:].T
    num_snaps = dof_u.shape[1]
    num_batches = num_snaps // batch_size
    if num_snaps % batch_size != 0:
        num_batches += 1

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

    diag_opp_indx = [6, 7, 4, 5, 2, 3, 0, 1]
    IJK = IJK[:, diag_opp_indx]

    nskip = config.nskip
    dof_node = config.dof_node # DOFs/node
    dof_elem = config.dof_elem # DOFs/element

    with h5py.File(rec_path, 'w') as rec_file:
        if 'Q_rec' in rec_file.keys():
            del rec_file['Q_rec']
        rec_file.create_dataset('Q_rec', (dof_u.shape[-1], config.nx_t, config.ny_t, config.nz_t, config.num_vars), dtype='float32')

        t_time = 0.0

        for id in range(num_batches):
            snap_start = id * batch_size
            snap_end = min((id + 1) * batch_size, num_snaps)
            start_time = time.time()
            sys.stdout.write(f'Processing batch {id+1}/{num_batches}, batch size: {batch_size}')
            sys.stdout.flush()

            Q_rec_u = np.zeros((config.nx_t, config.ny_t, config.nz_t, snap_end - snap_start))
            Q_rec_v = np.zeros((config.nx_t, config.ny_t, config.nz_t, snap_end - snap_start))
            Q_rec_w = np.zeros((config.nx_t, config.ny_t, config.nz_t, snap_end - snap_start))

            for i in range(config.nx_g - 1): # Loop over GFEM elements, x direction
                for j in range(config.ny_g - 1): # Loop over GFEM elements, y direction
                    for k in range(config.nz_g - 1): # Loop over GFEM elements, z direction
                        start_time_elem = time.time()
                        lltogl = np.zeros(dof_elem, dtype=int)
                        for lindx in range(8):
                            indx_dof_start = ((i+IJK[lindx, 0])*config.ny_g*config.nz_g + (j+IJK[lindx, 1])*config.nz_g + (k+IJK[lindx, 2]))*dof_node
                            idx_dof_end = indx_dof_start + dof_node

                            lltogl[lindx*dof_node:(lindx+1)*dof_node] = np.arange(indx_dof_start, idx_dof_end)

                        dof_local_u = dof_u[lltogl, snap_start:snap_end]
                        dof_local_v = dof_v[lltogl, snap_start:snap_end]
                        dof_local_w = dof_w[lltogl, snap_start:snap_end]

                        
                        
                        Q_local_u_vec = config.modemat_local_u @ dof_local_u
                        Q_local_v_vec = config.modemat_local_v @ dof_local_v
                        Q_local_w_vec = config.modemat_local_w @ dof_local_w

                        Q_local_u = Q_local_u_vec.reshape((nskip+1, nskip+1, nskip+1, snap_end - snap_start), order='F')
                        Q_local_v = Q_local_v_vec.reshape((nskip+1, nskip+1, nskip+1, snap_end - snap_start), order='F')
                        Q_local_w = Q_local_w_vec.reshape((nskip+1, nskip+1, nskip+1, snap_end - snap_start), order='F')

                        indx_cell = i*nskip 
                        indy_cell = j*nskip
                        indz_cell = k*nskip

                        Q_rec_u[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, indz_cell:indz_cell+nskip+1, :] = Q_local_u
                        Q_rec_v[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, indz_cell:indz_cell+nskip+1, :] = Q_local_v
                        Q_rec_w[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, indz_cell:indz_cell+nskip+1, :] = Q_local_w
                        end_time_elem = time.time()

                        t_time += end_time_elem - start_time_elem

            rec_file['Q_rec'][snap_start:snap_end, :, :, :, 0] = Q_rec_u.transpose(3,0,1,2)
            rec_file['Q_rec'][snap_start:snap_end, :, :, :, 1] = Q_rec_v.transpose(3,0,1,2)
            rec_file['Q_rec'][snap_start:snap_end, :, :, :, 2] = Q_rec_w.transpose(3,0,1,2)

            end_time = time.time()
            batch_time = end_time - start_time
            sys.stdout.write(f', processed in {batch_time:.2f}s')
            if id+1 != num_batches:
                proj_time = (num_batches - (id + 1)) * batch_time / 60 # in minutes
                # convert to min:sec format
                proj_time_str = f'{int(proj_time)}m {int((proj_time - int(proj_time)) * 60)}s'
                sys.stdout.write(f' -> Proj. time: {proj_time_str}')
            sys.stdout.write('\n')
            sys.stdout.flush()

        sys.stdout.write(f'Total reconstruction time not with saving to disk: {t_time:.2f}s\n')
        sys.stdout.write('\n')