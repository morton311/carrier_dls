"""
Generate volume weights for a general hexahedral mesh.

This script computes the volume of each hexahedral element in a 3D mesh.
Input: grid.mat with x_grid, y_grid, z_grid (nx, ny, nz)
Output: weights_grid.mat with weights_grid (nx, ny, nz) - volume of each element
"""

import numpy as np
import scipy.io as sio
from scipy.spatial import ConvexHull

def hex_volume(vertices):
    """
    Compute volume of hexahedron by decomposing into 5 tetrahedra.
    vertices: (8, 3) array of corner coordinates
    """
    def tet_volume(v0, v1, v2, v3):
        # Volume = |det(v1-v0, v2-v0, v3-v0)| / 6
        return np.abs(np.linalg.det(np.array([v1-v0, v2-v0, v3-v0]))) / 6.0
    
    v = vertices
    # Decompose into 5 tetrahedra
    return (tet_volume(v[0], v[1], v[3], v[4]) +
            tet_volume(v[1], v[2], v[3], v[6]) +
            tet_volume(v[1], v[3], v[4], v[6]) +
            tet_volume(v[3], v[4], v[6], v[7]) +
            tet_volume(v[1], v[4], v[5], v[6]))

def generate_weights_grid_3d(x_grid, y_grid, z_grid):
    """
    Compute volume weights for a general hexahedral mesh.
    
    Parameters
    ----------
    x_grid, y_grid, z_grid : ndarray, shape (nx, ny, nz)
        Original grid coordinates
    
    Returns
    -------
    weights_grid : ndarray, shape (nx, ny, nz)
        Volume of each hexahedral element
    """
    nx, ny, nz = x_grid.shape
    
    # Create control grid (cell face centers/vertices) - one larger in each dimension
    xc = np.zeros((nx + 1, ny + 1, nz + 1))
    yc = np.zeros((nx + 1, ny + 1, nz + 1))
    zc = np.zeros((nx + 1, ny + 1, nz + 1))
    
    # =========================================================================
    # Interior points: average of 8 surrounding grid points
    # =========================================================================
    xc[1:nx, 1:ny, 1:nz] = 0.125 * (
        x_grid[:-1, :-1, :-1] + x_grid[1:, :-1, :-1] +
        x_grid[1:, 1:, :-1] + x_grid[:-1, 1:, :-1] +
        x_grid[:-1, :-1, 1:] + x_grid[1:, :-1, 1:] +
        x_grid[1:, 1:, 1:] + x_grid[:-1, 1:, 1:]
    )
    
    yc[1:nx, 1:ny, 1:nz] = 0.125 * (
        y_grid[:-1, :-1, :-1] + y_grid[1:, :-1, :-1] +
        y_grid[1:, 1:, :-1] + y_grid[:-1, 1:, :-1] +
        y_grid[:-1, :-1, 1:] + y_grid[1:, :-1, 1:] +
        y_grid[1:, 1:, 1:] + y_grid[:-1, 1:, 1:]
    )
    
    zc[1:nx, 1:ny, 1:nz] = 0.125 * (
        z_grid[:-1, :-1, :-1] + z_grid[1:, :-1, :-1] +
        z_grid[1:, 1:, :-1] + z_grid[:-1, 1:, :-1] +
        z_grid[:-1, :-1, 1:] + z_grid[1:, :-1, 1:] +
        z_grid[1:, 1:, 1:] + z_grid[:-1, 1:, 1:]
    )
    
    # =========================================================================
    # Face centers: average of 4 surrounding grid points
    # =========================================================================
    
    # XY planes (k=0 and k=nz)
    xc[1:nx, 1:ny, 0] = 0.25 * (
        x_grid[:-1, :-1, 0] + x_grid[1:, :-1, 0] +
        x_grid[1:, 1:, 0] + x_grid[:-1, 1:, 0]
    )
    xc[1:nx, 1:ny, nz] = 0.25 * (
        x_grid[:-1, :-1, -1] + x_grid[1:, :-1, -1] +
        x_grid[1:, 1:, -1] + x_grid[:-1, 1:, -1]
    )
    
    yc[1:nx, 1:ny, 0] = 0.25 * (
        y_grid[:-1, :-1, 0] + y_grid[1:, :-1, 0] +
        y_grid[1:, 1:, 0] + y_grid[:-1, 1:, 0]
    )
    yc[1:nx, 1:ny, nz] = 0.25 * (
        y_grid[:-1, :-1, -1] + y_grid[1:, :-1, -1] +
        y_grid[1:, 1:, -1] + y_grid[:-1, 1:, -1]
    )
    
    zc[1:nx, 1:ny, 0] = 0.25 * (
        z_grid[:-1, :-1, 0] + z_grid[1:, :-1, 0] +
        z_grid[1:, 1:, 0] + z_grid[:-1, 1:, 0]
    )
    zc[1:nx, 1:ny, nz] = 0.25 * (
        z_grid[:-1, :-1, -1] + z_grid[1:, :-1, -1] +
        z_grid[1:, 1:, -1] + z_grid[:-1, 1:, -1]
    )
    
    # XZ planes (j=0 and j=ny)
    xc[1:nx, 0, 1:nz] = 0.25 * (
        x_grid[:-1, 0, :-1] + x_grid[1:, 0, :-1] +
        x_grid[1:, 0, 1:] + x_grid[:-1, 0, 1:]
    )
    xc[1:nx, ny, 1:nz] = 0.25 * (
        x_grid[:-1, -1, :-1] + x_grid[1:, -1, :-1] +
        x_grid[1:, -1, 1:] + x_grid[:-1, -1, 1:]
    )
    
    yc[1:nx, 0, 1:nz] = 0.25 * (
        y_grid[:-1, 0, :-1] + y_grid[1:, 0, :-1] +
        y_grid[1:, 0, 1:] + y_grid[:-1, 0, 1:]
    )
    yc[1:nx, ny, 1:nz] = 0.25 * (
        y_grid[:-1, -1, :-1] + y_grid[1:, -1, :-1] +
        y_grid[1:, -1, 1:] + y_grid[:-1, -1, 1:]
    )
    
    zc[1:nx, 0, 1:nz] = 0.25 * (
        z_grid[:-1, 0, :-1] + z_grid[1:, 0, :-1] +
        z_grid[1:, 0, 1:] + z_grid[:-1, 0, 1:]
    )
    zc[1:nx, ny, 1:nz] = 0.25 * (
        z_grid[:-1, -1, :-1] + z_grid[1:, -1, :-1] +
        z_grid[1:, -1, 1:] + z_grid[:-1, -1, 1:]
    )
    
    # YZ planes (i=0 and i=nx)
    xc[0, 1:ny, 1:nz] = 0.25 * (
        x_grid[0, :-1, :-1] + x_grid[0, 1:, :-1] +
        x_grid[0, 1:, 1:] + x_grid[0, :-1, 1:]
    )
    xc[nx, 1:ny, 1:nz] = 0.25 * (
        x_grid[-1, :-1, :-1] + x_grid[-1, 1:, :-1] +
        x_grid[-1, 1:, 1:] + x_grid[-1, :-1, 1:]
    )
    
    yc[0, 1:ny, 1:nz] = 0.25 * (
        y_grid[0, :-1, :-1] + y_grid[0, 1:, :-1] +
        y_grid[0, 1:, 1:] + y_grid[0, :-1, 1:]
    )
    yc[nx, 1:ny, 1:nz] = 0.25 * (
        y_grid[-1, :-1, :-1] + y_grid[-1, 1:, :-1] +
        y_grid[-1, 1:, 1:] + y_grid[-1, :-1, 1:]
    )
    
    zc[0, 1:ny, 1:nz] = 0.25 * (
        z_grid[0, :-1, :-1] + z_grid[0, 1:, :-1] +
        z_grid[0, 1:, 1:] + z_grid[0, :-1, 1:]
    )
    zc[nx, 1:ny, 1:nz] = 0.25 * (
        z_grid[-1, :-1, :-1] + z_grid[-1, 1:, :-1] +
        z_grid[-1, 1:, 1:] + z_grid[-1, :-1, 1:]
    )
    
    # =========================================================================
    # Edge centers: average of 2 grid points
    # =========================================================================
    
    # Edges parallel to i-axis
    xc[1:nx, 0, 0] = 0.5 * (x_grid[:-1, 0, 0] + x_grid[1:, 0, 0])
    xc[1:nx, ny, 0] = 0.5 * (x_grid[:-1, -1, 0] + x_grid[1:, -1, 0])
    xc[1:nx, 0, nz] = 0.5 * (x_grid[:-1, 0, -1] + x_grid[1:, 0, -1])
    xc[1:nx, ny, nz] = 0.5 * (x_grid[:-1, -1, -1] + x_grid[1:, -1, -1])
    
    yc[1:nx, 0, 0] = 0.5 * (y_grid[:-1, 0, 0] + y_grid[1:, 0, 0])
    yc[1:nx, ny, 0] = 0.5 * (y_grid[:-1, -1, 0] + y_grid[1:, -1, 0])
    yc[1:nx, 0, nz] = 0.5 * (y_grid[:-1, 0, -1] + y_grid[1:, 0, -1])
    yc[1:nx, ny, nz] = 0.5 * (y_grid[:-1, -1, -1] + y_grid[1:, -1, -1])
    
    zc[1:nx, 0, 0] = 0.5 * (z_grid[:-1, 0, 0] + z_grid[1:, 0, 0])
    zc[1:nx, ny, 0] = 0.5 * (z_grid[:-1, -1, 0] + z_grid[1:, -1, 0])
    zc[1:nx, 0, nz] = 0.5 * (z_grid[:-1, 0, -1] + z_grid[1:, 0, -1])
    zc[1:nx, ny, nz] = 0.5 * (z_grid[:-1, -1, -1] + z_grid[1:, -1, -1])
    
    # Edges parallel to j-axis
    xc[0, 1:ny, 0] = 0.5 * (x_grid[0, :-1, 0] + x_grid[0, 1:, 0])
    xc[nx, 1:ny, 0] = 0.5 * (x_grid[-1, :-1, 0] + x_grid[-1, 1:, 0])
    xc[0, 1:ny, nz] = 0.5 * (x_grid[0, :-1, -1] + x_grid[0, 1:, -1])
    xc[nx, 1:ny, nz] = 0.5 * (x_grid[-1, :-1, -1] + x_grid[-1, 1:, -1])
    
    yc[0, 1:ny, 0] = 0.5 * (y_grid[0, :-1, 0] + y_grid[0, 1:, 0])
    yc[nx, 1:ny, 0] = 0.5 * (y_grid[-1, :-1, 0] + y_grid[-1, 1:, 0])
    yc[0, 1:ny, nz] = 0.5 * (y_grid[0, :-1, -1] + y_grid[0, 1:, -1])
    yc[nx, 1:ny, nz] = 0.5 * (y_grid[-1, :-1, -1] + y_grid[-1, 1:, -1])
    
    zc[0, 1:ny, 0] = 0.5 * (z_grid[0, :-1, 0] + z_grid[0, 1:, 0])
    zc[nx, 1:ny, 0] = 0.5 * (z_grid[-1, :-1, 0] + z_grid[-1, 1:, 0])
    zc[0, 1:ny, nz] = 0.5 * (z_grid[0, :-1, -1] + z_grid[0, 1:, -1])
    zc[nx, 1:ny, nz] = 0.5 * (z_grid[-1, :-1, -1] + z_grid[-1, 1:, -1])
    
    # Edges parallel to k-axis
    xc[0, 0, 1:nz] = 0.5 * (x_grid[0, 0, :-1] + x_grid[0, 0, 1:])
    xc[nx, 0, 1:nz] = 0.5 * (x_grid[-1, 0, :-1] + x_grid[-1, 0, 1:])
    xc[0, ny, 1:nz] = 0.5 * (x_grid[0, -1, :-1] + x_grid[0, -1, 1:])
    xc[nx, ny, 1:nz] = 0.5 * (x_grid[-1, -1, :-1] + x_grid[-1, -1, 1:])
    
    yc[0, 0, 1:nz] = 0.5 * (y_grid[0, 0, :-1] + y_grid[0, 0, 1:])
    yc[nx, 0, 1:nz] = 0.5 * (y_grid[-1, 0, :-1] + y_grid[-1, 0, 1:])
    yc[0, ny, 1:nz] = 0.5 * (y_grid[0, -1, :-1] + y_grid[0, -1, 1:])
    yc[nx, ny, 1:nz] = 0.5 * (y_grid[-1, -1, :-1] + y_grid[-1, -1, 1:])
    
    zc[0, 0, 1:nz] = 0.5 * (z_grid[0, 0, :-1] + z_grid[0, 0, 1:])
    zc[nx, 0, 1:nz] = 0.5 * (z_grid[-1, 0, :-1] + z_grid[-1, 0, 1:])
    zc[0, ny, 1:nz] = 0.5 * (z_grid[0, -1, :-1] + z_grid[0, -1, 1:])
    zc[nx, ny, 1:nz] = 0.5 * (z_grid[-1, -1, :-1] + z_grid[-1, -1, 1:])
    
    # =========================================================================
    # Corners: just copy from original grid
    # =========================================================================
    corners_i = [0, nx]
    corners_j = [0, ny]
    corners_k = [0, nz]
    
    for i in corners_i:
        for j in corners_j:
            for k in corners_k:
                gi = 0 if i == 0 else -1
                gj = 0 if j == 0 else -1
                gk = 0 if k == 0 else -1
                
                xc[i, j, k] = x_grid[gi, gj, gk]
                yc[i, j, k] = y_grid[gi, gj, gk]
                zc[i, j, k] = z_grid[gi, gj, gk]
    
    # =========================================================================
    # Compute hexahedron volumes using ConvexHull
    # =========================================================================
    weights_grid = np.zeros((nx, ny, nz))
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # 8 corners of the hexahedron
                vertices = np.array([
                    [xc[i+1, j, k+1], yc[i+1, j, k+1], zc[i+1, j, k+1]],
                    [xc[i+1, j+1, k+1], yc[i+1, j+1, k+1], zc[i+1, j+1, k+1]],
                    [xc[i+1, j+1, k], yc[i+1, j+1, k], zc[i+1, j+1, k]],
                    [xc[i+1, j, k], yc[i+1, j, k], zc[i+1, j, k]],
                    [xc[i, j, k+1], yc[i, j, k+1], zc[i, j, k+1]],
                    [xc[i, j+1, k+1], yc[i, j+1, k+1], zc[i, j+1, k+1]],
                    [xc[i, j+1, k], yc[i, j+1, k], zc[i, j+1, k]],
                    [xc[i, j, k], yc[i, j, k], zc[i, j, k]],
                ])
                
                weights_grid[i, j, k] = hex_volume(vertices)
    
    return weights_grid


def generate_weights_grid_2d(x_grid, y_grid):
    """
    Compute area weights for a general quadrilateral mesh.
    
    Parameters
    ----------
    x_grid, y_grid : ndarray, shape (nx, ny)
        Original grid coordinates
    
    Returns
    -------
    weights_grid : ndarray, shape (nx, ny)
        Area of each quadrilateral element
    """
    nx, ny = x_grid.shape
    
    # Create control grid (cell face centers/vertices) - one larger in each dimension
    xc = np.zeros((nx + 1, ny + 1))
    yc = np.zeros((nx + 1, ny + 1))

    # Interior points: average of 4 surrounding grid points
    xc[1:nx, 1:ny] = 0.25 * (
        x_grid[:-1, :-1] + x_grid[1:, :-1] +
        x_grid[1:, 1:] + x_grid[:-1, 1:]
    )
    yc[1:nx, 1:ny] = 0.25 * (
        y_grid[:-1, :-1] + y_grid[1:, :-1] +
        y_grid[1:, 1:] + y_grid[:-1, 1:]
    )

    # Face centers: average of 2 surrounding grid points
    xc[1:nx, 0] = 0.5 * (x_grid[:-1, 0] + x_grid[1:, 0])
    xc[1:nx, ny] = 0.5 * (x_grid[:-1, -1] + x_grid[1:, -1])
    yc[1:nx, 0] = 0.5 * (y_grid[:-1, 0] + y_grid[1:, 0])
    yc[1:nx, ny] = 0.5 * (y_grid[:-1, -1] + y_grid[1:, -1])
    xc[0, 1:ny] = 0.5 * (x_grid[0, :-1] + x_grid[0, 1:])
    xc[nx, 1:ny] = 0.5 * (x_grid[-1, :-1] + x_grid[-1, 1:])
    yc[0, 1:ny] = 0.5 * (y_grid[0, :-1] + y_grid[0, 1:])
    yc[nx, 1:ny] = 0.5 * (y_grid[-1, :-1] + y_grid[-1, 1:])

    # Corners: just copy from original grid
    xc[0, 0] = x_grid[0, 0]
    yc[0, 0] = y_grid[0, 0]
    xc[nx, 0] = x_grid[-1, 0]
    yc[nx, 0] = y_grid[-1, 0]
    xc[0, ny] = x_grid[0, -1]
    yc[0, ny] = y_grid[0, -1]
    xc[nx, ny] = x_grid[-1, -1]
    yc[nx, ny] = y_grid[-1, -1]

    # Compute quadrilateral areas using convex hull
    weights_grid = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            vertices = np.array([
                [xc[i+1, j], yc[i+1, j]],
                [xc[i+1, j+1], yc[i+1, j+1]],
                [xc[i, j+1], yc[i, j+1]],
                [xc[i, j], yc[i, j]],
            ])
            hull = ConvexHull(vertices)
            weights_grid[i, j] = hull.volume  # In 2D, 'volume' is the area

    
    return weights_grid


if __name__ == "__main__":
    # Load grid from MATLAB file
    data = sio.loadmat("grid.mat")
    x_grid = data["x_grid"]
    y_grid = data["y_grid"]
    z_grid = data["z_grid"]
    
    print(f"Grid shape: {x_grid.shape}")
    
    # Compute weights
    weights_grid = generate_weights_grid_3d(x_grid, y_grid, z_grid)
    
    print(f"Weights shape: {weights_grid.shape}")
    print(f"Min weight: {weights_grid.min():.6e}")
    print(f"Max weight: {weights_grid.max():.6e}")
    print(f"Mean weight: {weights_grid.mean():.6e}")
    print(f"Total volume: {weights_grid.sum():.6e}")
    
    # Save to MATLAB file
    sio.savemat("weights_grid.mat", {"weights_grid": weights_grid})
    print("\nSaved to weights_grid.mat")



