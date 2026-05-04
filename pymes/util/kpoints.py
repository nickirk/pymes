import numpy as np
import spglib as spg

from numba import jit, prange, get_num_threads, config
from pymes.log import print_title, print_logging_info

def gen_ir_ks(mesh=None, lattice=None, positions=None, number=None, kpoints='irreducible', is_shift=False):
    """
    Generate a uniform Monkhorst k-mesh in the 1.B.Z., or the
    irreducible wedges from a uniform Monkhorst k-mesh in the 1.B.Z (spglib).
    Args:
        mesh: list of 3 integers 
            Dimension of the uniform k-mesh in one direction. n_ks[1] x n_ks[2] x n_ks[3].
        lattice: np float array of size 3 x 3
            Defining the lattice vectors of the primitive cell.
        positions: np float array
            Defining the positions of the atoms in the primitive cell.
        number: list of integers
            Defining the species of the atoms in the cell.
        kpoints: string
            Type of k-point generation. 'uniform' for uniform Monkhorst k-mesh,
            'irreducible' for irreducible k-points from a uniform Monkhorst k-mesh (spglib).
        is_shift: bool
            Whether to shift the k-mesh (Monkhorst-Pack).
            is_shift = False Gamma-centered mesh θ[i,j,k]=[i/n_ks[1], j/n_ks[2], k/n_ks[3]] ; i=0,...,n_ks[1]-1 / j=0,...,n_ks[2]-1 / k=0,...,n_ks[3]-1.
            is_shift = True shifted mesh θ[i,j,k]=[(i+0.5)/n_ks[1], (j+0.5)/n_ks[2], (k+0.5)/n_ks[3]] ; i=0,...,n_ks[1]-1 / j=0,...,n_ks[2]-1 / k=0,...,n_ks[3]-1.

    Returns:
        frac_grid: list of np arrays of size 3
            The fractional coordinations of the irreducible k-points.
        weight: list of floats 
            The weight of each irreducible k-point. It is the number of equivalent k-points divided
            by the total number of k-points.
    """
    algo_name = "gen_ir_ks"
    # Default: 3 x 3 x 3 uniform k-mesh.
    if mesh is None:
        mesh = [3,] * 3
    # If mesh is given as an integer, convert it to a list of 3 integers.
    if isinstance(mesh, int):
        mesh = [mesh, ] * 3
    # Default: one atom of species 1(H) at the origin.
    if number is None:
        number = [1]
    # Default: one atom of species 1 at the origin.
    if positions is None:
        positions = [[0., 0., 0.]]
    # Default: simple cubic lattice with one atom at the origin.
    if lattice is None:
        lattice = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    # Shift for Monkhorst-Pack mesh.
    if is_shift:
        shift = [0.5, 0.5, 0.5]
    else:
        shift = [0., 0., 0.]

    print_logging_info(algo_name, ": Generating k-points with mesh=%s, shift=%s, and type '%s'." % (mesh, shift, kpoints), level=0)
    if kpoints == 'irreducible':
        cell = (lattice, positions, number)
        mapping, grid = spg.get_ir_reciprocal_mesh(mesh, cell, is_shift=shift)
        # Get the weight of the irreducible k-points.
        unique_inds = np.unique(mapping)
        weight = []
        total_n_ks = np.prod(mesh)
        for uind in unique_inds:
            locs = np.where(mapping == uind)
            num = len(locs[0])
            weight.append(num)
        weight = np.array(weight)
        assert (np.sum(weight) == total_n_ks)
        weight = np.array(weight) / total_n_ks
        # All k-points and mapping to ir-grid points.
        for i, (ir_gp_id, gp) in enumerate(zip(mapping, grid)):
            print_logging_info("%3d ->%3d %s" % (i, ir_gp_id, gp.astype(float) / mesh), level=1)
        # Irreducible k-points.
        frac_grid = grid[np.unique(mapping)] / np.array(mesh, dtype=float)
        print_logging_info("Number of ir-kpoints: %d" % len(np.unique(mapping)), level=1)
    elif kpoints == 'uniform':
        # Generate the uniform k-mesh in fractional coordinates.
        i = np.arange(mesh[0])
        j = np.arange(mesh[1])
        k = np.arange(mesh[2])
        grid = np.array(np.meshgrid(i, j, k, indexing='ij')).reshape(3, -1).T
        # Apply the shift (Monkhorst-Pack).
        grid = grid + np.array(shift)
        for i, gp in enumerate(grid):
            print_logging_info("%3d %s" % (i, gp.astype(float) / mesh), level=2)
        frac_grid = grid / np.array(mesh)
        weight = np.ones(len(frac_grid)) / len(frac_grid)
        print_logging_info("Number of uniform k-points: %d" % len(frac_grid), level=1)
    else:
        raise ValueError("Invalid type for k-point generation: %s" % kpoints)

    return frac_grid, weight

@jit(nopython=True)
def inverse_spherical_FT(r, f_k, kpoints, dk):
    """
    Function to compute the inverse (spherical) Fourier Transform
    of a function f(k) on a point r based on a grid of kpoints.
    f(r) = ∫ d³k f(k) exp(i k*r) = 1/(2π²r) ∫ dk k*sin(k*r) f(k)
    
    Args:
        r: float
            The distance at which to evaluate the inverse Fourier Transform.
        f_k: np array of floats
            The function values at the k-points.
        kpoints: np array of floats
            The k-points at which f(k) is evaluated.
        dk: float
            The spacing between the k-points in the grid.
    Returns:
        f_r: float
            The value of the inverse Fourier Transform at distance r.
    """
    if r < 1.e-12:
        return 0.0
    prefac = dk / (2.0 * np.pi**2 * r)
    integrand = kpoints * np.sin(kpoints * r) * f_k
    return np.sum(integrand) * prefac