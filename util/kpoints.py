import numpy as np
import spglib as spg

from pymes.log import print_title, print_logging_info


def gen_ir_ks(mesh=None, lattice=None, positions=None, number=None):
    """
    Generate irreducible wedges from a uniform Monkhorst k-mesh in the 1.B.Z.
    Args:
        mesh: list of 3 integers. Dimension of the uniform k-mesh in one direction. n_ks[1] x n_ks[2] x n_ks[3]
        lattice: np float array of size 3 x 3. Defining the lattice vectors of the primitive cell
        positions: np float array. Defining the positions of the atoms in the primitive cell
        number: list of integers. Defining the species of the atoms in the cell

    Returns:
        frac_grid: list of np arrays of size 3. The fractional coordinations of the irreducible k-points
        weight: list of floats. The weight of each irreducible k-point. It is the number of equivalent k-points divided
        by the total number of k-points.
    """

    if mesh is None:
        mesh = [3,] * 3

    if isinstance(mesh, int):
        mesh = [mesh, ] * 3

    if number is None:
        number = [1]

    if positions is None:
        positions = [[0., 0., 0.]]

    if lattice is None:
        lattice = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    cell = (lattice, positions, number)
    mapping, grid = spg.get_ir_reciprocal_mesh(mesh, cell, is_shift=[0, 0, 0])
    # get the weight of the irreducible k-points
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

    # All k-points and mapping to ir-grid points
    for i, (ir_gp_id, gp) in enumerate(zip(mapping, grid)):
        print_logging_info("%3d ->%3d %s" % (i, ir_gp_id, gp.astype(float) / mesh), level=2)

    # Irreducible k-points
    frac_grid = grid[np.unique(mapping)] / np.array(mesh, dtype=float)
    print_logging_info("Number of ir-kpoints: %d" % len(np.unique(mapping)), level=2)
    return frac_grid, weight
