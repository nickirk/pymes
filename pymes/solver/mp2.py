import time
import numpy as np
from functools import partial

einsum = partial(np.einsum, optimize=True)

from pymes.log import print_logging_info

def solve(t_epsilon_i, t_epsilon_a, t_V_ijab, t_V_abij, leve_shift=0., **kwargs):
    """
    dense mp2 algorithm
    Note that t_V_ijab and t_V_abij are not necessarily
    the same, e.g. in transcorrelated Hamiltonian.

    """
    t_T_abij = t_V_abij.copy()
    t_D_abij = t_epsilon_i[None, None, :, None] + t_epsilon_i[None, None, None, :] - t_epsilon_a[:, None, None, None] - t_epsilon_a[None, :, None, None]
    t_T_abij /= (t_D_abij + leve_shift)
    eDir = 2.0*einsum('abij, ijab->',t_T_abij, t_V_ijab)
    eExc = -1.0*einsum('abij, jiab->',t_T_abij, t_V_ijab)
    eTotal = eDir + eExc
    return [eTotal, t_T_abij]

def solve_sp(t_epsilon_i, t_epsilon_a, t_V_ijab, t_V_abij, leve_shift=0., sp=0, nv_part_size=None, **kwargs):
    """
    sparse mp2 algorithm: not debugged yet
    Note that t_V_ijab and t_V_abij are not necessarily the same, e.g. in transcorrelated Hamiltonian.
    -------------
    Parameters:
       t_epsilon_i: 1D ctf tensor. The occupied orbital energies
       t_epsilon_a: 1D ctf tensor. The unoccupied orbital energies
       t_V_ijab: ctf tensor. oovv 2-body integrals
       t_V_abij: ctf tensor. vvoo 2-body integrals
       leve_shift: float.
       sp: 0 or 1. Sparsity of ctf tensors
       nv_part_size: integer. The partition size of the virtual index in calculating the MP2 energies, to save memory.
                     The default value is 0, which means no partition is used. It will be set to nv in the algorithm.
    """

    algoName = "mp2.solve"
    timeMp2 = time.time()
    print_logging_info(algoName,level=0)

    if "debug_level" in kwargs:
        debug_level = kwargs["debug_level"]
    else:
        debug_level = 3

    no = t_epsilon_i.size
    nv = t_epsilon_a.size

    # the following ctf expression calcs the outer sum, as wanted.
    print_logging_info("Creating D_abij", level = 1, debug_level=debug_level)

    # memory efficient implementation for sparse V_abij, validity for dense still need to be tested.
    print_logging_info("Calculating T_abij", level = 1, debug_level=debug_level)
    inds, vals = t_V_abij.read_local_nnz()
    del t_V_abij
    epsilon_i = t_epsilon_i.to_nparray()
    epsilon_a = t_epsilon_a.to_nparray()

    print_logging_info("Looping through nnz in t_V_abij", level = 1)
    print_logging_info("Total nnz entries on rank 0 = ", len(inds), level = 1)

    for ind in range(len(inds)):
        if ind % num_proc == 0:
            print_logging_info("Completed {:.2f} percent...".format(ind/len(inds)*100), level=2, debug_level=debug_level)
        global_ind = inds[ind]
        [a, b, i, j] = get_orb_inds(global_ind, [nv, nv, no, no])
        vals[ind] /= (epsilon_i[i] + epsilon_i[j] - epsilon_a[a] - epsilon_a[b] + leve_shift)

    del epsilon_i, t_epsilon_i
    del epsilon_a, t_epsilon_a

    t_T_abij = np.zeros([nv,nv,no,no], dtype=t_V_ijab.dtype)
    t_T_abij.write(inds, vals)

    if nv_part_size is None:
        n_part = 1
        nv_part_size = nv
    else:
        n_part = int(nv//nv_part_size) + 1
    eDir = 0.
    eExc = 0.
    print_logging_info("Summing direct and exchange contributions", level=1)
    print_logging_info("Partitioning T and V tensors", level=1)
    for n in range(n_part):
        n_lower = n * nv_part_size
        if n_lower >= nv:
            break
        n_higher = (n + 1) * nv_part_size
        if n_higher > nv:
            n_higher = nv
        print_logging_info("n_lower = ", n_lower, ", n_higher = ", n_higher, level = 2)
        t_T_nmij = t_T_abij[n_lower:n_higher, :, :, :]
        t_V_ijnm = t_V_ijab[:, :, n_lower:n_higher, :]

        eDir += 2.0*einsum('abij, ijab->',t_T_nmij, t_V_ijnm)
        eExc += -1.0*einsum('abij, jiab->',t_T_nmij, t_V_ijnm)


    print_logging_info("Direct contribution = {:.12f}".format(np.real(eDir)),\
                       level=1)
    print_logging_info("Exchange contribution = {:.12f}".format(np.real(eExc)),\
                       level=1)
    print_logging_info("MP2 energy = {:.12f}".format(np.real(eDir+eExc)), level=1)
    print_logging_info("{:.3f} seconds spent on "\
                       .format((time.time()-timeMp2))+algoName, level=1)
    return [eDir+eExc, t_T_abij]

def get_orb_inds(global_ind, dims):
    """
    Args:
        global_ind: int, the global index of an entry on a ctf tensor
        dims: list of ints, the dimensions of the ctf tensor

    Returns:
        inds: list of ints, the corresponding indices of the entry on the tensor
    """

    inds = []
    for i in range(len(dims)):
        ind = global_ind // np.prod(dims[i+1:])
        global_ind -= ind * np.prod(dims[i+1:])
        inds.append(int(ind))
    return inds