import time
import numpy as np
import ctf
import os, psutil

from pymes.log import print_logging_info

def solve(tEpsilon_i, tEpsilon_a, t_V_ijab, t_V_abij, levelShift=0., sp=0):
    '''
    mp2 algorithm
    '''
    algoName = "mp2.solve"
    world = ctf.comm()
    timeMp2 = time.time()
    print_logging_info(algoName,level=0)

    no = tEpsilon_i.size
    nv = tEpsilon_a.size

    #t_T_abij = t_V_abij.copy()
    t_T_abij = ctf.tensor([nv,nv,no,no],dtype=t_V_abij.dtype,sp=t_V_abij.sp)


    # the following ctf expression calcs the outer sum, as wanted.
    print_logging_info("Creating D_abij", level = 1)
    process = psutil.Process(os.getpid())

    # memory efficient implementation for sparse V_abij, validity for dense still need to be tested.
    print_logging_info("Calculating T_abij", level = 1)
    inds, vals = t_V_abij.read_local_nnz()
    del t_V_abij
    epsilon_i = tEpsilon_i.to_nparray()
    epsilon_a = tEpsilon_a.to_nparray()

    print_logging_info("Looping through nnz in t_V_abij", level = 1)
    print_logging_info("Total nnz entries on rank 0 = ", len(inds), level = 1)
    num_proc = world.np()

    for ind in range(len(inds)):
        if ind % num_proc == 0:
            print_logging_info("Completed ", ind/len(inds)*100, " percent...", level=2)
            print_logging_info("Current memory usage ", process.memory_info().rss / 1024. ** 2, " MB", level=2)
        global_ind = inds[ind]
        [a, b, i, j] = get_orb_inds(global_ind, [nv, nv, no, no])
        vals[ind] /= (epsilon_i[i] + epsilon_i[j] - epsilon_a[a] - epsilon_a[b] + levelShift)

    t_T_abij.write(inds, vals)

    #t_T_abij.i("abij") << tEpsilon_i.i("i") + tEpsilon_i.i("j")-tEpsilon_a.i("a")-tEpsilon_a.i("b")
    #t_D_abij = ctf.tensor([no,no,nv,nv],dtype=complex, sp=1)
    #t_T_abij = 1./(t_T_abij+levelShift)
    # why the ctf contraction is not used here?
    # let's see if the ctf contraction does the same job

    # the implementation below is memory heavy
    #t_T_abij = ctf.einsum('abij,abij->abij', t_V_abij, t_T_abij)
    #t_T_abij = t_V_abij / (t_T_abij + levelShift)



    print_logging_info("Calculating direct energy", level = 1)
    eDir  = 2.0*ctf.einsum('abij,ijab->',t_T_abij, t_V_ijab)
    print_logging_info("Calculating exchange energy", level = 1)
    eExc = -1.0*ctf.einsum('abij,ijba->',t_T_abij, t_V_ijab)
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