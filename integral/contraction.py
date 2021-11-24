import ctf
from ctf.core import *
import itertools

from pymes.logging import print_logging_info



'''
This module deal with the contractions in the 3-body integrals arising
from the transcorrelation.
Expressions derived using diagrams. Assuming the 3-body term is
-1/6 L^{opq}_{rst}


Author: Ke Liao <ke.liao.life@gmail.com>
        Evelin Schristlmaier
'''


def get_single_contraction(no, t_L_opqrst):
    nb = t_L_opqrst.shape[0]
    t_D_pqrs = ctf.tensor([nb,nb,nb,nb], dtype=t_L_opqrst.dtype, sp=t_L_opqrst.sp)
    # hole lines = 1, loops = 1, sign = 1, equavilent diagrams= 3
    t_D_pqrs += 2**1*3.0*ctf.einsum("ipqirs->pqrs", t_L_opqrst[:no,:,:,:no,:,:])
    # hole lines = 1, loops = 0, sign = -1, equavilent diagrams= 3
    t_D_pqrs += -3.0*ctf.einsum("ipqirs->pqrs", t_L_opqrst[:no,:,:,:no,:,:])

    return t_D_pqrs

def get_double_contraction(no, t_L_opqrst):
    nb = t_L_opqrst.shape[0]
    t_S_pq = ctf.tensor([nb, nb], dtype=t_L_opqrst.dtype, sp=t_L_opqrst.sp)
    # hole lines = 2, loops = 2, sign = 1, spin fac = 2**2, equavilent diagrams= 3
    t_S_pq += 2.0**2*3.0*ctf.einsum("ijpijq->pq", t_L_opqrst[:no,:no,:,:no,:no,:])
    # hole lines = 2, loops = 0, sign = 1, spin fac = 2**0, equavilent diagrams= 3
    t_S_pq += 3.0*ctf.einsum("ijpqij->pq", t_L_opqrst[:no,:no,:,:,:no,:no])
    # hole lines = 2, loops = 1, sign = -1, spin fac = 2**1, equavilent diagrams= 3
    t_S_pq += -2**1*3.0*ctf.einsum("ijpjiq->pq", t_L_opqrst[:no,:no,:,:no,:no,:])
    # hole lines = 2, loops = 1, sign = -1, equavilent diagrams= 3
    t_S_pq += -2**1*3.0*ctf.einsum("pjiijq->pq", t_L_opqrst[:,:no,:no,:no,:no,:])

    return t_S_pq

def get_triple_contraction(no, t_L_opqrst):
    
    t_T_0 = 0.
    print_logging_info("Triple contraction")
    # hole lines = 3, loops =3, (-1)^(3+3)=1, spin fac = 2**3, equ diagrams = 1, 2*6 from sym tensor
    t_L_ijkijk = t_L_opqrst[:no,:no,:no,:no,:no,:no]
    t_T_0 = 2**3*ctf.einsum("ijkijk->", t_L_opqrst[:no,:no,:no,:no,:no,:no])
    # hole lines = 3, loops =2, (-1)^(3+2)=-1, spin fac = 2**2, equ diagrams = 3
    t_T_0 -= 2**2*3.0*ctf.einsum("ijkjik->", t_L_opqrst[:no,:no,:no,:no,:no,:no]) 
    # hole lines = 3, loops =1, (-1)^(3+1)=1, spin fac = 2**1, equ diagrams = 2 (mirror)
    t_T_0 += 4.0*ctf.einsum("ijkjki->", t_L_opqrst[:no,:no,:no,:no,:no,:no])
    
    return t_T_0

def recover_L(t_L_sym_opqrst):
    '''
    This function handles the 48-fold symmetry presents in L tensor
    
    Parameters:
    -----------
    t_L_sym_opqrst: a part of a symmetric ctf sparse tensor
    

    Returns:
    --------
    t_L_opqrst: ctf full tensor
    '''
    world = ctf.comm()
    t_L_opqrst = ctf.tensor(t_L_sym_opqrst.shape, sp=1)
    inds_sym, vals_sym = t_V_opqrst.read_local_nnz()

    inds_full = []
    vals_full = []
    shape = t_L_sym_opqrst.shape
    for idx, val in inds_sym, vals_sym:
        list_ind = global_ind_2_list_inds(idx, )shape)
        equiv_list_inds = gen_sym_idx(list_ind)

def global_ind_2_list_inds(global_ind, shape):
    '''
    Decompose global ctf indices to individual indices

    Parameters:
    -----------
    global_ind: int
    shape: list of ints, shape of the ctf tensor
    
    Returns:
    list_inds: list of ints, length the same as the length of shape, the individual indices
    '''
    list_inds = []

    for n in range(len(shape)-1):
        list_inds.append(int(global_ind//np.prod(shape[n+1:])))
        global_ind -= list_inds[-1]*np.prod(shape[n+1:])
    list_inds.append(global_ind)

    return list_inds

def list_inds_2_global_ind(list_inds, shape):
    '''
    Calculate the global index of a list of indices of a tensor

    Parameters:
    -----------
    list_inds: list of int
    shape: list of ints, shape of the ctf tensor
    
    Returns:
    global_ind: int, the global index of a non-zero entry
    '''
    global_ind = 0

    for i in range(len(list_inds)-1):
        global_ind += i*np.prod(shape[i+1:])
    
    global_ind += list_inds[-1]

    return global_ind

def gen_sym_inds(list_inds):
    '''
    This function generates all the indices related by the 48-fold symmetries.

    Parameters:
    -----------
    list_inds: a list of ints, the indices of a tensor entry

    Returns:
    --------
    sym_related_inds: list of list of ints, all the possible symmetry related inds
    '''
    # two indices exchange symmetry
    sym_related_inds = []
    
    for i in range(3):
        tmp_inds = list_inds
        tmp_inds[i], tmp_inds[i+3] = tmp_inds[i+3], tmp_inds[i]
        sym_related_inds.append(tmp_inds)
    
    # electron exchange symmetries
    for per in itertools.permutations(range(3)):
        per = list[per]+[x+3 for x in per]
        tmp_inds = list_inds[per]
        sym_related_inds.append(tmp_inds)
    