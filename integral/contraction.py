import ctf
from ctf.core import *
import itertools

from pymes.logging import print_logging_info



'''
This module deal with the contractions in the 3-body integrals arising
from the transcorrelation.
Expressions derived using diagrams. Assuming the 3-body term is
- L^{opq}_{rst}


Author: Ke Liao <ke.liao.whu@gmail.com>
        Evelin Schristlmaier
'''


def get_single_contraction(no, t_L_opqrst):
    nb = t_L_opqrst.shape[0]
    t_D_pqrs = ctf.tensor([nb,nb,nb,nb], dtype=t_L_opqrst.dtype, sp=t_L_opqrst.sp)
    # hole lines = 1, loops = 1, sign = 1, equavilent diagrams= 3
    t_D_pqrs += 2**1*3.0*ctf.einsum("ipqirs->pqrs", t_L_opqrst[:no,:,:,:no,:,:])
    # hole lines = 1, loops = 0, sign = -1, equavilent diagrams= 3
    # TODO: wrong diagram/expression, it cannot be the same as the previous one
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

def get_triple_contraction(no, t_L_orpsqt):
    '''
    Parameters:
    -----------
    no: int, number of occupied orbitals
    t_L_orpsqt: ctf tensor, sym = [SY,NS,SY,NS,SY,NS]

    Returns:
    --------
    t_T_0: float
    '''
    
    t_T_0 = 0.
    
    print_logging_info("Triple contraction")
    # hole lines = 3, loops =3, (-1)^(3+3)=1, spin fac = 2**3, equ diagrams = 1, 2*6 from sym tensor

    t_L_ijklmn = ctf.tensor([no,no,no,no,no,no]) 
    t_L_ijklmn = t_L_orpsqt[:no,:no,:no,:no,:no,:no]

    t_T_0 += 2.**3*ctf.einsum("iijjkk->", t_L_ijklmn)
    # hole lines = 3, loops =2, (-1)^(3+2)=-1, spin fac = 2**2, equ diagrams = 3
    t_T_0 += -2**2*3.0*ctf.einsum("ijjikk->", t_L_ijklmn)
    # hole lines = 3, loops =1, (-1)^(3+1)=1, spin fac = 2**1, equ diagrams = 2 (mirror)
    t_T_0 += 2.0*2.0*ctf.einsum("ijjkki->", t_L_ijklmn)
    
    return -t_T_0/6.


def recover_L(t_L_sym_opqrst, shape):
    '''
    This function handles the 48-fold symmetry presents in L tensor
    
    Parameters:
    -----------
    t_L_sym_opqrst: the symmetric part of a symmetric ctf sparse tensor
    shape: list of ints, the shape of the desired slice of the full tensor

    Returns:
    --------
    t_L_opqrst: ctf full tensor, with shape equal to the requested one
    '''
    world = ctf.comm()
    t_L_opqrst = ctf.tensor(t_L_sym_opqrst.shape, sp=1)
    inds_sym, vals_sym = t_V_opqrst.read_local_nnz()

    inds_full = []
    vals_full = []
    shape = t_L_sym_opqrst.shape
    for idx, val in inds_sym, vals_sym:
        list_ind = global_ind_2_list_inds(idx, shape)
        equiv_list_inds = gen_sym_idx(list_ind)

    return t_L_opqrst


def global_ind_2_list_inds(global_ind, shape):
    '''
    Decompose global ctf indices to individual indices

    Parameters:
    -----------
    global_ind: int
    shape: list of ints, shape of the ctf tensor
    
    Returns:
    --------
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
    --------
    global_ind: int, the global index of a non-zero entry
    '''
    global_ind = 0

    for i in range(len(list_inds)-1):
        global_ind += i*np.prod(shape[i+1:])
    
    global_ind += list_inds[-1]

    return global_ind


def gen_sym_int_inds(list_inds):
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

    return

def sym_contraction(ein_inds, t_L_opqrst):
    '''
    ein_inds: string, containing the indices to contract over
    '''
    return


def gen_sym_str_inds(string_inds):
    '''
    This function generates all 48 symmetry related indices.

    Parameters:
    -----------
    string_inds: string, containing six characters

    Returns:
    --------
    full_sym_inds: list of strings
    '''
    full_sym_inds = [string_inds]
    # exchange of two indices
    i = 0
    while i < 3:
        num_ele = len(full_sym_inds)
        for ind in range(num_ele):
            tmp_inds = list(full_sym_inds[ind])
            tmp_inds[i], tmp_inds[i+3] = tmp_inds[i+3], tmp_inds[i]
            full_sym_inds.append(''.join(tmp_inds))
        i += 1
        
    # exchange of a pair of indices
    tmp_inds = []
    for i in full_sym_inds:
        i = list(i)
        for per_1, per_2 in zip(itertools.permutations(i[:3]), itertools.permutations(i[3:])):
            
            per = ''.join(per_1+per_2)
            tmp_inds.append(per)
    full_sym_inds = tmp_inds
    return full_sym_inds

def gen_sym_diag_str_inds(string_inds, sorted_string=None):
    '''
    This function generates the overlap block of string_inds relative to the sorted string.
    E.g. if string_inds = "ibcajk", the sorted string is "abcijk". This function detects "a" and "i"
    are swapped, so it will output abcajk. There are two types of symmetry operations among the six characters,
    1. exchange between a and i, b and j, c and k
    2. exchange between the pairs (a,i) and (b,j) and (c,k)

    If any of the two operations relative to the sorted strings are detected, the exchanged indices will be 
    set to be the same. For example, if a and i are exchanged, we will set i->a; if (a,i) and (b,j) are exchanged,
    we will set (b,j)->(a,i). The two operations can be compounded. 

    Parameters:
    -----------
    string_inds: string, containing six characters, 

    Returns:
    --------
    full_sym_inds: list of strings
    '''
    if sorted_string is None:
        sorted_string = sorted(list(string_inds))
    string_inds = list(string_inds)
    output_string = string_inds.copy()
    # first detect if any type 1 symmetries are applied:
    for i in range(3):
        if string_inds[i+3] <= string_inds[i]:
            output_string[i], output_string[i+3] = output_string[i+3], output_string[i+3]
            string_inds[i], string_inds[i+3] = string_inds[i+3], string_inds[i]
            print(string_inds, output_string)
    
    # now swaping pairs symmetry
    for j in range(3):
        for i in range(2):
            if string_inds[i] >= string_inds[i+1]:
                string_inds[i], string_inds[i+1] = string_inds[i+1], string_inds[i]
                string_inds[i+3], string_inds[i+4] = string_inds[i+4], string_inds[i+3]
                output_string[i], output_string[i+1] = output_string[i+1], output_string[i+1]
                output_string[i+3], output_string[i+4] = output_string[i+4], output_string[i+4]
                print(string_inds, output_string)
    output_string = ''.join(output_string)
    assert(string_inds == sorted_string)

    return output_string