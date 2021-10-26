import ctf
from ctf.core import *



'''
This module deal with the contractions in the 3-body integrals arising
from the transcorrelation.
'''

def get_single_contraction(no, t_V_opqrst):
    nb = t_V_opqrst.shape[0]
    t_D_pqrs = ctf.tensor([nb,nb,nb,nb], dtype=t_V_opqrst.dtype, sp=t_V_opqrst.sp)
    return t_D_pqrs

def get_double_contraction(no, t_V_opqrst):
    t_S_pq = ctf.tensor([nb, nb], dtype=t_V_opqrst.dtype, sp=t_V_opqrst.sp)
    return t_S_pq

def get_triple_contraction(no, t_V_opqrst):
    t_T_0 = 0.
    return t_T_0