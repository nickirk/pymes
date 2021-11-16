import ctf
from ctf.core import *



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
    # hole lines = 3, loops =3, (-1)^(3+3)=1, spin fac = 2**3, equ diagrams = 1
    t_T_0 = 2**3*ctf.einsum("ijkijk->", t_L_opqrst[:no,:no,:no,:no,:no,:no])
    # hole lines = 3, loops =2, (-1)^(3+2)=-1, spin fac = 2**2, equ diagrams = 3
    t_T_0 -= 2**2*3.0*ctf.einsum("ijkjik->", t_L_opqrst[:no,:no,:no,:no,:no,:no]) 
    # hole lines = 3, loops =1, (-1)^(3+1)=1, spin fac = 2**1, equ diagrams = 1
    t_T_0 += 2.0*ctf.einsum("ijkjki->", t_L_opqrst[:no,:no,:no,:no,:no,:no])
    return t_T_0