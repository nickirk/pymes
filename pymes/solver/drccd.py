import time
import numpy as np

from pymes.solver import mp2
from pymes.log import print_logging_info
from functools import partial

einsum = partial(np.einsum, optimize=True)

def get_residual(tEpsilon_i, tEpsilon_a, tT_abij, tV_abij, tV_aijb, tV_iabj, \
        tV_ijab):

    algoName = "drccd.getResidual"
    no = tT_abij.shape[-1]
    nv = tT_abij.shape[0]

    tFock_ab = np.zeros([nv,nv], dtype=tEpsilon_a.dtype)
    tFock_ij = np.zeros([no,no], dtype=tEpsilon_i.dtype)
    tFock_ab = np.diag(tEpsilon_a)
    tFock_ij = np.diag(tEpsilon_i)
    tR_abij = np.zeros([nv,nv,no,no],dtype=tV_abij.dtype)

    #tR_abij.i("abij") << tV_abij.i("abij") + tFock_ab.i("ad")*tT_abij.i("dbij")\
    #        - tFock_ij.i("ik")*tT_abij.i("abkj") \
    #        + tFock_ab.i("bd")*tT_abij.i("daji")\
    #        - tFock_ij.i("jk")*tT_abij.i("baki") \
    #        +tV_aijb.i("akic") * tT_abij.i("cbkj")\
    #        + tV_iabj.i("kbcj") * tT_abij.i("acij") \
    #        + tT_abij.i("acij")*tV_ijab.i("klcd") * tT_abij.i("dblj")
    tR_abij = tV_abij + einsum("ad, dbij -> abij", tFock_ab, tT_abij) 
    tR_abij -= einsum("ik, abkj -> abij", tFock_ij, tT_abij)
    tR_abij += einsum("bd, daji -> abij", tFock_ab, tT_abij)
    tR_abij -= einsum("jk, baki -> abij", tFock_ij, tT_abij)
    tR_abij += einsum("akic, cbkj -> abij", tV_aijb, tT_abij)
    tR_abij += einsum("kbcj, acij -> abij", tV_iabj, tT_abij)
    tR_abij += einsum("acij, klcd, dblj -> abij", tT_abij, tV_ijab, tT_abij)
    

    return tR_abij

def getEnergy(tT_abij, tV_ijab):
    '''
    calculate the dr-CCD energy, using the converged amplitudes
    '''
    tDirCcdE = 2. * einsum("abij, ijab ->", tT_abij, tV_ijab)
    tExCcdE  = 0.
    #tExCcdE  = -1. * einsum("abij, ijba ->", tT_abij, tV_ijab)
    return [tDirCcdE, tExCcdE]
