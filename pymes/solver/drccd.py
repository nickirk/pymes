import time
import numpy as np

from pymes.solver import mp2
#from pymes.solver import ccd
from pymes.log import print_logging_info

#def solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=0., sp=0,  fDcd=True, maxIter=100, fDiis=True, amps=None, bruekner=False, epsilonE=1e-8):
#    '''
#    drccd algorithm
#    tV_ijkl = V^{ij}_{kl}
#    tV_abij = V^{ab}_{ij}
#    tT_abij = T^{ab}_{ij}
#    the upper indices refer to conjugation
#    '''
#    algoName = "drccd.solve"
#    timeDcd = time.time()
#    return ccd.solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=levelShift, sp=sp,  fDrccd=True, maxIter=maxIter, fDiis=fDiis,amps=amps, bruekner=bruekner, epsilonE=epsilonE)


def getResidual(tEpsilon_i, tEpsilon_a, tT_abij, tV_abij, tV_aijb, tV_iabj, \
        tV_ijab):

    algoName = "drccd.getResidual"
    no = tT_abij.shape[-1]
    nv = tT_abij.shape[0]

    tFock_ab = np.zeros([nv,nv], dtype=tEpsilon_a.dtype)
    tFock_ij = np.zeros([no,no], dtype=tEpsilon_i.dtype)
    tFock_ab = np.diagonal(tEpsilon_a)
    tFock_ij = np.diagonal(tEpsilon_i)
    tR_abij = np.zeros([nv,nv,no,no],dtype=tV_abij.dtype)

    #tR_abij.i("abij") << tV_abij.i("abij") + tFock_ab.i("ad")*tT_abij.i("dbij")\
    #        - tFock_ij.i("ik")*tT_abij.i("abkj") \
    #        + tFock_ab.i("bd")*tT_abij.i("daji")\
    #        - tFock_ij.i("jk")*tT_abij.i("baki") \
    #        +tV_aijb.i("akic") * tT_abij.i("cbkj")\
    #        + tV_iabj.i("kbcj") * tT_abij.i("acij") \
    #        + tT_abij.i("acij")*tV_ijab.i("klcd") * tT_abij.i("dblj")
    tR_abij = tV_abij + np.einsum("ad, dbij -> abij", tFock_ab, tT_abij) 
    tR_abij -= np.einsum("ik, abkj -> abij", tFock_ij, tT_abij)
    tR_abij += np.einsum("bd, daji -> abij", tFock_ab, tT_abij)
    tR_abij -= np.einsum("jk, baki -> abij", tFock_ij, tT_abij)
    tR_abij += np.einsum("akic, cbkj -> abij", tV_aijb, tT_abij)
    tR_abij += np.einsum("kbcj, acij -> abij", tV_iabj, tT_abij)
    tR_abij += np.einsum("acij, klcd, dblj -> abij", tT_abij, tV_ijab, tT_abij)
    

    return tR_abij

def getEnergy(tT_abij, tV_ijab):
    '''
    calculate the dr-CCD energy, using the converged amplitudes
    '''
    tDirCcdE = 2. * np.einsum("abij, ijab ->", tT_abij, tV_ijab)
    tExCcdE  = 0.
    #tExCcdE  = -1. * np.einsum("abij, ijba ->", tT_abij, tV_ijab)
    return [tDirCcdE, tExCcdE]
