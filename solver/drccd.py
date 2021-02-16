import time
import numpy as np
import ctf
from ctf.core import *
from pymes.solver import mp2
from pymes.mixer import diis
from pymes.logging import print_logging_info

def solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=0., sp=0,  fDcd=True, maxIter=100, fDiis=True, amps=None, bruekner=False, epsilonE=1e-8):
    '''
    drccd algorithm
    tV_ijkl = V^{ij}_{kl}
    tV_abij = V^{ab}_{ij}
    tT_abij = T^{ab}_{ij}
    the upper indices refer to conjugation
    '''
    algoName = "drccd.solve"
    timeDcd = time.time()
    return ccd.solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=levelShift, sp=sp,  fDrccd=True, maxIter=maxIter, fDiis=fDiis,amps=amps, bruekner=bruekner, epsilonE=epsilonE)


def getResidual(tT_abij, tV_abij, tV_aijb, tV_iabj, \
        tV_ijab):

    algoName = "drccd.getResidual"
    no = tT_abij.shape[-1]
    nv = tT_abij.shape[0]
    tR_abij = ctf.tensor([nv,nv,no,no],dtype=tV_klij.dtype,sp=tT_abij.sp)

    tR_abij.i("abij") << tV_abij.i("abij") + tV_aijb.i("akic") * tT_abij.i("cbkj")\
            + tV_iabj.i("kbcj") * tT_abij.i("acij") \
            + tT_abij.i("acij")*tV_ijab.i("klcd") * tT_abij.i("dblj") 

    return tR_abij

def getEnergy(tT_abij, tV_ijab):
    '''
    calculate the dr-CCD energy, using the converged amplitudes
    '''
    tDirCcdE = 2. * ctf.einsum("abij, ijab ->", tT_abij, tV_ijab)
    tExCcdE  = 0.
    #tExCcdE  = -1. * ctf.einsum("abij, ijba ->", tT_abij, tV_ijab)
    return [tDirCcdE, tExCcdE]
