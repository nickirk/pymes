import time
import numpy as np
import ctf
from ctf.core import *
from pymes.solver import ccd

def solve(tGamma_pqG, tEpsilon_i, tEpsilon_a, fDcd=True):
    '''
    dcd algorithm
    tV_ijkl = V^{ij}_{kl}
    tV_abij = V^{ab}_{ij}
    tT_abij = T^{ab}_{ij}
    the upper indices refer to conjugation
    '''
    algoName = "dcd.solve"
    timeDcd = time.time()
    return ccd.solve(tGamma_pqG, tEpsilon_i, tEpsilon_a, fDcd)

