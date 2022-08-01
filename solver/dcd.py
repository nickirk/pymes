import time

from ctf.core import *
from pymes.solver import ccd


def solve(t_epsilon_i, t_epsilon_a, t_V_pqrs, level_shift=0., sp=0, \
          is_dcd=True, max_iter=100, is_diis=True, amps=None, is_bruekner=False, \
          epsilon_e=1e-8):
    '''
    dcd algorithm
    t_V_ijkl = V^{ij}_{kl}
    t_V_abij = V^{ab}_{ij}
    tT_abij = T^{ab}_{ij}
    the upper indices refer to conjugation
    '''
    algo_name = "dcd.solve"
    time_dcd = time.time()
    return ccd.solve(t_epsilon_i, \
                     t_epsilon_a, \
                     t_V_pqrs, \
                     level_shift=level_shift,\
                     sp=sp, \
                     is_dcd=True, \
                     max_iter=max_iter, \
                     is_diis=is_diis, \
                     amps=amps, \
                     is_bruekner=is_bruekner, \
                     epsilon_e=epsilon_e)

