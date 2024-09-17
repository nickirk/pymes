import time

from pymes.solver import ccd
import ctf


class DCD(ccd.CCD):
    def __init__(self, no, delta_e=1e-8, is_dcd=True, is_diis=True, is_dr_ccd=False, is_bruekner=False):
        super().__init__(no, delta_e, is_dcd, is_diis, is_dr_ccd, is_bruekner)
    
