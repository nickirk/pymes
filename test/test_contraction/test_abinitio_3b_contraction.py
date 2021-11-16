import ctf
from ctf.core import *
import numpy as np

from pymes.logging import print_logging_info
from pymes.util import tcdump
from pymes.integral import contraction
from pymes.util import fcidump
from pymes.mean_field import hf

def main(file_name):

    # first read in a TCDUMP file
    t_V_opqrst = tcdump.read_from_tcdump(file_name)
     
    n_elec, n_orb, e_core, epsilon, h_pq, V_pqrs = fcidump.read_fcidump(is_tc=True)
    no = int(n_elec/2)
    print("no=",no)
    #t_S_pqrs = contraction.get_single_contraction(no, t_V_opqrst)
    #t_D_pq = contraction.get_double_contraction(no, t_V_opqrst)
    t_T_0 = contraction.get_triple_contraction(no, t_V_opqrst)
    t_h_pq = ctf.astensor(h_pq)
    t_V_pqrs = ctf.astensor(V_pqrs)
    hf_e = hf.calc_hf_e(no, e_core, t_h_pq, t_V_pqrs)
    #hf_e += t_T_0
    print(t_h_pq)
    print("Reference energy = -75.47094")
    print("Calculated from integrals =", hf_e)


if __name__ == '__main__':
    main("TCDUMP")