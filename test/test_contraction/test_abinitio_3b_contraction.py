import ctf
from ctf.core import *
import numpy as np

from pymes.logging import print_logging_info
from pymes.util import tcdump
from pymes.integral import contraction
from pymes.util import fcidump
from pymes.mean_field import hf

def main(fcidump_file="FCIDUMP.c2", tcdump_file="TCDUMP.c2"):

    # first read in a TCDUMP file
    t_V_opqrst = tcdump.read_from_tcdump(tcdump_file, sp=0)
     
    n_elec, n_orb, e_core, epsilon, h_pq, V_pqrs = \
        fcidump.read_fcidump(fcidump_file,is_tc=True)
    no = int(n_elec/2)
    print("no=",no)
    t_h_pq = ctf.astensor(h_pq)
    t_V_pqrs = ctf.astensor(V_pqrs)
    hf_e = hf.calc_hf_e(no, e_core, t_h_pq, t_V_pqrs)
    #t_S_pqrs = contraction.get_single_contraction(no, t_V_opqrst)
    #t_D_pq = contraction.get_double_contraction(no, t_V_opqrst)
    #no=1
    t_T_0 = contraction.get_triple_contraction(no, t_V_opqrst)
    #hf_e -= t_T_0
    
    print("Reference energy = -75.6192902200410")
    print("One body contr: <D0|T|D0> = -131.925903392947")
    print("Two body contr: <D0|U|D0> = 40.9942285257871")
    print("One and two body contr: ", -131.925903392947+40.9942285257871+e_core)
    print("Three body contr: <D0|L|D0> = -1.991281213307161E-002")

    print("Calculated from integrals: <D0|T+U|D0> =", hf_e)
    print("Calculated from integrals: <D0|L|D0> =", -t_T_0)
    print("Calculated reference energy: <D0|H|D0> = ", hf_e-t_T_0)


if __name__ == '__main__':
    main(fcidump_file="FCIDUMP", tcdump_file="TCDUMP")