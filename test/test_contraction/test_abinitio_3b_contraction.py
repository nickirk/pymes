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
    t_L_orpsqt = tcdump.read_from_tcdump(tcdump_file, sp=1)
    t_L_orpsqt_d = tcdump.read_from_tcdump(tcdump_file, sp=0)

    assert(np.allclose(t_L_orpsqt.to_nparray(), t_L_orpsqt_d.to_nparray()))
    tcdump.write_2_tcdump(t_L_orpsqt, file_name="TCDUMP.w")
    
    print("sym of L =", t_L_orpsqt.sym)
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
    t_T_0 = contraction.get_triple_contraction(no, t_L_orpsqt)
    #hf_e -= t_T_0
    e_ref = -14.6160234970329
    e_one_body = -19.4188606946810
    e_two_body = 4.78424329682590
    e_three_body = 1.859390082220713E-002
    print("Reference energy = ", e_ref)
    print("One body contr: <D0|T|D0> = ", e_one_body)
    print("Two body contr: <D0|U|D0> = ", e_two_body)
    print("One and two body contr= ", e_one_body+e_two_body+e_core)
    print("Three body contr: <D0|L|D0> = ", e_three_body)

    print("Calculated from integrals: <D0|T+U|D0> =", hf_e)
    print("Calculated from integrals: <D0|L|D0> =", t_T_0)
    print("Calculated from integrals: <D0|L|D0>/ref =", t_T_0/e_three_body)
    print("Calculated reference energy: <D0|H|D0> = ", hf_e+t_T_0)

    t_T_0 = contraction.get_triple_contraction(no, t_L_orpsqt_d)
    #hf_e -= t_T_0
    e_ref = -14.6160234970329
    e_one_body = -19.4188606946810
    e_two_body = 4.78424329682590
    e_three_body = 1.859390082220713E-002
    print("Reference energy = ", e_ref)
    print("One body contr: <D0|T|D0> = ", e_one_body)
    print("Two body contr: <D0|U|D0> = ", e_two_body)
    print("One and two body contr= ", e_one_body+e_two_body+e_core)
    print("Three body contr: <D0|L|D0> = ", e_three_body)

    print("Calculated from integrals: <D0|T+U|D0> =", hf_e)
    print("Calculated from integrals: <D0|L|D0> =", t_T_0)
    print("Calculated from integrals: <D0|L|D0>/ref =", t_T_0/e_three_body)
    print("Calculated reference energy: <D0|H|D0> = ", hf_e+t_T_0)

if __name__ == '__main__':
    main(fcidump_file="FCIDUMP", tcdump_file="TCDUMP")
    main(fcidump_file="FCIDUMP", tcdump_file="TCDUMP.w")