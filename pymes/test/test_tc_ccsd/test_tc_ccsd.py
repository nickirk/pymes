import ctf
from ctf import *
import numpy as np
from pymes.util import fcidump, tcdump
from pymes.solver import ccsd, ccd, mp2
from pymes.mean_field import hf
from pymes.integral import contraction

def test_tc_ref_energy(fcidump_file="FCIDUMP.LiH.tc", tcdump_file="TCDUMP.LiH_FNO", ref_e=None):
    # known values
    ref_e = -8.042996662464
    n_elec, nb, e_core, e_orb, h_pq, V_pqrs = fcidump.read(fcidump_file, is_tc=True)
    no = int(n_elec / 2)

    t_L_opqrst = tcdump.read(tcdump_file, sp=0)
    t_T_0 = contraction.get_triple_contraction(no, t_L_opqrst)
    
    t_V_pqrs = ctf.astensor(V_pqrs)
    t_h_pq = ctf.astensor(h_pq)



    # make sure HF energy is correct first
    hf_e = hf.calc_hf_e(no, e_core, t_h_pq, t_V_pqrs)
    hf_e += t_T_0
    print(hf_e)
    assert np.abs(ref_e - hf_e) < 1.e-8

    # then MP2 and CCSD energies
    t_fock_pq = hf.construct_hf_matrix(no, t_h_pq, t_V_pqrs)

    # correction to Fock matrix
    t_delta_eps = contraction.get_double_contraction(no, t_L_opqrst)
    print(t_delta_eps)
    t_fock_pq += t_delta_eps
    t_V_pqrs += contraction.get_single_contraction(no, t_L_opqrst)
    mycc = ccsd.CCSD(no)
    ccsd_e = mycc.solve(t_fock_pq, t_V_pqrs)["ccsd e"]
    #if ref_e is None:
    #    ref_e = -0.02035412567214456
    #assert np.abs(ccsd_e - ref_e) < 1.e-7


#def test_tc_ccsd_fno(fcidump_file="fcidump.no"):

