import numpy as np
from pymes.integral import contraction
from pymes.mean_field import hf
from pymes.solver import ccsd
from pymes.util import fcidump, tcdump

# see test_ccsd
# 3-body integrals are not used in practice
# this test was based on the old code

# for new xTC FCIDUMP files, ccsd and other solvers can be used 
# as normal CCSD and other solvers to calculate TC energies

# deprecated

print("Warning: test_tc_ccd.py is deprecated. ")
def test_tc_ref_energy(fcidump_file="pymes/test/test_tc_ccsd/FCIDUMP.LiH.tc", tcdump_file="pymes/test/test_tc_ccsd/TCDUMP.LiH_FNO", ref_e = -8.042996662464):
    # known values

    n_elec, nb, e_core, e_orb, h_pq, V_pqrs = fcidump.read(fcidump_file, is_tc=True)
    no = int(n_elec / 2)

    t_L_opqrst = tcdump.read(tcdump_file, sp=0)
    t_T_0 = contraction.get_triple_contraction(no, t_L_opqrst)
    print(t_T_0)
    t_V_pqrs = V_pqrs
    t_h_pq = h_pq

    # make sure HF energy is correct first
    hf_e = hf.calc_hf_e(no, e_core, t_h_pq, t_V_pqrs)
    print("HF e without T_0 = ", hf_e)
    hf_e += t_T_0
    print("HF e + T_0 = ", hf_e)

    assert np.abs(ref_e - hf_e) < 1.e-8
    return 0


def test_tc_ccsd_energy(fcidump_file="pymes/test/test_tc_ccsd/FCIDUMP.LiH.tc", tcdump_file="pymes/test/test_tc_ccsd/TCDUMP.LiH_FNO", ref_e = -0.010391224684):

    n_elec, nb, e_core, e_orb, h_pq, V_pqrs = fcidump.read(fcidump_file, is_tc=True)
    no = int(n_elec / 2)

    t_V_pqrs = V_pqrs
    t_h_pq = h_pq

    t_L_opqrst = tcdump.read(tcdump_file, sp=0)

    # then MP2 and CCSD energies
    t_fock_pq = hf.construct_hf_matrix(no, t_h_pq, t_V_pqrs)

    # correction to Fock matrix
    t_delta_eps = contraction.get_double_contraction(no, t_L_opqrst)
    print(np.sum(np.abs(t_delta_eps)))
    t_fock_pq += t_delta_eps
    t_delta_V = contraction.get_single_contraction(no, t_L_opqrst)
    print(np.sum(np.abs(t_delta_V)))
    t_V_pqrs += t_delta_V

    mycc = ccsd.CCSD(no)
    ccsd_e = mycc.solve(t_fock_pq, t_V_pqrs, delta_e=1.e-11 )["ccsd e"]

    assert np.abs(ccsd_e - ref_e) < 1.e-7

def test_tc_ccsd_h2():
    test_tc_ref_energy(fcidump_file="pymes/test/test_tc_ccsd/FCIDUMP.H2.tc", tcdump_file="pymes/test/test_tc_ccsd/TCDUMP.H2.tc", ref_e = -1.1660095160466279)
    test_tc_ccsd_energy(fcidump_file="pymes/test/test_tc_ccsd/FCIDUMP.H2.tc", tcdump_file="pymes/test/test_tc_ccsd/TCDUMP.H2.tc", ref_e =-0.005919199166)

if __name__ == "__main__":
    test_tc_ccsd_h2()
    test_tc_ref_energy()
    test_tc_ccsd_energy()
    print("All tests passed")