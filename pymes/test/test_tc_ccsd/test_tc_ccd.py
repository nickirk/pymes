import numpy as np
from pymes.integral import contraction
from pymes.mean_field import hf
from pymes.solver import ccd
from pymes.util import fcidump, tcdump



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
    # get hf_e from fock matrix
    t_fock_pq = hf.construct_hf_matrix(no, t_h_pq, t_V_pqrs)
    hf_e_from_fock = 2 * np.einsum("ii->", t_fock_pq[:no, :no]) + e_core
    dirHFE = 2. * np.einsum('jiji->', t_V_pqrs[:no, :no, :no, :no])
    excHFE = -1. * np.einsum('ijji->', t_V_pqrs[:no, :no, :no, :no])
    hf_e_from_fock -= dirHFE + excHFE

    assert np.abs(hf_e - hf_e_from_fock) < 1.e-8
    print("HF e without T_0 = ", hf_e)
    hf_e += t_T_0
    print("HF e + T_0 = ", hf_e)
    assert np.abs(ref_e - hf_e) < 1.e-8
    return 0


def test_tc_ccd_energy(fcidump_file="pymes/test/test_eom_ccsd/FCIDUMP.LiH.tc", tcdump_file="pymes/test/test_eom_ccsd/TCDUMP.LiH_FNO", ref_e = -0.010648023717):

    n_elec, nb, e_core, e_orb, h_pq, V_pqrs = fcidump.read(fcidump_file, is_tc=True)
    no = int(n_elec / 2)
    t_V_pqrs = np.astensor(V_pqrs)
    t_h_pq = np.astensor(h_pq)
    t_L_opqrst = tcdump.read(tcdump_file, sp=0)

    # then MP2 and CCSD energies
    t_fock_pq = hf.construct_hf_matrix(no, t_h_pq, t_V_pqrs)
    # correction to Fock matrix
    t_delta_eps = contraction.get_double_contraction(no, t_L_opqrst)
    t_fock_pq += t_delta_eps
    t_delta_V = contraction.get_single_contraction(no, t_L_opqrst)
    t_V_pqrs += t_delta_V

    mycc = ccd.CCD(no)
    ccd_result = mycc.solve(t_fock_pq, t_V_pqrs, delta_e=1.e-11)

    ccd_e = ccd_result["ccd e"]


    assert np.abs(ccd_e - ref_e) < 1.e-9

def test_tc_ccd_h2():
    test_tc_ref_energy(fcidump_file="pymes/test/test_tc_ccsd/FCIDUMP.H2.tc", tcdump_file="pymes/test/test_tc_ccsd/TCDUMP.H2.tc", ref_e = -1.1660095160466279 )
    test_tc_ccd_energy(fcidump_file="pymes/test/test_tc_ccsd/FCIDUMP.H2.tc", tcdump_file="pymes/test/test_tc_ccsd/TCDUMP.H2.tc", ref_e =-0.005919199166)

def test_tc_ccd_h2_hdf5():
    test_tc_ref_energy(fcidump_file="pymes/test/test_tc_ccsd/FCIDUMP.H2.tc", tcdump_file="pymes/test/test_util/tcdump.h5", ref_e=-1.1660095160466279)
    test_tc_ccd_energy(fcidump_file="pymes/test/test_tc_ccsd/FCIDUMP.H2.tc", tcdump_file="pymes/test/test_util/tcdump.h5", ref_e=-0.005919199166)
