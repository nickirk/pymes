import ctf
import numpy as np
from pymes.integral import contraction
from pymes.mean_field import hf
from pymes.solver import ccd
from pymes.util import fcidump, tcdump



def test_tc_ref_energy(fcidump_file="FCIDUMP.LiH.tc", tcdump_file="TCDUMP.LiH_FNO", ref_e = -8.042996662464):
    # known values

    n_elec, nb, e_core, e_orb, h_pq, V_pqrs = fcidump.read(fcidump_file, is_tc=True)
    no = int(n_elec / 2)

    t_L_opqrst = tcdump.read(tcdump_file, sp=0)
    t_T_0 = contraction.get_triple_contraction(no, t_L_opqrst)
    print(t_T_0)
    t_V_pqrs = ctf.astensor(V_pqrs)
    t_h_pq = ctf.astensor(h_pq)

    # make sure HF energy is correct first
    hf_e = hf.calc_hf_e(no, e_core, t_h_pq, t_V_pqrs)
    # get hf_e from fock matrix
    t_fock_pq = hf.construct_hf_matrix(no, t_h_pq, t_V_pqrs)
    hf_e_from_fock = 2 * ctf.einsum("ii->", t_fock_pq[:no, :no]) + e_core
    dirHFE = 2. * ctf.einsum('jiji->', t_V_pqrs[:no, :no, :no, :no])
    excHFE = -1. * ctf.einsum('ijji->', t_V_pqrs[:no, :no, :no, :no])
    hf_e_from_fock -= dirHFE + excHFE

    assert np.abs(hf_e - hf_e_from_fock) < 1.e-8
    print("HF e without T_0 = ", hf_e)
    hf_e += t_T_0
    print("HF e + T_0 = ", hf_e)
    assert np.abs(ref_e - hf_e) < 1.e-8
    return 0


def test_tc_ccd_energy(fcidump_file="FCIDUMP.LiH.tc", tcdump_file="TCDUMP.LiH_FNO", ref_e = -0.010032224361999909):

    n_elec, nb, e_core, e_orb, h_pq, V_pqrs = fcidump.read(fcidump_file, is_tc=True)
    no = int(n_elec / 2)
    print("V_pqrs (22|11) = ", V_pqrs[1, 0, 1, 0])
    print("V_pqrs (11|22) = ", V_pqrs[0, 1, 0, 1])
    t_V_pqrs = ctf.astensor(V_pqrs)
    t_h_pq = ctf.astensor(h_pq)
    print("h_pq=\n", t_h_pq)
    t_L_opqrst = tcdump.read(tcdump_file, sp=0)

    # then MP2 and CCSD energies
    t_fock_pq = hf.construct_hf_matrix(no, t_h_pq, t_V_pqrs)
    #print("t_fock_pq_t = \n", t_fock_pq_t)
    #t_fock_pq = ctf.einsum("pq -> qp", t_fock_pq_t)
    print("t_fock_pq = \n", t_fock_pq)
    # correction to Fock matrix
    t_delta_eps = contraction.get_double_contraction(no, t_L_opqrst)
    print("t_delta_eps = ", t_delta_eps)
    t_fock_pq += t_delta_eps
    print("t_fock_pq + t_delta_eps = \n", t_fock_pq)
    t_delta_V = contraction.get_single_contraction(no, t_L_opqrst)
    print("V_pqrs (22|11) = ", t_V_pqrs[1, 0, 1, 0])
    print("V_pqrs (11|22) = ", t_V_pqrs[0, 1, 0, 1])
    print("V_pqrs (21|12) = ", t_V_pqrs[1, 0, 0, 1])
    print("V_pqrs (12|12) = ", t_V_pqrs[0, 0, 1, 1])

    print("t_delta_V (11|11) = ", t_delta_V[0, 0, 0, 0])
    print("t_delta_V (22|11) = ", t_delta_V[1, 0, 1, 0])
    print("t_delta_V (11|22) = ", t_delta_V[0, 1, 0, 1])
    print("t_delta_V (21|12) = ", t_delta_V[1, 0, 0, 1])
    print("t_delta_V (12|12) = ", t_delta_V[0, 0, 1, 1])

    t_V_pqrs += t_delta_V
    print("modified V_pqrs (11|11) = ", t_V_pqrs[0, 0, 0, 0])
    print("modified V_pqrs (22|11) = ", t_V_pqrs[1, 0, 1, 0])
    print("modified V_pqrs (11|22) = ", t_V_pqrs[0, 1, 0, 1])
    print("modified V_pqrs (21|12) = ", t_V_pqrs[1, 0, 0, 1])
    print("modified V_pqrs (12|12) = ", t_V_pqrs[0, 0, 1, 1])

    mycc = ccd.CCD(no)
    ccd_result = mycc.solve(t_fock_pq, t_V_pqrs, epsilon_e=1.e-11)

    ccd_e = ccd_result["ccd e"]
    t2 = ccd_result["t2 amp"]



    assert np.abs(ccd_e - ref_e) < 1.e-9

def test_tc_ccd_h2():
    test_tc_ref_energy(fcidump_file="FCIDUMP.H2.tc", tcdump_file="TCDUMP.H2.tc", ref_e = -1.1660095160466279 )
    test_tc_ccd_energy(fcidump_file="FCIDUMP.H2.tc", tcdump_file="TCDUMP.H2.tc", ref_e =-0.005919199166)

def test_tc_ccd_ueg():
    test_tc_ref_energy(fcidump_file="14E.RS0.5.CO2.KCDEFAULT.TC.FCIDUMP", tcdump_file="14E.RS0.5.CO2.KCDEFAULT.TC.TCDUMP", ref_e = 58.437681570270)
    test_tc_ccd_energy(fcidump_file="14E.RS0.5.CO2.KCDEFAULT.TC.FCIDUMP", tcdump_file="14E.RS0.5.CO2.KCDEFAULT.TC.TCDUMP", ref_e =-0.307587144706)