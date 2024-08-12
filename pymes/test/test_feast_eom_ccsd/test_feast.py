import numpy as np

import ctf
from pymes.util import fcidump
from pymes.solver import ccsd, feast_eom_ccsd
from pymes.mean_field import hf
from pymes.integral.partition import part_2_body_int

def driver(fcidump_file="pymes/test/test_eom_ccsd/FCIDUMP.LiH.321g", 
           ref_e={  "hf_e": -7.92958534362757, 
                    "ccsd_e": -0.0190883270951031,
                    "ee": [0.1180867117168979, 0.154376205595602]}):
    hf_ref_e = ref_e["hf_e"]
    n_elec, nb, e_core, e_orb, h_pq, V_pqrs = fcidump.read(fcidump_file)

    t_V_pqrs = ctf.astensor(V_pqrs)
    t_h_pq = ctf.astensor(h_pq)


    no = int(n_elec/2)
    # make sure HF energy is correct first
    hf_e = hf.calc_hf_e(no, e_core, t_h_pq, t_V_pqrs)
    print("HF e = ", hf_e)
    #assert np.isclose(hf_e, hf_ref_e)

    # CCSD energies
    t_fock_pq = hf.construct_hf_matrix(no, t_h_pq, t_V_pqrs)
    mycc = ccsd.CCSD(no)
    mycc.delta_e = 1e-12
    mycc.max_iter = 200
    ccsd_result = mycc.solve(t_fock_pq, t_V_pqrs, max_iter=200)
    ccsd_e = ccsd_result["ccsd e"]

    ccsd_e_ref = ref_e["ccsd_e"]
    #assert np.isclose(ccsd_e, ccsd_e_ref)

    # construct a EOM-CCSD instance
    # current formulation requires the singles dressed fock and V tensors
    # partition V integrals

    t_T_ai = ccsd_result["t1"].copy()
    t_T_abij = ccsd_result["t2"].copy()

    dict_t_V = part_2_body_int(no, t_V_pqrs)

    t_fock_dressed_pq = mycc.get_T1_dressed_fock(t_fock_pq, t_T_ai, dict_t_V)
    dict_t_V_dressed = mycc.get_T1_dressed_V(t_T_ai, dict_t_V)#, dict_t_V_dressed)

    n_e = 2
    eom_cc = feast_eom_ccsd.FEAST_EOM_CCSD(no, e_c=0.5, e_r=0.1, n_trial=4, max_iter=100, tol=1e-8)
    eom_cc.linear_solver = "jacobi"
    e_excit = eom_cc.solve(t_fock_dressed_pq, dict_t_V_dressed, t_T_abij)
    print("Excited state energies = ", e_excit)
    e_excit_ref = ref_e["ee"]
    assert np.allclose(e_excit, e_excit_ref[:n_e])

def test_feast_model_ham():
    no = 2
    eom_cc = feast_eom_ccsd.FEAST_EOM_CCSD(no, e_c=3.5, e_r=1, n_trial=2, max_iter=100, tol=1e-8)
    eom_cc.solve_test(nv=4)

if __name__ == "__main__":
    #test_feast_model_ham()
    driver()