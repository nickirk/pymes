import ctf
import numpy as np
from pymes.util import fcidump
from pymes.solver import ccsd, ccd, mp2, eom_ccsd
from pymes.mean_field import hf
from pymes.integral.partition import part_2_body_int

def test_eom_ccsd_energy(fcidump_file="../test_ccsd/FCIDUMP.LiH.bare", ref_e=None):
    # known values
    hf_ref_e = -7.95197153899133
    n_elec, nb, e_core, e_orb, h_pq, V_pqrs = fcidump.read(fcidump_file)

    t_V_pqrs = ctf.astensor(V_pqrs)
    t_h_pq = ctf.astensor(h_pq)


    no = int(n_elec/2)
    # make sure HF energy is correct first
    hf_e = hf.calc_hf_e(no, e_core, t_h_pq, t_V_pqrs)
    print(hf_e)

    # CCSD energies
    t_fock_pq = hf.construct_hf_matrix(no, t_h_pq, t_V_pqrs)
    mycc = ccsd.CCSD(no)
    ccsd_e = mycc.solve(t_fock_pq, t_V_pqrs)["ccsd e"]


    # construct a EOM-CCSD instance
    # current formulation requires the singles dressed fock and V tensors
    # partition V integrals
    dict_t_V = part_2_body_int(no, t_V_pqrs)
    t_fock_dressed_pq = mycc.get_T1_dressed_fock(t_fock_pq, mycc.t_T_ai, dict_t_V)

    dict_t_V_dressed= {}.fromkeys(dict_t_V.keys(), None)
    dict_t_V_dressed.update({"ijka": None, "iabj": None})
    dict_t_V_dressed = mycc.get_T1_dressed_V(mycc.t_T_ai, dict_t_V, dict_t_V_dressed)
    eom_cc = eom_ccsd.EOM_CCSD(no, n_excit=1)
    e_excit = eom_cc.solve(t_fock_dressed_pq, dict_t_V_dressed, mycc.t_T_abij)
    print("Excited state energies = ", e_excit)
#def test_ccsd_fno(fcidump_file="fcidump.no"):
#    test_ccsd_energy(fcidump_file, ref_e=-0.01931436971985408)