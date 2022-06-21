import ctf
import numpy as np
from pymes.util import fcidump
from pymes.solver import ccsd, ccd, mp2, eom_ccsd
from pymes.mean_field import hf
from pymes.integral.partition import part_2_body_int

def test_eom_ccsd_energy(fcidump_file="./FCIDUMP.HF.augccpvdz",
                         ref_e={"hf_e": -100.017027694409, "ccsd_e": -0.222001880893871}):
    # known values
    #hf_ref_e = -7.95197478868981
    hf_ref_e = ref_e["hf_e"]
    n_elec, nb, e_core, e_orb, h_pq, V_pqrs = fcidump.read(fcidump_file)

    t_V_pqrs = ctf.astensor(V_pqrs)
    t_h_pq = ctf.astensor(h_pq)


    no = int(n_elec/2)
    # make sure HF energy is correct first
    hf_e = hf.calc_hf_e(no, e_core, t_h_pq, t_V_pqrs)
    assert np.isclose(hf_e, hf_ref_e)

    # CCSD energies
    t_fock_pq = hf.construct_hf_matrix(no, t_h_pq, t_V_pqrs)
    mycc = ccsd.CCSD(no)
    mycc.delta_e = 1e-9
    ccsd_e = mycc.solve(t_fock_pq, t_V_pqrs)["ccsd e"]
    #ccsd_e_ref = -0.02035412476830058
    #ccsd_e_ref = -0.02035251845411305
    ccsd_e_ref = ref_e["ccsd_e"]
    assert np.isclose(ccsd_e, ccsd_e_ref)

    # construct a EOM-CCSD instance
    # current formulation requires the singles dressed fock and V tensors
    # partition V integrals
    dict_t_V = part_2_body_int(no, t_V_pqrs)
    t_fock_dressed_pq = mycc.get_T1_dressed_fock(t_fock_pq, mycc.t_T_ai, dict_t_V)

    dict_t_V_dressed= {}.fromkeys(dict_t_V.keys(), None)
    dict_t_V_dressed.update({"ijka": None, "iabj": None})
    dict_t_V_dressed = mycc.get_T1_dressed_V(mycc.t_T_ai, dict_t_V, dict_t_V_dressed)
    n_e = 1
    eom_cc = eom_ccsd.EOM_CCSD(no, n_excit=n_e)
    e_excit = eom_cc.solve(t_fock_dressed_pq, dict_t_V_dressed, mycc.t_T_abij)
    print("Excited state energies = ", e_excit)
    e_excit_ref = np.asarray([0.1333705757546808, 0.1841464947563311])
    assert np.allclose(e_excit, e_excit_ref[:n_e])

def test_davidson():
    nv = 4
    no = 4
    eom_cc = eom_ccsd.EOM_CCSD(no, n_excit=3)
    eom_cc.test_davidson()
#def test_ccsd_fno(fcidump_file="fcidump.no"):
#    test_ccsd_energy(fcidump_file, ref_e=-0.01931436971985408)