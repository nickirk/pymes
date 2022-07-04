import ctf
import numpy as np
from pymes.util import fcidump
from pymes.solver import ccsd, ccd, mp2, eom_ccsd
from pymes.mean_field import hf
from pymes.integral.partition import part_2_body_int

def test_eom_ccsd_energy():
    dump_files = ["FCIDUMP.LiH.321g","FCIDUMP.H2.ccpvdz"]
    ref_e = [{"hf_e": -7.92958534362757, "ccsd_e": -0.0190883270951031, "ee": [0.1180867117168979, 0.154376205595602]},
             {"hf_e":-0.984996038191858, "ccsd_e":-0.06504241987110972,
              "ee": [0.3383783048040098, 0.440635057573434, 0.7506212674602769]}]
    #for file, e in zip(dump_files, ref_e):
    #    driver(fcidump_file=file, ref_e=e)
    driver(fcidump_file=dump_files[0], ref_e=ref_e[0])
def driver(fcidump_file="./FCIDUMP.H2.sto6g", ref_e={"hf_e": -0.891589185800039, "ccsd_e": -0.1012250926230937}):
    # known values
    #hf_ref_e = -7.95197478868981
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
    t_V_pqrs_orig = t_V_pqrs.copy()
    t_fock_pq = hf.construct_hf_matrix(no, t_h_pq, t_V_pqrs)
    mycc = ccsd.CCSD(no)
    mycc.delta_e = 1e-12
    mycc.max_iter = 200
    t_fock_copy_pq = t_fock_pq.copy()
    ccsd_result = mycc.solve(t_fock_pq, t_V_pqrs)
    ccsd_e = ccsd_result["ccsd e"]

    #assert np.allclose(t_fock_copy_pq.to_nparray(), t_fock_pq.to_nparray())
    ccsd_e_ref = ref_e["ccsd_e"]
    #assert np.isclose(ccsd_e, ccsd_e_ref)

    # construct a EOM-CCSD instance
    # current formulation requires the singles dressed fock and V tensors
    # partition V integrals

    #mycc.t_T_ai[:] = 0.
    t_T_ai = ccsd_result["t1"].copy()
    t_T_abij = ccsd_result["t2"].copy()
    t_T_abij[:] = 0.
    #t_T_ai[:] = 0.
    t_V_pqrs_orig = ctf.zeros(t_V_pqrs.shape)
    dict_t_V = part_2_body_int(no, t_V_pqrs_orig)
    #t_fock_pq[no:, :no] = 0.
    #t_fock_pq[:no, no:] = 0.

    t_fock_dressed_pq = mycc.get_T1_dressed_fock(t_fock_pq, t_T_ai, dict_t_V)
    #t_fock_dressed_pq = ctf.zeros(t_fock_pq.shape)
    #
    dict_t_V_dressed = mycc.get_T1_dressed_V(t_T_ai, dict_t_V)#, dict_t_V_dressed)
    #dict_t_V_dressed["klij"] = ctf.zeros(dict_t_V["klij"].shape)
    #dict_t_V_dressed["abcd"] = ctf.zeros(dict_t_V["abcd"].shape)
    #dict_t_V_dressed["ijab"] = ctf.zeros(dict_t_V["ijab"].shape)
    #dict_t_V_dressed["abij"] = ctf.zeros(dict_t_V["abij"].shape)

    n_e = 2
    eom_cc = eom_ccsd.EOM_CCSD(no, n_excit=n_e)
    #e_excit = eom_cc.solve(t_fock_dressed_pq, dict_t_V_dressed, t_fock_pq, t_T_abij)
    e_excit = eom_cc.solve(t_fock_dressed_pq, dict_t_V_dressed, t_fock_pq, t_T_abij)
    print("Excited state energies = ", e_excit)
    e_excit_ref = ref_e["ee"]
    #e_excit_ref = np.asarray([0.1333705757546808, 0.1841464947563311])
    #assert np.allclose(e_excit, e_excit_ref[:n_e])


def test_davidson():
    nv = 8
    no = 4
    eom_cc = eom_ccsd.EOM_CCSD(no, n_excit=3)
    eom_cc.test_davidson()
#def test_ccsd_fno(fcidump_file="fcidump.no"):
#    test_ccsd_energy(fcidump_file, ref_e=-0.01931436971985408)