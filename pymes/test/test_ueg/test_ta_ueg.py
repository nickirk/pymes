#!/usr/bin/python3 -u

import time
import numpy as np


from pymes.solver import mp2
from pymes.model import ueg
from pymes.mean_field import hf
from pymes.log import print_title, print_logging_info
from pymes.util.kpoints import gen_ir_ks


def compute_kinetic_energy(ueg):
    """
    input: ueg, an UEG class;
    return: an array of size of basis_fns/2
    """
    num_spatial_orb = int(len(ueg.basis_fns) / 2)
    G = []
    kinetic_G = []
    for i in range(num_spatial_orb):
        G.append(ueg.basis_fns[2 * i].k)
        kinetic_G.append(ueg.basis_fns[2 * i].kinetic)
    kinetic_G = np.asarray(kinetic_G)
    return kinetic_G


def test_zero_shift(shift=[0., 0., 0.], e_ref = [7.59923631, 1.33429356, 0.89665277]):
    gamma = None
    nel = 14
    rs = 1.0
    k_f = 1. / 2 * (3 * nel / np.pi) ** (1. / 3)
    cutoff = (k_f * 1.2) ** 2
    kc = 1.0
    tc_hf_e, contr_3b, tc_mp2_e = driver(nel, cutoff, rs, gamma, kc, shift)
    assert (np.abs(tc_hf_e - e_ref[0]) < 1.e-8)
    assert (np.abs(contr_3b - e_ref[1]) < 1.e-8)
    assert (np.abs(tc_mp2_e - e_ref[2]) < 1.e-8)

def test_nonzero_shift(shift=[0.1, 0.25, 0.5], e_ref=[10.43225777093217, 1.1470242894883573, 0.234320519158]):
    gamma = None
    nel = 14
    rs = 1.0
    k_f = 1. / 2 * (3 * nel / np.pi) ** (1. / 3)
    cutoff = (k_f * 1.2) ** 2
    #cutoff = 2
    kc = 1.0
    tc_hf_e, contr_3b, tc_mp2_e = driver(nel, cutoff, rs, gamma, kc, shift)
    assert (np.abs(tc_hf_e - e_ref[0]) < 1.e-8)
    assert (np.abs(contr_3b - e_ref[1]) < 1.e-8)
    assert (np.abs(tc_mp2_e - e_ref[2]) < 1.e-8)
    return tc_hf_e, contr_3b, tc_mp2_e

def test_twisted_average():
    ta_e_bag = []
    gamma = None
    nel = 14
    rs = 1.0
    k_f = 1. / 2 * (3 * nel / np.pi) ** (1. / 3)
    cutoff = (k_f * 1.2) ** 2
    #cutoff = 2
    kc = 1.0
    for ns in range(3, 5):
        ir_ks, weight = gen_ir_ks(ns)
        ta_e = np.zeros(3)
        for ks, w in zip(ir_ks, weight):
            tc_hf_e, contr_3b, tc_mp2_e = driver(nel, cutoff, rs, gamma, kc, ks)
            ta_e += np.array([tc_hf_e, contr_3b, tc_mp2_e]) * w
        print("TA mesh = %d x %d x %d".format(ns, ns, ns))
        print("Twisted-averaged energy = ", ta_e)
        ta_e_bag.append(ta_e)
    print("ns = ", [3, 4])
    print("ta energies = ", ta_e_bag)
    assert ((np.abs(ta_e_bag[0] - ta_e_bag[1])/14/27.2114) < 1.e-3).all()
    print("Twist-average energies converged down to 0.001 eV/electron.")


def driver(nel, cutoff, rs, gamma, kc, shift):
    no = int(nel / 2)
    nalpha = int(nel / 2)
    nbeta = int(nel / 2)

    time_set_sys = time.time()
    ueg_model = ueg.UEG(nel, nalpha, nbeta, rs)
    print_title("System Information Summary", '=')
    print_logging_info("Number of electrons = {}".format(nel))
    print_logging_info("rs = {}".format(rs))
    print_logging_info("Volume of the box = {}".format(ueg_model.Omega))
    print_logging_info("Length of the box = {}".format(ueg_model.L))
    print_logging_info("{:.3f} seconds spent on setting up model" \
                       .format((time.time() - time_set_sys)))

    time_init_basis = time.time()
    ueg_model.init_single_basis(cutoff, shift)

    num_spatial_orb = int(len(ueg_model.basis_fns) / 2)
    nP = num_spatial_orb
    nGOrb = num_spatial_orb

    nv = nP - no
    print_title('Basis set', '=')
    print_logging_info('Number of spin orbitals = {}' \
                       .format(int(len(ueg_model.basis_fns))))
    print_logging_info('Number of spatial orbitals (plane waves) = {}' \
                       .format(num_spatial_orb))
    print_logging_info('Ratio Norb/Nel = {:.3f}'.format(num_spatial_orb / nel))
    print_logging_info('k shift = ', shift)
    print_logging_info("{:.3f} seconds spent on generating basis." \
                       .format((time.time() - time_init_basis)))

    time_coulomb = time.time()

    # Getting the kinetic energy array (1-body operator)
    kinetic_G = compute_kinetic_energy(ueg_model)
    print("Kinetic energies =", kinetic_G)

    t_h_pq = (np.diag(kinetic_G))
    time_pure_2_body_int = time.time()
    ueg_model.gamma = gamma
    # specify the k_c in the correlator
    ueg_model.k_cutoff = kc

    print_title('Evaluating pure 2-body integrals', '=')
    print_logging_info("kCutoff = {}".format(ueg_model.k_cutoff))

    # consider only true two body operators (excluding the singly contracted
    # 3-body integrals). This integral will be used to compute the HF energy
    t_V_pqrs = ueg_model.eval_2b_integrals(correlator=ueg_model.gaskell, \
                                           is_only_2b=True, sp=1)

    print_logging_info("{:.3f} seconds spent on evaluating pure 2-body integrals" \
                       .format((time.time() - time_pure_2_body_int)))

    print_title('Evaluating HF energy', '=')
    t_fock_pq = hf.construct_hf_matrix(no, t_h_pq, t_V_pqrs)

    time_hf = time.time()

    t_epsilon_i = t_fock_pq.diagonal()[:no].copy()
    t_epsilon_a = t_fock_pq.diagonal()[no:].copy()
    print_logging_info("HF orbital energies:")
    print_logging_info(t_epsilon_i)
    print_logging_info(t_epsilon_a)

    t_hf_e = hf.calc_hf_e(no, 0., t_h_pq, t_V_pqrs)
    print_logging_info("HF energy = {}".format(t_hf_e))

    print_logging_info("{:.3f} seconds spent on evaluating HF energy" \
                       .format((time.time() - time_hf)))

    orbital_energy_correction = True
    # the correction to the one particle energies from doubly contracted 3-body
    # integrals
    contr_from_doubly_contra_3b = ueg_model.double_contractions_in_3_body()
    contr_from_triply_contra_3b = ueg_model.triple_contractions_in_3_body()
    print_logging_info("Mean field contributions from 3 body to total energy:")
    print_logging_info(contr_from_triply_contra_3b)
    print_logging_info("Contributions from 3 body to 1 particle energies:")
    print_logging_info(contr_from_doubly_contra_3b)

    # print_logging_info("Original gap = ", original_gap)
    print_logging_info("2b Corrected gap = ", t_epsilon_a[0] - t_epsilon_i[-1])
    if orbital_energy_correction:
        t_epsilon_i += contr_from_doubly_contra_3b[:no]
        t_epsilon_a += contr_from_doubly_contra_3b[no:]
    gap = t_epsilon_a[0] - t_epsilon_i[-1]
    print_logging_info("2b+3b Corrected gap = ", gap)
    print_logging_info("Corrected single-particle energies:")
    print_logging_info(t_epsilon_i)
    print_logging_info(t_epsilon_a)

    # perparing effective 2b
    t_V_pqrs += ueg_model.eval_2b_integrals(correlator=ueg_model.gaskell,
                                            is_rpa_approx=True, sp=1)

    print_logging_info("Starting MP2")

    t_V_ijab = t_V_pqrs[:no, :no, no:, no:]
    t_V_abij = t_V_pqrs[no:, no:, :no, :no]
    del t_V_pqrs
    tc_mp2_e, tc_mp2Amp = mp2.solve(t_epsilon_i, t_epsilon_a, t_V_ijab, t_V_abij, sp=1)
    t2_norm = np.sqrt(np.einsum("abij, abij->", tc_mp2Amp, tc_mp2Amp))

    total_e = t_hf_e + tc_mp2_e + contr_from_triply_contra_3b

    print_title("Summary of results", "=")
    print_logging_info("Num spin orb={}, rs={}, kCutoff={}".format(len(ueg_model.basis_fns), rs,
                                                                   ueg_model.k_cutoff))
    print_logging_info("TC-HF E = {:.8f}".format(t_hf_e))
    print_logging_info("TC MP2 correlation E = {:.8f}".format(tc_mp2_e))
    print_logging_info("3-body mean-field E = {:.8f}"
                       .format(contr_from_triply_contra_3b))
    print_logging_info("Total TC-MP2 E = {:.8f}".format(total_e))

    return t_hf_e, contr_from_triply_contra_3b, tc_mp2_e

if __name__ == "__main__":
    test_zero_shift()
    test_nonzero_shift()
    test_twisted_average()
    print("All tests passed.")