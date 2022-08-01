#!/usr/bin/python3 -u

import time
import numpy as np


import ctf

from pymes.solver import mp2
from pymes.model import ueg
from pymes.solver import ccd
from pymes.mean_field import hf
from pymes.log import print_title, print_logging_info


##############################################################################
#   1. ctf tensors starts with a lower case t
#   2. indices follow each variable after a _
#   3. numpy's nparray runs fastest with the right most index
#   4. for small tensors, use nparrays, only when large contractions are needed
#      then use ctf tensors. In case in the future some other tensor engines
#      might be used
##############################################################################



def compute_kinetic_energy(ueg):
    """
    input: ueg, an UEG class;
    return: an array of size of basis_fns/2
    """
    num_spatial_orb = int(len(ueg.basis_fns)/2)
    G = []
    kinetic_G = []
    for i in range(num_spatial_orb):
        G.append(ueg.basis_fns[2*i].k)
        kinetic_G.append(ueg.basis_fns[2*i].kinetic)
    kinetic_G = np.asarray(kinetic_G)
    return kinetic_G


def test_sym_2b(nel, cutoff,rs, gamma, kc, amps):
    world=ctf.comm()
    no = int(nel/2)
    nalpha = int(nel/2)
    nbeta = int(nel/2)
    rs = rs

    # Cutoff for the single-particle basis set.
    cutoff = cutoff

    # Symmetry of the many-particle wavefunction: consider gamma-point only.
    time_set_sys = time.time()
    ueg_model = ueg.UEG(nel, nalpha, nbeta, rs)
    print_title("System Information Summary",'=')
    print_logging_info("Number of electrons = {}".format(nel))
    print_logging_info("rs = {}".format(rs))
    print_logging_info("Volume of the box = {}".format(ueg_model.Omega))
    print_logging_info("Length of the box = {}".format(ueg_model.L))
    print_logging_info("{:.3f} seconds spent on setting up model"\
                       .format((time.time()-time_set_sys)))


    time_init_basis = time.time()
    ueg_model.init_single_basis(cutoff)

    num_spatial_orb = int(len(ueg_model.basis_fns)/2)
    nP = num_spatial_orb
    nGOrb = num_spatial_orb

    nv = nP - no
    print_title('Basis set', '=')
    print_logging_info('Number of spin orbitals = {}'\
                       .format(int(len(ueg_model.basis_fns))))
    print_logging_info('Number of spatial orbitals (plane waves) = {}'\
                       .format(num_spatial_orb))
    print_logging_info("{:.3f} seconds spent on generating basis."\
                       .format((time.time()-time_init_basis)))


    time_pure_2_body_int = time.time()
    ueg_model.gamma = gamma
    # specify the k_c in the correlator
    ueg_model.k_cutoff = 1.0 #ueg_model.L/(2*np.pi)*2.3225029893472993/rs

    print_title('Evaluating pure 2-body integrals','=')
    print_logging_info("kCutoff = {}".format(ueg_model.k_cutoff))

    # consider only true two body operators (excluding the singly contracted
    # 3-body integrals). This integral will be used to compute the HF energy
    t_V_pqrs = ueg_model.eval_2b_integrals(correlator=ueg_model.trunc,\
                                     is_only_2b=True, sp=0)
    #t_V_sym_pqrs = ctf.tensor(t_V_pqrs.shape, sp=t_V_pqrs.sp)
    #t_V_sym_pqrs.i("pqrs") << 0.5 *( t_V_pqrs.i("pqrs") + t_V_pqrs.i("qpsr"))
    #t_V_pqrs = t_V_sym_pqrs

    print_logging_info("{:.3f} seconds spent on evaluating pure 2-body integrals"\
                       .format((time.time()-time_pure_2_body_int)))

    print_title('Evaluating HF energy','=')
    # Getting the kinetic energy array (1-body operator)
    kinetic_G = compute_kinetic_energy(ueg_model)

    time_hf = time.time()

    print_logging_info("Partitioning V_pqrs")
    t_V_ijkl = t_V_pqrs[:no,:no,:no,:no]

    print_logging_info("Calculating hole and particle energies")
    tEpsilon_i = hf.calcOccupiedOrbE(kinetic_G, t_V_ijkl, no)
    holeEnergy = np.real(tEpsilon_i.to_nparray())

    t_V_aibj = t_V_pqrs[no:,:no,no:,:no]
    t_V_aijb = t_V_pqrs[no:,:no,:no,no:]
    tEpsilon_a = hf.calcVirtualOrbE(kinetic_G, t_V_aibj, t_V_aijb, no, nv)
    particleEnergy = np.real(tEpsilon_a.to_nparray())

    print_logging_info("HF orbital energies:")
    print_logging_info(tEpsilon_i.to_nparray())
    print_logging_info(tEpsilon_a.to_nparray())


    ### calculate HF energy: E_{HF} = \sum_i epsilon_i +\sum_ij (2*V_{ijij}-V_{ijji})
    print_logging_info("Calculating HF energy")
    tEHF = 2*ctf.einsum('i->',tEpsilon_i)
    t_V_klij = t_V_pqrs[:no,:no,:no,:no]
    t_fock_pq = hf.construct_hf_matrix(no, ctf.astensor(np.diag(kinetic_G)), t_V_pqrs)
    print_logging_info("Calculating dir and exc HF energy")

    dirHFE = 2. * ctf.einsum('jiji->',t_V_klij.to_nparray())
    excHFE = -1. * ctf.einsum('ijji->',t_V_klij.to_nparray())

    print_logging_info("Summing dir and exc HF energy")

    tEHF = tEHF-(dirHFE + excHFE)
    print_logging_info("Direct = {}".format(dirHFE))
    print_logging_info("Exchange = {}".format(excHFE))
    print_logging_info("HF energy = {}".format(tEHF))

    print_logging_info("{:.3f} seconds spent on evaluating HF energy"\
                       .format((time.time()-time_hf)))

    # Now add the contributions from the 3-body integrals into the diagonal and
    # two body operators, also to the total energy, corresponding to 3 orders
    # of contractions
    print_title('Evaluating effective 2-body integrals','=')
    time_eff_2_body = time.time()
    # before calculating new integrals, delete the old one to release memory
    t_V_asym_pqrs = ueg_model.eval_2b_integrals(correlator=ueg_model.trunc,\
                                     is_effect_2b=True, sp=0)
    print_logging_info("{:.3f} seconds spent on evaluating effective 2-body integrals"\
                       .format((time.time()-time_eff_2_body)))
    print_logging_info("Symmetrizing t_V_pqrs with respect to electron numbering")
    
    #t_V_eff_pqrs = t_V_asym_pqrs - t_V_pqrs
    t_V_sym_pqrs = ctf.tensor(t_V_pqrs.shape)
    t_V_sym_pqrs.i("pqrs") << 0.5 *( t_V_asym_pqrs.i("pqrs") + t_V_asym_pqrs.i("qpsr"))
    t_V_pqrs += t_V_sym_pqrs

    orbital_energy_correction = True
    # the correction to the one particle energies from doubly contracted 3-body
    # integrals
    contr_from_doubly_contra_3b = ueg_model.double_contractions_in_3_body()
    contr_from_triply_contra_3b = ueg_model.triple_contractions_in_3_body()
    print_logging_info("Mean field contributions from 3 body to total energy:")
    print_logging_info(contr_from_triply_contra_3b)
    print_logging_info("Contributions from 3 body to 1 particle energies:")
    print_logging_info(contr_from_doubly_contra_3b)

    if orbital_energy_correction:
        tEpsilon_i += contr_from_doubly_contra_3b[:no]
        tEpsilon_a += contr_from_doubly_contra_3b[no:]
        t_fock_pq += np.diag(contr_from_doubly_contra_3b)



    print_logging_info("Starting MP2")

    mp2_e, mp2Amp = mp2.solve(tEpsilon_i, tEpsilon_a, t_V_pqrs)
    ccd_e = 0.
    dcd_e = 0.

    ls = -(np.log(rs)*0.8+1.0)
    print_logging_info("Starting CCD")

    mycc = ccd.CCD(no)
    ccd_results = mycc.solve(t_fock_pq, t_V_pqrs)

    # unpacking
    ccd_e = ccd_results["ccd e"]
    ccd_amp = ccd_results["t2 amp"]
    ccd_dE = ccd_results["dE"]

    #print_logging_info("Starting DCD")
    #dcd_results = dcd.solve(tEpsilon_i, tEpsilon_a, t_V_pqrs, levelShift=ls,\
    #                        sp=0, maxIter=70, fDiis=True,amps=ccd_amp)
    #dcd_e = dcd_results["ccd e"]
    #dcd_amp = dcd_results["t2 amp"]
    #dcd_dE = dcd_results["dE"]

    print_title("Summary of results","=")
    print_logging_info("Num spin orb={}, rs={}, kCutoff={}".format(len(ueg_model.basis_fns),rs,\
                        ueg_model.k_cutoff))
    print_logging_info("HF E = {:.8f}".format(tEHF))
    print_logging_info("CCD correlation E = {:.8f}".format(ccd_e))

    print_logging_info("3-body mean-field E = {:.8f}"\
                       .format(contr_from_triply_contra_3b))
    print_logging_info("Total CCD E = {:.8f}".format(tEHF+ccd_e+contr_from_triply_contra_3b))

    assert np.abs(tEHF - 58.143779330795965) < 1.e-8
    assert np.abs(contr_from_triply_contra_3b - 0.07218268772824925) < 1.e-8

    known_doubly_contr_3b = np.array([0.0079401, 0.01672232, 0.01672232, 0.01672232, 0.01672232, 0.01672232,
        0.01672232, 0.01166044, 0.01166044, 0.01166044, 0.01166044, 0.01166044,
        0.01166044, 0.01166044, 0.01166044, 0.01166044, 0.01166044, 0.01166044,
        0.01166044, 0.01826549, 0.01826549, 0.01826549, 0.01826549, 0.01826549,
        0.01826549, 0.01826549, 0.01826549, 0.00796643, 0.00796643, 0.00796643,
        0.00796643, 0.00796643, 0.00796643, 0.01309416, 0.01309416, 0.01309416,
        0.01309416, 0.01309416, 0.01309416, 0.01309416, 0.01309416, 0.01309416,
        0.01309416, 0.01309416, 0.01309416, 0.01309416, 0.01309416, 0.01309416,
        0.01309416, 0.01309416, 0.01309416, 0.01309416, 0.01309416, 0.01309416,
        0.01309416, 0.01309416, 0.01309416])
    assert np.sum(np.abs(contr_from_doubly_contra_3b - known_doubly_contr_3b)) < 1.e-8
    assert np.abs(mp2_e - -0.327226965969) < 1.e-8
    assert np.abs(ccd_e - -0.256670836708) < 1.e-8

    return 0

if __name__ == '__main__':
  #for gamma in None:
  gamma = None
  amps = None
  nel = 14
  for rs in [0.5]:
    for cutoff in [5]:
      kCutoffFraction = None
      test_sym_2b(nel,cutoff,rs, gamma, kCutoffFraction,amps)
  ctf.MPI_Stop()
