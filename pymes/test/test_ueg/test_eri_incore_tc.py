#!/usr/bin/python3 -u

import time
import numpy as np


from pymes.solver import mp2
from pymes.model import ueg
from pymes.solver import ccd, dcd
from pymes.mean_field import hf
from pymes.integral import eri
from pymes.log import print_title, print_logging_info


def compare_tensors(tensor1, tensor2):
    """
    Compare two tensors and print the maximum, mean, and RMS differences.
    """
    assert tensor1.shape == tensor2.shape, "Tensors have different shapes: {} vs {}".format(tensor1.shape, tensor2.shape)

    is_consistent = np.allclose(tensor1, tensor2, rtol=1e-5, atol=1e-8)
    if is_consistent:
        print_logging_info("Tensors are consistent.", level=1)
    else:
        print_logging_info("Tensors are NOT consistent.", level=1)
        print_logging_info("Max difference: {}".format(np.max(np.abs(tensor1 - tensor2))), level=1)
        print_logging_info("Mean difference: {}".format(np.mean(np.abs(tensor1 - tensor2))), level=1)
        print_logging_info("RMS difference: {}".format(np.sqrt(np.mean((tensor1 - tensor2)**2))), level=1)


def main(nel, cutoff, rs, gamma, kc, amps):
    no     = int(nel/2)
    nalpha = int(nel/2)
    nbeta  = int(nel/2)
    rs     = rs

    # Cutoff for the single-particle basis set.
    cutoff = cutoff

    # Symmetry of the many-particle wavefunction: consider gamma-point only.
    time_set_sys = time.time()
    ueg_model = ueg.UEG(nel, nalpha, nbeta, rs, is_tc=True)
    print_title("System Information Summary",'=')
    print_logging_info("Number of electrons = {}".format(nel))
    print_logging_info("rs = {}".format(rs))
    print_logging_info("Volume of the box = {}".format(ueg_model.Omega))
    print_logging_info("Length of the box = {}".format(ueg_model.L))
    if ueg_model.is_tc:
        print_logging_info("Using the TC-Method.")
    else:
        print_logging_info("Using the non-TC method.")
    print_logging_info("{:.3f} seconds spent on setting up model"\
                       .format((time.time()-time_set_sys)))

    # Initializing the basis set.
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

    # Initializing the ERI integrals.
    print_title('Evaluating the Electron Repulsion Integrals','=')

    time_init_eri = time.time()
    if ueg_model.is_tc:
       print_logging_info("Using the TC method to calculate ERI.")
       ueg_model.correlator = ueg_model.trunc
       ueg_model.k_cutoff = kc
       ueg_model.gamma = gamma
    
    myERI = eri.ERI(ueg_model)
    myERI.calc_eri(incore=True)

    print_logging_info("{:.3f} seconds spent on constructing in-core ERI."\
                       .format((time.time()-time_init_eri)))

    HFE = myERI.EHF
    print_title("Summary of current results","-")
    print_logging_info("Reference Energy [HFE + ET] = {:.8f}".format(HFE))

    # Calculate Fock matrix and Coulomb tensor the standard way.
    print_title('Calculating the Fock matrix and Coulomb tensor STANDARD WAY','=')

    print_title('Evaluating pure 2-body integrals','-')
    time_pure_2_body_int = time.time()
    t_V_pqrs = ueg_model.eval_2b_integrals(correlator=ueg_model.trunc,\
                                           is_only_2b=True,sp=1)
    print_logging_info("{:.3f} seconds spent on evaluating pure 2-body integrals"\
                       .format((time.time()-time_pure_2_body_int)))
    
    print_title('Evaluating HF energy','-')
    kinetic_G = ueg_model.compute_kinetic_energy()
    time_ehf = time.time()
    print_logging_info("Partitioning V_pqrs", level=0)
    tV_ijkl = t_V_pqrs[:no,:no,:no,:no]
    tV_aibj = t_V_pqrs[no:,:no,no:,:no]
    tV_aijb = t_V_pqrs[no:,:no,:no,no:]
    print_logging_info("Calculating hole and particle energies", level =0)
    tEpsilon_i = hf.calcOccupiedOrbE(kinetic_G, tV_ijkl, no)
    tEpsilon_a = hf.calcVirtualOrbE(kinetic_G, tV_aibj, tV_aijb, no, nv)
    print_logging_info("HF orbital energies:", level=0)
    print_logging_info(tEpsilon_i, level=1)
    print_logging_info(tEpsilon_a, level=1)
    print_logging_info("Calculating HF energy", level=0)
    tEHF = 2*np.einsum('i->',tEpsilon_i)
    print_logging_info("Calculating dir and exc HF energy", level=1)
    dirHFE = 2. * np.einsum('jiji->',tV_ijkl)
    excHFE = -1. * np.einsum('ijji->',tV_ijkl)
    print_logging_info("Summing dir and exc HF energy", level=1)
    tEHF = tEHF-(dirHFE + excHFE)
    print_logging_info("Direct = {}".format(dirHFE), level=0)
    print_logging_info("Exchange = {}".format(excHFE), level=0)
    print_logging_info("HF energy = {}".format(tEHF), level=0)
    print_logging_info("{:.3f} seconds spent on evaluating HF energy"\
                       .format((time.time()-time_ehf)))

 #   print_title('Evaluating effective 2-body integrals','-')
 #   time_eff_2_body = time.time()
 #   t_V_pqrs += ueg_model.eval_2b_integrals(correlator=ueg_model.trunc,\
 #                                           is_effect_2b=True,sp=1)
 #   print_logging_info("{:.3f} seconds spent on evaluating effective 2-body integrals"\
 #                      .format((time.time()-time_eff_2_body)))
    
    print_title('Correcting the orbital energies','-')
    contr_from_doubly_contra_3b = ueg_model.double_contractions_in_3_body()
    contr_from_triply_contra_3b = ueg_model.triple_contractions_in_3_body()
    print_logging_info("Mean field contributions from 3 body to total energy:", level=0)
    print_logging_info(contr_from_triply_contra_3b)
    print_logging_info("Contributions from 3 body to 1 particle energies:", level=0)
    print_logging_info(contr_from_doubly_contra_3b)
    tEpsilon_i += contr_from_doubly_contra_3b[:no]
    tEpsilon_a += contr_from_doubly_contra_3b[no:]
    
    print_title('Constructing the Fock matrix','-')
    time_hf = time.time()
    t_fock_pq = hf.construct_hf_matrix(no, np.diag(kinetic_G), t_V_pqrs)
    print_logging_info("{:.3f} seconds spent on constructing the Fock matrix"\
                       .format((time.time()-time_hf)))

    # Compare STANDARD WAY with the ERI class.
    print_title('Compare the STANDARD WAY and ERI class','=')

    # Compare orbital energies.
    print_title('Comparing the occupied orbital energies','-')
    compare_tensors(tEpsilon_i, myERI.eps_occ)
    print_title('Comparing the virtual orbital energies','-')
    compare_tensors(tEpsilon_a, myERI.eps_virt)

    # Compare the Fock matrix with the one from the ERI class.
    print_title('Comparing the Fock matrix','-')
    compare_tensors(t_fock_pq, myERI.fock)

    # Compare the Coulomb tensor with the one from the ERI class.
    print_title('Comparing the Coulomb tensor','-')

    print_logging_info("Comparing [oooo] block", level=0)
    compare_tensors(myERI.oooo, t_V_pqrs[:no,:no,:no,:no])

    print_logging_info("Comparing [vovo] block", level=0)
    compare_tensors(myERI.vovo, t_V_pqrs[no:,:no,no:,:no])

    print_logging_info("Comparing [voov] block", level=0)
    compare_tensors(myERI.voov, t_V_pqrs[no:,:no,:no,no:])

    print_logging_info("Comparing [ovvo] block", level=0)
    compare_tensors(myERI.ovvo, t_V_pqrs[:no,no:,no:,:no])

    print_logging_info("Comparing [oovv] block", level=0)
    compare_tensors(myERI.oovv, t_V_pqrs[:no,:no,no:,no:])

    print_logging_info("Comparing [vvoo] block", level=0)
    compare_tensors(myERI.vvoo, t_V_pqrs[no:,no:,:no,:no])

    print_logging_info("Comparing [ovov] block", level=0)
    compare_tensors(myERI.ovov, t_V_pqrs[:no,no:,:no,no:])

    print_logging_info("Comparing [vvvv] block", level=0)
    compare_tensors(myERI.vvvv, t_V_pqrs[no:,no:,no:,no:])
 
    print_title('Evaluating the MP2 energies','=')
    print_logging_info("Starting MP2", level=0)
    time_mp2 = time.time()
    mp2_energy, mp2_Amp = mp2.solve(myERI.eps_occ, myERI.eps_virt, \
                                    myERI.oovv, myERI.vvoo)
    print_logging_info("{:.3f} seconds spent on MP2"\
                       .format((time.time()-time_mp2)), level=0)
    print_logging_info("MP2 energy = {:.8f}".format(mp2_energy), lvel=0)


if __name__ == '__main__':
  #for gamma in None:
  gamma = None
  amps  = None
  nel   = 14
  for rs in [0.5]:
    for cutoff in [2]:
      kCutoffFraction = 1
      main(nel,cutoff,rs, gamma, kCutoffFraction,amps)
