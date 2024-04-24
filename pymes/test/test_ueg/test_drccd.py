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


def main(nel, cutoff,rs, gamma, kc):
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


    # consider only true two body operators (excluding the singly contracted
    # 3-body integrals). This integral will be used to compute the HF energy
    tV_pqrs = ueg_model.eval2BodyIntegrals(sp=1)

    print_logging_info("{:.3f} seconds spent on evaluating Coulomb integrals"\
                       .format((time.time()-time_pure_2_body_int)))

    print_title('Evaluating HF energy','=')
    # Getting the kinetic energy array (1-body operator)
    kinetic_G = compute_kinetic_energy(ueg_model)

    time_hf = time.time()

    print_logging_info("Partitioning V_pqrs")
    tV_ijkl = tV_pqrs[:no,:no,:no,:no]

    print_logging_info("Calculating hole and particle energies")
    tEpsilon_i = hf.calc_occ_orb_e(kinetic_G, tV_ijkl, no)
    holeEnergy = np.real(tEpsilon_i.to_nparray())

    tV_aibj = tV_pqrs[no:,:no,no:,:no]
    tV_aijb = tV_pqrs[no:,:no,:no,no:]
    tEpsilon_a = hf.calc_vir_orb_e(kinetic_G, tV_aibj, tV_aijb, no, nv)
    particleEnergy = np.real(tEpsilon_a.to_nparray())

    print_logging_info("HF orbital energies:")
    print_logging_info(tEpsilon_i.to_nparray())
    print_logging_info(tEpsilon_a.to_nparray())


    ### calculate HF energy: E_{HF} = \sum_i epsilon_i +\sum_ij (2*V_{ijij}-V_{ijji})
    print_logging_info("Calculating HF energy")
    tEHF = 2*ctf.einsum('i->',tEpsilon_i)
    tV_klij = tV_pqrs[:no,:no,:no,:no]

    print_logging_info("Calculating dir and exc HF energy")

    dirHFE = 2. * ctf.einsum('jiji->',tV_klij.to_nparray())
    excHFE = -1. * ctf.einsum('ijji->',tV_klij.to_nparray())

    print_logging_info("Summing dir and exc HF energy")

    tEHF = tEHF-(dirHFE + excHFE)
    print_logging_info("Direct = {}".format(dirHFE))
    print_logging_info("Exchange = {}".format(excHFE))
    print_logging_info("HF energy = {}".format(tEHF))

    print_logging_info("{:.3f} seconds spent on evaluating HF energy"\
                       .format((time.time()-time_hf)))

    mp2E, mp2Amp = mp2.solve(tEpsilon_i, tEpsilon_a, tV_pqrs)
    ccdE = 0.
    dcdE = 0.

    print_logging_info("Starting CCD")
    ccd_results = ccd.solve(tEpsilon_i, tEpsilon_a, tV_pqrs, \
                                               fDrccd=True,levelShift=-1., sp=0, maxIter=60, fDiis=True)
    ccd_e = ccd_results["ccd e"]
    ccd_amp = ccd_results["t2 amp"]
    ccd_dE = ccd_results["dE"]

    print_title("Summary of results","=")
    print_logging_info("HF E = {:.8f}".format(tEHF))
    print_logging_info("CCD correlation E = {:.8f}".format(ccd_e))
    print_logging_info("Total CCD E = {:.8f}".format(tEHF+ccd_e))


if __name__ == '__main__':
  #for gamma in None:
  gamma = None
  nel = 14
  for rs in [5.0]:
    for cutoff in [8]:
      kCutoffFraction = None
      main(nel,cutoff,rs, gamma, kCutoffFraction)
  ctf.MPI_Stop()
