#!/usr/bin/python3 -u

import time
import numpy as np


from pymes.solver import mp2
from pymes.model import ueg
from pymes.solver import ccd, dcd
from pymes.mean_field import hf
from pymes.integral import eri
from pymes.log import print_title, print_logging_info


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
    time_init_eri = time.time()
    if ueg_model.is_tc:
       ueg_model.correlator = ueg_model.trunc
       ueg_model.k_cutoff = kc
       ueg_model.gamma = gamma
    
    print_title('Evaluating the Electron Repulsion Integrals','=')
    print_logging_info("kCutoff = {}".format(ueg_model.k_cutoff))
    print_logging_info("Gamma = {}".format(ueg_model.gamma))
    
    myERI = eri.ERI(ueg_model)
    myERI.calc_eri(incore=True)

    print_logging_info("{:.3f} seconds spent on constructing in-core ERI."\
                       .format((time.time()-time_init_eri)))

    HFE = myERI.EHF
    print_title("Summary of current results","=")
    print_logging_info("Reference Energy [HFE + ET] = {:.8f}".format(HFE))


if __name__ == '__main__':
  #for gamma in None:
  gamma = None
  amps  = None
  nel   = 14
  for rs in [0.5]:
    for cutoff in [2]:
      kCutoffFraction = 1
      main(nel,cutoff,rs, gamma, kCutoffFraction,amps)
