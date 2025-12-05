#!/usr/bin/python3 -u

import time
import sys
import os
import numpy as np
import psutil

from pymes.solver import mp2
from pymes.model import ueg
from pymes.solver import ccd, dcd
from pymes.mean_field import hf
from pymes.integral import eri
from pymes.util.tensors import write_one_index_tensor, \
                                write_two_index_tensor, \
                                write_four_index_tensor
from pymes.util.parallel_tasks import print_threading_info
from pymes.log import print_title, print_logging_info

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print_logging_info("ERROR: NUMBA is not available.", level=0)
    sys.exit(1)

try:
    import pytblis
    TBLIS_AVAILABLE = True
except ImportError:
    TBLIS_AVAILABLE = False
    print_logging_info("WARNING: PyTBLIS is not available.", level=0)
    sys.exit(1)

sys.stdout.flush()

def main(nel, rs, \
        basis_cutoff, k_cutoff, \
        kmesh_fac, kmesh_cutoff, \
        gamma, amps, \
        eri_incore=True, \
        write_tensors=False):
    
    no     = int(nel/2)
    nalpha = int(nel/2)
    nbeta  = int(nel/2)
    rs     = rs

    # Symmetry of the many-particle wavefunction: consider gamma-point only.
    time_set_sys = time.time()
    ueg_model = ueg.UEG(nel, nalpha, nbeta, rs, is_tc=True)
    print_title("System Information Summary",'=')
    print_logging_info("Number of electrons = {}".format(nel))
    print_logging_info("rs = {}".format(rs))
    print_logging_info("Volume of the box = {}".format(ueg_model.Omega))
    print_logging_info("Length of the box = {}".format(ueg_model.L))
    print_logging_info("Madelung constant = {}".format(ueg_model.madelung(rs,nel)))
    if ueg_model.is_tc:
        print_logging_info("Using the TC-Method.")
    else:
        print_logging_info("Using the non-TC method.")
    print_logging_info("{:.3f} seconds spent on setting up model"\
                       .format((time.time()-time_set_sys)))
    sys.stdout.flush()
    
    # Print the number of cores and memory available.

    print_title("System Resources Information",'=')
    print_threading_info()
    sys.stdout.flush()

    # Initializing the basis set.
    time_init_basis = time.time()
    ueg_model.init_single_basis(basis_cutoff)

    num_spatial_orb = int(len(ueg_model.basis_fns)/2)
    nP = num_spatial_orb
    nGOrb = num_spatial_orb

    nv = nP - no
    print_title('Basis set', '=')
    print_logging_info('Number of spin orbitals = {}'\
                        .format(int(len(ueg_model.basis_fns))))
    print_logging_info('Number of spatial orbitals (plane waves) = {}'\
                        .format(num_spatial_orb))
    print_logging_info('Number of occupied orbitals = {}'\
                        .format(no))
    print_logging_info('Number of virtual orbitals = {}'\
                        .format(nv))
    print_logging_info("{:.3f} seconds spent on generating basis."\
                       .format((time.time()-time_init_basis)))
    sys.stdout.flush()

    # Initializing the ERI integrals.
    print_title('Evaluating the Electron Repulsion Integrals','=')

    time_init_eri = time.time()
    if ueg_model.is_tc:
       print_logging_info("Using the TC method to calculate ERI.")
       ueg_model.correlator = ueg_model.trunc
       ueg_model.k_cutoff = k_cutoff
       ueg_model.kmesh_fac = kmesh_fac
       ueg_model.kmesh_cutoff = kmesh_cutoff
       ueg_model.gamma = gamma
       ueg_model.init_kPrime()
    
    myERI = eri.ERI(ueg_model)
    myERI.calc_eri(incore=eri_incore)

    print_logging_info("{:.3f} seconds spent on constructing in-core ERI."\
                       .format((time.time()-time_init_eri)))

    HFE = myERI.EHF
    sys.stdout.flush()
 
    print_title(' MP2 ','=')
    print_logging_info('Evaluating the MP2 energy from ERI', level=0)
    print_logging_info("Starting MP2", level=0)
    time_mp2 = time.time()
    mp2_energy, mp2_Amp = mp2.solve(myERI.eps_occ, myERI.eps_virt, \
                                    myERI.oovv, myERI.vvoo)
    print_logging_info("{:.3f} seconds spent on MP2"\
                       .format((time.time()-time_mp2)), level=0)
    print_logging_info("MP2 energy = {:.8f}".format(mp2_energy), lvel=0)
    sys.stdout.flush()

    print_title(' CCD ','=')

    print_logging_info('Evaluating the CCD energy from ERI', level=0)
    print_logging_info("Starting CCD", level=0)
    ccd_e = 0.
    ls = -0.2
    myCCD = ccd.CCD(no)
    ccd_results = myCCD.solve(myERI, level_shift=ls, \
                              sp=0, max_iter=100, is_diis=True, amps=amps, epsilon_e=1e-7)
    
    print_logging_info("Unpacking CCD results", level=0)
    ccd_e = ccd_results["ccd e"]
    ccd_amp = ccd_results["t2 amp"]
    ccd_dE = ccd_results["dE"]

    print_logging_info("Manipulating CCD T2 amplitude norms", level=0)
    ccd_t2_norm = 2.*np.einsum("abij,abij->", ccd_amp,ccd_amp)
    ccd_t2_norm -= np.einsum("abij,baij->", ccd_amp,ccd_amp)
    ccd_t2_norm = ccd_t2_norm**(1./2)

    print_title("Summary of CCD results","=")
    print_logging_info("Num spin orb={}, rs={}, kCutoff={}".format(len(ueg_model.basis_fns),rs,\
                        ueg_model.k_cutoff))
    print_logging_info("Ref. E = {:.8f}".format(myERI.EHF))
    print_logging_info("MP2 E = {:.8f}".format(mp2_energy))
    print_logging_info("CCD correlation E = {:.8f}".format(ccd_e))
    print_logging_info("CCD T2 norm = ",ccd_t2_norm)
    print_logging_info("Total CCD E = {:.8f}".format(myERI.EHF+ccd_e))
    print_logging_info("CCD dE = {:.8f}".format(ccd_dE))
    sys.stdout.flush()

    print_title(' DCD ','=')

    print_logging_info('Evaluating the DCD energy from ERI', level=0)
    ls = -1
    print_logging_info("Starting DCD with level shift = ", ls , level=0)
    dcd_e = 0.
    myDCD = dcd.DCD(no)
    dcd_results = myDCD.solve(myERI, level_shift=ls, \
                              sp=0, max_iter=100, is_diis=True, amps=ccd_amp, epsilon_e=1e-7)
    
    print_logging_info("Unpacking DCD results", level=0)
    dcd_e = dcd_results["ccd e"]
    dcd_amp = dcd_results["t2 amp"]
    dcd_dE = dcd_results["dE"]

    print_logging_info("Manipulating DCD T2 amplitude norms", level=0)
    dcd_t2_norm = 2.*np.einsum("abij,abij->", dcd_amp,dcd_amp)
    dcd_t2_norm -= np.einsum("abij,baij->", dcd_amp,dcd_amp)
    dcd_t2_norm = dcd_t2_norm**(1./2)

    print_title("Summary of DCD results","=")
    print_logging_info("Num spin orb={}, rs={}, kCutoff={}".format(len(ueg_model.basis_fns),rs,\
                        ueg_model.k_cutoff))
    print_logging_info("Ref. E = {:.8f}".format(myERI.EHF))
    print_logging_info("MP2 E = {:.8f}".format(mp2_energy))
    print_logging_info("DCD correlation E = {:.8f}".format(dcd_e))
    print_logging_info("DCD T2 norm = ",dcd_t2_norm)
    print_logging_info("Total DCD E = {:.8f}".format(myERI.EHF+dcd_e))
    print_logging_info("DCD dE = {:.8f}".format(dcd_dE))
    sys.stdout.flush()

    print_title("Summary of CCD/DCD results","=")
    print_logging_info("Num spin orb={}, rs={}, kCutoff={}".format(len(ueg_model.basis_fns),rs,\
                        ueg_model.k_cutoff))
    print_logging_info("Ref. E = {:.8f}".format(myERI.EHF))
    print_logging_info("MP2 E = {:.8f}".format(mp2_energy))
    print_logging_info("CCD correlation E = {:.8f}".format(ccd_e))
    print_logging_info("CCD T2 norm = ",ccd_t2_norm)
    print_logging_info("DCD correlation E = {:.8f}".format(dcd_e))
    print_logging_info("DCD T2 norm = ",dcd_t2_norm)
    print_logging_info("Total CCD E = {:.8f}".format(myERI.EHF+ccd_e))
    print_logging_info("Total DCD E = {:.8f}".format(myERI.EHF+dcd_e))
    sys.stdout.flush()

    #print_title("Summary of NUMBA JIT-Compulation","=")
    #if NUMBA_AVAILABLE:
        #print(ueg._get_2b_int.parallel_diagnostics(level=4))

    if write_tensors:
        print_logging_info("Writing tensors to files", level=0)
        write_one_index_tensor( myERI.eps_occ, "OCC.txt")
        write_one_index_tensor( myERI.eps_virt, "VIRT.txt")
        write_two_index_tensor( myERI.fock, "FOCK.txt")
        write_four_index_tensor(myERI.oooo, "OOOO.txt")
        write_four_index_tensor(myERI.vovo, "VOVO.txt")
        write_four_index_tensor(myERI.ovov, "OVOV.txt")
        write_four_index_tensor(myERI.voov, "VOOV.txt")
        write_four_index_tensor(myERI.ovvo, "OVVO.txt")
        write_four_index_tensor(myERI.vvoo, "VVOO.txt")
        write_four_index_tensor(myERI.oovv, "OOVV.txt")
        write_four_index_tensor(myERI.vvvv, "VVVV.txt")


if __name__ == '__main__':
  
  nel   = 14
  k_cutoff = 1
  kmesh_frac = 1
  # cutoff=841 gives ipmax=30 to compare with legacy implementation.
  kmesh_cutoff = 841 

  gamma = None
  amps  = None

  eri_incore = False
  write_tensors = False


  for rs in [0.5]:
    for basis_cutoff in [2]:
      main(nel, rs, \
            basis_cutoff, k_cutoff, \
            kmesh_frac, kmesh_cutoff,
            gamma, amps, \
            eri_incore=eri_incore, \
            write_tensors=write_tensors)
