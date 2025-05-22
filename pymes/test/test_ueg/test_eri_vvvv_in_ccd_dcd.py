#!/usr/bin/python3 -u

import time
import numpy as np
from functools import partial
import psutil

from pymes.util import tensors
from pymes.model import ueg
from pymes.integral import eri
from pymes.log import print_title, print_logging_info

einsum = partial(np.einsum, optimize=True)

def main(nel, cutoff, rs, gamma, kc, amps, \
            eri_incore, \
            dtype=np.float64):
    
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
       ueg_model.init_kPrime()
    
    myERI = eri.ERI(ueg_model)
    #myERI.calc_eri(incore=eri_incore)

    print_logging_info("{:.3f} seconds spent on constructing in-core ERI."\
                       .format((time.time()-time_init_eri)))

    print_title(' Test CCD Residual R_abij ','=')

    R_abij = np.zeros([nv, nv, no, no], dtype=dtype)
    T_abij = np.random.rand(nv, nv, no, no).astype(dtype)

    print_logging_info("Number of occupied orbitals = {}".format(no), level=0)
    print_logging_info("Number of virtual orbitals = {}".format(nv), level=0)
    print_logging_info("Number of spatial orbitals = {}".format(num_spatial_orb), level=0)
    print_logging_info("Number of elements in the VVVV tensor = {}".format(nv*nv*nv*nv), level=0)

    # Calculate block size dynamically to optimize memory usage.
    element_size = T_abij.dtype.itemsize  # Size of one element in bytes
    total_elements_dimension = nv           # Total elements along the first axis.
    block_size = tensors.calculate_block_size(total_elements_dimension, element_size,
                                              is_shared_memory=True)

    available_memory = psutil.virtual_memory().available

    print_logging_info("Available memory = {:.2f} GB".format(available_memory / (1024**3)), level=0)
    print_logging_info("Element size = {} bytes".format(element_size), level=0)
    print_logging_info("Block size = {}".format(block_size), level=0)
    print_logging_info("Block mem. size = {} GB".format(block_size * nv**3  * element_size / (1024**3)), level=0)

    # Process tensor 'vvvv'-contribution in blocks.
    for block_start in range(0, nv, block_size):
        print_logging_info("Processing block {} of size {}".format(block_start, block_size), level=1)
        block_end = min(block_start + block_size, nv)
        indx = tuple((block_start, block_end, 0, nv, 0, nv, 0, nv))
        print_logging_info("Calling get_vvvv with index {}".format(indx), level=2)
        start_time = time.time()
        t_V_xbcd = myERI.get_vvvv(indx)
        end_time = time.time()
        print_logging_info("Time taken to get_vvvv = {:.3f} seconds".format(end_time - start_time), level=2)
        print_logging_info("t_V_xbcd shape = {}".format(t_V_xbcd.shape), level=2)
        print_logging_info("Contracting t_V_xbcd with T_abij", level=2)
        t_R_xbij = einsum("xbcd, cdij -> xbij", t_V_xbcd, T_abij)
        R_abij[block_start:block_end, :, :, :] += t_R_xbij


if __name__ == '__main__':
   
   gamma = None
   amps  = None
   nel   = 54
   for rs in [0.5]:
      for cutoff in [40]:
         kCutoffFraction = 1
         main(nel, cutoff, rs, gamma, kCutoffFraction, amps, \
                    eri_incore=False, \
                    dtype=np.float64)
