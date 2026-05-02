#!/usr/bin/python3 -u

import time
import sys
import os
import numpy as np
import argparse
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

def main(nel, rs, 
         k_symm = 'gamma', \
         tc_type=None,\
         corr=None, \
         k_cutoff=None, \
         gamma=None, \
         Rcut=None, \
         nr=None, \
         write_madelung=False, \
         filename='madelung.out'):

    no     = int(nel/2)
    nalpha = int(nel/2)
    nbeta  = int(nel/2)
    rs     = rs

    # Symmetry of the many-particle wavefunction: consider gamma-point or baldereschi-point.
    time_set_sys = time.time()
    ueg_model = ueg.UEG(nel, nalpha, nbeta, rs, k_symm=k_symm, tc=tc_type)
    print_title("System Information Summary",'=')
    print_logging_info("Number of electrons = {}".format(nel))
    print_logging_info("rs = {}".format(rs))
    print_logging_info("Volume of the box = {}".format(ueg_model.Omega))
    print_logging_info("Length of the box = {}".format(ueg_model.L))
    if ueg_model.is_tc:
        print_logging_info("Using the TC-Method.")
        if corr == 'trunc':
            ueg_model.correlator = ueg_model.trunc
        elif corr == 'coulomb-yukawa':
            ueg_model.correlator = ueg_model.coulomb_yukawa
        elif corr == 'rpa':
            ueg_model.correlator = ueg_model.RPA
        else:
            raise ValueError("Invalid correlator type. Choose from 'trunc', 'coulomb-yukawa', or 'rpa'.")
        ueg_model.k_cutoff = k_cutoff
        ueg_model.gamma = gamma
        print_logging_info("TC correlator: {}".format(ueg_model.correlator.__name__), level=1)
        print_logging_info("K-cutoff in correlator: {:.12e} [2π/L]".format(ueg_model.k_cutoff), level=1)
    else:
        print_logging_info("Using the non-TC method.")
    print_logging_info("{:.3f} seconds spent on setting up model"\
                       .format((time.time()-time_set_sys)))
    sys.stdout.flush()

    # Compute the Madelung Constant.
    print_title("Computing Madelung Constant", '=')
    time_madelung = time.time()
    madelung_constant = ueg_model.madelung(Rcut=Rcut, nr=nr)
    print_logging_info("Madelung constant [Ha/e] = {:.8f}".format(madelung_constant))
    print_logging_info("{:.3f} seconds spent on computing Madelung constant"\
                        .format((time.time()-time_madelung)))
    if write_madelung:
        print_logging_info( "Writing Madelung constant to file: {}".format(filename))
        with open(filename, 'a') as f:
            f.write("{:.3f} {} {:.12e}\n".format(rs, nel, madelung_constant))

if __name__ == '__main__':

    # TC options.
    k_symm = 'gamma'
    tc_type = 'long-range'
    k_cutoff = 1.e-12
    gamma = None
    # Madelung options.
    Rcut = 200
    nr = 2000000
    write_madelung = True
    filename='madelung.out'


    for correlator in ['coulomb-yukawa', 'rpa']:
        filename = 'madelung.{}.out'.format(correlator)
        if write_madelung:
            with open(filename, 'w') as f:
                f.write("# 1. rs.____| 2. N____| 3. Mc [Ha/e]\n")
        for rs in [2.0]:
            for nel in [2, 14, 38, 54, 66, 114, 162, 186, 246, 294, 342, 358, 406, 502, 514, 610, 682, 730, 778, 874, 922, 970, 1030, 1174, 1238, 1382, 1478, 1502, 1598, 1694, 1790, 1850, 1898, 2042, 2090, 2282, 2378, 2426, 2474, 2618, 2714, 2730, 2838, 3006, 3102, 3150, 3294, 3486, 3582, 3678, 3726, 3870]:
                main(nel, rs, \
                    k_symm=k_symm, \
                    tc_type=tc_type,\
                    corr=correlator,
                    k_cutoff=k_cutoff, \
                    gamma=gamma, \
                    Rcut=Rcut, 
                    nr=nr, \
                    write_madelung=write_madelung, \
                    filename=filename)
