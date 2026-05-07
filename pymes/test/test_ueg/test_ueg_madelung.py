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
         veff=None, \
         rpoints=None, \
         Rcut=None, \
         nr=None, \
         rmax=None, \
         nrpoints=None, \
         write_madelung=False, \
         write_veff=False, \
         filename='madelung.out',
         mode='madelung'):

    
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
            ueg_model.init_ConvMesh(nx=300, dkfac=100, kmaxfac=100)
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

    if mode == 'madelung':
        print_title("Computing Madelung Constant", '=')
        time_madelung = time.time()
        madelung_constant = ueg_model.madelung(Rcut=Rcut, nr=nr, veff=veff, rpoints=rpoints)
        print_logging_info("Madelung constant [Ha/e] = {:.8f}".format(madelung_constant))
        print_logging_info("{:.3f} seconds spent on computing Madelung constant"\
                        .format((time.time()-time_madelung)))
        if write_madelung:
            print_logging_info( "Writing Madelung constant to file: {}".format(filename))
            with open(filename, 'a') as f:
                f.write("{:.3f} {} {:.12e}\n".format(rs, nel, madelung_constant))
    elif mode == 'veff':
        print_title("Computing Effective Potential on a Grid", '=')
        print_logging_info( "Using rmax = {} and nrpoints = {}".format(rmax, nrpoints))
        time_veff = time.time()
        rpoints = np.geomspace(0.001, rmax, nrpoints)
        v1, v2, v3 = ueg_model.get_veff_rspace(rpoints, rc=-1)
        veff = v2 + v3
        print_logging_info("Computed effective potential on a grid with {} points.".format(len(rpoints)))
        print_logging_info("{:.3f} seconds spent on computing effective potential".\
                        format((time.time()-time_veff)))
        if write_veff:
            print_logging_info( "Writing effective potential to file: veff_rspace.dat")
            with open('veff_rspace.dat', 'w') as f:
                f.write("# 1. r [Bohr] | 2. v_eff [a.u.]\n")
                for r, v in zip(rpoints, veff):
                    f.write("{:.15e} {:.15e}\n".format(r, v))
        return veff, rpoints


if __name__ == '__main__':

    # TC options.
    k_symm = 'gamma'
    tc_type = 'long-range'
    k_cutoff = 1.e-12
    gamma = None
    # Madelung options.
    Rcut = 200
    nr = 1000000
    write_madelung = True
    filename='madelung.out'
    # Potential options.
    precalc_veff = True
    nrpoints = 100000
    rmax = 0.0

    if k_symm == 'gamma':
        nelec = [2, 14, 38, 54, 66, 114, 162, 186, 246, 294, 342, 358, 406, 502, 514, 610, 682, 730, 778, 874, 922, 970, 1030, 1174, 1238, 1382, 1478, 1502, 1598, 1694, 1790, 1850, 1898, 2042, 2090, 2282, 2378, 2426, 2474, 2618, 2714, 2730, 2838, 3006, 3102, 3150, 3294, 3486, 3582, 3678, 3726, 3870]
    elif k_symm == 'baldereschi':
        nelec = [2, 8, 14, 22, 34, 40, 52, 70, 76, 90, 108, 120, 138, 156, 168, 180, 210, 228, 242, 266, 272, 302, 332, 344, 368, 392, 410, 434, 464, 476, 502, 544, 568, 580, 610, 628, 652, 700, 718, 754, 778, 796, 832, 862, 886, 912, 960, 978, 1008, 1056, 1068, 1104, 1158, 1170, 1194, 1224, 1260, 1308, 1350, 1380, 1404, 1458, 1476, 1502, 1538, 1568, 1622, 1676, 1694, 1718, 1772, 1802, 1850, 1892, 1916, 1946, 2006, 2036, 2060, 2120, 2138, 2186, 2252, 2276, 2332, 2362, 2392, 2452, 2488, 2512, 2536, 2608, 2638, 2692, 2764, 2788, 2824, 2884, 2914, 2938, 2998, 3040, 3100, 3142, 3178, 3214, 3274, 3316, 3354, 3426, 3444, 3498, 3582, 3600, 3648, 3708, 3750, 3798, 3828, 3852, 3906, 3996] 
    else:
        raise ValueError("Invalid k-symmetry. Choose from 'gamma' or 'baldereschi'.")

    for correlator in ['rpa']:
        filename = 'madelung.{}.out'.format(correlator)
        if write_madelung:
            with open(filename, 'w') as f:
                f.write("# 1. rs.____| 2. N____| 3. Mc [Ha/e]\n")
        for rs in [0.5]:
            if precalc_veff:
                # Calculate effective potential on a grid:
                nmax = np.max(nelec)
                Lmax = (4*np.pi*nmax/3)**(1/3)*rs
                rmax = Rcut*Lmax + 0.5
                veff, rpoints = main(2, rs, \
                        k_symm=k_symm, \
                        tc_type=tc_type,\
                        corr=correlator, \
                        k_cutoff=k_cutoff, \
                        gamma=gamma, \
                        veff=None, \
                        rpoints=None, \
                        Rcut=Rcut, 
                        nr=nr, \
                        rmax=rmax, \
                        nrpoints=nrpoints, \
                        write_madelung=False,
                        write_veff=True, \
                        mode='veff')
            else:
                veff = None
                rpoints = None
            for nel in nelec:
                main(nel, rs, \
                    k_symm=k_symm, \
                    tc_type=tc_type,\
                    corr=correlator,
                    k_cutoff=k_cutoff, \
                    gamma=gamma, \
                    veff=veff, \
                    rpoints=rpoints, \
                    Rcut=Rcut, 
                    nr=nr, \
                    write_madelung=write_madelung, \
                    write_veff=False, \
                    filename=filename, \
                    mode='madelung')
