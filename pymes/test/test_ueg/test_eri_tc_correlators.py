#!/usr/bin/python3 -u

import time
import sys
import os
import numpy as np

from pymes.model import ueg
from pymes.integral import eri
from pymes.util.parallel_tasks import print_threading_info
from pymes.log import print_title, print_logging_info

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print_logging_info("ERROR: NUMBA is not available.", level=0)
    sys.exit(1)

sys.stdout.flush()

def main(nel, rs, basis_cutoff, k_cutoff, gamma, correlator_type='RPA',
         nx=200, dkfac=60, kmaxfac=20.0):
    """
    Test to evaluate correlators and convolution integrals on predefined k-vectors.
    
    Parameters
    ----------
    nel : int
        Number of electrons
    rs : float
        Density parameter
    basis_cutoff : float
        Plane wave basis cutoff
    k_cutoff : float
        Cutoff parameter for correlator
    gamma : float or None
        Correlator parameter
    correlator_type : str
        Type of correlator: 'RPA', 'trunc', 'coulomb', 'coulomb_yukawa'
    nx : int
        Number of points in cos(theta) grid for ConvMesh
    dkfac : int
        Determines k' grid spacing as dk = k_F/dkfac
    kmaxfac : float
        Maximum k' value: kmax = k_F * kmaxfac
    """
    
    no = int(nel/2)
    nalpha = int(nel/2)
    nbeta = int(nel/2)
    
    # Initialize UEG model with transcorrelation
    time_set_sys = time.time()
    ueg_model = ueg.UEG(nel, nalpha, nbeta, rs, is_tc=True)
    
    print_title("System Information Summary", '=')
    print_logging_info("Number of electrons = {}".format(nel))
    print_logging_info("rs = {}".format(rs))
    print_logging_info("Volume of the box = {:.8f}".format(ueg_model.Omega))
    print_logging_info("Length of the box = {:.8f}".format(ueg_model.L))
    print_logging_info("dk0 = 2*pi/L = {:.8f}".format(ueg_model.dk0))
    print_logging_info("Fermi wave vector k_F = {:.8f}".format(ueg_model.kFermi))
    print_logging_info("{:.3f} seconds spent on setting up model"\
                       .format((time.time()-time_set_sys)))
    sys.stdout.flush()
    
    # Initialize basis set
    time_init_basis = time.time()
    ueg_model.init_single_basis(basis_cutoff)
    
    num_spatial_orb = int(len(ueg_model.basis_fns)/2)
    nv = num_spatial_orb - no
    
    print_title('Basis set', '=')
    print_logging_info('Number of spatial orbitals (plane waves) = {}'\
                        .format(num_spatial_orb))
    print_logging_info('Number of occupied orbitals = {}'\
                        .format(no))
    print_logging_info('Number of virtual orbitals = {}'\
                        .format(nv))
    print_logging_info("Max. kinetic energy (ecut) = {:.6f} [a.u.]"
                       .format(ueg_model.cutoff * (2 * np.pi / ueg_model.L) ** 2 / 2.))
    print_logging_info("{:.3f} seconds spent on generating basis."\
                       .format((time.time()-time_init_basis)))
    sys.stdout.flush()
    
    # Set up correlator
    print_title('Correlator Setup', '=')
    if correlator_type.upper() == 'RPA':
        ueg_model.correlator = ueg_model.RPA
    elif correlator_type.lower() == 'trunc':
        ueg_model.correlator = ueg_model.trunc
    elif correlator_type.lower() == 'coulomb':
        ueg_model.correlator = ueg_model.coulomb
    elif correlator_type.lower() == 'coulomb_yukawa':
        ueg_model.correlator = ueg_model.coulomb_yukawa
    else:
        raise ValueError("Unknown correlator type: {}".format(correlator_type))
    
    ueg_model.k_cutoff = k_cutoff
    ueg_model.gamma = gamma
    
    print_logging_info("Correlator type: {}".format(correlator_type))
    print_logging_info("k_cutoff = {}".format(k_cutoff))
    print_logging_info("gamma = {}".format(gamma))
    sys.stdout.flush()
    
    # Initialize ConvMesh for convolution integrals
    print_title('Convolution Mesh Initialization', '=')
    time_init_mesh = time.time()
    ueg_model.init_kPrime(cutoff=1) # Dummy
    ueg_model.init_ConvMesh(nx=nx, dkfac=dkfac, kmaxfac=kmaxfac)
    print_logging_info("ConvMesh parameters:")
    print_logging_info("  nx (cos(theta) points) = {}".format(nx))
    print_logging_info("  dkfac = {}".format(dkfac))
    print_logging_info("  kmaxfac = {}".format(kmaxfac))
    print_logging_info("  k' grid points = {}".format(len(ueg_model.kpts_mesh)))
    print_logging_info("  dk' = {:.8f}".format(ueg_model.dkpts))
    print_logging_info("  k'_max = {:.8f}".format(ueg_model.kptsmax))
    print_logging_info("{:.3f} seconds spent on ConvMesh initialization"\
                       .format((time.time()-time_init_mesh)))
    sys.stdout.flush()
    
    # Compute Fk0 (convolution at k=0)
    print_title('Computing F{(∇u)²}(k=0)', '=')
    time_fk0 = time.time()
    kGamma = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    Fk0 = ueg_model.intNablaUSquare(kGamma)
    ueg_model.Fk0_conv = Fk0
    Fk0_value = 0.375355528756647  # Known benchmark value
    abs_err_fk0 = abs(-Fk0 - Fk0_value)
    rel_err_fk0 = abs_err_fk0 / abs(Fk0_value) * 100.0
    print_logging_info("Known benchmark F{{(∇u)²}}(k=0) = {:.12e}".format(Fk0_value))
    print_logging_info("Computed F{{(∇u)²}}(k=0) = {:.12e}".format(Fk0))
    print_logging_info("Absolute error = {:.12e}".format(abs_err_fk0))
    print_logging_info("Relative error = {:.6f} %".format(rel_err_fk0))
    print_logging_info("{:.3f} seconds spent on F{{(∇u)²}}(k=0) computation"\
                       .format((time.time()-time_fk0)))
    sys.stdout.flush()

    # =============================================================================
    # STAGE 1: Evaluate correlators on scalar k values
    # =============================================================================
    print_title('STAGE 1: Correlator Evaluation on Scalar k Values', '=')
    
    # Define scalar k values and u(k) benchmark (in a.u.) 
    k_values = [ 
                1.919164816580675E-002,
                3.838329633161350E-002,
                5.757494449742025E-002,
                7.676659266322700E-002,
                9.595824082903374E-002,
                0.115149888994840,
                0.134341537160647,
                0.153533185326454,                     
                0.172724833492261,                     
                0.191916481658067,                     
                0.211108129823874,                     
                0.230299777989681,                     
                0.249491426155488
    ]

    u_values = [
                -6894.87228655380,                     
                -1706.52693483727,                     
                -750.893434925046,                     
                -418.166906220201,                    
                -264.959800325587,                    
                -182.167021650696,                     
                -132.504530200823,                     
                -100.439384444868,                     
                -78.5704503542809,                     
                -63.0097726009929,                     
                -51.5572763687703,                     
                -42.8928010343729,                     
                -36.1856399593403
    ]
    
    print_logging_info("Evaluating correlator u(k) for k values (a.u.):")
    print_logging_info("")
    print_logging_info("{:>15s} {:>20s} {:>20s} {:>15s}".format("k [a.u.]", "u(k)", "Abs. Err.", "% Err."))
    print_logging_info("-" * 80)
    
    for k, u in zip(k_values, u_values):
        k_square = k ** 2
        u_k = ueg_model.correlator(k_square)
        abs_err = abs(u_k - u)
        rel_err = abs_err / abs(u) * 100.0
        print_logging_info("{:20.12e} {:20.12e} {:20.12e}  {:4.2e} %".format(k, u_k, abs_err, rel_err))
    
    print_logging_info("")
    sys.stdout.flush()
    
    # =============================================================================
    # STAGE 2: Evaluate u(k) and convolution on 3D grid vectors
    # =============================================================================
    print_title('STAGE 2: Evaluation on 3D Grid Vectors', '=')
    
    # Define 3D grid vectors (in units of dk0 = 2*pi/L)
    # Components are integers: [nx, ny, nz]
    grid_vectors = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 2],
        [0, 2, 0],
        [0, 2, 1],
        [0, 2, 2],
        [1, 0, 0],
        [1, 0, 1],
        [1, 0, 2],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 0],
        [1, 2, 1],
        [1, 2, 2],
        [2, 0, 0],
        [2, 0, 1],
        [2, 0, 2],
        [2, 1, 0],
        [2, 1, 1],
        [2, 1, 2],
        [2, 2, 0],
        [2, 2, 1],
        [2, 2, 2]
    ]

    conv_values = [
            0.375355528756647,      
            4.328133621187336E-002,
            8.907082003577486E-003,
            4.328133621187336E-002,
            2.224441895227200E-002,
            6.161423310997848E-003,
            8.907082003577486E-003,
            6.161423310997848E-003,
            2.573377645688652E-003,
            4.328133621187336E-002,
            2.224441895227200E-002,
            6.161423310997848E-003,
            2.224441895227200E-002,
            1.353955784159294E-002,
            4.444896149135484E-003,
            6.161423310997848E-003,
            4.444896149135484E-003,
            2.040636128890584E-003,
            8.907082003577486E-003,
            6.161423310997848E-003,
            2.573377645688652E-003,
            6.161423310997848E-003,
            4.444896149135484E-003,
            2.040636128890584E-003,
            2.573377645688652E-003,
            2.040636128890584E-003,
            1.143061660769310E-003
    ]
    
    print_logging_info("Evaluating u(k) and F{{(∇u)²}}(k) on 3D grid:")
    print_logging_info("Grid vectors in units of dk0 = {:.8f}".format(ueg_model.dk0))
    print_logging_info("")
    print_logging_info("{:>4s} {:>4s} {:>4s} {:>15s} {:>15s} {:>20s} {:>20s} {:>15s} {:>15s}".format(
        "nx", "ny", "nz", "|k| [a.u.]", "u(k)", "F{(∇u)²}(k)", "Abs. Err.", "% Err.", "Time [s]"))
    print_logging_info("-" * 125)
    
    for kvec, conv in zip(grid_vectors, conv_values):
        nx_int, ny_int, nz_int = kvec
        
        # Convert to k-vector in absolute units
        k_vec = np.array([nx_int, ny_int, nz_int], dtype=np.float64) * ueg_model.dk0
        k_mag = np.linalg.norm(k_vec)
        k_square = k_mag ** 2
        
        # Evaluate correlator u(k²)
        u_k = ueg_model.correlator(k_square)
        
        # Evaluate convolution integral F{(∇u)²}(k)
        time_conv_start = time.time()
        if k_mag < 1.e-12:
            # Use precomputed Fk0
            conv_k = Fk0
        else:
            conv_k = ueg_model.intNablaUSquare(k_vec)
        time_conv = time.time() - time_conv_start

        abs_err = abs(-conv_k - conv)
        rel_err = abs_err / abs(conv) * 100.0
        
        print_logging_info("{:4d} {:4d} {:4d} {:15.8f} {:20.12e} {:20.12e} {:20.12e}  {:4.2e} % {:10.3f}".format(
            nx_int, ny_int, nz_int, k_mag, u_k, -conv_k, abs_err, rel_err, time_conv))
        sys.stdout.flush()
    
    print_logging_info("")
    print_logging_info("=" * 125)

    # =============================================================================
    # STAGE 3: Evaluating the xTC-HF Mean Field (Reference) Energy
    # =============================================================================
    print_title('STAGE 3: Evaluation of the xTC-HF (Mean Field) Reference Energy', '=')

    print_logging_info("Using known benchmark conv Gamma k=0.'")
    ueg_model.Fk0_conv = -0.3753885175131227

    time_init_eri = time.time()
    myERI = eri.ERI(ueg_model)
    myERI.calc_eri(incore=False)
    print_logging_info("{:.3f} seconds spent on constructing in-core ERI."\
                       .format((time.time()-time_init_eri)))
    print_logging_info("")
    HFE = myERI.EHF
    xTCHF = 54.218771372414587  # Known benchmark value
    abs_err_hfe = abs(HFE - xTCHF)
    rel_err_hfe = abs_err_hfe / abs(xTCHF) * 100.0
    print_logging_info("Known benchmark xTC-HF (Mean Field) Energy = {:.12f}".format(xTCHF))
    print_logging_info("Computed xTC-HF (Mean Field) Energy = {:.12f}".format(HFE))
    print_logging_info("Absolute error = {:.12e}".format(abs_err_hfe))
    print_logging_info("Relative error = {:.6f} %".format(rel_err_hfe))
    print_logging_info("=" * 125)

    print_logging_info("Test completed successfully!")
    sys.stdout.flush()


if __name__ == '__main__':
    
    # Default parameters
    nel = 14
    rs = 0.5
    basis_cutoff = 2
    k_cutoff = 1e-12
    gamma = None
    correlator_type = 'RPA'
    
    # ConvMesh parameters
    nx = 200
    dkfac = 60
    kmaxfac = 50
    
    main(nel, rs, basis_cutoff, k_cutoff, gamma, 
         correlator_type=correlator_type,
         nx=nx, dkfac=dkfac, kmaxfac=kmaxfac)
