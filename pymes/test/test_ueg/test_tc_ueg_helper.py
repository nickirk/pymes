#!/usr/bin/python3 -u

"""
Test script to verify that Numba-compiled helper functions in ueg_helper.py
produce identical results to the original UEG class methods.

Tests:
    - _trunc_correlator vs UEG.trunc
    - _coulomb_correlator vs UEG.coulomb
    - _coulomb_yukawa_correlator vs UEG.coulomb_yukawa
    - _RPA_correlator vs UEG.RPA
    - _sumNablaUSquare vs UEG.sumNablaUSquare
    - _intNablaUSquare vs UEG.intNablaUSquare
    - _contract_exchange_3_body vs UEG.contract_exchange_3_body
    - _contractP_KWithQ vs UEG.contractP_KWithQ
    - _double_contractions_in_3_body vs UEG.double_contractions_in_3_body
    - _triple_contractions_in_3_body vs UEG.triple_contractions_in_3_body
"""

import sys
import time
import numpy as np
from pymes.model import ueg
from pymes.model.ueg_helper import (_trunc_correlator,
                                     _coulomb_correlator,
                                     _coulomb_yukawa_correlator,
                                     _RPA_correlator,
                                     _sumNablaUSquare,
                                     _intNablaUSquare,
                                     _contract_exchange_3_body,
                                     _contractP_KWithQ,
                                     _double_contractions_in_3_body,
                                     _triple_contractions_in_3_body)
from pymes.log import print_title, print_logging_info

def test_trunc_correlator(ueg_model, tolerance=1e-12):
    """
    Test that _trunc_correlator gives the same results as UEG.trunc.
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model
    tolerance : float
        Absolute tolerance for comparison
    
    Returns
    -------
    bool : True if test passes, False otherwise
    """
    print_title("Testing _trunc_correlator", "=")
    
    # Prepare test data
    L = ueg_model.L
    k_cutoff = ueg_model.k_cutoff
    gamma = ueg_model.gamma
    k_cutoffSquare = (k_cutoff * 2 * np.pi / L) ** 2
    
    # Test with various k-vectors
    test_k_values = [ 
        0.0,                                           # Zero
        1.0e-12,                                       # Very small (at threshold)
        1.0e-8,                                        # Small
        1.0e-4,                                        # Small
        0.01,                                          # Small
        0.5,                                           # Small-medium
        k_cutoffSquare * 0.5,                         # Below cutoff
        k_cutoffSquare * 0.9,                         # Just below cutoff
        k_cutoffSquare * 1.0001,                      # Just above cutoff
        k_cutoffSquare * 1.5,                         # Above cutoff
        k_cutoffSquare * 2.0,                         # Well above cutoff
        (2 * np.pi / L) ** 2,                         # Typical value
        (5 * np.pi / L) ** 2,                         # Larger value
        (10 * np.pi / L) ** 2,                        # Large value
        (15 * np.pi / L) ** 2,                        # Very large value
        (20 * np.pi / L) ** 2,                        # Very large value
        (50 * np.pi / L) ** 2,                        # Extremely large value
    ]
    
    all_pass = True
    max_diff = 0.0
    
    print_logging_info(f"Testing with {len(test_k_values)} k^2 values", level=1)
    
    for i, kSquare in enumerate(test_k_values):
        # Original function (scalar)
        result_original = ueg_model.trunc(kSquare)
        
        # Numba-compiled function
        result_compiled = _trunc_correlator(kSquare, k_cutoffSquare, gamma)
        
        # Compare
        diff = np.abs(result_original - result_compiled)
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print_logging_info(f"  Test {i+1} FAILED: k^2={kSquare:.6e}, "
                             f"original={result_original:.12e}, "
                             f"compiled={result_compiled:.12e}, "
                             f"diff={diff:.12e}", level=2)
            all_pass = False
        else:
            print_logging_info(f"  Test {i+1} PASSED: k^2={kSquare:.6e}, "
                             f"diff={diff:.12e}", level=2)
    
    # Test with array input
    print_logging_info("Testing with array input", level=1)
    kSquare_array = np.array(test_k_values)
    result_original_array = ueg_model.trunc(kSquare_array)
    
    result_compiled_array = np.array([_trunc_correlator(k, k_cutoffSquare, gamma) 
                                      for k in kSquare_array])
    
    diff_array = np.abs(result_original_array - result_compiled_array)
    max_diff_array = np.max(diff_array)
    max_diff = max(max_diff, max_diff_array)
    
    if max_diff_array > tolerance:
        print_logging_info(f"  Array test FAILED: max_diff={max_diff_array:.12e}", level=2)
        all_pass = False
    else:
        print_logging_info(f"  Array test PASSED: max_diff={max_diff_array:.12e}", level=2)
    
    print_logging_info(f"Maximum difference: {max_diff:.12e}", level=1)
    print_logging_info(f"Test result: {'PASSED' if all_pass else 'FAILED'}", level=1)
    
    return all_pass


def test_coulomb_correlator(ueg_model, tolerance=1e-12):
    """
    Test that _coulomb_correlator gives the same results as UEG.coulomb.
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model
    tolerance : float
        Absolute tolerance for comparison
    
    Returns
    -------
    bool : True if test passes, False otherwise
    """
    print_title("Testing _coulomb_correlator", "=")
    
    # Prepare test data
    L = ueg_model.L
    k_cutoff = ueg_model.k_cutoff
    gamma = ueg_model.gamma
    k_cutoffSquare = (k_cutoff * 2 * np.pi / L) ** 2
    
    # Test with various k-vectorsa
    test_k_values = [
        0.0,                                           # Zero (should return 0)
        1e-15,                                         # Very small (threshold test)
        1e-12,                                         # At threshold
        1e-10,                                         # Small
        1e-6,                                          # Small
        0.001,                                         # Small-medium
        0.1,                                           # Medium
        (2 * np.pi / L) ** 2,                         # Typical value
        (5 * np.pi / L) ** 2,                         # Larger value
        (10 * np.pi / L) ** 2,                        # Large value
        (15 * np.pi / L) ** 2,                        # Very large value
        (20 * np.pi / L) ** 2,                        # Very large value
        (50 * np.pi / L) ** 2,                        # Extremely large value
    ]
    
    all_pass = True
    max_diff = 0.0
    
    print_logging_info(f"Testing with {len(test_k_values)} k^2 values", level=1)
    
    for i, kSquare in enumerate(test_k_values):
        # Original function
        result_original = ueg_model.coulomb(kSquare)
        
        # Numba-compiled function
        result_compiled = _coulomb_correlator(kSquare, k_cutoffSquare, gamma)
        
        # Compare
        diff = np.abs(result_original - result_compiled)
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print_logging_info(f"  Test {i+1} FAILED: k^2={kSquare:.6e}, "
                             f"original={result_original:.12e}, "
                             f"compiled={result_compiled:.12e}, "
                             f"diff={diff:.12e}", level=2)
            all_pass = False
        else:
            print_logging_info(f"  Test {i+1} PASSED: k^2={kSquare:.6e}, "
                             f"diff={diff:.12e}", level=2)
    
    print_logging_info(f"Maximum difference: {max_diff:.12e}", level=1)
    print_logging_info(f"Test result: {'PASSED' if all_pass else 'FAILED'}", level=1)
    
    return all_pass


def test_coulomb_yukawa_correlator(ueg_model, tolerance=1e-12):
    """
    Test that _coulomb_yukawa_correlator gives the same results as UEG.coulomb_yukawa.
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model
    tolerance : float
        Absolute tolerance for comparison
    
    Returns
    -------
    bool : True if test passes, False otherwise
    """
    print_title("Testing _coulomb_yukawa_correlator", "=")
    
    # Prepare test data
    L = ueg_model.L
    k_cutoff = ueg_model.k_cutoff
    rho = ueg_model.n_ele / ueg_model.Omega
    gamma = ueg_model.gamma
    k_cutoffSquare = (k_cutoff * 2 * np.pi / L) ** 2
    
    # Test with various k-vectors
    test_k_values = [
        1e-15,                                         # Very small (threshold test)
        1e-12,                                         # At threshold
        1e-10,                                         # Small
        1e-6,                                          # Small
        0.001,                                         # Small-medium
        0.1,                                           # Medium
        (0.5 * np.pi / L) ** 2,                       # Small k
        (2 * np.pi / L) ** 2,                         # Typical value
        (5 * np.pi / L) ** 2,                         # Larger value
        (10 * np.pi / L) ** 2,                        # Large value
        (15 * np.pi / L) ** 2,                        # Very large value
        (20 * np.pi / L) ** 2,                        # Very large value
        (50 * np.pi / L) ** 2,                        # Extremely large value
    ]
    
    all_pass = True
    max_diff = 0.0
    
    print_logging_info(f"Testing with {len(test_k_values)} k^2 values", level=1)
    
    for i, kSquare in enumerate(test_k_values):
        # Original function
        result_original = ueg_model.coulomb_yukawa(kSquare)
        
        # Numba-compiled function
        result_compiled = _coulomb_yukawa_correlator(kSquare, rho, k_cutoffSquare, gamma)
        
        # Compare
        diff = np.abs(result_original - result_compiled)
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print_logging_info(f"  Test {i+1} FAILED: k^2={kSquare:.6e}, "
                             f"original={result_original:.12e}, "
                             f"compiled={result_compiled:.12e}, "
                             f"diff={diff:.12e}", level=2)
            all_pass = False
        else:
            print_logging_info(f"  Test {i+1} PASSED: k^2={kSquare:.6e}, "
                             f"diff={diff:.12e}", level=2)
    
    # Test with array input
    print_logging_info("Testing with array input", level=1)
    kSquare_array = np.array(test_k_values)
    result_original_array = ueg_model.coulomb_yukawa(kSquare_array)
    
    result_compiled_array = np.array([_coulomb_yukawa_correlator(k, rho, k_cutoffSquare, gamma) 
                                      for k in kSquare_array])
    
    diff_array = np.abs(result_original_array - result_compiled_array)
    max_diff_array = np.max(diff_array)
    max_diff = max(max_diff, max_diff_array)
    
    if max_diff_array > tolerance:
        print_logging_info(f"  Array test FAILED: max_diff={max_diff_array:.12e}", level=2)
        all_pass = False
    else:
        print_logging_info(f"  Array test PASSED: max_diff={max_diff_array:.12e}", level=2)
    
    print_logging_info(f"Maximum difference: {max_diff:.12e}", level=1)
    print_logging_info(f"Test result: {'PASSED' if all_pass else 'FAILED'}", level=1)
    
    return all_pass


def test_RPA_correlator(ueg_model, tolerance=1e-12):
    """
    Test that _RPA_correlator gives the same results as UEG.RPA.
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model
    tolerance : float
        Absolute tolerance for comparison
    
    Returns
    -------
    bool : True if test passes, False otherwise
    """
    print_title("Testing _RPA_correlator", "=")
    
    # Prepare test data
    L = ueg_model.L
    k_cutoff = ueg_model.k_cutoff
    rho = ueg_model.n_ele / ueg_model.Omega
    gamma = ueg_model.gamma
    k_cutoffSquare = (k_cutoff * 2 * np.pi / L) ** 2
    kFermi = (3.0 * np.pi**2 * rho) ** (1.0 / 3.0)
    
    # Test with various k-vectors (relative to k_Fermi)
    test_k_values = [
        1e-15,                                         # Very small (threshold test)
        1e-12,                                         # At threshold
        1e-10,                                         # Small
        1e-6,                                          # Small
        0.001,                                         # Small-medium
        0.1,                                           # Medium
        (0.1 * kFermi) ** 2,                          # Well below k_F
        (0.5 * kFermi) ** 2,                          # Below 2*k_F
        (1.0 * kFermi) ** 2,                          # At k_F
        (1.5 * kFermi) ** 2,                          # Below 2*k_F
        (1.99 * kFermi) ** 2,                         # Just below 2*k_F (critical!)
        (2.0 * kFermi) ** 2,                          # At 2*k_F (critical point!)
        (2.01 * kFermi) ** 2,                         # Just above 2*k_F (critical!)
        (2.5 * kFermi) ** 2,                          # Above 2*k_F
        (5.0 * kFermi) ** 2,                          # Well above 2*k_F
        (10.0 * kFermi) ** 2,                         # Very large
        (20.0 * kFermi) ** 2,                         # Extremely large
    ]
    
    all_pass = True
    max_diff = 0.0
    
    print_logging_info(f"Testing with {len(test_k_values)} k^2 values", level=1)
    print_logging_info(f"k_Fermi = {kFermi:.6f}", level=1)
    
    for i, kSquare in enumerate(test_k_values):
        # Original function
        result_original = ueg_model.RPA(kSquare)
        
        # Numba-compiled function
        result_compiled = _RPA_correlator(kSquare, rho, k_cutoffSquare, gamma)
        
        # Compare
        diff = np.abs(result_original - result_compiled)
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print_logging_info(f"  Test {i+1} FAILED: k^2={kSquare:.6e}, "
                             f"original={result_original:.12e}, "
                             f"compiled={result_compiled:.12e}, "
                             f"diff={diff:.12e}", level=2)
            all_pass = False
        else:
            print_logging_info(f"  Test {i+1} PASSED: k^2={kSquare:.6e}, "
                             f"diff={diff:.12e}", level=2)
    
    # Test with array input
    print_logging_info("Testing with array input", level=1)
    kSquare_array = np.array(test_k_values)
    result_original_array = ueg_model.RPA(kSquare_array)
    
    result_compiled_array = np.array([_RPA_correlator(k, rho, k_cutoffSquare, gamma) 
                                      for k in kSquare_array])
    
    diff_array = np.abs(result_original_array - result_compiled_array)
    max_diff_array = np.max(diff_array)
    max_diff = max(max_diff, max_diff_array)
    
    if max_diff_array > tolerance:
        print_logging_info(f"  Array test FAILED: max_diff={max_diff_array:.12e}", level=2)
        all_pass = False
    else:
        print_logging_info(f"  Array test PASSED: max_diff={max_diff_array:.12e}", level=2)
    
    print_logging_info(f"Maximum difference: {max_diff:.12e}", level=1)
    print_logging_info(f"Test result: {'PASSED' if all_pass else 'FAILED'}", level=1)
    
    return all_pass


def test_sumNablaUSquare(ueg_model, tolerance=1e-10):
    """
    Test that _sumNablaUSquare gives the same results as UEG.sumNablaUSquare.
    
    NOTE: This test is only applicable for canonical TC (is_l_tc=False).
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model
    tolerance : float
        Absolute tolerance for comparison
    
    Returns
    -------
    bool : True if test passes, False otherwise
    """
    print_title("Testing _sumNablaUSquare", "=")
    
    # Check if this test is applicable
    if ueg_model.tc_type != 'canonical':
        print_logging_info("SKIPPING: sumNablaUSquare is for canonical TC only (tc_type='canonical')", level=1)
        return True
    
    # Validate prerequisites
    if ueg_model.kPrime is None:
        print_logging_info("ERROR: kPrime not initialized! Call init_kPrime() first.", level=1)
        return False
    
    # Prepare test data
    rho = ueg_model.n_ele / ueg_model.Omega
    Omega = ueg_model.Omega
    L = ueg_model.L
    kPrime = ueg_model.kPrime.astype(np.float64) * ( 2 * np.pi / L)
    k_cutoff = ueg_model.k_cutoff
    gamma = ueg_model.gamma
    k_cutoffSquare = (k_cutoff * 2 * np.pi / L) ** 2
    correlator_idx = ueg_model.get_correlator_idx()
    
    # Test with various k-vectors
    nP = len(ueg_model.basis_fns) // 2
    test_indices = [0, nP//4, nP//2, 3*nP//4, nP-1]
    
    all_pass = True
    max_diff = 0.0
    
    print_logging_info(f"Testing with {len(test_indices)} different k-vectors", level=1)
    print_logging_info(f"Using correlator index: {correlator_idx}", level=1)
    
    for i, idx in enumerate(test_indices):
        kVec = ueg_model.basis_fns[idx * 2].kp
        
        # Original function
        start = time.time()
        result_original = ueg_model.sumNablaUSquare(kVec)
        time_original = time.time() - start
        
        # Numba-compiled function
        start = time.time()
        result_compiled = _sumNablaUSquare(kVec, rho, Omega, kPrime, k_cutoffSquare, gamma, correlator_idx)
        time_compiled = time.time() - start
        
        # Compare
        diff = np.abs(result_original - result_compiled)
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print_logging_info(f"  Test {i+1} FAILED: idx={idx}, "
                             f"original={result_original:.12e}, "
                             f"compiled={result_compiled:.12e}, "
                             f"diff={diff:.12e}", level=2)
            all_pass = False
        else:
            print_logging_info(f"  Test {i+1} PASSED: idx={idx}, "
                             f"diff={diff:.12e}", level=2)
    
    print_logging_info(f"Maximum difference: {max_diff:.12e}", level=1)
    print_logging_info(f"Test result: {'PASSED' if all_pass else 'FAILED'}", level=1)
    
    return all_pass


def test_intNablaUSquare(ueg_model, tolerance=1e-10):
    """
    Test that _intNablaUSquare gives the same results as UEG.intNablaUSquare.
    
    NOTE: This test is only applicable for long-range TC (tc_type='long-range').
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model
    tolerance : float
        Absolute tolerance for comparison
    
    Returns
    -------
    bool : True if test passes, False otherwise
    """
    print_title("Testing _intNablaUSquare", "=")
    
    # Check if this test is applicable
    if ueg_model.tc_type != 'long-range':
        print_logging_info("SKIPPING: intNablaUSquare is for long-range TC only (tc_type='long-range')", level=1)
        return True
    
    # Validate prerequisites
    if ueg_model.kpts_mesh is None or ueg_model.xtheta_mesh is None:
        print_logging_info("ERROR: Convolution mesh not initialized! Call init_ConvMesh() first.", level=1)
        return False
    
    # Prepare test data
    rho = ueg_model.n_ele / ueg_model.Omega
    L = ueg_model.L
    k_cutoff = ueg_model.k_cutoff
    gamma = ueg_model.gamma
    k_cutoffSquare = (k_cutoff * 2 * np.pi / L) ** 2
    correlator_idx = ueg_model.get_correlator_idx()
    
    kpts_mesh = ueg_model.kpts_mesh
    xtheta_mesh = ueg_model.xtheta_mesh
    dkpts = ueg_model.dkpts
    dxtheta = ueg_model.dxtheta
    
    # Test with various k-vectors (INCLUDING k=0)
    nP = len(ueg_model.basis_fns) // 2
    test_indices = [0, nP//4, nP//2, 3*nP//4, nP-1]
    
    # Build test cases from basis functions (all should have |k| > 0)
    test_cases = []
    # Include k=0
    test_cases.append(np.array([0.0, 0.0, 0.0]))
    # Include small k values
    test_cases.append(np.array([1e-8, 0.0, 0.0]))
    test_cases.append(np.array([1e-8, 1e-8, 0.0]))
    test_cases.append(np.array([1e-8, 1e-8, 1e-8]))
    test_cases.append(np.array([1e-5, 0.0, 0.0]))
    test_cases.append(np.array([1e-5, 1e-5, 0.0]))
    test_cases.append(np.array([1e-5, 1e-5, 1e-5]))
    # Include allowed momentum transfer vectors
    for idx in test_indices:
        kVec = ueg_model.basis_fns[idx * 2].kp
        # Only add if |k| > 0
        k_mag = np.sqrt(kVec.dot(kVec))
        if k_mag > 1e-12:
            test_cases.append(kVec)
    
    all_pass = True
    max_diff = 0.0
    
    print_logging_info(f"Testing with {len(test_cases)} different k-vectors (k ≠ 0)", level=1)
    print_logging_info(f"Using correlator index: {correlator_idx}", level=1)
    print_logging_info(f"Grid sizes: n_kp={len(kpts_mesh)}, n_x={len(xtheta_mesh)}", level=1)
    
    for i, kVec in enumerate(test_cases):
        k_mag = np.sqrt(kVec.dot(kVec))
        
        # Original function
        start = time.time()
        result_original = ueg_model.intNablaUSquare(kVec)
        time_original = time.time() - start
        
        # Numba-compiled function
        start = time.time()
        result_compiled = _intNablaUSquare(kVec, kpts_mesh, xtheta_mesh, dkpts, dxtheta, \
                                          rho, k_cutoffSquare, gamma, correlator_idx)
        time_compiled = time.time() - start
        
        # Compare
        diff = np.abs(result_original - result_compiled)
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print_logging_info(f"  Test {i+1} FAILED: |k|={k_mag:.6e}, "
                             f"original={result_original:.12e}, "
                             f"compiled={result_compiled:.12e}, "
                             f"time_orig={time_original:.4f}s, "
                             f"time_comp={time_compiled:.4f}s, "
                             f"diff={diff:.12e}", level=2)
            all_pass = False
        else:
            speedup = time_original/time_compiled if time_compiled > 0 else float('inf')
            print_logging_info(f"  Test {i+1} PASSED: |k|={k_mag:.6e}, "
                             f"diff={diff:.12e}, "
                             f"time_orig={time_original:.4f}s, "
                             f"time_comp={time_compiled:.4f}s, "
                             f"speedup={speedup:.2f}x", level=2)
    
    print_logging_info(f"Maximum difference: {max_diff:.12e}", level=1)
    print_logging_info(f"Test result: {'PASSED' if all_pass else 'FAILED'}", level=1)
    
    return all_pass


def test_contract_exchange_3_body(ueg_model, tolerance=1e-10):
    """
    Test that _contract_exchange_3_body gives the same results as 
    UEG.contract_exchange_3_body.
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model
    tolerance : float
        Absolute tolerance for comparison
    
    Returns
    -------
    bool : True if test passes, False otherwise
    """
    print_title("Testing _contract_exchange_3_body", "=")
    
    # Prepare test data
    rho = ueg_model.n_ele / ueg_model.Omega
    Omega = ueg_model.Omega
    L = ueg_model.L
    k_cutoff = ueg_model.k_cutoff
    gamma = ueg_model.gamma
    k_cutoffSquare = (k_cutoff * 2 * np.pi / L) ** 2
    correlator_idx = ueg_model.get_correlator_idx()
    
    no = ueg_model.n_ele // 2
    basis_occ_Kp = np.array([ueg_model.basis_fns[i * 2].kp for i in range(no)], 
                            dtype=np.float64)
    
    # Test with various k-vectors
    nP = len(ueg_model.basis_fns) // 2
    test_cases = [
        (0, nP//2),
        (nP//4, 3*nP//4),
        (no-1, no),
        (no, 3*nP//4),
        (0, nP-1),
    ]
    
    all_pass = True
    max_diff = 0.0
    
    print_logging_info(f"Testing with {len(test_cases)} different (p, q) pairs", level=1)
    print_logging_info(f"Using correlator index: {correlator_idx}", level=1)
    
    for i, (p_idx, q_idx) in enumerate(test_cases):
        p_vec = ueg_model.basis_fns[p_idx * 2].kp
        q_vec = ueg_model.basis_fns[q_idx * 2].kp
        kVec = p_vec - q_vec
        
        # Original function
        start = time.time()
        result_original = ueg_model.contract_exchange_3_body(p_vec, kVec)
        time_original = time.time() - start
        
        # Numba-compiled function
        start = time.time()
        result_compiled = _contract_exchange_3_body(p_vec, kVec, basis_occ_Kp, 
                                                   rho, Omega, k_cutoffSquare, gamma, correlator_idx)
        time_compiled = time.time() - start
        
        # Compare
        diff = np.abs(result_original - result_compiled)
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print_logging_info(f"  Test {i+1} FAILED: (p,q)=({p_idx},{q_idx}), "
                             f"original={result_original:.12e}, "
                             f"compiled={result_compiled:.12e}, "
                             f"diff={diff:.12e}", level=2)
            all_pass = False
        else:
            speedup = time_original/time_compiled if time_compiled > 0 else float('inf')
            print_logging_info(f"  Test {i+1} PASSED: (p,q)=({p_idx},{q_idx}), "
                             f"diff={diff:.12e}, "
                             f"speedup={speedup:.2f}x", level=2)
    
    print_logging_info(f"Maximum difference: {max_diff:.12e}", level=1)
    print_logging_info(f"Test result: {'PASSED' if all_pass else 'FAILED'}", level=1)
    
    return all_pass


def test_contractP_KWithQ(ueg_model, tolerance=1e-10):
    """
    Test that _contractP_KWithQ gives the same results as UEG.contractP_KWithQ.
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model
    tolerance : float
        Absolute tolerance for comparison
    
    Returns
    -------
    bool : True if test passes, False otherwise
    """
    print_title("Testing _contractP_KWithQ", "=")
    
    # Prepare test data
    rho = ueg_model.n_ele / ueg_model.Omega
    Omega = ueg_model.Omega
    L = ueg_model.L
    k_cutoff = ueg_model.k_cutoff
    gamma = ueg_model.gamma
    k_cutoffSquare = (k_cutoff * 2 * np.pi / L) ** 2
    correlator_idx = ueg_model.get_correlator_idx()
    
    no = ueg_model.n_ele // 2
    basis_occ_Kp = np.array([ueg_model.basis_fns[i * 2].kp for i in range(no)], 
                            dtype=np.float64)
    
    # Test with various k-vectors
    nP = len(ueg_model.basis_fns) // 2
    test_cases = [
        (0, nP//2),
        (nP//4, 3*nP//4),
        (no-1, no),
        (no, 3*nP//4),
        (0, nP-1),
    ]
    
    all_pass = True
    max_diff = 0.0
    
    print_logging_info(f"Testing with {len(test_cases)} different (p, q) pairs", level=1)
    print_logging_info(f"Using correlator index: {correlator_idx}", level=1)
    
    for i, (p_idx, q_idx) in enumerate(test_cases):
        p_vec = ueg_model.basis_fns[p_idx * 2].kp
        q_vec = ueg_model.basis_fns[q_idx * 2].kp
        kVec = p_vec - q_vec
        
        # Original function
        start = time.time()
        result_original = ueg_model.contractP_KWithQ(p_vec, kVec)
        time_original = time.time() - start
        
        # Numba-compiled function
        start = time.time()
        result_compiled = _contractP_KWithQ(p_vec, kVec, basis_occ_Kp, 
                                           rho, Omega, k_cutoffSquare, gamma, correlator_idx)
        time_compiled = time.time() - start
        
        # Compare
        diff = np.abs(result_original - result_compiled)
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print_logging_info(f"  Test {i+1} FAILED: (p,q)=({p_idx},{q_idx}), "
                             f"original={result_original:.12e}, "
                             f"compiled={result_compiled:.12e}, "
                             f"diff={diff:.12e}", level=2)
            all_pass = False
        else:
            speedup = time_original/time_compiled if time_compiled > 0 else float('inf')
            print_logging_info(f"  Test {i+1} PASSED: (p,q)=({p_idx},{q_idx}), "
                             f"diff={diff:.12e}, "
                             f"speedup={speedup:.2f}x", level=2)
    
    print_logging_info(f"Maximum difference: {max_diff:.12e}", level=1)
    print_logging_info(f"Test result: {'PASSED' if all_pass else 'FAILED'}", level=1)
    
    return all_pass

def test_double_contractions_3_body(ueg_model, tolerance=1e-10):
    """
    Test that _double_contractions_3_body gives the same results as 
    UEG.double_contractions_3_body.
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model
    tolerance : float
        Absolute tolerance for comparison
    
    Returns
    -------
    bool : True if test passes, False otherwise
    """
    print_title("Testing _double_contractions_3_body", "=")
    nP = ueg_model.n_ele // 2
    wp_original =  np.zeros(nP)
    wp_compiled =  np.zeros(nP)
    # Original function
    start_original = time.time()
    wp_original = ueg_model.double_contractions_in_3_body()
    time_original = time.time() - start_original
    # Numba-compiled function
    _ = ueg_model.get_double_contractions_3b_int()
    start_compiled = time.time()
    wp_compiled = ueg_model.get_double_contractions_3b_int()
    time_compiled = time.time() - start_compiled
    # Compare
    diff = np.abs(wp_original - wp_compiled)
    max_diff = np.max(diff)
    if max_diff > tolerance:
        print_logging_info(f"  Test FAILED: max_diff={max_diff:.12e}", level=2)
        all_pass = False
    else:
        speedup = time_original/time_compiled if time_compiled > 0 else float('inf')
        print_logging_info(f"  Test PASSED: max_diff={max_diff:.12e}, "
                         f"time_orig={time_original:.4f}s, "
                         f"time_comp={time_compiled:.4f}s, "
                         f"speedup={speedup:.2f}x", level=2)
        all_pass = True
    print_logging_info(f"Maximum difference: {max_diff:.12e}", level=1)
    print_logging_info(f"Test result: {'PASSED' if all_pass else 'FAILED'}", level=1)
    return all_pass

def test_triple_contractions_3_body(ueg_model, tolerance=1e-10):
    """
    Test that _triple_contractions_3_body gives the same results as 
    UEG.triple_contractions_3_body.
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model
    tolerance : float
        Absolute tolerance for comparison
    
    Returns
    -------
    bool : True if test passes, False otherwise
    """
    print_title("Testing _triple_contractions_3_body", "=")
    nP = ueg_model.n_ele // 2
    # Original function
    start_original = time.time()
    ET_original = ueg_model.triple_contractions_in_3_body()
    time_original = time.time() - start_original
    # Numba-compiled function
    _ = ueg_model.get_triple_contractions_3b_int()
    start_compiled = time.time()
    ET_compiled = ueg_model.get_triple_contractions_3b_int()
    time_compiled = time.time() - start_compiled
    # Compare
    diff = np.abs(ET_original - ET_compiled)
    if diff > tolerance:
        print_logging_info(f"  Test FAILED: diff={diff:.12e}", level=2)
        all_pass = False
    else:
        speedup = time_original/time_compiled if time_compiled > 0 else float('inf')
        print_logging_info(f"  Test PASSED: diff={diff:.12e}, "
                         f"time_orig={time_original:.4f}s, "
                         f"time_comp={time_compiled:.4f}s, "
                         f"speedup={speedup:.2f}x", level=2)
        all_pass = True
    print_logging_info(f"Difference: {diff:.12e}", level=1)
    print_logging_info(f"Test result: {'PASSED' if all_pass else 'FAILED'}", level=1)
    return all_pass


def main(nel=14, cutoff=2, rs=0.5, gamma=None, kc=1, correlator='trunc', tc='canonical'):
    """
    Main test function.
    
    Parameters
    ----------
    nel : int
        Number of electrons
    cutoff : float
        Kinetic energy cutoff for basis
    rs : float
        Wigner-Seitz radius
    gamma : float or None
        Gamma parameter for correlator
    kc : float
        K-cutoff fraction for correlator
    correlator : str
        Correlator type: 'trunc', 'coulomb', 'coulomb_yukawa', 'RPA'
    tc : str
        TC type: 'canonical' or 'long-range'
    """
    print_title("Testing Numba JIT-Compiled UEG Helper Functions", "=")
    
    # Setup UEG model
    print_logging_info("Setting up UEG model", level=0)
    nalpha = nel // 2
    nbeta = nel // 2
    ueg_model = ueg.UEG(nel, nalpha, nbeta, rs, tc=tc)
    
    print_logging_info(f"Number of electrons: {nel}", level=1)
    print_logging_info(f"Cutoff: {cutoff}", level=1)
    print_logging_info(f"rs: {rs}", level=1)
    print_logging_info(f"Omega: {ueg_model.Omega:.6f}", level=1)
    print_logging_info(f"L: {ueg_model.L:.6f}", level=1)
    print_logging_info(f"TC type: {'l-TC [long-range]' if tc == 'long-range' else 'canonical TC'}", level=1)
    
    # Initialize basis
    print_logging_info("Initializing basis set", level=0)
    ueg_model.init_single_basis(cutoff)
    nP = len(ueg_model.basis_fns) // 2
    print_logging_info(f"Number of spatial orbitals: {nP}", level=1)
    
    # Setup TC parameters
    print_logging_info("Setting up TC parameters", level=0)
    if correlator == 'trunc':
        ueg_model.correlator = ueg_model.trunc
    elif correlator == 'coulomb':
        ueg_model.correlator = ueg_model.coulomb
    elif correlator == 'coulomb_yukawa':
        ueg_model.correlator = ueg_model.coulomb_yukawa
    elif correlator == 'RPA':
        ueg_model.correlator = ueg_model.RPA
    else:
        raise ValueError(f"Unknown correlator type: {correlator}")
    ueg_model.k_cutoff = kc
    ueg_model.gamma = gamma if gamma is not None else 1.0
    
    print_logging_info(f"Correlator: {ueg_model.correlator.__name__}", level=1)
    print_logging_info(f"k_cutoff: {ueg_model.k_cutoff}", level=1)
    print_logging_info(f"gamma: {ueg_model.gamma}", level=1)
    
    # TC type-specific initialization
    if tc == 'long-range':
        print_logging_info("Using long-range TC (long-range transcorrelated) method", level=0)
        print_logging_info("Initializing convolution mesh for long-range TC", level=1)
        ueg_model.init_ConvMesh(nx=200, dkfac=60, kmaxfac=20)
        print_logging_info(f"kpts_mesh size: {len(ueg_model.kpts_mesh)}", level=1)
        print_logging_info(f"xtheta_mesh size: {len(ueg_model.xtheta_mesh)}", level=1)
    else:
        print_logging_info("Using canonical TC method", level=0)
        print_logging_info("Initializing kPrime for canonical TC", level=1)
        ueg_model.init_kPrime()
        print_logging_info(f"kPrime shape: {ueg_model.kPrime.shape}", level=1)
    
    sys.stdout.flush()
    
    # Run tests
    test_results = {}
    
    # Test correlators (common to both TC types)
    test_results['trunc'] = test_trunc_correlator(ueg_model)
    sys.stdout.flush()
    
    test_results['coulomb'] = test_coulomb_correlator(ueg_model)
    sys.stdout.flush()
    
    test_results['coulomb_yukawa'] = test_coulomb_yukawa_correlator(ueg_model)
    sys.stdout.flush()
    
    test_results['RPA'] = test_RPA_correlator(ueg_model)
    sys.stdout.flush()
    
    # Test helper functions based on TC type
    if tc == 'canonical':
        # Canonical TC tests
        print_logging_info("Testing canonical TC helper function: sumNablaUSquare", level=0)
        test_results['sumNablaUSquare'] = test_sumNablaUSquare(ueg_model)
        sys.stdout.flush()
    else:
        # long-range TC tests
        print_logging_info("Testing long-range TC helper function: intNablaUSquare", level=0)
        test_results['intNablaUSquare'] = test_intNablaUSquare(ueg_model)
        sys.stdout.flush()
    
    # Test 3-body contraction functions (common to both TC types)
    test_results['contract_exchange_3_body'] = test_contract_exchange_3_body(ueg_model)
    sys.stdout.flush()
    
    test_results['contractP_KWithQ'] = test_contractP_KWithQ(ueg_model)
    sys.stdout.flush()

    test_results['double_contractions_3_body'] = test_double_contractions_3_body(ueg_model)
    sys.stdout.flush()

    test_results['triple_contractions_3_body'] = test_triple_contractions_3_body(ueg_model)
    sys.stdout.flush()
    
    # Summary
    print_title("Test Summary", "=")
    all_passed = all(test_results.values())
    
    for test_name, passed in test_results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
        print_logging_info(f"{test_name:30s}: {status}", level=0)
    
    print_logging_info("", level=0)
    if all_passed:
        print_logging_info("ALL TESTS PASSED ✓✓✓", level=0)
        return 0
    else:
        print_logging_info("SOME TESTS FAILED ✗✗✗", level=0)
        return 1


if __name__ == '__main__':
    # Test both canonical TC and long-range TC methods
    print_title("=" * 80, "=")
    print_title("TESTING CANONICAL TC METHOD", "=")
    print_title("=" * 80, "=")
    exit_code_tc = main(nel=14, cutoff=2, rs=0.5, gamma=None, kc=1e-12, correlator='RPA', tc='canonical')
    
    print("\n" * 3)
    
    print_title("=" * 80, "=")
    print_title("TESTING LONG-RANGE TC METHOD", "=")
    print_title("=" * 80, "=")
    exit_code_ltc = main(nel=14, cutoff=2, rs=0.5, gamma=None, kc=1e-12, correlator='RPA', tc='long-range')
    
    # Return non-zero if either test failed
    sys.exit(max(exit_code_tc, exit_code_ltc))
