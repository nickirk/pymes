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
    - _contract_exchange_3_body vs UEG.contract_exchange_3_body
    - _contractP_KWithQ vs UEG.contractP_KWithQ
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
                                     _contract_exchange_3_body,
                                     _contractP_KWithQ)
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
    denom_thrs = ueg_model.denom_thrs
    k_cutoffSquare = (k_cutoff * 2 * np.pi / L) ** 2
    
    # Test with various k-vectors
    test_k_values = [
        0.0,                                           # Zero
        k_cutoffSquare * 0.5,                         # Below cutoff
        k_cutoffSquare * 1.0001,                      # Just above cutoff
        k_cutoffSquare * 2.0,                         # Well above cutoff
        (2 * np.pi / L) ** 2,                         # Typical value
        (5 * np.pi / L) ** 2,                         # Larger value
    ]
    
    all_pass = True
    max_diff = 0.0
    
    print_logging_info(f"Testing with {len(test_k_values)} k^2 values", level=1)
    
    for i, kSquare in enumerate(test_k_values):
        # Original function (scalar)
        result_original = ueg_model.trunc(kSquare)
        
        # Numba-compiled function
        result_compiled = _trunc_correlator(kSquare, k_cutoffSquare, gamma, denom_thrs)
        
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
    
    result_compiled_array = np.array([_trunc_correlator(k, k_cutoffSquare, gamma, denom_thrs) 
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
    gamma = ueg_model.gamma
    denom_thrs = ueg_model.denom_thrs
    
    # Test with various k-vectors
    test_k_values = [
        0.0,                                           # Zero (should return 0)
        1e-15,                                         # Very small (should return 0)
        (2 * np.pi / L) ** 2,                         # Typical value
        (5 * np.pi / L) ** 2,                         # Larger value
        (10 * np.pi / L) ** 2,                        # Even larger
    ]
    
    all_pass = True
    max_diff = 0.0
    
    print_logging_info(f"Testing with {len(test_k_values)} k^2 values", level=1)
    
    for i, kSquare in enumerate(test_k_values):
        # Original function
        result_original = ueg_model.coulomb(kSquare)
        
        # Numba-compiled function
        result_compiled = _coulomb_correlator(kSquare, gamma, denom_thrs)
        
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
    rho = ueg_model.n_ele / ueg_model.Omega
    gamma = ueg_model.gamma
    denom_thrs = ueg_model.denom_thrs
    
    # Test with various k-vectors
    test_k_values = [
        1e-15,                                         # Very small (threshold test)
        (2 * np.pi / ueg_model.L) ** 2,               # Typical value
        (5 * np.pi / ueg_model.L) ** 2,               # Larger value
        (10 * np.pi / ueg_model.L) ** 2,              # Even larger
    ]
    
    all_pass = True
    max_diff = 0.0
    
    print_logging_info(f"Testing with {len(test_k_values)} k^2 values", level=1)
    
    for i, kSquare in enumerate(test_k_values):
        # Original function
        result_original = ueg_model.coulomb_yukawa(kSquare)
        
        # Numba-compiled function
        result_compiled = _coulomb_yukawa_correlator(kSquare, rho, gamma, denom_thrs)
        
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
    
    result_compiled_array = np.array([_coulomb_yukawa_correlator(k, rho, gamma, denom_thrs) 
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
    rho = ueg_model.n_ele / ueg_model.Omega
    gamma = ueg_model.gamma
    denom_thrs = ueg_model.denom_thrs
    kFermi = (3.0 * np.pi**2 * rho) ** (1.0 / 3.0)
    
    # Test with various k-vectors (relative to k_Fermi)
    test_k_values = [
        1e-15,                                         # Very small (threshold test)
        (0.5 * kFermi) ** 2,                          # Below 2*k_F
        (1.5 * kFermi) ** 2,                          # Below 2*k_F
        (2.0 * kFermi) ** 2,                          # At 2*k_F
        (2.5 * kFermi) ** 2,                          # Above 2*k_F
        (5.0 * kFermi) ** 2,                          # Well above 2*k_F
    ]
    
    all_pass = True
    max_diff = 0.0
    
    print_logging_info(f"Testing with {len(test_k_values)} k^2 values", level=1)
    print_logging_info(f"k_Fermi = {kFermi:.6f}", level=1)
    
    for i, kSquare in enumerate(test_k_values):
        # Original function
        result_original = ueg_model.RPA(kSquare)
        
        # Numba-compiled function
        result_compiled = _RPA_correlator(kSquare, rho, gamma, denom_thrs)
        
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
    
    result_compiled_array = np.array([_RPA_correlator(k, rho, gamma, denom_thrs) 
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
    
    # Prepare test data
    rho = ueg_model.n_ele / ueg_model.Omega
    Omega = ueg_model.Omega
    L = ueg_model.L
    kPrime = ueg_model.kPrime.astype(np.float64) * ( 2 * np.pi / L)
    k_cutoff = ueg_model.k_cutoff
    gamma = ueg_model.gamma
    denom_thrs = ueg_model.denom_thrs
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
        result_compiled = _sumNablaUSquare(kVec, rho, Omega, kPrime, k_cutoffSquare, gamma, correlator_idx, denom_thrs)
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
            speedup = time_original/time_compiled if time_compiled > 0 else float('inf')
            print_logging_info(f"  Test {i+1} PASSED: idx={idx}, "
                             f"diff={diff:.12e}, "
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
    denom_thrs = ueg_model.denom_thrs
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
                                                   rho, Omega, k_cutoffSquare, gamma, correlator_idx, denom_thrs)
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
    denom_thrs = ueg_model.denom_thrs
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
                                           rho, Omega, k_cutoffSquare, gamma, correlator_idx, denom_thrs)
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


def main(nel=14, cutoff=2, rs=0.5, gamma=None, kc=1):
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
    """
    print_title("Testing Numba-Compiled UEG Helper Functions", "=")
    
    # Setup UEG model
    print_logging_info("Setting up UEG model", level=0)
    nalpha = nel // 2
    nbeta = nel // 2
    ueg_model = ueg.UEG(nel, nalpha, nbeta, rs, is_tc=True)
    
    print_logging_info(f"Number of electrons: {nel}", level=1)
    print_logging_info(f"Cutoff: {cutoff}", level=1)
    print_logging_info(f"rs: {rs}", level=1)
    print_logging_info(f"Omega: {ueg_model.Omega:.6f}", level=1)
    print_logging_info(f"L: {ueg_model.L:.6f}", level=1)
    
    # Initialize basis
    print_logging_info("Initializing basis set", level=0)
    ueg_model.init_single_basis(cutoff)
    nP = len(ueg_model.basis_fns) // 2
    print_logging_info(f"Number of spatial orbitals: {nP}", level=1)
    
    # Setup TC parameters
    print_logging_info("Setting up TC parameters", level=0)
    ueg_model.correlator = ueg_model.trunc
    ueg_model.k_cutoff = kc
    ueg_model.gamma = gamma if gamma is not None else 1.0
    ueg_model.init_kPrime()
    
    print_logging_info(f"Correlator: {ueg_model.correlator.__name__}", level=1)
    print_logging_info(f"k_cutoff: {ueg_model.k_cutoff}", level=1)
    print_logging_info(f"gamma: {ueg_model.gamma}", level=1)
    print_logging_info(f"kPrime shape: {ueg_model.kPrime.shape}", level=1)
    
    sys.stdout.flush()
    
    # Run tests
    test_results = {}
    
    # Test correlators
    test_results['trunc'] = test_trunc_correlator(ueg_model)
    sys.stdout.flush()
    
    test_results['coulomb'] = test_coulomb_correlator(ueg_model)
    sys.stdout.flush()
    
    test_results['coulomb_yukawa'] = test_coulomb_yukawa_correlator(ueg_model)
    sys.stdout.flush()
    
    test_results['RPA'] = test_RPA_correlator(ueg_model)
    sys.stdout.flush()
    
    # Test helper functions
    test_results['sumNablaUSquare'] = test_sumNablaUSquare(ueg_model)
    sys.stdout.flush()
    
    test_results['contract_exchange_3_body'] = test_contract_exchange_3_body(ueg_model)
    sys.stdout.flush()
    
    test_results['contractP_KWithQ'] = test_contractP_KWithQ(ueg_model)
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
    # Test with small system for quick validation
    exit_code = main(nel=14, cutoff=2, rs=0.5, gamma=None, kc=1)
    sys.exit(exit_code)
