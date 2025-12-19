#!/usr/bin/python3 -u

"""
Test script for convergence of intNablaUSquare (TDL convolution integral)
with respect to integration grid parameters.

This script tests the convolution integral:
F{(∇u)²}(k) = ∫ d³k' (k'·(k-k')) u(k') u(|k-k'|)

The integral is computed using spherical coordinates with parameters:
    - nx: number of x = cos(θ) grid points ∈ [-1, 1]
    - dkfac: determines k' grid spacing as dk = k_F/dkfac
    - kmaxfac: maximum k' value as kmax = k_F × kmaxfac

The test systematically varies ONE parameter at a time while keeping others at default values.
For each k-point, three separate files are generated:
    - k{value}_nx.dat: varying nx
    - k{value}_dkfac.dat: varying dkfac
    - k{value}_kmaxfac.dat: varying kmaxfac
"""

import sys
import time
import numpy as np
from pymes.model import ueg
from pymes.log import print_title, print_logging_info


def sweep_nx_parameter(ueg_model, k_vec, k_mag, nx_values, nx_default, dkfac_default, kmaxfac_default, output_file):
    """
    Sweep nx parameter while keeping dkfac and kmaxfac at default values.
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model with correlator set
    k_vec : numpy.ndarray
        K-vector to test (3D array)
    k_mag : float
        Magnitude of k-vector
    nx_values : list of int
        List of nx values to test
    nx_default : int
        Default nx value (not used in this sweep)
    dkfac_default : int
        Default dkfac value (held constant)
    kmaxfac_default : float
        Default kmaxfac value (held constant)
    output_file : str
        Output filename
    
    Returns
    -------
    results : dict
        Dictionary containing sweep results
    """
    print_title(f"Sweeping nx parameter for |k| = {k_mag:.6f}", "-")
    print_logging_info(f"Fixed parameters: dkfac={dkfac_default}, kmaxfac={kmaxfac_default}", level=1)
    
    results = {
        'k_vector': k_vec.copy(),
        'k_magnitude': k_mag,
        'parameter': 'nx',
        'parameter_values': [],
        'integral_values': [],
        'computation_times': [],
        'n_kp_points': [],
        'n_x_points': [],
        'dx_values': [],
        'fixed_dkfac': dkfac_default,
        'fixed_kmaxfac': kmaxfac_default,
    }
    
    for nx in nx_values:
        print_logging_info(f"Testing nx={nx}", level=2)
        
        # Initialize grid with current nx and default values for other parameters
        ueg_model.init_ConvMesh(nx=nx, dkfac=dkfac_default, kmaxfac=kmaxfac_default)
        
        n_kp = len(ueg_model.kpts_mesh)
        n_x = len(ueg_model.xtheta_mesh)
        dx = ueg_model.dxtheta
        
        # Compute integral
        start_time = time.time()
        integral_value = ueg_model.intNablaUSquare(k_vec)
        elapsed_time = time.time() - start_time
        
        # Store results
        results['parameter_values'].append(nx)
        results['integral_values'].append(integral_value)
        results['computation_times'].append(elapsed_time)
        results['n_kp_points'].append(n_kp)
        results['n_x_points'].append(n_x)
        results['dx_values'].append(dx)
        
        print_logging_info(f"  nx={nx}, dx={dx:.6f}, Integral: {integral_value:.12e}, Time: {elapsed_time:.3f} s", level=3)
        sys.stdout.flush()
    
    # Write results to file
    write_sweep_results_to_file(results, output_file)
    
    return results


def sweep_dkfac_parameter(ueg_model, k_vec, k_mag, dkfac_values, nx_default, dkfac_default, kmaxfac_default, output_file):
    """
    Sweep dkfac parameter while keeping nx and kmaxfac at default values.
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model with correlator set
    k_vec : numpy.ndarray
        K-vector to test (3D array)
    k_mag : float
        Magnitude of k-vector
    dkfac_values : list of int
        List of dkfac values to test
    nx_default : int
        Default nx value (held constant)
    dkfac_default : int
        Default dkfac value (not used in this sweep)
    kmaxfac_default : float
        Default kmaxfac value (held constant)
    output_file : str
        Output filename
    
    Returns
    -------
    results : dict
        Dictionary containing sweep results
    """
    print_title(f"Sweeping dkfac parameter for |k| = {k_mag:.6f}", "-")
    print_logging_info(f"Fixed parameters: nx={nx_default}, kmaxfac={kmaxfac_default}", level=1)
    
    results = {
        'k_vector': k_vec.copy(),
        'k_magnitude': k_mag,
        'parameter': 'dkfac',
        'parameter_values': [],
        'integral_values': [],
        'computation_times': [],
        'n_kp_points': [],
        'n_x_points': [],
        'fixed_nx': nx_default,
        'fixed_kmaxfac': kmaxfac_default,
    }
    
    for dkfac in dkfac_values:
        print_logging_info(f"Testing dkfac={dkfac}", level=2)
        
        # Initialize grid with current dkfac and default values for other parameters
        ueg_model.init_ConvMesh(nx=nx_default, dkfac=dkfac, kmaxfac=kmaxfac_default)
        
        n_kp = len(ueg_model.kpts_mesh)
        n_x = len(ueg_model.xtheta_mesh)
        
        # Compute integral
        start_time = time.time()
        integral_value = ueg_model.intNablaUSquare(k_vec)
        elapsed_time = time.time() - start_time
        
        # Store results
        results['parameter_values'].append(dkfac)
        results['integral_values'].append(integral_value)
        results['computation_times'].append(elapsed_time)
        results['n_kp_points'].append(n_kp)
        results['n_x_points'].append(n_x)
        
        print_logging_info(f"  Integral: {integral_value:.12e}, Time: {elapsed_time:.3f} s", level=3)
        sys.stdout.flush()
    
    # Write results to file
    write_sweep_results_to_file(results, output_file)
    
    return results


def sweep_kmaxfac_parameter(ueg_model, k_vec, k_mag, kmaxfac_values, nx_default, dkfac_default, kmaxfac_default, output_file):
    """
    Sweep kmaxfac parameter while keeping nx and dkfac at default values.
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model with correlator set
    k_vec : numpy.ndarray
        K-vector to test (3D array)
    k_mag : float
        Magnitude of k-vector
    kmaxfac_values : list of float
        List of kmaxfac values to test
    nx_default : int
        Default nx value (held constant)
    dkfac_default : int
        Default dkfac value (held constant)
    kmaxfac_default : float
        Default kmaxfac value (not used in this sweep)
    output_file : str
        Output filename
    
    Returns
    -------
    results : dict
        Dictionary containing sweep results
    """
    print_title(f"Sweeping kmaxfac parameter for |k| = {k_mag:.6f}", "-")
    print_logging_info(f"Fixed parameters: nx={nx_default}, dkfac={dkfac_default}", level=1)
    
    results = {
        'k_vector': k_vec.copy(),
        'k_magnitude': k_mag,
        'parameter': 'kmaxfac',
        'parameter_values': [],
        'integral_values': [],
        'computation_times': [],
        'n_kp_points': [],
        'n_x_points': [],
        'fixed_nx': nx_default,
        'fixed_dkfac': dkfac_default,
    }
    
    for kmaxfac in kmaxfac_values:
        print_logging_info(f"Testing kmaxfac={kmaxfac:.1f}", level=2)
        
        # Initialize grid with current kmaxfac and default values for other parameters
        ueg_model.init_ConvMesh(nx=nx_default, dkfac=dkfac_default, kmaxfac=kmaxfac)
        
        n_kp = len(ueg_model.kpts_mesh)
        n_x = len(ueg_model.xtheta_mesh)
        
        # Compute integral
        start_time = time.time()
        integral_value = ueg_model.intNablaUSquare(k_vec)
        elapsed_time = time.time() - start_time
        
        # Store results
        results['parameter_values'].append(kmaxfac)
        results['integral_values'].append(integral_value)
        results['computation_times'].append(elapsed_time)
        results['n_kp_points'].append(n_kp)
        results['n_x_points'].append(n_x)
        
        print_logging_info(f"  Integral: {integral_value:.12e}, Time: {elapsed_time:.3f} s", level=3)
        sys.stdout.flush()
    
    # Write results to file
    write_sweep_results_to_file(results, output_file)
    
    return results


def write_sweep_results_to_file(results, output_file):
    """
    Write parameter sweep results to a file.
    
    Parameters
    ----------
    results : dict
        Dictionary containing sweep results
    output_file : str
        Name of output file
    """
    print_logging_info(f"Writing results to: {output_file}", level=2)
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("# Convergence Test Results for intNablaUSquare\n")
        f.write("# TDL Convolution Integral: F{(∇u)²}(k) = ∫ d³k' (k'·(k-k')) u(k') u(|k-k'|)\n")
        f.write("#\n")
        f.write(f"# Parameter sweep: {results['parameter']}\n")
        f.write(f"# k-vector: [{results['k_vector'][0]:.8e}, {results['k_vector'][1]:.8e}, {results['k_vector'][2]:.8e}] a.u.⁻¹\n")
        f.write(f"# |k| = {results['k_magnitude']:.8e} a.u.⁻¹\n")
        f.write("#\n")
        
        # Write fixed parameters
        if results['parameter'] == 'nx':
            f.write(f"# Fixed parameters: dkfac={results['fixed_dkfac']}, kmaxfac={results['fixed_kmaxfac']}\n")
            f.write("#\n")
            f.write("# Columns:\n")
            f.write(f"#   1: {results['parameter']}\n")
            f.write("#   2: dx (computed from nx)\n")
            f.write("#   3: integral_value [a.u.]\n")
            f.write("#   4: computation_time [s]\n")
            f.write("#   5: n_kp_points (number of k' grid points)\n")
            f.write("#   6: n_x_points (number of x grid points)\n")
            f.write("#\n")
            f.write(f"{'# nx':>18s} {'dx':>20s} {'integral':>20s} {'time[s]':>12s} {'n_kp':>10s} {'n_x':>10s}\n")
        elif results['parameter'] == 'dkfac':
            f.write(f"# Fixed parameters: nx={results['fixed_nx']}, kmaxfac={results['fixed_kmaxfac']}\n")
            f.write("#\n")
            f.write("# Columns:\n")
            f.write(f"#   1: {results['parameter']}\n")
            f.write("#   2: integral_value [a.u.]\n")
            f.write("#   3: computation_time [s]\n")
            f.write("#   4: n_kp_points (number of k' grid points)\n")
            f.write("#   5: n_x_points (number of x grid points)\n")
            f.write("#\n")
            f.write(f"{'# dkfac':>18s} {'integral':>20s} {'time[s]':>12s} {'n_kp':>10s} {'n_x':>10s}\n")
        elif results['parameter'] == 'kmaxfac':
            f.write(f"# Fixed parameters: nx={results['fixed_nx']}, dkfac={results['fixed_dkfac']}\n")
            f.write("#\n")
            f.write("# Columns:\n")
            f.write(f"#   1: {results['parameter']}\n")
            f.write("#   2: integral_value [a.u.]\n")
            f.write("#   3: computation_time [s]\n")
            f.write("#   4: n_kp_points (number of k' grid points)\n")
            f.write("#   5: n_x_points (number of x grid points)\n")
            f.write("#\n")
            f.write(f"{'# kmaxfac':>18s} {'integral':>20s} {'time[s]':>12s} {'n_kp':>10s} {'n_x':>10s}\n")
        
        # Write data
        for i in range(len(results['parameter_values'])):
            if results['parameter'] == 'nx':
                f.write(f"{results['parameter_values'][i]:20d} ")
                f.write(f"{results['dx_values'][i]:20.8f} ")
                f.write(f"{results['integral_values'][i]:20.12e} "
                        f"{results['computation_times'][i]:12.3f} "
                        f"{results['n_kp_points'][i]:10d} {results['n_x_points'][i]:10d}\n")
            elif results['parameter'] == 'dkfac':
                f.write(f"{results['parameter_values'][i]:20d} ")
                f.write(f"{results['integral_values'][i]:20.12e} "
                        f"{results['computation_times'][i]:12.3f} "
                        f"{results['n_kp_points'][i]:10d} {results['n_x_points'][i]:10d}\n")
            else:  # kmaxfac
                f.write(f"{results['parameter_values'][i]:20.2f} ")
                f.write(f"{results['integral_values'][i]:20.12e} "
                        f"{results['computation_times'][i]:12.3f} "
                        f"{results['n_kp_points'][i]:10d} {results['n_x_points'][i]:10d}\n")
    
    print_logging_info(f"Results written successfully", level=3)


def test_convergence_intNablaUSquare(ueg_model, k_test_values, 
                                      nx_values, dkfac_values, kmaxfac_values,
                                      nx_default, dkfac_default, kmaxfac_default,
                                      output_prefix='k'):
    """
    Test convergence of intNablaUSquare by sweeping one parameter at a time.
    For each k-point, three files are generated with separate parameter sweeps.
    
    Parameters
    ----------
    ueg_model : UEG
        Initialized UEG model with correlator set
    k_test_values : list of arrays
        List of k-vectors to test (each is a 3D numpy array)
    nx_values : list of int
        List of nx values to test
    dkfac_values : list of int
        List of dkfac values to test
    kmaxfac_values : list of float
        List of kmaxfac values to test
    nx_default : int
        Default nx value
    dkfac_default : int
        Default dkfac value
    kmaxfac_default : float
        Default kmaxfac value
    output_prefix : str
        Prefix for output files
    
    Returns
    -------
    all_results : dict
        Dictionary containing all sweep results
    """
    print_title("Testing Convergence of intNablaUSquare", "=")
    
    print_logging_info(f"Default parameters: nx={nx_default}, dkfac={dkfac_default}, kmaxfac={kmaxfac_default}", level=0)
    print_logging_info(f"k_F = {ueg_model.kFermi:.6f} a.u.⁻¹", level=0)
    print_logging_info(f"Testing {len(k_test_values)} different k-vectors", level=0)
    print_logging_info(f"Parameter ranges:", level=0)
    print_logging_info(f"  nx: {nx_values}", level=1)
    print_logging_info(f"  dkfac: {dkfac_values}", level=1)
    print_logging_info(f"  kmaxfac: {kmaxfac_values}", level=1)
    
    total_sweeps = len(k_test_values) * 3  # 3 sweeps per k-point
    sweep_count = 0
    
    start_time_total = time.time()
    
    all_results = {}
    
    for k_idx, k_vec in enumerate(k_test_values):
        k_mag = np.sqrt(k_vec.dot(k_vec))
        print_title(f"K-vector {k_idx+1}/{len(k_test_values)}: |k| = {k_mag:.6f}", "=")
        
        # Generate output filenames for this k-point
        output_file_nx = f"{output_prefix}{k_mag:.4f}_nx.dat"
        output_file_dkfac = f"{output_prefix}{k_mag:.4f}_dkfac.dat"
        output_file_kmaxfac = f"{output_prefix}{k_mag:.4f}_kmaxfac.dat"
        
        # Sweep 1: nx
        sweep_count += 1
        print_logging_info(f"Sweep {sweep_count}/{total_sweeps}", level=0)
        results_nx = sweep_nx_parameter(
            ueg_model, k_vec, k_mag, nx_values,
            nx_default, dkfac_default, kmaxfac_default,
            output_file_nx
        )
        
        # Sweep 2: dkfac
        sweep_count += 1
        print_logging_info(f"Sweep {sweep_count}/{total_sweeps}", level=0)
        results_dkfac = sweep_dkfac_parameter(
            ueg_model, k_vec, k_mag, dkfac_values,
            nx_default, dkfac_default, kmaxfac_default,
            output_file_dkfac
        )
        
        # Sweep 3: kmaxfac
        sweep_count += 1
        print_logging_info(f"Sweep {sweep_count}/{total_sweeps}", level=0)
        results_kmaxfac = sweep_kmaxfac_parameter(
            ueg_model, k_vec, k_mag, kmaxfac_values,
            nx_default, dkfac_default, kmaxfac_default,
            output_file_kmaxfac
        )
        
        # Store results
        all_results[f'k{k_idx}_nx'] = results_nx
        all_results[f'k{k_idx}_dkfac'] = results_dkfac
        all_results[f'k{k_idx}_kmaxfac'] = results_kmaxfac
        
        print_logging_info(f"Completed all sweeps for |k| = {k_mag:.6f}", level=0)
        sys.stdout.flush()
    
    total_time = time.time() - start_time_total
    print_title("Convergence Test Completed", "=")
    print_logging_info(f"Total time: {total_time:.2f} s", level=0)
    print_logging_info(f"Average time per sweep: {total_time/total_sweeps:.3f} s", level=0)
    
    return all_results


def analyze_convergence(all_results):
    """
    Analyze convergence behavior from test results.
    
    Parameters
    ----------
    all_results : dict
        Dictionary containing all sweep results
    """
    print_title("Convergence Analysis Summary", "=")
    
    for key, results in all_results.items():
        print_title(f"Analysis: {key}", "-")
        print_logging_info(f"Parameter: {results['parameter']}", level=1)
        print_logging_info(f"|k| = {results['k_magnitude']:.6f} a.u.⁻¹", level=1)
        
        integral_vals = np.array(results['integral_values'])
        
        print_logging_info(f"Number of tests: {len(integral_vals)}", level=1)
        print_logging_info(f"Integral range: [{integral_vals.min():.8e}, {integral_vals.max():.8e}]", level=1)
        print_logging_info(f"Mean: {integral_vals.mean():.12e}", level=1)
        print_logging_info(f"Std. dev.: {integral_vals.std():.12e}", level=1)
        if np.abs(integral_vals.mean()) > 1e-15:
            print_logging_info(f"Relative std. dev.: {integral_vals.std()/np.abs(integral_vals.mean())*100:.4f}%", level=1)


def main(nel=14, rs=0.5, basis_cutoff=2, k_cutoff=1, gamma=None,
         output_prefix='k'):
    """
    Main function to set up UEG model and run convergence tests.
    
    Parameters
    ----------
    nel : int
        Number of electrons
    rs : float
        Wigner-Seitz radius
    basis_cutoff : float
        Kinetic energy cutoff for basis
    k_cutoff : float
        K-cutoff fraction for correlator
    gamma : float or None
        Gamma parameter for correlator
    output_prefix : str
        Prefix for output files
    """
    print_title("TDL Convolution Integral Convergence Test", "=")
    
    # 1. Setup UEG model
    print_title("Setting up UEG model", "-")
    nalpha = nel // 2
    nbeta = nel // 2
    ueg_model = ueg.UEG(nel, nalpha, nbeta, rs, is_tc=True)
    
    print_logging_info(f"Number of electrons: {nel}", level=1)
    print_logging_info(f"rs: {rs}", level=1)
    print_logging_info(f"Volume (Ω): {ueg_model.Omega:.6f} a.u.³", level=1)
    print_logging_info(f"Length (L): {ueg_model.L:.6f} a.u.", level=1)
    print_logging_info(f"Density (ρ): {ueg_model.n_ele / ueg_model.Omega:.6f} a.u.⁻³", level=1)
    print_logging_info(f"Fermi wave vector (k_F): {ueg_model.kFermi:.6f} a.u.⁻¹", level=1)
    
    # Initialize basis set
    print_title("Initializing basis set", "-")
    ueg_model.init_single_basis(basis_cutoff)
    nP = len(ueg_model.basis_fns) // 2
    print_logging_info(f"Number of spatial orbitals: {nP}", level=1)
    
    # Setup correlator
    print_title("Setting up correlator", "-")
    ueg_model.correlator = ueg_model.RPA
    ueg_model.k_cutoff = k_cutoff
    ueg_model.gamma = gamma if gamma is not None else 1.0
    
    print_logging_info(f"Correlator: {ueg_model.correlator.__name__}", level=1)
    print_logging_info(f"k_cutoff: {ueg_model.k_cutoff}", level=1)
    print_logging_info(f"gamma: {ueg_model.gamma}", level=1)
    
    # 2. Define test k-vectors
    print_title("Defining test k-vectors", "-")
    
    # Test various k-vectors (in units of k_F)
    k_test_relative = [
        np.array([0.0, 0.0, 0.0]),   # k = 0 Gamma point
        np.array([0.1, 0.0, 0.0]),   # Very small k along x
        np.array([0.5, 0.0, 0.0]),   # Small k along x
        np.array([1.0, 0.0, 0.0]),   # k ~ k_F along x
        np.array([1.5, 0.0, 0.0]),   # k > k_F along x
        np.array([0.1, 0.1, 0.0]),   # Very small k in xy-plane
        np.array([0.5, 0.5, 0.0]),   # Small k in xy-plane
        np.array([1.0, 1.0, 0.0]),   # Larger k in xy-plane
    ]
    
    # Convert to absolute units
    k_test_values = [k * ueg_model.kFermi for k in k_test_relative]
    
    for i, k in enumerate(k_test_values):
        k_mag = np.sqrt(k.dot(k))
        print_logging_info(f"k_{i+1} = [{k[0]:.4f}, {k[1]:.4f}, {k[2]:.4f}], |k| = {k_mag:.4f} a.u.⁻¹", level=1)
    
    # 3. Define default values and sweep ranges
    print_title("Defining grid parameters", "-")
    
    # Default values (used when parameter is not being swept)
    nx_default = 200
    dkfac_default = 60
    kmaxfac_default = 20.0
    
    print_logging_info(f"Default values:", level=1)
    print_logging_info(f"  nx_default = {nx_default}", level=2)
    print_logging_info(f"  dkfac_default = {dkfac_default}", level=2)
    print_logging_info(f"  kmaxfac_default = {kmaxfac_default}", level=2)
    
    # Parameter ranges for sweeps
    nx_values = [10, 20, 50, 100, 200, 400, 800, 1000, 2000, 5000, 10000]
    dkfac_values = [10, 20, 40, 80, 160, 320]
    kmaxfac_values = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0]
    
    print_logging_info(f"Sweep ranges:", level=1)
    print_logging_info(f"  nx values: {nx_values}", level=2)
    print_logging_info(f"  dkfac values: {dkfac_values}", level=2)
    print_logging_info(f"  kmaxfac values: {kmaxfac_values}", level=2)
    
    # 4. Run convergence tests
    all_results = test_convergence_intNablaUSquare(
        ueg_model, k_test_values,
        nx_values, dkfac_values, kmaxfac_values,
        nx_default, dkfac_default, kmaxfac_default,
        output_prefix=output_prefix
    )
    
    # 5. Analyze results
    analyze_convergence(all_results)
    
    print_title("Test Completed Successfully!", "=")
    print_logging_info(f"Output files: {output_prefix}{{magnitude}}_{{parameter}}.dat", level=0)


if __name__ == '__main__':
    # Run with default parameters
    main(
        nel=14,
        rs=0.5,
        basis_cutoff=2,
        k_cutoff=1e-12,
        gamma=None,
        output_prefix='k'
    )
