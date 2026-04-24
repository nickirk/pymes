import numpy as np
from numba import jit, prange, get_num_threads, config

"""
Helper functions for UEG model system JIT-compiled with Numba for better performance.
    This module includes functions to compute two-body integrals in a transcorrelated framework
    in a JIT-compiled manner using Numba for enhanced performance.
Main functions:
    _get_orbital_energies: Numba JIT-compiled calculation of orbital energies and HF components.
    _get_2b_int: Numba JIT-compiled version of the get_2b_int().
    _get_2b_int_kernel: kernel function for computing two-body integrals.
    _triple_contractions_in_3_body: Numba JIT-compiled version of triple_contractions_in_3_body().
    _double_contractions_in_3_body: Numba JIT-compiled version of double_contractions_in_3_body().
Auxiliary functions:
    _sumNablaUSquare: Numba JIT-compiled version of the sumNablaUSquare().
    _contract_exchange_3_body: Numba JIT-compiled version of the contract_exchange_3_body().
    _contractP_KWithQ: Numba JIT-compiled version of the contractP_KWithQ().
Correlators:
    _calc_correlator: wrapper function to select the correlator type.
"""

# TRIPLY AND DOUBLY CONTRACTED 3-BODY INTEGRALS -------------------------

@jit(nopython=True, parallel=True)
def _triple_contractions_in_3_body(basis_occ_Kp, n_ele, Omega, rho, k_cutoffSquare, gamma, correlator_idx):
    """Numba JIT-compiled version of UEG.triple_contractions_in_3_body().

    Computes the triply contracted 3-body interactions (a scalar energy contribution).

    Parameters
    ----------
    basis_occ_Kp : ndarray, shape (n_occ, 3), float64
        k-vectors of occupied orbitals (spin-up only, i.e. n_ele/2 entries).
    n_ele : int
        Total number of electrons.
    Omega : float
        Volume of the simulation cell.
    rho : float
        Electron density.
    k_cutoffSquare : float
        Square of the k-space cutoff for the correlator.
    gamma : float
        Correlator amplitude parameter.
    correlator_idx : int
        Integer identifier selecting the correlator type (see _calc_correlator).

    Returns
    -------
    result : float
        Triply contracted 3-body energy contribution.
    """
    no = basis_occ_Kp.shape[0]

    # Build tables: p_qSquare[p,q] = |k_occ[p] - k_occ[q]|^2
    #               up_q_pq[p,q]   = correlator(p_qSquare[p,q])
    p_qSquare = np.zeros((no, no))
    up_q_pq   = np.zeros((no, no))
    for p in prange(no):
        for q in range(no):
            sq = 0.0
            for kk in range(3):
                d = basis_occ_Kp[p, kk] - basis_occ_Kp[q, kk]
                sq += d * d
            p_qSquare[p, q] = sq
            up_q_pq[p, q]   = _calc_correlator(correlator_idx, sq, k_cutoffSquare, rho, gamma)

    # Direct diagram: factor 2 from spin sum
    # dirE = sum_{p,q} u(p-q)^2 * |p-q|^2  *  (n_ele/2) / Omega^2 * 2
    dirE = np.sum(up_q_pq * up_q_pq * p_qSquare) * n_ele / 2.0 / Omega**2 * 2.0

    # Exchange diagram: sum_{p,q,o} [(p-q)·(p-o)] * u(p-q) * u(p-o)
    # Original: -2*2 * einsum("pqo,pqo->", p_o_dot_p_q, u_pq_u_po) / 2 / Omega^2
    # Accumulate per-p slice so each prange iteration is independent.
    excE_arr = np.zeros(no)
    for p in prange(no):
        for q in range(no):
            for o in range(no):
                dot_qo = 0.0
                for kk in range(3):
                    dot_qo += (basis_occ_Kp[p, kk] - basis_occ_Kp[q, kk]) * (basis_occ_Kp[p, kk] - basis_occ_Kp[o, kk])
                excE_arr[p] += dot_qo * up_q_pq[p, q] * up_q_pq[p, o]
    excE = -4.0 * np.sum(excE_arr) / 2.0 / Omega**2

    return dirE + excE


@jit(nopython=True, parallel=True)
def _double_contractions_in_3_body(basis_occ_Kp, basis_Kp, n_ele, Omega, rho, k_cutoffSquare, gamma, correlator_idx):
    """Numba JIT-compiled version of UEG.double_contractions_in_3_body().

    Computes the doubly contracted 3-body integrals, yielding one-body energy corrections
    for all orbitals (occupied + virtual).  Four diagram types are computed: perl, wave,
    shield, and frog.

    Parameters
    ----------
    basis_occ_Kp : ndarray, shape (n_occ, 3), float64
        k-vectors of occupied orbitals (spin-up only, i.e. n_ele/2 entries).
    basis_Kp : ndarray, shape (n_p, 3), float64
        k-vectors of all orbitals (occupied + virtual, spin-up only).
    n_ele : int
        Total number of electrons.
    Omega : float
        Volume of the simulation cell.
    rho : float
        Electron density.
    k_cutoffSquare : float
        Square of the k-space cutoff for the correlator.
    gamma : float
        Correlator amplitude parameter.
    correlator_idx : int
        Integer identifier selecting the correlator type (see _calc_correlator).

    Returns
    -------
    one_particle_energies : ndarray, shape (n_p,), float64
        One-body energy corrections from doubly contracted 3-body integrals.
    """
    no = basis_occ_Kp.shape[0]
    nP   = basis_Kp.shape[0]

    # --- Pre-compute (p,i) tables: differences between all and occupied k-vectors ---
    # u_diff_pi[p,i]  = correlator(|k_all[p] - k_occ[i]|^2)
    # sq_diff_pi[p,i] = |k_all[p] - k_occ[i]|^2
    u_diff_pi  = np.zeros((nP, no))
    sq_diff_pi = np.zeros((nP, no))
    for p in prange(nP):
        for i in range(no):
            sq = 0.0
            for kk in range(3):
                d = basis_Kp[p, kk] - basis_occ_Kp[i, kk]
                sq += d * d
            sq_diff_pi[p, i] = sq
            u_diff_pi[p, i]  = _calc_correlator(correlator_idx, sq, k_cutoffSquare, rho, gamma)

    # --- Perl diagram ---
    # e_perl[p] = sum_i u(|k_all[p]-k_occ[i]|^2)^2 * |k_all[p]-k_occ[i]|^2
    e_perl = np.zeros(nP)
    for p in prange(nP):
        for i in range(no):
            e_perl[p] += u_diff_pi[p, i] * u_diff_pi[p, i] * sq_diff_pi[p, i]
    e_perl = 2.0 * n_ele / Omega**2 / 2.0 * e_perl

    # --- Wave diagram ---
    # e_wave[p] = sum_{i,j} [(k_all[p]-k_occ[i])·(k_all[p]-k_occ[j])] * u_diff_pi[p,i] * u_diff_pi[p,j]
    e_wave = np.zeros(nP)
    for p in prange(nP):
        for i in range(no):
            for j in range(no):
                dot_ij = 0.0
                for kk in range(3):
                    dot_ij += (basis_Kp[p, kk] - basis_occ_Kp[i, kk]) * (basis_Kp[p, kk] - basis_occ_Kp[j, kk])
                e_wave[p] += dot_ij * u_diff_pi[p, i] * u_diff_pi[p, j]
    e_wave = -e_wave * 2.0 / Omega**2 / 2.0

    # --- Pre-compute (i,j) tables: differences within occupied k-vectors ---
    # u_diff_ij[i,j]  = correlator(|k_occ[i] - k_occ[j]|^2)
    # sq_diff_ij[i,j] = |k_occ[i] - k_occ[j]|^2
    u_diff_ij  = np.zeros((no, no))
    sq_diff_ij = np.zeros((no, no))
    for i in prange(no):
        for j in range(no):
            sq = 0.0
            for kk in range(3):
                d = basis_occ_Kp[i, kk] - basis_occ_Kp[j, kk]
                sq += d * d
            sq_diff_ij[i, j] = sq
            u_diff_ij[i, j]  = _calc_correlator(correlator_idx, sq, k_cutoffSquare, rho, gamma)

    # --- Shield diagram (independent of p) ---
    # shield_val = sum_{i,j} u_diff_ij[i,j]^2 * sq_diff_ij[i,j]
    shield_val = 0.0
    for i in range(no):
        for j in range(no):
            shield_val += u_diff_ij[i, j] * u_diff_ij[i, j] * sq_diff_ij[i, j]
    # factor: 2 (spin) / 2 (symmetry) / Omega^2 — same net factor as in the original
    e_shield = np.ones(nP) * (2.0 * shield_val / 2.0 / Omega**2)

    # --- Frog diagram ---
    # e_frog[p] = sum_{i,j} [(k_occ[i]-k_occ[j])·(k_occ[i]-k_all[p])] * u_diff_ij[i,j] * u_diff_pi[p,i]
    # (the -(diff_vec_pi) used in the original is equivalent to k_occ[i]-k_all[p])
    e_frog = np.zeros(nP)
    for p in prange(nP):
        for i in range(no):
            for j in range(no):
                dot_ijip = 0.0
                for kk in range(3):
                    dot_ijip += (basis_occ_Kp[i, kk] - basis_occ_Kp[j, kk]) * (basis_occ_Kp[i, kk] - basis_Kp[p, kk])
                e_frog[p] += dot_ijip * u_diff_ij[i, j] * u_diff_pi[p, i]
    e_frog = -e_frog * 4.0 / Omega**2 / 2.0

    return e_perl + e_wave + e_shield + e_frog

# ORBITAL ENERGIES ---------------------------------------------------------------------

@jit(nopython=True, parallel=True)
def _get_orbital_energies(kinetic_G, n_ele, Omega, L, rho,
                          imax, k_cutoff, gamma,
                          UMAT, basis_indices_map,
                          basis_occ_Kp, basis_Kvec, basis_Kp,
                          is_only_2b, is_tc, correlator_idx,
                          dtype=np.float64):
    """
    Numba JIT-compiled function to compute the orbital energies for the UEG model system.

    NOTE: This function computes the orbital energies including contributions from kinetic energy,
    pure 2-body integrals from transcorrelation involving contractions with the Coulomb tensor.
    This means that the double contractions from the 3-body integrals are not computed here,
    and must be added a posteriori. Adding the single contractions from the 3-body integrals 
    to the Coulomb integrals and contracting them with the density matrix will yield a double 
    counting of an already normal-ordered operator.

    Parameters
    ----------
    kinetic_G: nparray of float dtype
        kinetic energy grid.
    n_ele: int
        number of electrons.
    Omega: float
        volume of the cubic simulation cell.
    L: float
        length of the cubic simulation cell.
    rho: float
        electron density.
    imax: int
        maximum k-point index in each direction.
    k_cutoff: float
        plane wave vector cutoff inside the correlaor function trunc.
    gamma: float
        parameter in the correlator function.
    UMAT: nparray of float dtype
        pre-computed U matrix for TC (canonical/long-range) integrals.
    basis_indices_map: nparray of int dtype
        an array to store indices of basis functions (plane waves) for
        later lookup. Size Nx*Ny*Nz, Nx, Ny, Nz are the k-vector points
        in x, y, z directions.
    basis_occ_Kp: nparray of float dtype
        an array to store (shifted) k-vectors of occupied orbitals.
    basis_Kvec: nparray of int dtype
        an array to store k-vector indices (quantum numbers) of all basis functions.
    basis_Kp: nparray of float dtype
        an array to store (shifted) k-vectors of all basis functions.
    is_only_2b: bool
        parameter which determines to include only the additional
        pure 2-body tc integrals, besides the Coulomb integrals.
    is_tc: bool
        parameter which determines whether transcorrelated framework is
        active or not for the calculation of the integrals.
    correlator_idx: int
        identifier for the correlator type.

    Returns
    -------
    EHF: float
        Hartree Fock energy computed from the orbital energies and the direct and exchange contributions.
    epsilon_i: nparray of float dtype
        orbital energies for the occupied orbitals.
    epsilon_a: nparray of float dtype
        orbital energies for the virtual orbitals.
    """
    no = int(n_ele // 2)
    nP = int(basis_Kp.shape[0])
    nv = nP - no

    k_cutoffSquare = (2 * np.pi * k_cutoff / L)**2
    idx_shift = 2*imax

    epsilon_i = np.zeros(no, dtype=dtype)
    epsilon_a = np.zeros(nv, dtype=dtype)

    HF_dirE = 0.0
    HF_exE = 0.0

    for p in prange(nP):
        # Start with kinetic energy.
        e_p = kinetic_G[p]
        e_dir = 0.0
        e_exc = 0.0
        for i in range(no):
            # Direct term V_pipi (only non-zero for transcorrelated integrals).
            d_int_k = basis_Kvec[p] - basis_Kvec[p]
            d_k_vec = basis_Kp[p] - basis_Kp[p]
            dk_square = d_k_vec[0]**2 + d_k_vec[1]**2 + d_k_vec[2]**2
            u_mat = 0.
            if is_tc:
                idx_shift = 2 * imax
                ix = d_int_k[0] + idx_shift
                iy = d_int_k[1] + idx_shift
                iz = d_int_k[2] + idx_shift
                u_mat = UMAT[ix, iy, iz]
            V_pipi = _get_2b_int_kernel(p, i, p, i,
                                        dk_square, d_k_vec, u_mat,
                                        n_ele, Omega, rho,
                                        k_cutoffSquare, gamma,
                                        basis_occ_Kp, basis_Kp,
                                        is_only_2b, False, 
                                        is_tc, correlator_idx)
            e_dir += V_pipi
            # Exchange term V_ippi.
            d_int_k = basis_Kvec[i] - basis_Kvec[p]
            d_k_vec = basis_Kp[i] - basis_Kp[p]
            dk_square = d_k_vec[0]**2 + d_k_vec[1]**2 + d_k_vec[2]**2
            u_mat = 0.
            if is_tc:
                idx_shift = 2 * imax
                ix = d_int_k[0] + idx_shift
                iy = d_int_k[1] + idx_shift
                iz = d_int_k[2] + idx_shift
                u_mat = UMAT[ix, iy, iz]
            V_piip = _get_2b_int_kernel(p, i, i, p,
                                        dk_square, d_k_vec, u_mat,
                                        n_ele, Omega, rho,
                                        k_cutoffSquare, gamma,
                                        basis_occ_Kp, basis_Kp,
                                        is_only_2b, False, 
                                        is_tc, correlator_idx)
            e_exc += V_piip
        e_p += 2 * e_dir - e_exc
        if p < no:
            HF_dirE += 2. * e_dir
            HF_exE += -1. * e_exc
            epsilon_i[p] = e_p
        else:
            epsilon_a[p-no] = e_p

    EHF = 2. * np.sum(epsilon_i) - HF_dirE - HF_exE

    return EHF, epsilon_i, epsilon_a

# PURE AND EFFECTIVETWO-BODY INTEGRALS -------------------------------------------------

@jit(nopython=True, parallel=True)
def _get_2b_int( idx, n_ele, Omega, L, rho, 
                    imax, k_cutoff, gamma,
                    UMAT, basis_indices_map,
                    basis_occ_Kp, basis_Kvec, basis_Kp,
                    is_only_2b, is_effect_2b, is_tc, correlator_idx,
                    dtype=np.float64):
    """ Numba JIT-compiled version of the get_2b_int function for better performance.
    Parameters
    ----------
    idx: tuple of int
        indices of the block of the Coulomb tensor to be computed.
    n_ele: int
        number of electrons
    Omega: float
        volume of the cubic simulation cell
    L: float
        length of the cubic simulation cell
    rho: float
        electron density
    imax: int
        maximum k-point index in each direction
    k_cutoff: float
        plane wave vector cutoff inside the correlaor function trunc.
    gamma: float
        parameter in the correlator function.
    UMAT: nparray of float dtype
        pre-computed U matrix for TC (canonical/long-range) integrals.
    basis_indices_map: nparray of int dtype
        an array to store indices of basis functions (plane waves) for
        later lookup. Size Nx*Ny*Nz, Nx, Ny, Nz are the k-vector points
        in x, y, z directions.
    basis_occ_Kp: nparray of float dtype
        an array to store (shifted) k-vectors of occupied orbitals.
    basis_Kvec: nparray of int dtype
        an array to store k-vector indices (quantum numbers) of all basis functions.
    basis_Kp: nparray of float dtype
        an array to store (shifted) k-vectors of all basis functions.
    is_only_2b: bool
        parameter which determines to include only the additional
        pure 2-body tc integrals, besides the Coulomb integrals.
    is_effect_2b: bool
        parameter which determines to include the effective 2-body integrals 
        as a result of single contractions from the 3-body integrals. 
        There are four types of single contractions in the 3-body integrals: 
        RPA type and 3 exchange types.
    is_tc: bool
        parameter which determines whether transcorrelated framework is
        active or not for the calculation of the integrals.
    correlator_idx: int
        identifier for the correlator type.

    Returns
    -------
    V_pqrs: tensor object (tensor by default)
        of size [ idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7] ], np array.
    """
    num_k_in_each_dir = imax * 2 + 1
    V_pqrs = np.zeros((idx[1]-idx[0], idx[3]-idx[2], idx[5]-idx[4], idx[7]-idx[6]), dtype=dtype)
    k_cutoffSquare = (2 * np.pi * k_cutoff / L)**2
    idx_shift = 2*imax

    #p_range = idx[1] - idx[0]
    #r_range = idx[5] - idx[4]

    for r in prange(idx[4], idx[5]):
        loc_r_idx = r - idx[4]
    #for p in prange(idx[0], idx[1]):
    #    loc_p_idx = p - idx[0]
    #    for r in range(idx[4], idx[5]):
        for p in range(idx[0], idx[1]):
            loc_p_idx = p - idx[0]
    #for pr in prange(p_range * r_range):
    #        p = idx[0] + pr // r_range
    #        r = idx[4] + pr % r_range
    #        loc_p_idx = p - idx[0]
            #loc_r_idx = r - idx[4]
            d_int_k = basis_Kvec[r] - basis_Kvec[p]
            d_k_vec = basis_Kp[r] - basis_Kp[p]
            dk_square = d_k_vec[0]**2 + d_k_vec[1]**2 + d_k_vec[2]**2
            u_mat = 0.
            if is_tc:
                ix = d_int_k[0] + idx_shift
                iy = d_int_k[1] + idx_shift
                iz = d_int_k[2] + idx_shift
                u_mat = UMAT[ix, iy, iz]
                #if abs(dk_square) < 1.e-24: 
                #    u_mat = Fk0
                #else:
                #    u_mat = _intNablaUSquare(d_k_vec, kpts_mesh, xtheta_mesh, dkpts, dxtheta, \
                #                            rho, k_cutoffSquare, gamma, correlator_idx)
                #    #u_mat = _sumNablaUSquare(d_k_vec, rho, Omega, kPrime, k_cutoffSquare, gamma, correlator_idx)
            for q in range(idx[2], idx[3]):
                loc_q_idx = q - idx[2]
                int_ks = basis_Kvec[q] - d_int_k
                # [s] index to basis_indices_map.
                loc_s = num_k_in_each_dir ** 2 * (int_ks[0] + imax) + \
                        num_k_in_each_dir * (int_ks[1] + imax) + \
                        int_ks[2] + imax
                # check if ks-vector is in the basis set.
                if len(basis_indices_map) > loc_s >= 0:
                    # check if s index of ks-vector is in the range of
                    # the block of the Coulomb tensor to be computed.
                    s = int(basis_indices_map[loc_s])
                    if s < idx[6] or s >= idx[7]:
                        continue
                else:
                    continue
                loc_s_idx = s - idx[6]
                #dk_square = d_k_vec[0]**2 + d_k_vec[1]**2 + d_k_vec[2]**2
                
                V_pqrs[loc_p_idx,
                        loc_q_idx,
                        loc_r_idx,
                        loc_s_idx] = _get_2b_int_kernel(p, q, r, s,
                                                        dk_square, d_k_vec, u_mat,
                                                        n_ele, Omega, rho,
                                                        k_cutoffSquare, gamma,
                                                        basis_occ_Kp, basis_Kp,
                                                        is_only_2b, is_effect_2b, 
                                                        is_tc, correlator_idx)
    return V_pqrs   

@jit(nopython=True)
def _get_2b_int_kernel(p,q,r,s, 
                        dk_square, d_k_vec, u_mat,
                        n_ele, Omega, rho,
                        k_cutoffSquare, gamma,
                        basis_occ_Kp, basis_Kp,
                        is_only_2b, is_effect_2b, 
                        is_tc, correlator_idx):
    """Kernel function to compute the two-body integral for given indices p,q,r,s ;
    their corresponding k-vector differences and the pre-computed UMAT matrix 
    value for the TC contribution.
    
    NOTE: This function must me called with indices (p,q,r,s) of V_{pq}^{rs} of an allowed
            non-zero matrix element, i.e. with k_p - k_r = k_q - k_s, so that the corresponding 
            k-vector differences and UMAT value are correctly computed and passed in. This
            check must be performed in the calling function (e.g. _get_2b_int) to avoid 
            redundant calculations of the k-vector differences and UMAT values for zero matrix elements.
            No checks on this are performed inside this function.

    """
    w = 0.0
    if is_tc:
        if is_only_2b:
            # Pure 2-body TC integrals.
            if np.abs(dk_square) > 0.:
                rs_dk = basis_Kp[r] - basis_Kp[s]
                rs_dk_dot_d_k_vec = rs_dk[0]*d_k_vec[0] + rs_dk[1]*d_k_vec[1] + rs_dk[2]*d_k_vec[2]
                corr_dk_square = _calc_correlator(correlator_idx, dk_square, k_cutoffSquare, rho, gamma)
                w = 4. * np.pi / dk_square \
                    + u_mat \
                    + (dk_square - rs_dk_dot_d_k_vec) \
                    * corr_dk_square
                w = w / Omega
            else:
                w = u_mat / Omega
        elif is_effect_2b:
            # Effective 2-body integrals from single contractions of 3-body TC integrals.
            if np.abs(dk_square) > 0.:
                corr_dk_square = _calc_correlator(correlator_idx, dk_square, k_cutoffSquare, rho, gamma)
                w_pqrs = -(n_ele) * dk_square \
                        * corr_dk_square**2 / Omega \
                        + 2. * _contract_exchange_3_body( basis_Kp[r], d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx) \
                        - 2. * _contract_exchange_3_body( basis_Kp[p], d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx) \
                        + 2. * _contractP_KWithQ( basis_Kp[r], d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx)
                w_qpsr = -(n_ele) * dk_square \
                        * corr_dk_square**2 / Omega \
                        + 2. * _contract_exchange_3_body( basis_Kp[s], -d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx) \
                        - 2. * _contract_exchange_3_body( basis_Kp[q], -d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx) \
                        + 2. * _contractP_KWithQ( basis_Kp[s], -d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx)
                w = 0.5 * (w_pqrs + w_qpsr)
            else:
                w = u_mat
                w_pqrs = 2. * _contractP_KWithQ( basis_Kp[r],  d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx)
                w_qpsr = 2. * _contractP_KWithQ( basis_Kp[s], -d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx)
                w = 0.5 * (w_pqrs + w_qpsr)
            w = w / Omega
        else:
            # Full 2-body integrals including both Coulomb and TC contributions.
            if np.abs(dk_square) > 0.:
                corr_dk_square = _calc_correlator(correlator_idx, dk_square, k_cutoffSquare, rho, gamma)
                rs_dk = basis_Kp[r] - basis_Kp[s]
                rs_dk_dot_d_k_vec = rs_dk[0]*d_k_vec[0] + rs_dk[1]*d_k_vec[1] + rs_dk[2]*d_k_vec[2]
                w = 4. * np.pi / dk_square
                w += + u_mat \
                    + (dk_square - rs_dk_dot_d_k_vec) \
                    * corr_dk_square
                w_pqrs = -(n_ele) * dk_square \
                        * corr_dk_square**2 / Omega \
                        + 2. * _contract_exchange_3_body( basis_Kp[r], d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx) \
                        - 2. * _contract_exchange_3_body( basis_Kp[p], d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx) \
                        + 2. * _contractP_KWithQ( basis_Kp[r], d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx)
                w_qpsr = -(n_ele) * dk_square \
                        * corr_dk_square**2 / Omega \
                        + 2. * _contract_exchange_3_body( basis_Kp[s], -d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx) \
                        - 2. * _contract_exchange_3_body( basis_Kp[q], -d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx) \
                        + 2. * _contractP_KWithQ( basis_Kp[s], -d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx)
                w += 0.5 * (w_pqrs + w_qpsr)
            else:
                w = u_mat
                w_pqrs = 2. * _contractP_KWithQ( basis_Kp[r],  d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx)
                w_qpsr = 2. * _contractP_KWithQ( basis_Kp[s], -d_k_vec, basis_occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx)
                w += 0.5 * (w_pqrs + w_qpsr)
            w = w / Omega
    else:
        # Coulomb integrals only.
        if np.abs(dk_square) > 0.:
            w = 4. * np.pi / dk_square / Omega

    return w

@jit(nopython=True)
def _contract_exchange_3_body(pVec, kVec, occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx):
    """ Numba JIT-compiled version of the contract_exchange_3_body function for better performance.
    Computes the single contraction of the exchange type from the 3-body integrals.
    Parameters
    ----------
    pVec: nparray of float dtype
        k-vector of basis function p.
    kVec: nparray of float dtype
        difference of two k-vectors (k_p - k_q).
    occ_Kp: nparray of float dtype
        an array to store shifted k-vectors of occupied orbitals.
    rho: float
        electron density.
    Omega: float
        volume of the cubic simulation cell.
    k_cutoffSquare: float
        plane wave vector cutoff inside the correlaor function trunc.
    gamma: float
        parameter in the correlator function.
    correlator_idx: int
        identifier for the correlator type.
    Returns
    -------
    w: float
        value of the single contraction of the exchange type from the 3-body integrals.
    """
    kVecSquare = kVec[0]**2 + kVec[1]**2 + kVec[2]**2
    corr_kVec = _calc_correlator(correlator_idx, kVecSquare, k_cutoffSquare, rho, gamma)
    w = 0.0
    for i in range(occ_Kp.shape[0]):
        pDiff = pVec - occ_Kp[i]
        pDiffSquare = pDiff[0]**2 + pDiff[1]**2 + pDiff[2]**2
        pDiffDotkVec = pDiff[0]*kVec[0] + pDiff[1]*kVec[1] + pDiff[2]*kVec[2]

        corr_pDiff = _calc_correlator(correlator_idx, pDiffSquare, k_cutoffSquare, rho, gamma)

        w += pDiffDotkVec * corr_pDiff * corr_kVec
    
    return w / Omega

@jit(nopython=True)
def _contractP_KWithQ(pVec, kVec, occ_Kp, rho, Omega, k_cutoffSquare, gamma, correlator_idx):
    """ Numba JIT-compiled version of the contractP_KWithQ function for better performance.
    Parameters
    ----------
    pVec: nparray of float dtype
        k-vector of basis function p.
    kVec: nparray of float dtype
        difference of two k-vectors (k_p - k_q).
    occ_Kp: nparray of float dtype
        an array to store shifted k-vectors of occupied orbitals.
    rho: float
        electron density.
    Omega: float
        volume of the cubic simulation cell.
    k_cutoffSquare: float
        plane wave vector cutoff inside the correlaor function trunc.
    gamma: float
        parameter in the correlator function.
    correlator_idx: int
        identifier for the correlator type.
    Returns
    -------
    w: float
        value of the contraction over index 's' in the 3-body integrals.
    """
    w = 0.0
    for i in range(occ_Kp.shape[0]):
        vec1 = pVec - kVec - occ_Kp[i]
        vec2 = pVec - occ_Kp[i]
        vec1Square = vec1[0]**2 + vec1[1]**2 + vec1[2]**2
        vec2Square = vec2[0]**2 + vec2[1]**2 + vec2[2]**2
        vec1Dotvec2 = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]

        corr_vec1 = _calc_correlator(correlator_idx, vec1Square, k_cutoffSquare, rho, gamma)
        corr_vec2 = _calc_correlator(correlator_idx, vec2Square, k_cutoffSquare, rho, gamma)

        w += vec1Dotvec2 * corr_vec1 * corr_vec2

    return w / Omega

# UMATRIX FOR PURE AND EFFECTIVE TWO-BODY TC INTEGRALS -------------------------------------------------

@jit(nopython=True, parallel=True)
def _init_UMAT_TC(Omega, L, rho, imax, 
                    k_cutoff, gamma,
                    kPrime, correlator_idx,
                    dtype=np.float64):
    dim = 4 * imax + 1
    UMAT = np.zeros((dim, dim, dim), dtype=dtype)
    k_cutoffSquare = (2 * np.pi * k_cutoff / L)**2
    idx_shift = 2 * imax
    for i in prange(-2*imax, 2*imax+1):
        for j in range(-2*imax, 2*imax+1):
            for k in range(-2*imax, 2*imax+1):
                kVec = np.array([i, j, k], dtype=dtype) * (2 * np.pi / L)
                F = _sumNablaUSquare(kVec, rho, Omega, kPrime, k_cutoffSquare, gamma, correlator_idx)
                UMAT[i+idx_shift,
                     j+idx_shift,
                     k+idx_shift] = F
    return UMAT

@jit(nopython=True, parallel=True)
def _init_UMAT_lr_TC(L, rho, imax, 
                        k_cutoff, gamma,
                        kpts_mesh, xtheta_mesh,
                        dkpts, dxtheta,
                        correlator_idx,
                        dtype=np.float64):
    dim = 4 * imax + 1
    UMAT = np.zeros((dim, dim, dim), dtype=dtype)
    k_cutoffSquare = (2 * np.pi * k_cutoff / L)**2
    idx_shift = 2 * imax
    for i in prange(-2*imax, 2*imax+1):
        for j in range(-2*imax, 2*imax+1):
            for k in range(-2*imax, 2*imax+1):
                kVec = np.array([i, j, k], dtype=dtype) * (2 * np.pi / L)
                F = _intNablaUSquare(kVec, kpts_mesh, xtheta_mesh, dkpts, dxtheta, \
                                        rho, k_cutoffSquare, gamma, correlator_idx)
                UMAT[i+idx_shift,
                     j+idx_shift,
                     k+idx_shift] = F
    return UMAT


@jit(nopython=True)
def _sumNablaUSquare(kVec, rho, Omega, kPrime, k_cutoffSquare, gamma, correlator_idx):
    """ Numba JIT-compiled version of the sumNablaUSquare function for better performance.
    Computes: sum_k' (k1 · k2) * u(k1^2) * u(k2^2) / Omega
    Parameters
    ----------
    kVec: nparray of float dtype
        difference of two k-vectors (k_p - k_q).
    rho: float
        electron density.
    Omega: float
        volume of the cubic simulation cell.
    kPrime: nparray of float dtype
        an array to store a denser k'-point grid for integration.
    k_cutoffSquare: float
        plane wave vector cutoff inside the correlaor function trunc.
    gamma: float
        parameter in the correlator function.
    correlator_idx: int
        identifier for the correlator type.
    Returns
    -------
    u_mat: float
        value of the sum of the squared gradients of the correlator function
        in k-space.
    """
    # kPrime is already scaled by 2π/L.
    #k1 = 2 * np.pi * kPrime / L
    k1 = kPrime
    k2 = kVec - k1
    umat = 0.0
    for i in range(k1.shape[0]):
        k2 = kVec - k1[i]
        k1Square = k1[i,0]**2 + k1[i,1]**2 + k1[i,2]**2
        k2Square = k2[0]**2 + k2[1]**2 + k2[2]**2
        k1Dotk2 = k1[i,0]*k2[0] + k1[i,1]*k2[1] + k1[i,2]*k2[2]
        corr_k1 = _calc_correlator(correlator_idx, k1Square, k_cutoffSquare, rho, gamma)
        corr_k2 = _calc_correlator(correlator_idx, k2Square, k_cutoffSquare, rho, gamma)
        umat += k1Dotk2 * corr_k1 * corr_k2
    return umat / Omega

@jit(nopython=True)
def _intNablaUSquare(kVec, kpts_mesh, xtheta_mesh, dkpts, dxtheta, \
                        rho, k_cutoffSquare, gamma, correlator_idx,
                        w=4):
    
    """
    Numba JIT-compiled version of the intNablaUSquare function for better performance.
    Computes the convolution integral of the squared gradient of the correlator function in k-space,
    in the TDL:
    - F{(∇u)²}(k) = ∫ d³k' (k'·(k-k')) u(k') u(|k-k'|)
    
    See intNablaUSquare() in ueg.py for more details.
    
    Parameters
    ----------
    kVec : np.ndarray
        Momentum transfer vector k (3D array), must have |k| > 0
    kpts_mesh : np.ndarray
        k' grid points (1D array)
    xtheta_mesh : np.ndarray
        x = cos(θ) grid points (1D array)
    dkpts : float
        Spacing in k' grid
    dxtheta : float
        Spacing in x grid
    rho : float
        Electron density
    k_cutoffSquare : float
        Square of cutoff wave vector
    gamma : float
        Correlator parameter
    correlator_idx : int
        Correlator type identifier
    w : int
        Weight exponent for regularization (default=4)
    
    Returns
    -------
    result : float
        Value of the convolution integral
    """
    kSquare = kVec[0]**2 + kVec[1]**2 + kVec[2]**2
    k = np.sqrt(kSquare)
    # Treat k = 0 case separately.
    if abs(k) < 1.e-12:
        # F{(∇u)²}(k=0) of finer k'-mesh.
        rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
        if (abs(rs - 0.5) < 1.e-3):
            umat = -0.3753885175131227 # NOTE: VALUE FOR rs=0.5 AND RPA CORRELATOR [Mathematica].
        elif (abs(rs - 2.0) < 1.e-3):
            umat = -10.82451989533362 # NOTE: VALUE FOR rs=2.0 AND RPA CORRELATOR [Mathematica].
        return umat
        #dk = dkpts/5000
        #kmax = kpts_mesh[-1] * 500
        #nkp = int(kmax / dk) + 1
        #prefac = -1.0 * (1.0/(2.0 * np.pi**2)) * dk
        #umat = 0.0
        #for ikp in range(1, nkp+1):
        #    kp = ikp * dk
        #    kpSquare = kp**2
        #    u_kp = _calc_correlator(correlator_idx, kpSquare, k_cutoffSquare, rho, gamma)
        #    F = kp ** 4 * u_kp ** 2
        #    umat += F
        #umat *= prefac
        #return umat
    else:
        # Prefactor: (dkp * dx) / (2π)².
        prefac = (dkpts * dxtheta) / (2.0 * np.pi)**2
        nkp = kpts_mesh.shape[0]
        nx = xtheta_mesh.shape[0]
        umat = 0.0
        # Double loop: outer over k', inner over x = cos(θ).
        for ikp in range(nkp):
            kp = kpts_mesh[ikp]
            kpSquare = kp**2
            kpW = kp**w
            # Pre-compute u(kp²) once per kp.
            u_kp = _calc_correlator(correlator_idx, kpSquare, k_cutoffSquare, rho, gamma)
            inner_int = 0.0
            for ix in range(nx):
                x = xtheta_mesh[ix]
                # |k - k'|² = k² + k'² - 2k·k'·cos(θ).
                kMinusKpSquare = kSquare + kpSquare - 2.0 * k * kp * x
                # Skip if |k-k'|² is too small (singularity).
                if abs(kMinusKpSquare) < 1.e-12:
                    continue
                # Regularization weight: 2·|k-k'|^w / (|k-k'|^w + k'^w).
                kMinusKpW = kMinusKpSquare**(w/2)
                weight = (2.0 * kMinusKpW) / (kMinusKpW + kpW)
                # u(|k-k'|²).
                u_kMinusKp = _calc_correlator(correlator_idx, kMinusKpSquare, k_cutoffSquare, rho, gamma)
                # Integrand: (k·k'·x - k'²) · u(k') · u(|k-k'|) · k'² · weight.
                inner_int += (k * kp * x - kpSquare) * u_kMinusKp * weight
            # Add contribution from this k' point.
            umat += inner_int * u_kp * kpSquare
        umat *= prefac
    return umat

# CORRELATORS -----------------------------------------------------

@jit(nopython=True)
def _calc_correlator(correlator_idx, kSquare, k_cutoffSquare, rho, gamma):
    """
    Wrapper function to select and apply the appropriate correlator.

    Input:
    ------
    correlator_idx: int
        Identifier for the correlator type.
    kSquare: float
        Square of the wave vector k.
    k_cutoffSquare: float
        Square of the cutoff wave vector.
    rho: float
        Electron density.
    gamma: float
        Parameter in the correlator function.
    
    Parameters:
    -----------
    correlator_id: int
        0: None,
        1: trunc,
        2: coulomb,
        3: coulomb-yukawa,
        4: RPA.
    """
    if correlator_idx == 0:  # None
        return 0.0
    elif correlator_idx == 1:  # trunc
        return _trunc_correlator(kSquare, k_cutoffSquare, gamma)
    elif correlator_idx == 2:  # coulomb
        return _coulomb_correlator(kSquare, k_cutoffSquare, gamma)
    elif correlator_idx == 3:  # coulomb-yukawa
        return _coulomb_yukawa_correlator(kSquare, rho, k_cutoffSquare, gamma)
    elif correlator_idx == 4:  # RPA
        return _RPA_correlator(kSquare, rho, k_cutoffSquare, gamma)
    else:
        return 0.0

@jit(nopython=True)
def _trunc_correlator(kSquare, k_cutoffSquare, gamma):
    """ Numba JIT-compiled version of the trunc() function for better performance.
    Computes the truncated correlator function u(k), where k is SCALAR.
    See trunc() in ueg.py for more details.
    """
    if kSquare <= k_cutoffSquare * (1 + 0.00001):
        corr = 0.0
    elif kSquare > 1.e-12:
        corr = -4. * np.pi / (kSquare ** 2)
    else:
        corr = 0.0
    return corr * gamma

@jit(nopython=True)
def _coulomb_correlator(kSquare, k_cutoffSquare, gamma):
    """ Numba JIT-compiled version of the coulomb() function for better performance.
    Computes the coulomb correlator function u(k), where k is SCALAR.
    See coulomb() in ueg.py for more details.
    """
    if kSquare > k_cutoffSquare * (1 + 0.00001):
        corr = -4. * np.pi / kSquare
    else:
        corr = 0.0
    return corr * gamma

@jit(nopython=True)
def _coulomb_yukawa_correlator(kSquare, rho, k_cutoffSquare, gamma):
    """ Numba JIT-compiled version of the coulomb_yukawa() function for better performance.
    Computes the coulomb-yukawa correlator function u(k), where k is SCALAR.
    See coulomb_yukawa() in ueg.py for more details.
    """
    wp = np.sqrt(4. * np.pi * rho)
    k_cutoffDenom = k_cutoffSquare * (k_cutoffSquare + wp)
    a = - 4. * np.pi
    b = kSquare * (kSquare + wp)
    if np.abs(b) > k_cutoffDenom:
        corr = a / b
    else:
        corr = 0.0
    return corr * gamma

@jit(nopython=True)
def _RPA_correlator(kSquare, rho, k_cutoffSquare, gamma):
    """ Numba JIT-compiled version of the RPA() function for better performance.
    Computes the RPA correlator function u(k), where k is SCALAR.
    See RPA() in ueg.py for more details.
    """
    kVec = np.sqrt(kSquare)
    kFermi = (3.0 * np.pi**2 * rho) ** (1.0 / 3.0)
    k_cutoffDenom = 2. * rho * k_cutoffSquare * \
                    ( (3./4.)*(np.sqrt(k_cutoffSquare)/kFermi) - \
                    (1./16.)*(np.sqrt(k_cutoffSquare)/kFermi)**3 )
    if kVec > (2*kFermi):
        T2 = 1.0
    else:
        T2 = (3./4.)*(kVec/kFermi) - (1./16.)*(kVec/kFermi)**3
    a = kSquare - np.sqrt( (kSquare** 2) + 16. * np.pi * rho * (T2**2) )
    b = 2. * rho * T2 * kSquare
    if np.abs(b) > k_cutoffDenom:
        corr = a / b
    else:
        corr = 0.0
    return corr * gamma

#@jit(nopython=True)
#def _yukawa_correlator(kSquare, k_cutoffSquare, rho, gamma, multiply_by_k_square=False):
#    """ Numba JIT-compiled version of the yukawa_correlator function for better performance.
#    Computes the yukawa correlator function u(k^2), where k is SCALAR.
#
#    The G=0 terms need more consideration due to 1 / (G=0)^2 divergence.
#
#    Parameters
#    ----------
#    kSquare: float
#        square of the plane wave vector k.
#    k_cutoffSquare: float
#        plane wave vector cutoff.
#    rho: float
#        electron density.
#    gamma: float
#        parameter in the correlator function.
#    rho: float
#        electron density.
#    Returns
#    -------
#    corr: float
#        value of the yukawa correlator function u(k^2).
#    """
#    gamma_0 = np.sqrt(rho / 4. * np.pi)
#    a = -4. * np.pi
#    gamma_yukawa = gamma * gamma_0
#    k_cutoffSquare = max(k_cutoffSquare, 1e-12)
#    k_cutoffDenom = k_cutoffSquare + gamma_yukawa
#    if np.abs(k_cutoffDenom) < 1e-12:
#        k_cutoffDenom = 1e-12
#    if not multiply_by_k_square:
#        b = (kSquare + gamma_yukawa)
#        if (np.abs(b) > k_cutoffDenom):
#            corr = a / b
#        else:
#            corr = 0.0
#    else:
#        if kSquare > k_cutoffSquare * (1 + 0.00001):
#            corr = a / (kSquare + gamma_yukawa) * kSquare
#        else:
#            corr = 0.0
#    return corr
#
#@jit(nopython=True)
#def _yukawa_coulomb_correlator(kSquare, k_cutoffSquare, rho, gamma, multiply_by_k_square=False):
#    """ Numba JIT-compiled version of the yukawa_coulomb_correlator function for better performance.
#    Computes the yukawa-coulomb correlator function u(k^2), where k is SCALAR.
#
#    The G=0 terms need more consideration due to 1 / (G=0)^2 divergence.
#
#    Parameters
#    ----------
#    kSquare: float
#        square of the plane wave vector k.
#    k_cutoffSquare: float
#        plane wave vector cutoff.
#    rho: float
#        electron density.
#    gamma: float
#        parameter in the correlator function.
#    Returns
#    -------
#    corr: float
#        value of the yukawa-coulomb correlator function u(k^2).
#    """
#    # gamma is different due to gamma_0 = 1.5 in case self.gamma is None.
#    #if gamma == 1.0:
#    #    gamma_yukawa = 1.5
#    #else:
#    #    gamma_yukawa = gamma
#    gamma_yukawa = gamma
#    # A corresponds to 1/gamma**2 in Gruneis paper
#    A = np.sqrt(1.0 / (4.0 * np.pi * rho))
#    A = 1. / A * gamma_yukawa
#    # It has to be - and divided by gamm to satisfy the cusp condition
#    a = -4. * np.pi
#    k_cutoffSquare = max(k_cutoffSquare, 1e-12)
#    k_cutoffDenom = (k_cutoffSquare + A)
#    if np.abs(k_cutoffDenom) < 1e-12:
#        k_cutoffDenom = 1e-12
#    if not multiply_by_k_square:
#        b = (kSquare + A) * kSquare
#        if (np.abs(b) > k_cutoffDenom):
#            corr = a / b
#        else:
#            corr = 0.0
#    else:
#        if kSquare > k_cutoffSquare * (1 + 0.00001):
#            corr = a / (kSquare + A)
#        else:
#            corr = 0.0
#    return corr
#
#@jit(nopython=True)
#def _gaskell_correlator(kSquare, k_cutoffSquare, rho, gamma):
#    """ 
#    Placeholder for Gaskell correlator function.
#    Currently returns 0.0 for all inputs.
#    """
#    return 0.0
#
#@jit(nopython=True)
#def _gaskell_modified_correlator(kSquare, k_cutoffSquare, rho, gamma):
#    """
#    Placeholder for modified Gaskell correlator function.
#    Currently returns 0.0 for all inputs.
#    """
#    return 0.0  
#
#@jit(nopython=True)
#def _smooth_correlator(kSquare, k_cutoffSquare, rho, gamma):
#    """
#    Placeholder for smooth correlator function.
#    Currently returns 0.0 for all inputs.
#    """
#    return 0.0
