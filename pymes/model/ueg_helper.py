import numpy as np
from numba import jit, prange, get_num_threads, config

"""
Helper functions for UEG model system JIT-compiled with Numba for better performance.
    This module includes functions to compute two-body integrals in a transcorrelated framework 
    in a JIT-compiled manner using Numba for enhanced performance.
Main functions:
    _get_2b_int: Numba JIT-compiled version of the get_2b_int().
Auxiliary functions:
    _sumNablaUSquare: Numba JIT-compiled version of the sumNablaUSquare().
    _contract_exchange_3_body: Numba JIT-compiled version of the contract_exchange_3_body().
    _contractP_KWithQ: Numba JIT-compiled version of the contractP_KWithQ().
Correlators:
    _trunc_correlator: Numba JIT-compiled version of the trunc().
"""


@jit(nopython=True, parallel=True)
def _get_2b_int( idx, n_ele, Omega, L, imax, k_cutoff, gamma, 
                    kPrime, basis_indices_map,
                    basis_occ_Kp, basis_Kvec, basis_Kp,
                    is_only_2b, is_effect_2b, is_tc, dtype=np.float64):
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
    imax: int
        maximum k-point index in each direction
    k_cutoff: float
        plane wave vector cutoff inside the correlaor function trunc.
    gamma: float
        parameter in the correlator function
    kPrime: nparray of float dtype
        an array to store shifted k-vectors for later lookup.
    basis_indices_map: nparray of int dtype
        an array to store indices of basis functions (plane waves) for
        later lookup. Size Nx*Ny*Nz, Nx, Ny, Nz are the k-vector points
        in x, y, z directions.
    basis_occ_Kp: nparray of float dtype
        an array to store shifted k-vectors of occupied orbitals.
    basis_Kvec: nparray of int dtype
        an array to store k-vectors of all basis functions.
    basis_Kp: nparray of float dtype
        an array to store shifted k-vectors of all basis functions.
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

    Returns
    -------
    V_pqrs: tensor object (tensor by default)
        of size [ idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7] ], np array.
    """
    num_k_in_each_dir = imax * 2 + 1
    V_pqrs = np.zeros((idx[1]-idx[0], idx[3]-idx[2], idx[5]-idx[4], idx[7]-idx[6]), dtype=dtype)
    k_cutoffSquare = (2 * np.pi * k_cutoff / L)**2 

    #p_range = idx[1] - idx[0]
    #r_range = idx[5] - idx[4]

    for p in prange(idx[0], idx[1]):
        loc_p_idx = p - idx[0]
        for r in range(idx[4], idx[5]):
    #for pr in prange(p_range * r_range):
    #        p = idx[0] + pr // r_range
    #        r = idx[4] + pr % r_range
    #        loc_p_idx = p - idx[0]
            loc_r_idx = r - idx[4]
            d_int_k = basis_Kvec[r] - basis_Kvec[p]
            d_k_vec = basis_Kp[r] - basis_Kp[p]
            u_mat = 0.
            if is_tc:
                u_mat = _sumNablaUSquare(d_k_vec, Omega, L, kPrime, k_cutoffSquare,  gamma)
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
                dk_square = d_k_vec[0]**2 + d_k_vec[1]**2 + d_k_vec[2]**2
                w = 0.0
                if is_tc:
                    if is_only_2b:
                        # Pure 2-bidy TC integrals.
                        if np.abs(dk_square) > 0.:
                            rs_dk = basis_Kp[r] - basis_Kp[s]
                            rs_dk_dot_d_k_vec = rs_dk[0]*d_k_vec[0] + rs_dk[1]*d_k_vec[1] + rs_dk[2]*d_k_vec[2]
                            corr_dk_square = _trunc_correlator(dk_square, k_cutoffSquare, gamma, L)
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
                            corr_dk_square = _trunc_correlator(dk_square, k_cutoffSquare, gamma, L)
                            w_pqrs = -(n_ele) * dk_square \
                                    * corr_dk_square**2 / Omega \
                                    + 2. * _contract_exchange_3_body( basis_Kp[r], d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma) \
                                    - 2. * _contract_exchange_3_body( basis_Kp[p], d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma) \
                                    + 2. * _contractP_KWithQ( basis_Kp[r], d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma)
                            w_qpsr = -(n_ele) * dk_square \
                                    * corr_dk_square**2 / Omega \
                                    + 2. * _contract_exchange_3_body( basis_Kp[s], -d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma) \
                                    - 2. * _contract_exchange_3_body( basis_Kp[q], -d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma) \
                                    + 2. * _contractP_KWithQ( basis_Kp[s], -d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma)
                            w = 0.5 * (w_pqrs + w_qpsr)
                        else:
                            w = u_mat
                            w_pqrs = 2. * _contractP_KWithQ( basis_Kp[r],  d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma)
                            w_qpsr = 2. * _contractP_KWithQ( basis_Kp[s], -d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma)
                            w = 0.5 * (w_pqrs + w_qpsr)
                        w = w / Omega
                    else:
                        # Full 2-body integrals including both Coulomb and TC contributions.
                        if np.abs(dk_square) > 0.:
                            corr_dk_square = _trunc_correlator(dk_square, k_cutoffSquare, gamma, L)
                            rs_dk = basis_Kp[r] - basis_Kp[s]
                            rs_dk_dot_d_k_vec = rs_dk[0]*d_k_vec[0] + rs_dk[1]*d_k_vec[1] + rs_dk[2]*d_k_vec[2]
                            w = 4. * np.pi / dk_square
                            w += + u_mat \
                                + (dk_square - rs_dk_dot_d_k_vec) \
                                * corr_dk_square
                            w_pqrs = -(n_ele) * dk_square \
                                    * corr_dk_square**2 / Omega \
                                    + 2. * _contract_exchange_3_body( basis_Kp[r], d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma) \
                                    - 2. * _contract_exchange_3_body( basis_Kp[p], d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma) \
                                    + 2. * _contractP_KWithQ( basis_Kp[r], d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma)
                            w_qpsr = -(n_ele) * dk_square \
                                    * corr_dk_square**2 / Omega \
                                    + 2. * _contract_exchange_3_body( basis_Kp[s], -d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma) \
                                    - 2. * _contract_exchange_3_body( basis_Kp[q], -d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma) \
                                    + 2. * _contractP_KWithQ( basis_Kp[s], -d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma)
                            w += 0.5 * (w_pqrs + w_qpsr)
                        else:
                            w = u_mat
                            w_pqrs = 2. * _contractP_KWithQ( basis_Kp[r],  d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma)
                            w_qpsr = 2. * _contractP_KWithQ( basis_Kp[s], -d_k_vec, basis_occ_Kp, Omega, L, k_cutoffSquare, gamma)
                            w += 0.5 * (w_pqrs + w_qpsr)
                        w = w / Omega
                else:
                    # Coulomb integrals only.
                    if np.abs(dk_square) > 0.:
                        w = 4. * np.pi / dk_square / Omega
                V_pqrs[loc_p_idx,
                        loc_q_idx,
                        loc_r_idx,
                        loc_s_idx] = w
    return V_pqrs   

@jit(nopython=True)
def _contract_exchange_3_body(pVec, kVec, occ_Kp, Omega, L, k_cutoffSquare, gamma):
    """ Numba JIT-compiled version of the contract_exchange_3_body function for better performance.
    Computes the single contraction of the exchange type from the 3-body integrals.
    Parameters
    ----------
    pVec: nparray of float dtype
        k-vector of basis function p.
    kVec: nparray of float dtype
        difference of two k-vectors (k_p - k_q)
    occ_Kp: nparray of float dtype
        an array to store shifted k-vectors of occupied orbitals.
    Omega: float
        volume of the cubic simulation cell
    L: float
        length of the cubic simulation cell
    k_cutoffSquare: float
        plane wave vector cutoff inside the correlaor function trunc.
    gamma: float
        parameter in the correlator function.
    Returns
    -------
    w: float
        value of the single contraction of the exchange type from the 3-body integrals.
    """
    kVecSquare = kVec[0]**2 + kVec[1]**2 + kVec[2]**2
    corr_kVec = _trunc_correlator(kVecSquare, k_cutoffSquare, gamma, L)
    w = 0.0
    for i in range(occ_Kp.shape[0]):
        pDiff = pVec - occ_Kp[i]
        pDiffSquare = pDiff[0]**2 + pDiff[1]**2 + pDiff[2]**2
        pDiffDotkVec = pDiff[0]*kVec[0] + pDiff[1]*kVec[1] + pDiff[2]*kVec[2]

        corr_pDiff = _trunc_correlator(pDiffSquare, k_cutoffSquare, gamma, L)

        w += pDiffDotkVec * corr_pDiff * corr_kVec
    
    return w / Omega

@jit(nopython=True)
def _contractP_KWithQ(pVec, kVec, occ_Kp, Omega, L, k_cutoffSquare, gamma):
    """ Numba JIT-compiled version of the contractP_KWithQ function for better performance.
    Parameters
    ----------
    pVec: nparray of float dtype
        k-vector of basis function p.
    kVec: nparray of float dtype
        difference of two k-vectors (k_p - k_q)
    occ_Kp: nparray of float dtype
        an array to store shifted k-vectors of occupied orbitals.
    Omega: float
        volume of the cubic simulation cell
    L: float
        length of the cubic simulation cell
    k_cutoffSquare: float
        plane wave vector cutoff inside the correlaor function trunc.
    gamma: float
        parameter in the correlator function.
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

        corr_vec1 = _trunc_correlator(vec1Square, k_cutoffSquare, gamma, L)
        corr_vec2 = _trunc_correlator(vec2Square, k_cutoffSquare, gamma, L)

        w += vec1Dotvec2 * corr_vec1 * corr_vec2

    return w / Omega
    

@jit(nopython=True)
def _sumNablaUSquare(kVec, Omega, L, kPrime, k_cutoffSquare, gamma):
    """ Numba JIT-compiled version of the sumNablaUSquare function for better performance.
    Computes: sum_k' (k1 · k2) * u(k1^2) * u(k2^2) / Omega
    Parameters
    ----------
    kVec: nparray of float dtype
        difference of two k-vectors (k_p - k_q)
    Omega: float
        volume of the cubic simulation cell
    L: float
        length of the cubic simulation cell
    kPrime: nparray of float dtype
        an array to store shifted k-vectors for later lookup.
    k_cutoffSquare: float
        plane wave vector cutoff inside the correlaor function trunc.
    gamma: float
        parameter in the correlator function.
    Returns
    -------
    u_mat: float
        value of the sum of the squared gradients of the correlator function
        in k-space.
    """
    k1 = 2 * np.pi * kPrime / L
    k2 = kVec - k1
    umat = 0.0
    for i in range(k1.shape[0]):
        k2 = kVec - k1[i]
        k1Square = k1[i,0]**2 + k1[i,1]**2 + k1[i,2]**2
        k2Square = k2[0]**2 + k2[1]**2 + k2[2]**2
        k1Dotk2 = k1[i,0]*k2[0] + k1[i,1]*k2[1] + k1[i,2]*k2[2]
        corr_k1 = _trunc_correlator(k1Square, k_cutoffSquare, gamma, L)
        corr_k2 = _trunc_correlator(k2Square, k_cutoffSquare, gamma, L)
        umat += k1Dotk2 * corr_k1 * corr_k2

    return umat / Omega

# CORRELATORS -----------------------------------------------------

@jit(nopython=True)
def _trunc_correlator(kSquare, k_cutoffSquare, gamma, L):
    """ Numba JIT-compiled version of the trunc_correlator function for better performance.
    Computes the truncated correlator function u(k^2), where k is SCALAR.
    Parameters
    ----------
    kSquare: float
        square of the plane wave vector k.
    k_cutoff: float
        plane wave vector cutoff inside the correlator function trunc.
    gamma: float
        parameter in the correlator function.
    L: float
        length of the cubic simulation cell.
    Returns
    -------
    corr: float
        value of the truncated correlator function u(k^2).
    """
    if kSquare <= k_cutoffSquare * (1 + 0.00001):
        corr = 0.0
    elif kSquare > 1e-12:
        corr = -4. * np.pi / (kSquare ** 2)
    else:
        corr = 0.0
    return corr * gamma
