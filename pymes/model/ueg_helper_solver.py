import numpy as np
from numba import jit, prange, get_num_threads, config

from pymes.model.ueg_helper_int import _get_2b_int_kernel

"""
Helper functions for UEG model system JIT-compiled with Numba for better performance.
    This module includes functions to compute two-body integrals in a transcorrelated framework
    in a JIT-compiled manner using Numba for enhanced performance.
Main functions:
    _solve_mp2: Numba JIT-compiled function to compute the MP2 correlation energy contributions
                (both direct and exchange) for the UEG model system.
"""

@jit(nopython=True, parallel=True)
def _solve_mp2(n_ele, Omega, L, rho, 
                imax, k_cutoff, gamma,
                UMAT, basis_indices_map,
                basis_occ_Kp, basis_Kvec, basis_Kp,
                epsilon_i, epsilon_a,
                is_only_2b, is_effect_2b, is_tc, correlator_idx,
                dtype=np.float64):

    e_mp2_dir = 0.0
    e_mp2_exc = 0.0

    no = int(n_ele // 2)
    nP = int(basis_Kp.shape[0])
    nv = nP - no

    num_k_in_each_dir = imax * 2 + 1
    k_cutoffSquare = (2 * np.pi * k_cutoff / L)**2
    idx_shift = 2*imax

    for a in prange(no, nP):
        for i in range(no):
            d_int_k = basis_Kvec[i] - basis_Kvec[a]
            d_k_vec = basis_Kp[i] - basis_Kp[a]
            dk_square = d_k_vec[0]**2 + d_k_vec[1]**2 + d_k_vec[2]**2
            u_mat = 0.
            if is_tc:
                idx_shift = 2 * imax
                ix = d_int_k[0] + idx_shift
                iy = d_int_k[1] + idx_shift
                iz = d_int_k[2] + idx_shift
                u_mat = UMAT[ix, iy, iz]
            for b in range(no, nP):
                int_kj = basis_Kvec[b] - d_int_k
                loc_j = num_k_in_each_dir ** 2 * (int_kj[0] + imax) + \
                        num_k_in_each_dir * (int_kj[1] + imax) + \
                        int_kj[2] + imax
                if len(basis_indices_map) > loc_j >= 0:
                    j = int(basis_indices_map[loc_j])
                    if j < 0 or j >= no:
                        continue
                else:
                    continue
                denom  = epsilon_i[i] + epsilon_i[j] - epsilon_a[a-no] - epsilon_a[b-no]
                # Regularization to avoid singularity in the denominator.
                if abs(denom) < 1.e-12:
                    continue
                # Compute T2 amplitudes.
                T_abij = _get_2b_int_kernel(a, b, i, j,
                                            dk_square, d_k_vec, u_mat,
                                            n_ele, Omega, rho,
                                            k_cutoffSquare, gamma,
                                            basis_occ_Kp, basis_Kp,
                                            is_only_2b, is_effect_2b, is_tc, correlator_idx)
                T_abij = T_abij / denom
                # Compute Direct V_ijab: d_k_vec_ijab = k_a - k_i = k_j - k_b (from mom. cons.).
                # d_k_vec_ijab = - (k_i - k_a) = - d_k_vec.
                V_ijab = _get_2b_int_kernel(i, j, a, b,
                                            dk_square, -d_k_vec, u_mat,
                                            n_ele, Omega, rho,
                                            k_cutoffSquare, gamma,
                                            basis_occ_Kp, basis_Kp,
                                            is_only_2b, is_effect_2b, is_tc, correlator_idx)
                # Compute Exchange V_jiab: d_k_vec_jiab = k_a - k_j = k_i - k_b (from mom. cons.).
                # Must be recomputed here since j depends on b and differs from the (a,i) pair.
                d_int_k_jiab = basis_Kvec[a] - basis_Kvec[j]
                d_k_vec_jiab = basis_Kp[a] - basis_Kp[j]
                dk_square_jiab = d_k_vec_jiab[0]**2 + d_k_vec_jiab[1]**2 + d_k_vec_jiab[2]**2
                u_mat_jiab = 0.
                if is_tc:
                    ix_j = d_int_k_jiab[0] + idx_shift
                    iy_j = d_int_k_jiab[1] + idx_shift
                    iz_j = d_int_k_jiab[2] + idx_shift
                    u_mat_jiab = UMAT[ix_j, iy_j, iz_j]
                V_jiab = _get_2b_int_kernel(j, i, a, b,
                                            dk_square_jiab, d_k_vec_jiab, u_mat_jiab,
                                            n_ele, Omega, rho,
                                            k_cutoffSquare, gamma,
                                            basis_occ_Kp, basis_Kp,
                                            is_only_2b, is_effect_2b, is_tc, correlator_idx)
                # Accumulate MP2 energy contributions.
                e_mp2_dir += 2 * T_abij * V_ijab
                e_mp2_exc += -1 * T_abij * V_jiab
    return e_mp2_dir, e_mp2_exc