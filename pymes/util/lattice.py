import numpy as np
import spglib as spg
from numba import jit, prange, get_num_threads, config

from pymes.log import print_title, print_logging_info

def get_lattice_shells(L, Rmax, nmax, is_R0=False, use_jit=True, dtype=np.float64):
    """
    Function to compute the lattice shells and their multiplicities for a simple cubic lattice.
    Parameters
    ----------
    L : float
        Lattice constant (box length).
    Rmax : float
        Maximum radius for including lattice shells.
    nmax : int
        Maximum integer index for lattice points (defines the search range).
    is_R0 : bool, optional
        Whether to include the origin (R=0) in the shells.
    use_jit : bool, optional
        Whether to use the JIT-compiled version of the function for performance.
    dtype : np.dtype, optional
        Data type for the returned arrays.
    Returns
    -------
    R : np.ndarray
        Array of distances for each lattice shell.
    wR : np.ndarray
        Array of multiplicities (number of lattice points) for each shell.
    """
    if use_jit:
        R, wR = _get_lattice_shells_jit(L, Rmax, nmax, is_R0, dtype=dtype)
    else:
        R, wR = _get_lattice_shells(L, Rmax, nmax, is_R0, dtype=dtype)
    return R, wR

@jit(nopython=True)
def _get_lattice_shells_jit(L, Rmax, nmax, is_R0=False, dtype=np.float64):
    maxR2 = 3 * nmax * nmax
    counts = np.zeros(maxR2 + 1, np.int64)
    for n1 in range(-nmax, nmax + 1):
        for n2 in range(-nmax, nmax + 1):
            for n3 in range(-nmax, nmax + 1):
                if n1 == 0 and n2 == 0 and n3 == 0:
                    if not is_R0:
                        continue
                R2 = n1 * n1 + n2 * n2 + n3 * n3
                Rtmp = np.sqrt(R2) * L
                if Rtmp <= Rmax:
                    counts[R2] += 1
    # Count non-zero shells (skip R2 == 0)
    nz = 0
    for i in range(1, maxR2 + 1):
        if counts[i] != 0:
            nz += 1
    if nz == 0:
        return np.zeros(0, np.float64), np.zeros(0, np.int64)
    R = np.empty(nz, np.float64)
    wR = np.empty(nz, np.int64)
    idx = 0
    for i in range(1, maxR2 + 1):
        if counts[i] != 0:
            R[idx] = np.sqrt(i) * L
            wR[idx] = counts[i]
            idx += 1
    return R, wR

def _get_lattice_shells(L, Rmax, nmax, is_R0=False, dtype=np.float64):
    shells = {}
    for n1 in range ( -nmax, nmax+1):
        for n2 in range ( -nmax, nmax+1):
            for n3 in range ( -nmax, nmax+1):
                if n1 == 0 and n2 == 0 and n3 == 0:
                    if not is_R0:
                        continue
                R2 = n1**2 + n2**2 + n3**2
                R = np.sqrt(R2) * L
                if R <= Rmax:
                    shells[R2] = shells.get(R2, 0) + 1
    R2_unique = sorted(shells.keys())
    R = np.array([np.sqrt(R2) * L for R2 in R2_unique], dtype=dtype)
    wR = np.array([shells[R2] for R2 in R2_unique], dtype=np.int64)
    return R, wR