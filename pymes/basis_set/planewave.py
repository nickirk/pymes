import numpy as np

class BasisFunc:
    """Basis function with wavevector `2\pi(i,j,k)^{T}/L` of the desired spin.
    Args:
         i, j, k: integer. Labels (quantum numbers) of the wavevector
         L: float. Dimension of the cubic simulation cell of size `L\\times L \\times L`
         spin: integer. spin of the basis function (-1 for a down electron, +1 for an up electron)
         k_shift: size 3 list/np array of floats. The k-shift used in twist-average. The default values are relative to
         the 1.B.Z.
    """
    def __init__(self, i, j, k, L, spin, k_shift=[0., 0., 0.]):
        self.k = np.array([x for x in (i, j, k)])
        self.L = L
        self.kp = (self.k+k_shift)*2*np.pi/L
        #self.kp = (self.k)/L
        # remove 1/2 to be consistent with neci
        # self.kinetic = np.dot(self.kp, self.kp)
        self.kinetic = np.dot(self.kp, self.kp)/2.
        if not (spin == -1 or spin == 1):
            raise RuntimeError('spin not +1 or -1')
        self.spin = spin
    def __repr__(self):
        return (self.k, self.kinetic, self.spin).__repr__()
    def __lt__(self, other):
        return self.kinetic < other.kinetic

def is_closed_shell(N):
    """Check if the system is a closed shell.
    A closed shell is defined as having an even number of electrons (N) and
    a total (s) that can be expressed as a sum of three squares.
    Args:
        N (int): Number of electrons.
    Returns:
        bool: True if the system is a closed shell, False otherwise.
    """
    if N <= 0 or N % 2 != 0:
        return False

    cumulative = 0
    s = 0

    while True:
        total_degenerate_states = count_spatial_states(s)
        electrons_in_shell = total_degenerate_states * 2
        cumulative += electrons_in_shell
    
        if cumulative == N:
            return True
        elif cumulative > N:
            return False
    
        s += 1

def count_spatial_states(s):
    """Count the number of spatial states with a given total s = i^2 + j^2 + k^2, for i, j, k >= 0.
    and i, j, k are integers, i.e. the wavevector components of the basis function.
    The function counts the number of ways to express s as a sum of three squares (i^2 + j^2 + k^2)
    and considers the permutations of (i, j, k) and the sign variations.
    Args:
        s (int): Total (s = i^2 + j^2 + k^2).
    Returns:
        int: Number of spatial states.
    """
    n_states = 0
    max_i = int(s ** 0.5)
    for i in range(max_i + 1):
        remaining_i = s - i ** 2
        if remaining_i < 0:
            continue
        max_j = int(remaining_i ** 0.5)
        for j in range(min(i, max_j) + 1):
            remaining_j = remaining_i - j ** 2
            if remaining_j < 0:
                continue
            max_k = int(remaining_j ** 0.5)
            for k in range(min(j, max_k) + 1):
                if i ** 2 + j ** 2 + k ** 2 == s:
                    # Determine permutations
                    if i == j and j == k:
                        permutations = 1
                    elif i == j or j == k:
                        permutations = 3
                    else:
                        permutations = 6
                    # Count non-zero elements
                    non_zero = sum(x != 0 for x in (i, j, k))
                    sign_variations = 2 ** non_zero
                    n_states += permutations * sign_variations
    return n_states
