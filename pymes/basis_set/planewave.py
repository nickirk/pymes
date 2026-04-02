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

def is_closed_shell(N, k_shift=[0., 0., 0.]):
    """Check if the system is a closed shell.
    A closed shell is defined as having an even number of electrons (N) and
    a total (s) that can be expressed as a sum of three squares (possibly shifted).
    Args:
        N: int
            number of electrons.
        k_shift: size 3 list/np array of floats.
            the k-shift [sx, sy, sz] used for Gamma-point or Baldereschi point calculations, which 
            can affect the degeneracy of the spatial states and thus the closed-shell condition.
    Returns:
        bool: True if the system is a closed shell, False otherwise.
    """
    if N <= 0 or N % 2 != 0:
        return False

    cumulative = 0
    s = k_shift[0]**2 + k_shift[1]**2 + k_shift[2]**2
    ds = get_delta_s(k_shift)
    
    while True:
        total_degenerate_states = count_spatial_states(s, k_shift)
        electrons_in_shell = total_degenerate_states * 2
        cumulative += electrons_in_shell

        if cumulative == N:
            return True
        elif cumulative > N:
            return False
    
        s += ds

def get_delta_s(k_shift):
    """Calculate the smallest non-zero change in s due to the k-shift.
    Determines the spacing of energy levels and thus the closed-shell condition.
    Args:
        k_shift: size 3 list/np array of floats. 
                the k-shift [sx, sy, sz] of the lattice.
    Returns:
        float: The smallest non-zero change in s due to the k-shift.
    """
    deltas = []
    for i in range(3):
        if k_shift[i] != 0:
            deltas.append(2 * k_shift[i] + 1)  # Change from (n^2) to ((n+1)^2)
            deltas.append(2 * k_shift[i] - 1)  # Change from (n^2) to ((n-1)^2)
    return min(abs(delta) for delta in deltas) if deltas else 1

def count_spatial_states(s, k_shift=[0., 0., 0.]):
    """Count the number of spatial states with a given total s = (i+sx)^2 + (j+sy)^2 + (k+sz)^2,
    where i, j, k are integers and k_shift = [sx, sy, sz].
    
    For k_shift = [0, 0, 0] (Gamma point), this reduces to counting solutions to i^2 + j^2 + k^2 = s
    with proper permutation and sign symmetries.
    
    For non-zero k_shift, the symmetry is reduced and each state is counted individually.
    
    Args:
        s: float
            target kinetic energy parameter (i^2 + j^2 + k^2 for Gamma, or shifted sum otherwise).
        k_shift: size 3 list/np array of floats
            the k-shift [sx, sy, sz] of the lattice.
    
    Returns:
        int: Number of spatial states.
    """
    n_states = 0
    sx, sy, sz = k_shift[0], k_shift[1], k_shift[2]
    
    # Check if this is the Gamma-point case.
    is_gamma_point = np.allclose([sx, sy, sz], [0., 0., 0.])
    
    if is_gamma_point:
        # Efficient algorithm for Gamma-point: enumerate non-negative (i,j,k) with i <= j <= k.
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
                        # Determine permutations.
                        if i == j and j == k:
                            permutations = 1
                        elif i == j or j == k:
                            permutations = 3
                        else:
                            permutations = 6
                        # Count non-zero elements.
                        non_zero = sum(x != 0 for x in (i, j, k))
                        sign_variations = 2 ** non_zero
                        n_states += permutations * sign_variations
    else:
        # Non-Gamma case: search over full integer range with k-shift.
        # NOTE: this also works for the Gamma-point case.
        # Determine search range based on target s and shift magnitude.
        max_shift = max(abs(sx), abs(sy), abs(sz))
        search_range = int(np.sqrt(s) + max_shift + 2)
        
        # Use a set to avoid counting duplicate states.
        found_states = set()
        
        for i in range(-search_range, search_range + 1):
            for j in range(-search_range, search_range + 1):
                for k in range(-search_range, search_range + 1):
                    kinetic = (i + sx) ** 2 + (j + sy) ** 2 + (k + sz) ** 2
                    # Floating-point comparison.
                    if abs(kinetic - s) < 1e-10:
                        # Store as tuple to track uniqueness.
                        state = (i, j, k)
                        if state not in found_states:
                            found_states.add(state)
                            n_states += 1
    
    return n_states

def get_planewave_gap(basis_fns, no):
    """Calculate the gap between the highest occupied and lowest unoccupied spatial states.
    Args:
        basis_fns: list or tuple
            of BasisFunc objects, sorted by kinetic energy.
        no: int
            number of occupied spatial states (half the number of electrons).
    Returns:
        None: 
            if no is equal to the total number of spatial states (i.e., all states are occupied).
        float: 
            the energy gap between the highest occupied and lowest unoccupied spatial states.

    """

    if no <= 0 or no > len(basis_fns) // 2:
        raise ValueError("Invalid number of occupied states.")
    elif no == len(basis_fns) // 2:
        gap = None
    else:
        # Get the kinetic energy of the highest occupied and lowest unoccupied spatial states.
        occupied_kinetic = basis_fns[2*no - 1].kinetic
        unoccupied_kinetic = basis_fns[2*no].kinetic
        # Calculate the gap.
        gap = unoccupied_kinetic - occupied_kinetic
    return gap