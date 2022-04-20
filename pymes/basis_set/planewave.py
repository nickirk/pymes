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
