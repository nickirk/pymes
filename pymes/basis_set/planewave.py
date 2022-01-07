import numpy as np

class BasisFunc:
    '''Basis function with wavevector `2\pi(i,j,k)^{T}/L` of the desired spin.
    
    :param integer i, j, k: integer labels (quantum numbers) of the wavevector
    :param float L: dimension of the cubic simulation cell of size `L\\times L \\times L`
    :param integer spin: spin of the basis function (-1 for a down electron, +1 for an up electron)
    '''
    def __init__(self, i, j, k, L, spin):
        self.k = np.array([x for x in (i, j, k)])
        self.L = L
        self.kp = (self.k*2*np.pi)/L
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
