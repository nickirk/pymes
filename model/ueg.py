
import numpy as np
from pymes.basis_set import planewave

class UEG:

    def __init__(self, nel, nalpha, nbeta, rs):
        #: number of electrons
        self.nel = nel
        #: number of alpha (spin-up) electrons
        self.nalpha = nalpha
        #: number of beta (spin-down) electrons
        self.nbeta = nbeta
        #: electronic density
        self.rs = rs
        #: length of the cubic simulation cell containing nel electrons
        #: at the density of rs
        self.L = self.rs*((4*np.pi*self.nel)/3)**(1.0/3.0)
        #: volume of the cubic simulation cell containing nel electrons at the density
        #: of rs
        self.Omega = self.L**3


    #--- Basis set ---
    def init_single_basis(self, nMax, cutoff, sym):
        '''Create single-particle basis.
    
    :type self: :class:`UEG`
    :param sys: UEG system to be studied.
    
    :param nMax: the number of G vectors in each direction to loop over. eg. 1 gives -1, 0, 1
    
    :param float cutoff: energy cutoff, in units of `2*(2\pi/L)^2`, defining the single-particle basis.  Only single-particle basis functions with a kinetic energy equal to or less than the cutoff are considered.
    
    :type sym: np.array
    :param sym: integer vector defining the wavevector, in units of `2\pi/L`, representing the desired symmetry.  Only Hartree products and determinants of this symmetry are returned.
    
    :returns: basis_fns where:
    
        basis_fns
            tuple of relevant :class:`PlaneWaveFn` objects, ie the single-particle basis set.
    
    '''
    
        # Single particle basis within the desired energy cutoff.
        #cutoff = cutoff*(2*np.pi/self.L)**2
        imax = nMax #int(np.ceil(np.sqrt(cutoff*2)))
        basis_fns = []
        for i in range(-imax, imax+1):
            for j in range(-imax, imax+1):
                for k in range(-imax, imax+1):
                    bfn = planewave.BasisFunc(i, j, k, self.L, 1)
                    if bfn.k.dot(bfn.k) <= cutoff:
                        basis_fns.append(planewave.BasisFunc(i, j, k, self.L, 1))
                        basis_fns.append(planewave.BasisFunc(i, j, k, self.L, -1))
        # Sort in ascending order of kinetic energy.  Note that python's .sort()
        # (since 2.3) is guaranteed to be stable.
        basis_fns.sort()
        basis_fns = tuple(basis_fns)
    
        return basis_fns
