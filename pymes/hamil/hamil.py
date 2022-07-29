'''
hamil
=====

Base classes and functions for generating and interacting with Hamiltonian
matrices.

This class is only used for UEG model.
'''

# copyright (c) 2012 James Spencer.
# All rights reserved.
#
# Modified BSD License; see LICENSE for more details.

import numpy

#--- Construct Hamiltonian (base) ---

class Hamiltonian:
    '''Base Hamiltonian class.

The relevant subclass which provides the appropriate matrix elements should be
used.

:param sys: object describing system to be studied; used only in the virtual
    matrix element functions
:type basis: iterable of iterables of single-particle basis functions
:param basis: set of many-particle basis functions
:param float tau: timestep by which to propogate psip distributions in
    imaginary time

.. note::

    This is a base class; basis and sys must be appropriate to the actual subclass
    used and are specific to the required system and underlying many-particle
    basis set.
'''
    def __init__(self, sys, basis, tau=0.1):

        #: system to be studied
        self.sys = sys
        #: set of many-particle basis functions
        self.basis = basis
        #: number of many-particle basis functions (i.e. length basis)
        self.nbasis = len(self.basis)
        #: Hamiltonian matrix in the basis set of the many-particle basis functions
        self.hamil = numpy.zeros([self.nbasis, self.nbasis])
        #: timestep by which to propogate psip distributions in imaginary time
        self.tau = tau

        # Construct Hamiltonain
        for i in range(self.nbasis):
            bi = self.basis[i]
            self.hamil[i][i] = self.mat_fn_diag(bi)
            for j in range(i+1, self.nbasis):
                bj = self.basis[j]
                self.hamil[i][j] = self.mat_fn_offdiag(bi, bj)
                self.hamil[j][i] = self.hamil[i][j]

        propogator = numpy.identity(self.nbasis) - self.tau*self.hamil
        #: positive propogator matrix, `T^+` where `T = 1 - \tau H = T^+ - T^-` and `T^+` elements are non-negative.
        self.pos_propogator = (propogator > 0) * propogator
        #: negative propogator matrix, `T^-` where `T = 1 - \tau H = T^+ - T^-` and `T^-` elements are non-negative.
        self.neg_propogator = -( (propogator < 0) * propogator)


    def mat_fn_diag(self, b):
        '''Calculate a diagonal Hamiltonian matrix element.

.. warning::

    Virtual member.  Must be appropriately implemented in a subclass.

:type b: iterable of single-particle basis functions
:param b: a many-particle basis function, `|b\\rangle`

:rtype: float
:returns: `\langle b|H|b \\rangle`.
'''

        err = 'Should not be calling the base matrix element functions'
        raise RuntimeError(err)

    def mat_fn_offdiag(self, b1, b2):
        '''Calculate an off-diagonal Hamiltonian matrix element.

.. warning::

    Virtual member.  Must be appropriately implemented in a subclass.

:type b1: iterable of single-particle basis functions
:param b1: a many-particle basis function, `|b_1\\rangle`
:type b2: iterable of single-particle basis functions
:param b2: a many-particle basis function, `|b_2\\rangle`

:rtype: float
:returns: `\langle b_1|H|b_2 \\rangle`.
'''

        err = 'Should not be calling the base matrix element functions'
        raise RuntimeError(err)

    def eigh(self):
        ''':returns: (eigenvalues, eigenvectors) of the Hamiltonian matrix.'''

        return numpy.linalg.eigh(self.hamil)

    def eigvalsh(self):
        ''':returns: eigenvalues of the Hamiltonian matrix.'''

        return numpy.linalg.eigvalsh(self.hamil)

    def negabs_off_diagonal_elements(self):
        '''Set off-diagonal elements of the Hamiltonian matrix to be negative.

This converts the Hamiltonian into the lesser sign-problem matrix discussed by
Spencer, Blunt and Foulkes.
'''

        for i in range(self.nbasis):
            for j in range(i+1,self.nbasis):
                self.hamil[i][j] = -abs(self.hamil[i][j])
                self.hamil[j][i] = -abs(self.hamil[j][i])

    def negabs_diagonal_elements(self):
        '''Set diagonal elements of the Hamiltonian matrix to be negative.

This, when called after negabs_offdiagonal_elements, converts the Hamiltonian
into the greater sign-problem matrix discussed by Spencer, Blunt and Foulkes.
'''

        for i in range(self.nbasis):
            self.hamil[i][i] = -abs(self.hamil[i][i])

    def propogate(self, pos_psips, neg_psips):
        '''Propogates a psip (psi-particle) distribution for a single timestep.

:type pos_psips: 1D array or list (length nbasis)
:param pos_psips: distribution of positive psips at time `t = n\\tau` on the many-fermion basis set.
:type neg_psips: 1D array or list (length nbasis)
:param neg_psips: distribution of negative psips at time `t = n\\tau` on the many-fermion basis set.

:returns: (next_pos_psips, next_neg_psips) --- positive and negative psip distributions at time `t=(n+1)\\tau`.
'''

        next_pos_psips = numpy.dot(self.pos_propogator, pos_psips) + numpy.dot(self.neg_propogator, neg_psips)
        next_neg_psips = numpy.dot(self.pos_propogator, neg_psips) + numpy.dot(self.neg_propogator, pos_psips)

        return (next_pos_psips, next_neg_psips)


#--- Excitations between many-particle basis functions ---

def hartree_excitation(p1, p2):
    '''Find the excitation connecting two Hartree products.

:type p1: iterable of single-particle basis functions
:param p1: a Hartree product basis function
:type p2: iterable of single-particle basis functions
:param p2: a many-particle basis function

:returns: (from_1, to_1) where:

    from_1
        list of single-particle basis functions excited from p1
    to_2
        list of single-particle basis functions excited into p2
'''


    # Order matters in Hartree products, so the differing spin-orbitals
    # must appear in the same place.

    from_1 = []
    to_2 = []
    for i in range(len(p1)):
        if p1[i] != p2[i]:
            from_1.append(p1[i])
            to_2.append(p2[i])

    return (from_1, to_2)

def determinant_excitation(d1, d2):
    '''Find the excitation connecting two Slater determinants.

:type p1: iterable of single-particle basis functions
:param p1: a Slater determinant basis function
:type p2: iterable of single-particle basis functions
:param p2: a Slater determinant basis function

:returns: (from_1, to_1, nperm) where:

    from_1
        list of single-particle basis functions excited from d1
    to_2
        list of single-particle basis functions excited into d2
    nperm
        number of permutations required to align the two determinants such
        that the orders of single-particle basis functions are in maximum
        agreement.  This, in general, is not the minimal possible number of
        permutations but the parity of the permutations, which is all that is
        required for calculating matrix elements, is correct.
'''

    # Get excitation.
    # Also work out the number of permutations required to line up the two
    # derminants.  We do this by counting the number of permutations
    # required to move the spin-orbitals to the 'end' of each determinant.
    from_1 = []
    to_2 = []
    nperm = 0
    nfound = 0
    for (indx, basis) in enumerate(d1):
        if basis not in d2:
            from_1.append(basis)
            # Number of permutations required to move basis fn to the end.
            # Have to take into account if we've already moved one orbital
            # to the end.
            nperm += len(d1) - indx - 1 + nfound
            nfound += 1
    nfound = 0
    # Ditto for second determinant.
    for (indx, basis) in enumerate(d2):
        if basis not in d1:
            to_2.append(basis)
            nperm += len(d2) - indx - 1 + nfound
            nfound += 1

    return (from_1, to_2, nperm)

def permanent_excitation(p1, p2):
    '''Find the excitation connecting two permanents.

:type p1: iterable of single-particle basis functions
:param p1: a permanent basis function
:type p2: iterable of single-particle basis functions
:param p2: a permanent basis function

:returns: (from_1, to_1) where:

    from_1
        list of single-particle basis functions excited from p1
    to_2
        list of single-particle basis functions excited into p2
'''

    # Get excitation.
    # No sign change associated with permuting spin orbitals in a permanent
    # in order to line them up, so don't need to count the permutations
    # required.
    from_1 = []
    to_2 = []
    for (indx, basis) in enumerate(p1):
        if basis not in p2:
            from_1.append(basis)
    for (indx, basis) in enumerate(p2):
        if basis not in p1:
            to_2.append(basis)

    return (from_1, to_2)
