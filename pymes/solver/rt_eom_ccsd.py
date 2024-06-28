"""
This module contains the class RT_EOM_CCSD used for real time dynamics 
using the similarity-transformed Hamiltonian and the Cauchy integral formula
"""

import numpy as np
import time
from scipy.linalg import eig

from pymes.solver.feast_eom_ccsd import FEAST_EOM_CCSD, get_gauss_legendre_quadrature, normalize_amps
from pymes.log import print_title, print_logging_info

class RT_EOM_CCSD(FEAST_EOM_CCSD):
    """
    This class implements the real-time dynamics based 
    on the following Cauchy integral algorithm to propagate 
    the linear ansatz.
    exp(iHt) = \oint_C exp(Z)/(ZI-iHt) dZ

    It is a subclass of the FEAST_EOM_CCSD class

    Attributes
    ----------
    self.u_singles 
    self.u_doubles
    """

    def __init__(self, no, e_c=0., e_r=1, dt=0.1, tol=1e-12, max_iter=100,     
                 **kwargs):
        """
        Initialize the RT_EOM_CCSD object.

        Parameters
        ----------
        no: int, the number of occupied orbitals
        e_c: float, energy center
        e_r: float, radius of the energy window
        dt: float, the step size of time 
        max_iter : int, optional
            The maximum number of linear solver iterations.
        tol : float, optional
            The tolerance to stop the linear solver iterations.
        """
        self.no = no
        # energy center
        self.e_c = e_c
        # energy window radius
        self.e_r = e_r

        self.max_iter = max_iter
        self.tol = tol
        self.linear_solver = "Jacobi"

        # stored u vectors
        self.u_singles = None
        self.u_doubles = None

    def dump_log(self):
        """
        Dump the log of the FEAST algorithm.
        """
        pass

    def solve(self, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij, dt=0.1, u_singles=None,
              u_doubles=None):
        """
        Solve the EOM-CCSD equations using the non-Hermitian FEAST algorithm.
        """
        print_title("RT-EOM-CCSD Solver")
        time_init = time.time()

        # create initial guesses
        no = self.no
        t_epsilon_i = t_fock_dressed_pq.diagonal()[:no]
        t_epsilon_a = t_fock_dressed_pq.diagonal()[no:]
        nv = t_epsilon_a.shape[0]

        print_logging_info("Initialising u tensors...", level=1)
        if u_doubles is None or u_singles is None:
            raise RuntimeError("No initial state specified!")
        self.u_singles = [u_singles]
        self.u_doubles = [u_doubles]
        diag_ai = self.get_diag_singles(t_fock_dressed_pq, dict_t_V_dressed, t_T_abij).to_nparray()
        diag_abij = self.get_diag_doubles(t_fock_dressed_pq, dict_t_V_dressed, t_T_abij).to_nparray()

        
        # the solution to a set of linear systems are repeated
        # separating into real and imag problems can save some computation
        # for now, will use the full contour integral.
        # gauss-legrendre quadrature
        x, w = get_gauss_legendre_quadrature(8) 
        theta = -np.pi * x 
        # the quadrature points
        z = (self.e_c*1j + self.e_r * np.exp(1j * theta))*dt

        Q_singles = [np.zeros(diag_ai.shape, dtype=complex)]
        Q_doubles = [np.zeros(diag_abij.shape, dtype=complex)]

        time_iter_init = time.time()

        # solve for the linear system (Z_e-H)Qe = e^(Z_e)Y
        for e in range(len(z)):
            print_logging_info(f"e = {e}, z = {z[e]}, theta = {theta[e]}, w = {w[e]}", level=1)
            Qe_singles, Qe_doubles = self._jacobi(0, z[e], diag_ai, diag_abij, 
                                                  t_fock_dressed_pq, 
                                                  dict_t_V_dressed, t_T_abij, 
                                                  phase=np.exp(z[e]), is_rt=True, dt=dt)
        
            Q_singles[0] -= w[e]/2 * (self.e_r * np.exp(1j * theta[e])
                                        * Qe_singles)
            Q_doubles[0] -= w[e]/2 * (self.e_r * np.exp(1j * theta[e]) 
                                        * Qe_doubles)
        
        # check convergence
        u_norm= np.tensordot(np.conj(Q_singles[0]), Q_singles[0], axes=2)
        u_norm += np.tensordot(np.conj(Q_doubles[0]), Q_doubles[0], axes=4)
        print_logging_info("Norm of new u vec before normalization = ", u_norm)
        self.u_singles = Q_singles
        self.u_doubles = Q_doubles
        for l in range(len(self.u_singles)):
            self.u_singles[l], self.u_doubles[l] = normalize_amps(self.u_singles[l], self.u_doubles[l])

        u_norm= np.tensordot(np.conj(Q_singles[0]), Q_singles[0], axes=2)
        u_norm += np.tensordot(np.conj(Q_doubles[0]), Q_doubles[0], axes=4)
        print_logging_info("Norm of new u vec after normalization = ", u_norm)
        time_end = time.time()
        print_logging_info(f"RT-EOM-CCSD finished in {time_end - time_init:.2f} seconds.", level=0)

        return Q_singles[0], Q_doubles[0]
