"""
This module contains the class FEAST_EOM_CCSD to solve the EOM-CCSD equations using the FEAST algorithm.

The class FEAST_EOM_CCSD is a subclass of the class EOM_CCSD. It implements the method solve() to solve the EOM-CCSD equations using the FEAST algorithm.
"""

import numpy as np
import time
from scipy.linalg import eig

import ctf

from pymes.solver.eom_ccsd import EOM_CCSD
from pymes.log import print_title, print_logging_info


class FEAST_EOM_CCSD(EOM_CCSD):
    """
    This class implements the FEAST algorithm to solve the EOM-CCSD equations.

    Attributes
    ----------
    eigvals : ndarray
        The eigenvalues of the EOM-CCSD equations.
    eigvecs : ndarray
        The eigenvectors of the EOM-CCSD equations.
    """

    def __init__(self, no, e_c=0., e_r=1, max_iter=100, tol=1e-12, **kwargs):
        """
        Initialize the FEAST_EOM_CCSD object.

        Parameters
        ----------
        ccsd : CCSD
            The CCSD object.
        nroots : int, optional
            The number of eigenvalues and eigenvectors to compute.
        maxiter : int, optional
            The maximum number of iterations.
        tol : float, optional
            The tolerance to stop the iterations.
        """
        self.no = no
        # energy center
        self.e_c = e_c
        # energy window radius
        self.e_r = e_r
        # size of trial space (number of vectors)
        self.n_trial = 10

        self.max_iter = max_iter
        self.tol = tol

        # stored u vectors
        self.u_singles = []
        self.u_doubles = []

    def dump_log(self):
        """
        Dump the log of the FEAST algorithm.
        """
        pass

    def solve(self, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij):
        """
        Solve the EOM-CCSD equations using the non-Hermitian FEAST algorithm.
        """
        print_title("FEAST-EOM-CCSD Solver")
        time_init = time.time()

        # create initial guesses
        no = self.no
        t_epsilon_i = t_fock_dressed_pq.diagonal()[:no]
        t_epsilon_a = t_fock_dressed_pq.diagonal()[no:]
        nv = t_epsilon_a.shape[0]
        t_D_ai = ctf.tensor([nv, no])
        t_D_abij = ctf.tensor(t_T_abij.shape)
        t_D_ai.i("ai") << t_epsilon_i.i("i") - t_epsilon_a.i("a")
        t_D_abij.i("abij") << t_epsilon_i.i("i") + t_epsilon_i.i("j") \
                              - t_epsilon_a.i("a") - t_epsilon_a.i("b")

        print_logging_info("Initialising u tensors...", level=1)
        for l in range(self.n_trial):
            self.u_singles.append(ctf.astensor(np.random.rand(*t_D_ai.shape)))
            self.u_doubles.append(ctf.astensor(np.random.rand(*t_D_abij.shape)))

        # gauss-legrendre quadrature
        x, w = get_gauss_legendre_quadrature(10) 
        theta = -np.pi / 2 * (x - 1)
        z = self.e_c + self.e_r * np.exp(1j * theta)

        # start iteratons
        e_norm_prev = 1e10
        for i in range(self.max_iter):
            self.Q_singles = [ctf.zeros(t_D_ai.shape)] * self.n_trial
            self.Q_doubles = [ctf.zeros(t_D_abij.shape)] * self.n_trial
            time_iter_init = time.time()
            #self.u_singles, self.u_doubles = self.QR(self.u_singles, self.u_doubles)

            # solve for the linear system (z-H)Q = Y at z = z_e
            for e in range(len(z)//2):
                Qe_singles, Qe_doubles = self._solve_linear(z[e], self.u_singles, self.u_doubles, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij)
                for l in range(self.n_trial):
                    self.Q_singles[l] -= w[e]/2 * ctf.real(self.e_r * np.exp(1j * theta[e]) * Qe_singles[l])
                    self.Q_doubles[l] -= w[e]/2 * ctf.real(self.e_r * np.exp(1j * theta[e]) * Qe_doubles[l])
            # compute the projected Hamiltonian
            H_proj = np.zeros((self.n_trial, self.n_trial))
            B = np.zeros((self.n_trial, self.n_trial))
            w_singles = [ctf.zeros(self.u_singles[0].shape)] * self.n_trial
            w_doubles = [ctf.zeros(self.u_doubles[0].shape)] * self.n_trial
            for i in range(self.n_trial):
                w_singles[i] = self.update_singles(t_fock_dressed_pq,
                                                   dict_t_V_dressed, self.Q_singles[i],
                                                   self.Q_doubles[i], t_T_abij)
                w_doubles[i] = self.update_doubles(t_fock_dressed_pq,
                                                   dict_t_V_dressed, self.Q_singles[i],
                                                   self.Q_doubles[i], t_T_abij)
                for j in range(i):
                    H_proj[j, i] = ctf.einsum("ai, ai->", self.Q_singles[j], w_singles[i]) \
                              + ctf.einsum("abij, abij->", self.Q_doubles[j], w_doubles[i])
                    H_proj[i, j] = ctf.einsum("ai, ai->", self.Q_singles[i], w_singles[j]) \
                              + ctf.einsum("abij, abij->", self.Q_doubles[i], w_doubles[j])
                    B[i, j] = ctf.einsum("ai, ai->", self.Q_singles[i], self.Q_singles[j]) \
                                + ctf.einsum("abij, abij->", self.Q_doubles[i], self.Q_doubles[j])
                    B[j, i] = B[i, j]
                H_proj[l, l] = ctf.einsum("ai, ai->", self.Q_singles[i], w_singles[i]) \
                          + ctf.einsum("abij, abij->", self.Q_doubles[i], w_doubles[i])
                B[l, l] = ctf.einsum("ai, ai->", self.Q_singles[i], self.Q_singles[i]) \
                            + ctf.einsum("abij, abij->", self.Q_doubles[i], self.Q_doubles[i])
            # solve the eigenvalue problem

            eigvals, eigvecs = eig(H_proj, B)

            # update u_singles and u_doubles
            for l in range(self.n_trial):
                self.u_singles[l] = ctf.zeros(self.u_singles[0].shape)
                self.u_doubles[l] = ctf.zeros(self.u_doubles[0].shape)
                for i in range(self.n_trial):
                    self.u_singles[l] += eigvecs[i, l] * self.Q_singles[i]
                    self.u_doubles[l] += eigvecs[i, l] * self.Q_doubles[i]
            
            # check convergence
            e_norm = np.linalg.norm(eigvals)
            if np.abs(e_norm - e_norm_prev) < self.tol:
                break
            else:
                print_logging_info("FEAST iteration did not converge!")
                print_logging_info(f"Norm of eigenvalues: {e_norm}, Difference: {np.abs(e_norm - e_norm_prev)}", level=1)

            e_norm_prev = e_norm


        time_end = time.time()
        print_logging_info(f"FEAST-EOM-CCSD finished in {time_end - time_init:.2f} seconds.", level=0)
        self.e_excit = eigvals

        return eigvals
        

    def _solve_linear(self, ze, u_singles, u_doubles, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij):
        """
        Solve the linear system (z-H)Q = Y.
        ze : complex
        """
        Qe_singles = [ctf.zeros(u_singles[0].shape, dtype=complex)] * self.n_trial
        Qe_doubles = [ctf.zeros(u_doubles[0].shape, dtype=complex)] * self.n_trial
        for i in range(20):
            norm_singles = 0
            norm_doubles = 0
            for l in range(len(u_singles)):
                delta_singles = ctf.tensor(u_singles[0].shape, dtype=complex)
                delta_singles = ze * Qe_singles[l]
                delta_singles -= self.update_singles(t_fock_dressed_pq,
                                                   dict_t_V_dressed, Qe_singles[l],
                                                   Qe_doubles[l], t_T_abij)
                delta_singles -= u_singles[l]
                
                delta_doubles = ctf.tensor(u_doubles[0].shape, dtype=complex)
                delta_doubles = ze * Qe_doubles[l]
                delta_doubles -= self.update_doubles(t_fock_dressed_pq,
                                                   dict_t_V_dressed, Qe_singles[l],
                                                   Qe_doubles[l], t_T_abij)
                delta_doubles -= u_doubles[l]

                Qe_singles[l] += delta_singles
                Qe_doubles[l] += delta_doubles

                # check convergence
                norm_singles += np.linalg.norm(delta_singles.to_nparray())
                norm_doubles += np.linalg.norm(delta_doubles.to_nparray())

            if norm_singles + norm_doubles < self.tol:
                break
            else:
                print_logging_info(f"Norm of delta_singles: {norm_singles}, Norm of delta_doubles: {norm_doubles}", level=0)

        return Qe_singles, Qe_doubles


def get_gauss_legendre_quadrature(n):
    """
    Get the Gauss-Legendre quadrature points and weights.

    Parameters
    ----------
    n : int
        The number of quadrature points.

    Returns
    -------
    x : ndarray
        The quadrature points.
    w : ndarray
        The quadrature weights.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w