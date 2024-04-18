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

    def __init__(self, no, e_c=0., e_r=1, n_trial=5, max_iter=100, tol=1e-12, **kwargs):
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
        self.n_trial = n_trial

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
            self.u_singles.append(0.5-ctf.astensor(np.random.rand(*t_D_ai.shape)))
            self.u_doubles.append(0.5-ctf.astensor(np.random.rand(*t_D_abij.shape)))
        #self.u_singles, self.u_doubles = self.QR(self.u_singles, self.u_doubles)

        # gauss-legrendre quadrature
        x, w = get_gauss_legendre_quadrature(8) 
        theta = -np.pi / 2 * (x - 1)
        z = self.e_c + self.e_r * np.exp(1j * theta)

        # start iteratons
        e_norm_prev = 1e10
        for iter in range(self.max_iter):
            self.Q_singles = [ctf.tensor(t_D_ai.shape, dtype=float) for _ in  range(self.n_trial)]
            self.Q_doubles = [ctf.tensor(t_D_abij.shape, dtype=float) for _ in  range(self.n_trial)]

            time_iter_init = time.time()
            #self.u_singles, self.u_doubles = self.QR(self.u_singles, self.u_doubles)

            # solve for the linear system (z-H)Q = Y at z = z_e
            for e in range(len(z)):
                print_logging_info(f"e = {e}, z = {z[e]}, theta = {theta[e]}, w = {w[e]}", level=1)
                for l in range(self.n_trial):
                    Qe_singles, Qe_doubles = self._solve_linear(l, z[e], t_fock_dressed_pq, dict_t_V_dressed, t_T_abij)
                    self.Q_singles[l] -= w[e]/2 * ctf.real(self.e_r * np.exp(1j * theta[e]) * Qe_singles)
                    self.Q_doubles[l] -= w[e]/2 * ctf.real(self.e_r * np.exp(1j * theta[e]) * Qe_doubles)
            
            # normalize the trial vectors
            for l in range(self.n_trial):
                self.Q_singles[l], self.Q_doubles[l] = normalize_amps(self.Q_singles[l], self.Q_doubles[l])

            #self.Q_singles, self.Q_doubles = self.QR(self.Q_singles, self.Q_doubles)
            # compute the projected Hamiltonian
            H_proj = np.zeros((self.n_trial, self.n_trial))
            B = np.zeros((self.n_trial, self.n_trial))
            w_singles = [ctf.zeros(self.u_singles[0].shape) for _ in range(self.n_trial)]
            w_doubles = [ctf.zeros(self.u_doubles[0].shape) for _ in range(self.n_trial)]
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
                H_proj[i, i] = ctf.einsum("ai, ai->", self.Q_singles[i], w_singles[i]) \
                          + ctf.einsum("abij, abij->", self.Q_doubles[i], w_doubles[i])
                B[i, i] = ctf.einsum("ai, ai->", self.Q_singles[i], self.Q_singles[i]) \
                            + ctf.einsum("abij, abij->", self.Q_doubles[i], self.Q_doubles[i])
            # solve the eigenvalue problem
            eigvals, eigvecs = eig(H_proj, B)

            # update u_singles and u_doubles
            for l in range(self.n_trial):
                self.u_singles[l] = ctf.zeros(self.u_singles[0].shape)
                self.u_doubles[l] = ctf.zeros(self.u_doubles[0].shape)
                for i in range(self.n_trial):
                    self.u_singles[l] += np.real(eigvecs[i, l]) * self.Q_singles[i]
                    self.u_doubles[l] += np.real(eigvecs[i, l]) * self.Q_doubles[i]
            
            # check convergence
            e_norm = np.linalg.norm(eigvals)
            if np.abs(e_norm - e_norm_prev) < self.tol:
                break
            else:
                print_logging_info(f"Iter = {iter}, Eigenvalues: {eigvals}", level=1)
                print_logging_info(f"Norm of eigenvalues: {e_norm}, Difference: {np.abs(e_norm - e_norm_prev)}", level=1)

            e_norm_prev = e_norm


        time_end = time.time()
        print_logging_info(f"FEAST-EOM-CCSD finished in {time_end - time_init:.2f} seconds.", level=0)
        self.e_excit = eigvals

        return eigvals
        

    def _solve_linear(self, l, ze, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij):
        """
        Solve the linear system (z-H)Q = Y.
        Default algorithm is BiCGSTAB unconditioned.

        Parameters
        ----------
        l : int, trial vector index
        ze : complex
            The shift value.
        t_fock_dressed_pq : ctf.tensor, shape (norb, norb), dtype=float, dressed Fock matrix
        dict_t_V_dressed : dict, dressed two-body integrals
        t_T_abij : ctf.tensor, shape (norb, norb, norb, norb), dtype=float, T2 amplitudes


        Returns
        -------
        Qe_singles : dtype=ctf.tensor, single excitation vectors
        Qe_doubles : dtype=ctf.tensor, double excitation vectors
        """

        Qe_singles = ctf.zeros(self.u_singles[0].shape, dtype=complex)
        Qe_doubles = ctf.zeros(self.u_doubles[0].shape, dtype=complex)
        t_D_ai = ctf.tensor(self.u_singles[0].shape, dtype=float)
        epsilons_a = t_fock_dressed_pq.diagonal()[self.no:]
        epsilons_i = t_fock_dressed_pq.diagonal()[:self.no]
        t_D_ai.i("ai") << epsilons_a.i("a") - epsilons_i.i("i") 
        t_D_abij = ctf.tensor(self.u_doubles[0].shape, dtype=float)
        t_D_abij.i("abij") <<  epsilons_a.i("a") + epsilons_a.i("b")\
                               -epsilons_i.i("i") - epsilons_i.i("j")  

        def _get_residual(trial_singles, trial_doubles):
            """
            Get the residual of the linear system (z-H)Q = Y
            for the l-th trial vector.
            """
            delta_singles = ctf.zeros(self.u_singles[0].shape, dtype=complex) 
            delta_doubles = ctf.zeros(self.u_doubles[0].shape, dtype=complex) 
            delta_singles += self.u_singles[l]
            delta_singles -= ze * trial_singles[l]
            delta_singles += self.update_singles(t_fock_dressed_pq,
                                               dict_t_V_dressed, trial_singles,
                                               trial_doubles, t_T_abij)
            
            delta_doubles = ctf.tensor(self.u_doubles[0].shape, dtype=complex)
            delta_doubles += self.u_doubles[l]
            delta_doubles -= ze * trial_doubles[l]
            delta_doubles += self.update_doubles(t_fock_dressed_pq,
                                               dict_t_V_dressed, trial_singles,
                                               trial_doubles, t_T_abij)
            return delta_singles, delta_doubles
        
        delta_singles, delta_doubles = _get_residual(Qe_singles, Qe_doubles)
        rho = ctf.einsum("ai, ai->", ctf.conj(delta_singles), delta_singles)
        rho += ctf.einsum("abij, abij->", ctf.conj(delta_doubles), delta_doubles)
        p_singles = delta_singles.copy()
        p_doubles = delta_doubles.copy()
        r0_singles = delta_singles.copy()
        r0_doubles = delta_doubles.copy()
        r_singles = delta_singles.copy()
        r_doubles = delta_doubles.copy()
        for i in range(150):
            v_singles, v_doubles = _get_residual(p_singles, p_doubles)
            alpha = ctf.einsum("ai, ai->", ctf.conj(r0_singles), v_singles)
            alpha += ctf.einsum("abij, abij->", ctf.conj(r0_doubles), v_doubles)
            alpha = rho / alpha
            h_singles = Qe_singles + alpha * p_singles 
            h_doubles = Qe_doubles + alpha * p_doubles
            s_singles = r_singles - alpha * v_singles
            s_doubles = r_doubles - alpha * v_doubles
            s_norm = ctf.einsum("ai, ai->", ctf.conj(s_singles), s_singles)
            s_norm += ctf.einsum("abij, abij->", ctf.conj(s_doubles), s_doubles)
            if np.abs(s_norm) < 1e-8:
                print_logging_info(f"i = {i}, converged for s_norm", level=2)
                Qe_singles = h_singles
                Qe_doubles = h_doubles
                break
            t_singles, t_doubles = _get_residual(s_singles, s_doubles)
            omega = ctf.einsum("ai, ai->", ctf.conj(t_singles), s_singles)
            omega += ctf.einsum("abij, abij->", ctf.conj(t_doubles), s_doubles)
            t_norm = ctf.einsum("ai, ai->", ctf.conj(t_singles), t_singles)
            t_norm += ctf.einsum("abij, abij->", ctf.conj(t_doubles), t_doubles)
            omega /= t_norm
            Qe_singles = h_singles + omega * s_singles
            Qe_doubles = h_doubles + omega * s_doubles
            r_singles = s_singles - omega * t_singles
            r_doubles = s_doubles - omega * t_doubles
            r_norm = ctf.einsum("ai, ai->", ctf.conj(r_singles), r_singles)
            r_norm += ctf.einsum("abij, abij->", ctf.conj(r_doubles), r_doubles)
            if np.abs(r_norm) < 1e-8:
                print_logging_info(f"i = {i}, converged for r_norm", level=2)
                break
            #else:
            #    print_logging_info(f"i = {i}, r_norm: {np.abs(r_norm)}, s_norm: {np.abs(s_norm)}", level=2)
            # update rho and save the previous values
            rho_old = rho
            rho = ctf.einsum("ai, ai->", ctf.conj(r0_singles), r_singles)
            rho += ctf.einsum("abij, abij->", ctf.conj(r0_doubles), r_doubles)
            beta = (rho/rho_old) * (alpha/omega)
            p_singles = r_singles + beta * (p_singles - omega * v_singles)
            p_doubles = r_doubles + beta * (p_doubles - omega * v_doubles)
        print_logging_info(f"Linear Solver: l = {l}, tot_iter = {i}, s_norm = {np.abs(s_norm)}, r_norm = {np.abs(r_norm)}", level=2)
        return Qe_singles, Qe_doubles

    def solve_test(self, nv):
        """
        Solve the EOM-CCSD equations using the non-Hermitian FEAST algorithm.
        """
        print_title("FEAST-EOM-CCSD Solver Test")
        time_init = time.time()
        no = self.no

        ham = self.construct_fake_ham(nv, no)
        e_target, v_target = np.linalg.eig(ham)

        lowest_ex_ind_target = e_target.argsort()
        e_target = e_target[lowest_ex_ind_target]
        v_target = v_target[lowest_ex_ind_target]
        # create initial guesses
        s_shape = [nv, no]
        d_shape = [nv, nv, no, no]

        print_logging_info("Initialising u tensors...", level=1)
        for l in range(self.n_trial):
            self.u_singles.append(0.5-ctf.astensor(np.random.rand(*s_shape)))
            self.u_doubles.append(0.5-ctf.astensor(np.random.rand(*d_shape)))

        # gauss-legrendre quadrature
        x, w = get_gauss_legendre_quadrature(8) 
        theta = -np.pi / 2 * (x - 1)
        z = self.e_c + self.e_r * np.exp(1j * theta)

        # start iteratons
        e_norm_prev = 1e10
        for iter in range(self.max_iter):
            self.Q_singles = [ctf.tensor(s_shape, dtype=float) for _ in  range(self.n_trial)]
            self.Q_doubles = [ctf.tensor(d_shape, dtype=float) for _ in  range(self.n_trial)]
            time_iter_init = time.time()
            #self.u_singles, self.u_doubles = self.QR(self.u_singles, self.u_doubles)

            # solve for the linear system (z-H)Q = Y at z = z_e
            for e in range(len(z)):
                print_logging_info(f"e = {e}, z = {z[e]}, theta = {theta[e]}, w = {w[e]}", level=1)
                Qe_singles, Qe_doubles = self._solve_linear_test(z[e], self.u_singles, self.u_doubles, ham)
                #Qe_singles_conj, Qe_doubles_conj = self._solve_linear(np.conj(z[e]), self.u_singles, self.u_doubles, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij)
                for l in range(self.n_trial):
                    self.Q_singles[l] -= w[e]/2 * ctf.real(self.e_r * np.exp(1j * theta[e]) * Qe_singles[l])
                    self.Q_doubles[l] -= w[e]/2 * ctf.real(self.e_r * np.exp(1j * theta[e]) * Qe_doubles[l])
            #self.Q_singles, self.Q_doubles = self.QR(self.Q_singles, self.Q_doubles)
            # compute the projected Hamiltonian
            H_proj = np.zeros((self.n_trial, self.n_trial))
            B = np.zeros((self.n_trial, self.n_trial))
            w_singles = [ctf.zeros(self.u_singles[0].shape) for _ in range(self.n_trial)]
            w_doubles = [ctf.zeros(self.u_doubles[0].shape) for _ in range(self.n_trial)]
            for i in range(self.n_trial):
                w_singles[i] = self.update_singles_test(ham,
                                                   self.Q_singles[i],
                                                   self.Q_doubles[i])
                w_doubles[i] = self.update_doubles_test(ham,
                                                   self.Q_singles[i],
                                                   self.Q_doubles[i])
                for j in range(i):
                    H_proj[j, i] = ctf.einsum("ai, ai->", self.Q_singles[j], w_singles[i]) \
                              + ctf.einsum("abij, abij->", self.Q_doubles[j], w_doubles[i])
                    H_proj[i, j] = ctf.einsum("ai, ai->", self.Q_singles[i], w_singles[j]) \
                              + ctf.einsum("abij, abij->", self.Q_doubles[i], w_doubles[j])
                    B[i, j] = ctf.einsum("ai, ai->", self.Q_singles[i], self.Q_singles[j]) \
                                + ctf.einsum("abij, abij->", self.Q_doubles[i], self.Q_doubles[j])
                    B[j, i] = B[i, j]
                H_proj[i, i] = ctf.einsum("ai, ai->", self.Q_singles[i], w_singles[i]) \
                          + ctf.einsum("abij, abij->", self.Q_doubles[i], w_doubles[i])
                B[i, i] = ctf.einsum("ai, ai->", self.Q_singles[i], self.Q_singles[i]) \
                            + ctf.einsum("abij, abij->", self.Q_doubles[i], self.Q_doubles[i])
            # solve the eigenvalue problem

            eigvals, eigvecs = eig(H_proj, B)

            # update u_singles and u_doubles
            self.u_singles = []
            self.u_doubles = []
            # check the number of eigenvalues in the energy window
            valid_eig = np.where(np.in1d(eigvals[eigvals < self.e_c + self.e_r], eigvals[eigvals > self.e_c - self.e_r]))[0]
            print(eigvals)
            #if len(valid_eig) == 0:
            #    valid_eig = np.arange(self.n_trial)

            for l in range(self.n_trial):
                u_singles = ctf.tensor(s_shape, dtype=float)
                u_doubles = ctf.tensor(d_shape, dtype=float)
                for i in range(self.n_trial):
                    u_singles += np.real(eigvecs[i, l]) * self.Q_singles[i]
                    u_doubles += np.real(eigvecs[i, l]) * self.Q_doubles[i]
                self.u_singles.append(u_singles)
                self.u_doubles.append(u_doubles)
            
            
            # check convergence
            e_norm = np.linalg.norm(eigvals[valid_eig])
            if np.abs(e_norm - e_norm_prev) < self.tol:
                break
            else:
                print_logging_info(f"Iter = {iter}, Eigenvalues: {eigvals}", level=1)
                print_logging_info(f"Norm of eigenvalues: {e_norm}, Difference: {np.abs(e_norm - e_norm_prev)}", level=1)

            e_norm_prev = e_norm


        time_end = time.time()
        print_logging_info(f"FEAST-EOM-CCSD finished in {time_end - time_init:.2f} seconds.", level=0)
        self.e_excit = eigvals

        return eigvals
    def _solve_linear_test(self, ze, u_singles, u_doubles, ham):
        """
        Solve the linear system (z-H)Q = Y.
        ze : complex
        """
        Qe_singles = [ctf.zeros(u_singles[0].shape, dtype=complex) for _ in range(self.n_trial)]
        Qe_doubles = [ctf.zeros(u_doubles[0].shape, dtype=complex) for _ in range(self.n_trial)]
        Qe_singles_ref = [ctf.zeros(u_singles[0].shape, dtype=complex) for _ in range(self.n_trial)]
        Qe_doubles_ref = [ctf.zeros(u_doubles[0].shape, dtype=complex) for _ in range(self.n_trial)]
        # combine u_singles and u_doubles to Y 
        nv = u_singles[0].shape[0]
        no = u_singles[0].shape[1]
        Y = np.zeros([self.n_trial, nv*no+nv*nv*no*no], dtype=complex) 
        for l in range(self.n_trial):
            Y[l, :] = np.concatenate((u_singles[l].to_nparray().flatten(), u_doubles[l].to_nparray().flatten()))
        
        Q_e = np.linalg.solve(ze*np.diag(np.ones(ham.shape[0])) - ham, Y.T).T 
        for l in range(self.n_trial):
            Qe_singles_ref[l] = ctf.astensor(Q_e[l][:nv*no].reshape(nv, no), dtype=complex)
            Qe_doubles_ref[l] = ctf.astensor(Q_e[l][nv*no:].reshape(nv, nv, no, no), dtype=complex)
        for i in range(100):
            norm_singles = 0
            norm_doubles = 0
            for l in range(len(u_singles)):
                #X = np.concatenate((Qe_singles[l].to_nparray().flatten(), Qe_doubles[l].to_nparray().flatten()))
                #Hx = (ze*np.diag(np.ones(ham.shape[0])) - ham).dot(X)
                #Hx -= Y[l]
                ##Hx /= -(ze*np.ones(len(Hx)) - ham.diagonal())
                #delta_singles = ctf.astensor(Hx[:nv*no].reshape(nv, no), dtype=complex)
                #delta_doubles = ctf.astensor(Hx[nv*no:].reshape(nv, nv, no, no), dtype=complex)
                delta_singles = ctf.tensor(u_singles[0].shape, dtype=complex)
                delta_singles = ze * Qe_singles[l]
                delta_singles -= self.update_singles_test(ham,
                                                   Qe_singles[l],
                                                   Qe_doubles[l])
                delta_singles -= u_singles[l]
                
                delta_doubles = ctf.tensor(u_doubles[0].shape, dtype=complex)
                delta_doubles = ze * Qe_doubles[l]
                delta_doubles -= self.update_doubles_test(ham,
                                                   Qe_singles[l],
                                                   Qe_doubles[l])
                delta_doubles -= u_doubles[l]

                Qe_singles[l] -= 0.1 * (delta_singles/(ze - ham.diagonal()[:nv*no].reshape(nv, no))) 
                Qe_doubles[l] -= 0.1 * (delta_doubles/(ze - ham.diagonal()[nv*no:].reshape(nv*nv*no*no).reshape(nv, nv, no, no)))

                # check convergence
                norm_singles += np.linalg.norm(delta_singles.to_nparray())
                norm_doubles += np.linalg.norm(delta_doubles.to_nparray())

            if norm_singles + norm_doubles < self.tol:
                break
            #else:
        print_logging_info(f"|Delta Singles|: {norm_singles}, |Delta Doubles|: {norm_doubles}", level=2)

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

def normalize_amps(u_singles, u_doubles):
    norm_singles = ctf.einsum("ai, ai ->", u_singles, u_singles)
    norm_doubles = ctf.einsum("abij, abij ->", u_doubles, u_doubles)
    u_singles /= np.sqrt(norm_singles + norm_doubles)
    u_doubles /= np.sqrt(norm_singles + norm_doubles)
    return u_singles, u_doubles