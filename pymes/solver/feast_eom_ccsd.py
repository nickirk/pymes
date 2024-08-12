"""
This module contains the class FEAST_EOM_CCSD to 
solve the EOM-CCSD equations using the FEAST algorithm.

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

    def __init__(self, no, e_c=0., e_r=1, n_trial=5, max_iter=20, tol=1e-12, **kwargs):
        """
        Initialize the FEAST_EOM_CCSD object.

        Parameters
        ----------
        no: int, number of occupied orbitals
        e_c: float, the center of the energy window
        e_r: float, the radius of the energy window
        n_trial : int, optional
            The size of the trial space
        max_iter : int, optional
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
        self.n_excit = 2 

        self.max_iter = max_iter
        self.tol = tol
        self.linear_solver = "Jacobi"

        # stored u vectors
        self.u_singles = []
        self.u_doubles = []

        self.eigvals = np.array([self.e_c - self.e_r, self.e_c + self.e_r])
        self.eigvecs = None

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
        diag_ai = self.get_diag_singles(t_fock_dressed_pq, dict_t_V_dressed, t_T_abij).to_nparray()
        diag_abij = self.get_diag_doubles(t_fock_dressed_pq, dict_t_V_dressed, t_T_abij).to_nparray()

        print_logging_info("Initialising u tensors...", level=1)
        # TODO: more initialization schemes should be tested
        for l in range(self.n_excit):
            self.u_singles.append((0.5-(np.random.rand(*diag_ai.shape))))
            self.u_doubles.append((0.5-(np.random.rand(*diag_abij.shape)))*0.01)

        # normalize the trial vectors
        for l in range(len(self.u_singles)):
            self.u_singles[l], self.u_doubles[l] = normalize_amps(self.u_singles[l], self.u_doubles[l])
        # gauss-legrendre quadrature
        x, w = get_gauss_legendre_quadrature(8) 
        theta = -np.pi / 2 * (x - 1)
        z = self.e_c + self.e_r * np.exp(1j * theta)

        # start iteratons
        e_norm_prev = 1e10
        for iter in range(self.max_iter):
            self.Q_singles = [np.zeros(diag_ai.shape, dtype=float) for _ in  range(len(self.u_singles))]
            self.Q_doubles = [np.zeros(diag_abij.shape, dtype=float) for _ in  range(len(self.u_doubles))]

            time_iter_init = time.time()
            # normalize the trial vectors
            for l in range(len(self.u_singles)):
                self.u_singles[l], self.u_doubles[l] = normalize_amps(self.u_singles[l], self.u_doubles[l])

            # solve for the linear system (z-H)Q = Y at z = z_e
            for e in range(len(z)):
                print_logging_info(f"e = {e}, z = {z[e]}, theta = {theta[e]}, w = {w[e]}", level=1)
                for l in range(len(self.u_singles)):
                    if self.linear_solver.upper() == "BICGSTAB":
                        Qe_singles, Qe_doubles = self._bicgstab(l, z[e], t_fock_dressed_pq, dict_t_V_dressed, t_T_abij)
                    else:
                        Qe_singles, Qe_doubles = self._gcrotmk(l, z[e], diag_ai, diag_abij, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij)
                    self.Q_singles[l] -= w[e]/2 * np.real(self.e_r * np.exp(1j * theta[e]) * Qe_singles)
                    self.Q_doubles[l] -= w[e]/2 * np.real(self.e_r * np.exp(1j * theta[e]) * Qe_doubles)
            
            # compute the projected Hamiltonian
            H_proj = np.zeros((len(self.u_singles), len(self.u_singles)))
            B = np.zeros((len(self.u_singles), len(self.u_singles)))
            w_singles = [np.zeros(self.u_singles[0].shape) for _ in range(len(self.u_singles))]
            w_doubles = [np.zeros(self.u_doubles[0].shape) for _ in range(len(self.u_doubles))]
            for i in range(len(self.u_singles)):
                w_singles[i] = self.update_singles(t_fock_dressed_pq,
                                                   dict_t_V_dressed, ctf.astensor(self.Q_singles[i]),
                                                   ctf.astensor(self.Q_doubles[i]), t_T_abij).to_nparray()
                w_doubles[i] = self.update_doubles(t_fock_dressed_pq,
                                                   dict_t_V_dressed, ctf.astensor(self.Q_singles[i]),
                                                   ctf.astensor(self.Q_doubles[i]), t_T_abij).to_nparray()
                for j in range(i):
                    H_proj[j, i] = np.tensordot(self.Q_singles[j], w_singles[i], axes=2) \
                              + np.tensordot(self.Q_doubles[j], w_doubles[i], axes=4)
                    H_proj[i, j] = np.tensordot(self.Q_singles[i], w_singles[j], axes=2) \
                              + np.tensordot(self.Q_doubles[i], w_doubles[j], axes=4)
                    B[i, j] = np.tensordot(self.Q_singles[i], self.Q_singles[j], axes=2) \
                                + np.tensordot(self.Q_doubles[i], self.Q_doubles[j], axes=4)
                    B[j, i] = B[i, j]
                H_proj[i, i] = np.tensordot(self.Q_singles[i], w_singles[i], axes=2) \
                          + np.tensordot(self.Q_doubles[i], w_doubles[i], axes=4)
                B[i, i] = np.tensordot(self.Q_singles[i], self.Q_singles[i], axes=2) \
                            + np.tensordot(self.Q_doubles[i], self.Q_doubles[i], axes=4)
            # solve the eigenvalue problem
            self.eigvals, self.eigvecs = eig(H_proj, B)

            # update u_singles and u_doubles and to the trial vectors
            if len(self.u_singles) < self.n_trial:
                for l in range(len(self.eigvals)):
                    new_singles = np.zeros(self.u_singles[0].shape, dtype=float)
                    new_doubles = np.zeros(self.u_doubles[0].shape, dtype=float)
                    for i in range(len(self.eigvals)):
                        new_singles += np.real(self.eigvecs[i, l]) * self.Q_singles[i]
                        new_doubles += np.real(self.eigvecs[i, l]) * self.Q_doubles[i]
                    self.u_singles.append(new_singles)
                    self.u_doubles.append(new_doubles)
            else:
                for l in range(len(self.eigvals)):
                    for i in range(len(self.eigvals)):
                        self.u_singles[l] += np.real(self.eigvecs[i, l]) * self.Q_singles[i]
                        self.u_doubles[l] += np.real(self.eigvecs[i, l]) * self.Q_doubles[i]
            
            # check convergence
            e_norm = np.linalg.norm(self.eigvals)
            if np.abs(e_norm - e_norm_prev) < self.tol:
                break
            else:
                print_logging_info(f"Iter = {iter}, Eigenvalues: {self.eigvals}", level=1)
                print_logging_info(f"Norm of eigenvalues: {e_norm}, Difference: {np.abs(e_norm - e_norm_prev)}", level=1)

            e_norm_prev = e_norm


        time_end = time.time()
        print_logging_info(f"FEAST-EOM-CCSD finished in {time_end - time_init:.2f} seconds.", level=0)
        self.e_excit = self.eigvals

        return self.eigvals

    def get_residual(self, l, ze, trial_singles, trial_doubles, 
                      t_fock_dressed_pq, dict_t_V_dressed, t_T_abij,
                      phase=None, is_rt=False, dt=None):
        """
        Get the residual of the linear system (z-H)Q = Y
        for the l-th trial vector.
        """
        delta_singles = np.zeros(self.u_singles[0].shape, dtype=complex) 
        delta_doubles = np.zeros(self.u_doubles[0].shape, dtype=complex) 
        delta_singles += self.u_singles[l]
        if phase is not None:
            delta_singles *= phase
        delta_singles -= ze * trial_singles

        if is_rt and dt is not None:
            delta_singles += 1j*dt*self.update_singles(t_fock_dressed_pq,
                                           dict_t_V_dressed, ctf.astensor(trial_singles),
                                           ctf.astensor(trial_doubles), t_T_abij).to_nparray()
        else:
            delta_singles += self.update_singles(t_fock_dressed_pq,
                                               dict_t_V_dressed, ctf.astensor(trial_singles),
                                               ctf.astensor(trial_doubles), t_T_abij).to_nparray()
        
        delta_doubles = np.zeros(self.u_doubles[0].shape, dtype=complex)
        delta_doubles += self.u_doubles[l]
        if phase is not None:
            delta_doubles *= phase
        delta_doubles -= ze * trial_doubles
        if is_rt and dt is not None:
            delta_doubles += 1j*dt*self.update_doubles(t_fock_dressed_pq,
                                           dict_t_V_dressed, ctf.astensor(trial_singles),
                                           ctf.astensor(trial_doubles), t_T_abij).to_nparray()
        else:
            delta_doubles += self.update_doubles(t_fock_dressed_pq,
                                               dict_t_V_dressed, ctf.astensor(trial_singles),
                                               ctf.astensor(trial_doubles), t_T_abij).to_nparray()
        return delta_singles, delta_doubles

    def _opt_solver(self, l, ze, trial_singles, trial_doubles, 
                    t_fock_dressed_pq, dict_t_V_dressed, t_T_abij,
                    phase=None, is_rt=False, dt=None):
        """
        solve the linear system (z-H)Q = Y using scipy.optimize solvers
        """
        Qe_vec = np.concatenate((trial_singles.flatten(), trial_doubles.flatten()))

        def _get_residual(Qe_vec):
            """
            Get the residual of the linear system (z-H)Q = Y
            for the l-th trial vector.
            """
            Qe_singles = Qe_vec[:trial_singles.size].reshape(trial_singles.shape)
            Qe_doubles = Qe_vec[trial_singles.size:].reshape(trial_doubles.shape)
            delta_singles, delta_doubles = self.get_residual(
                l, ze, Qe_singles, Qe_doubles, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij,
                phase=phase, is_rt=is_rt, dt=dt)
            delta_vec = np.concatenate((delta_singles.flatten(), delta_doubles.flatten()))
            # calculate the norm of the residual
            return np.linalg.norm(delta_vec)

        from scipy.optimize import minimize
        res = minimize(_get_residual, Qe_vec, method='CG', tol=1e-4)
        Qe_vec = res.x
        Qe_singles = Qe_vec[:trial_singles.size].reshape(trial_singles.shape)
        Qe_doubles = Qe_vec[trial_singles.size:].reshape(trial_doubles.shape)
        
        return Qe_singles, Qe_doubles

    
    def _jacobi(self, l, ze, diag_ai, diag_abij, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij,
                phase=None, is_rt=False, dt=None, **kwargs):
        """
        Solve the linear system (z-H)Qe = Y.
        Parameters
        ----------
        l : int, trial vector index
        ze : complex
            The shift value.
        diag_ai : np.ndarray, shape (nv, no), dtype=float, diagonal elements of the singles dressed H matrix
        diag_abij : np.ndarray, shape (nv, nv, no, no), dtype=float, diagonal elements of the doubles dressed H matrix
        t_fock_dressed_pq : np.ndarray, shape (norb, norb), dtype=float, dressed Fock matrix
        dict_t_V_dressed : dict, dressed two-body integrals 
        t_T_abij : np.ndarray, shape (norb, norb, norb, norb), dtype=float, T2 amplitudes
        phase: complex, (z-H)Qe = phase * Y
        dt, the time step for the real-time propagation
        is_rt: bool, whether the solver is for real-time propagation
        """
        Qe_singles = np.zeros(self.u_singles[0].shape, dtype=complex)
        Qe_doubles = np.zeros(self.u_doubles[0].shape, dtype=complex)
        shift_abij = diag_abij
        shift_ai = diag_ai
        

        if is_rt and dt is not None:
            shift_abij = diag_abij*1j*dt 
            shift_ai = diag_ai*1j*dt
        
        for iter in range(200):
            delta_singles, delta_doubles = self.get_residual(
                l, ze, Qe_singles, Qe_doubles, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij,
                phase=phase, is_rt=is_rt, dt=dt)
            # preconditioner for the Jacobi method
            delta_singles /= (ze-shift_ai+0.01)
            delta_doubles /= (ze-shift_abij+0.01)
            Qe_singles += 0.01 * delta_singles 
            Qe_doubles += 0.01 * delta_doubles 
        print_logging_info(f"iter = {iter}, norm of delta_singles = {np.linalg.norm(delta_singles)}, norm of delta_doubles = {np.linalg.norm(delta_doubles)}", level=2)
        
        return Qe_singles, Qe_doubles

    def _gcrotmk(self, l, ze, diag_ai, diag_abij, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij,
                phase=None, is_rt=False, dt=None, **kwargs):
        from scipy.sparse.linalg import LinearOperator, gcrotmk
        from scipy.sparse import diags

        Qe_singles = np.zeros(self.u_singles[0].shape, dtype=complex)
        Qe_doubles = np.zeros(self.u_doubles[0].shape, dtype=complex)

        shift_abij = diag_abij
        shift_ai = diag_ai

        Qe_vec_init = np.concatenate((Qe_singles.flatten(), Qe_doubles.flatten()))
        b_vec = np.concatenate((self.u_singles[l].flatten(), self.u_doubles[l].flatten()), dtype=complex)
        if phase is not None:
            b_vec *= phase

        def matvec(Qe):
            """
            Matrix-vector product for the linear system (z-H)Q = Y
            """
            # unpack the vector
            trial_singles = Qe[:diag_ai.size].reshape(diag_ai.shape)
            trial_doubles = Qe[diag_ai.size:].reshape(diag_abij.shape)
            #delta_singles = np.zeros(self.u_singles[0].shape, dtype=complex) 
            #delta_doubles = np.zeros(self.u_doubles[0].shape, dtype=complex) 
            delta_singles = ze * trial_singles
            delta_doubles = ze * trial_doubles

            if is_rt and dt is not None:
                delta_singles -= 1j*dt*self.update_singles(t_fock_dressed_pq,
                                               dict_t_V_dressed, ctf.astensor(trial_singles),
                                               ctf.astensor(trial_doubles), t_T_abij).to_nparray()
            else:
                delta_singles -= self.update_singles(t_fock_dressed_pq,
                                                   dict_t_V_dressed, ctf.astensor(trial_singles),
                                                   ctf.astensor(trial_doubles), t_T_abij).to_nparray()
        
            delta_doubles = ze * trial_doubles
            if is_rt and dt is not None:
                delta_doubles -= 1j*dt*self.update_doubles(t_fock_dressed_pq,
                                               dict_t_V_dressed, ctf.astensor(trial_singles),
                                               ctf.astensor(trial_doubles), t_T_abij).to_nparray()
            else:
                delta_doubles -= self.update_doubles(t_fock_dressed_pq,
                                                   dict_t_V_dressed, ctf.astensor(trial_singles),
                                                   ctf.astensor(trial_doubles), t_T_abij).to_nparray()
            # convert to vector
            return np.concatenate((delta_singles.flatten(), delta_doubles.flatten()))
        A = LinearOperator((self.u_singles[0].size + self.u_doubles[0].size, self.u_singles[0].size + self.u_doubles[0].size), matvec=matvec)
        # construct a scipy sparse matrix M for the preconditioner using the diag_ai and diag_abij
        combined_diag = np.concatenate((1./(ze-diag_ai.flatten()+0.01), 1./(ze-diag_abij.flatten()+0.01)))
        M = diags(combined_diag, offsets=0)

        Qe_vec, exit_code = gcrotmk(A, b_vec, x0=Qe_vec_init, M=M, maxiter=3, tol=1e-4)
        print_logging_info("Linear Solver Info = ", exit_code, level=2)
        Qe_singles = Qe_vec[:self.u_singles[0].size].reshape(self.u_singles[0].shape)
        Qe_doubles = Qe_vec[self.u_singles[0].size:].reshape(self.u_doubles[0].shape)
        return Qe_singles, Qe_doubles
        

    def _bicgstab(self, l, ze, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij):
        """
        Solve the linear system (z-H)Q = Y.
        Default algorithm is BiCGSTAB unconditioned.

        Parameters
        ----------
        l : int, trial vector index
        ze : complex
            The shift value.
        t_fock_dressed_pq : np.ndarray, shape (norb, norb), dtype=float, dressed Fock matrix
        dict_t_V_dressed : dict, dressed two-body integrals
        t_T_abij : np.ndarray, shape (norb, norb, norb, norb), dtype=float, T2 amplitudes


        Returns
        -------
        Qe_singles : dtype=np.ndarray, single excitation vectors
        Qe_doubles : dtype=np.ndarray, double excitation vectors
        """

        Qe_singles = np.zeros(self.u_singles[0].shape, dtype=complex)
        Qe_doubles = np.zeros(self.u_doubles[0].shape, dtype=complex)

        def _get_residual(trial_singles, trial_doubles):
            """
            Get the residual of the linear system (z-H)Q = Y
            for the l-th trial vector.
            """
            delta_singles = np.zeros(self.u_singles[0].shape, dtype=complex) 
            delta_doubles = np.zeros(self.u_doubles[0].shape, dtype=complex) 
            delta_singles += self.u_singles[l]
            delta_singles -= ze * trial_singles
            delta_singles += self.update_singles(t_fock_dressed_pq,
                                               dict_t_V_dressed, ctf.astensor(trial_singles),
                                               ctf.astensor(trial_doubles), t_T_abij).to_nparray()
            
            delta_doubles = np.zeros(self.u_doubles[0].shape, dtype=complex)
            delta_doubles += self.u_doubles[l]
            delta_doubles -= ze * trial_doubles
            delta_doubles += self.update_doubles(t_fock_dressed_pq,
                                               dict_t_V_dressed, ctf.astensor(trial_singles),
                                               ctf.astensor(trial_doubles), t_T_abij).to_nparray()
            return delta_singles, delta_doubles
        
        delta_singles, delta_doubles = _get_residual(Qe_singles, Qe_doubles)
        rho = np.tensordot(np.conj(delta_singles), delta_singles, axes=2)
        rho += np.tensordot(np.conj(delta_doubles), delta_doubles, axes=4)
        p_singles = delta_singles.copy()
        p_doubles = delta_doubles.copy()
        r0_singles = delta_singles.copy()
        r0_doubles = delta_doubles.copy()
        r_singles = delta_singles.copy()
        r_doubles = delta_doubles.copy()
        r_norm = 0.
        s_norm = 0.
        for i in range(100):
            v_singles, v_doubles = _get_residual(p_singles, p_doubles)
            alpha = np.tensordot(np.conj(r0_singles), v_singles, axes=2)
            alpha += np.tensordot(np.conj(r0_doubles), v_doubles, axes=4)
            alpha = rho / alpha
            h_singles = Qe_singles + alpha * p_singles 
            h_doubles = Qe_doubles + alpha * p_doubles
            s_singles = r_singles - alpha * v_singles
            s_doubles = r_doubles - alpha * v_doubles
            s_norm = np.tensordot(np.conj(s_singles), s_singles, axes=2)
            s_norm += np.tensordot(np.conj(s_doubles), s_doubles, axes=4)
            if np.abs(s_norm) < 1e-8:
                print_logging_info(f"i = {i}, converged for s_norm", level=2)
                Qe_singles = h_singles
                Qe_doubles = h_doubles
                break
            t_singles, t_doubles = _get_residual(s_singles, s_doubles)
            omega = np.tensordot(np.conj(t_singles), s_singles, axes=2)
            omega += np.tensordot(np.conj(t_doubles), s_doubles, axes=4)
            t_norm = np.tensordot(np.conj(t_singles), t_singles, axes=2)
            t_norm += np.tensordot(np.conj(t_doubles), t_doubles, axes=4)
            omega /= t_norm
            Qe_singles = h_singles + omega * s_singles
            Qe_doubles = h_doubles + omega * s_doubles
            r_singles = s_singles - omega * t_singles
            r_doubles = s_doubles - omega * t_doubles
            r_norm = np.tensordot(np.conj(r_singles), r_singles, axes=2)
            r_norm += np.tensordot(np.conj(r_doubles), r_doubles, axes=4)
            if np.abs(r_norm) < 1e-8:
                print_logging_info(f"i = {i}, converged for r_norm", level=2)
                break
            #else:
            #    print_logging_info(f"i = {i}, r_norm: {np.abs(r_norm)}, s_norm: {np.abs(s_norm)}", level=2)
            # update rho and save the previous values
            rho_old = rho
            rho = np.tensordot(np.conj(r0_singles), r_singles, axes=2)
            rho += np.tensordot(np.conj(r0_doubles), r_doubles, axes=4)
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

        ham = self.construct_fake_non_sym_ham(nv, no)
        e_target, v_target = np.linalg.eig(ham)

        lowest_ex_ind_target = e_target.argsort()
        e_target = e_target[lowest_ex_ind_target]
        v_target = v_target[lowest_ex_ind_target]
        # create initial guesses
        s_shape = [nv, no]
        d_shape = [nv, nv, no, no]

        print_logging_info("Initialising u tensors...", level=1)
        for l in range(self.n_trial):
            self.u_singles.append(0.5-(np.random.rand(*s_shape)))
            self.u_doubles.append(0.5-(np.random.rand(*d_shape)))

        # gauss-legrendre quadrature
        x, w = get_gauss_legendre_quadrature(8) 
        theta = -np.pi / 2 * (x - 1)
        z = self.e_c + self.e_r * np.exp(1j * theta)

        # start iteratons
        e_norm_prev = 1e10
        for iter in range(self.max_iter):
            self.Q_singles = [np.zeros(s_shape, dtype=float) for _ in  range(self.n_trial)]
            self.Q_doubles = [np.zeros(d_shape, dtype=float) for _ in  range(self.n_trial)]
            time_iter_init = time.time()
            #self.u_singles, self.u_doubles = self.QR(self.u_singles, self.u_doubles)

            # solve for the linear system (z-H)Q = Y at z = z_e
            for e in range(len(z)):
                print_logging_info(f"e = {e}, z = {z[e]}, theta = {theta[e]}, w = {w[e]}", level=1)
                Qe_singles, Qe_doubles = self._solve_linear_test(z[e], self.u_singles, self.u_doubles, ham)
                #Qe_singles_conj, Qe_doubles_conj = self._solve_linear(np.conj(z[e]), self.u_singles, self.u_doubles, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij)
                for l in range(self.n_trial):
                    self.Q_singles[l] -= w[e]/2 * np.real(self.e_r * np.exp(1j * theta[e]) * Qe_singles[l])
                    self.Q_doubles[l] -= w[e]/2 * np.real(self.e_r * np.exp(1j * theta[e]) * Qe_doubles[l])
            #self.Q_singles, self.Q_doubles = self.QR(self.Q_singles, self.Q_doubles)
            # compute the projected Hamiltonian
            H_proj = np.zeros((self.n_trial, self.n_trial))
            B = np.zeros((self.n_trial, self.n_trial))
            w_singles = [np.zeros(self.u_singles[0].shape) for _ in range(self.n_trial)]
            w_doubles = [np.zeros(self.u_doubles[0].shape) for _ in range(self.n_trial)]
            for i in range(self.n_trial):
                w_singles[i] = self.update_singles_test(ham,
                                                   ctf.astensor(self.Q_singles[i]),
                                                   ctf.astensor(self.Q_doubles[i])).to_nparray()
                w_doubles[i] = self.update_doubles_test(ham,
                                                   ctf.astensor(self.Q_singles[i]),
                                                   ctf.astensor(self.Q_doubles[i])).to_nparray()
                for j in range(i):
                    H_proj[j, i] = np.tensordot(self.Q_singles[j], w_singles[i], axes=2) \
                              + np.tensordot(self.Q_doubles[j], w_doubles[i], axes=4)
                    H_proj[i, j] = np.tensordot(self.Q_singles[i], w_singles[j], axes=2) \
                              + np.tensordot(self.Q_doubles[i], w_doubles[j], axes=4)
                    B[i, j] = np.tensordot(self.Q_singles[i], self.Q_singles[j], axes=2) \
                                + np.tensordot(self.Q_doubles[i], self.Q_doubles[j], axes=4)
                    B[j, i] = B[i, j]
                H_proj[i, i] = np.tensordot(self.Q_singles[i], w_singles[i], axes=2) \
                          + np.tensordot(self.Q_doubles[i], w_doubles[i], axes=4)
                B[i, i] = np.tensordot(self.Q_singles[i], self.Q_singles[i], axes=2) \
                            + np.tensordot(self.Q_doubles[i], self.Q_doubles[i], axes=4)
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
                u_singles = np.zeros(s_shape, dtype=float)
                u_doubles = np.zeros(d_shape, dtype=float)
                for i in range(self.n_trial):
                    u_singles += np.real(eigvecs[i, l]) * self.Q_singles[i]
                    u_doubles += np.real(eigvecs[i, l]) * self.Q_doubles[i]
                self.u_singles.append(u_singles)
                self.u_doubles.append(u_doubles)
            
            
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

    def _solve_linear_test(self, ze, u_singles, u_doubles, ham):
        """
        Solve the linear system (z-H)Q = Y.
        ze : complex
        """
        Qe_singles = [np.zeros(u_singles[0].shape, dtype=complex) for _ in range(self.n_trial)]
        Qe_doubles = [np.zeros(u_doubles[0].shape, dtype=complex) for _ in range(self.n_trial)]
        Qe_singles_ref = [np.zeros(u_singles[0].shape, dtype=complex) for _ in range(self.n_trial)]
        Qe_doubles_ref = [np.zeros(u_doubles[0].shape, dtype=complex) for _ in range(self.n_trial)]
        # combine u_singles and u_doubles to Y 
        nv = u_singles[0].shape[0]
        no = u_singles[0].shape[1]
        Y = np.zeros([self.n_trial, nv*no+nv*nv*no*no], dtype=complex) 
        for l in range(self.n_trial):
            Y[l, :] = np.concatenate((u_singles[l].flatten(), u_doubles[l].flatten()))
        
        Q_e = np.linalg.solve(ze*np.diag(np.ones(ham.shape[0])) - ham, Y.T).T 
        for l in range(self.n_trial):
            Qe_singles_ref[l] = Q_e[l][:nv*no].reshape(nv, no)
            Qe_doubles_ref[l] = Q_e[l][nv*no:].reshape(nv, nv, no, no)
        for i in range(300):
            norm_singles = 0
            norm_doubles = 0
            for l in range(len(u_singles)):
                delta_singles = np.zeros(u_singles[0].shape, dtype=complex)
                delta_singles = ze * Qe_singles[l]
                delta_singles -= self.update_singles_test(ham,
                                                   ctf.astensor(Qe_singles[l]),
                                                   ctf.astensor(Qe_doubles[l])).to_nparray()
                delta_singles -= u_singles[l]
                
                delta_doubles = np.zeros(u_doubles[0].shape, dtype=complex)
                delta_doubles = ze * Qe_doubles[l]
                delta_doubles -= self.update_doubles_test(ham,
                                                   ctf.astensor(Qe_singles[l]),
                                                   ctf.astensor(Qe_doubles[l])).to_nparray()
                delta_doubles -= u_doubles[l]

                Qe_singles[l] -= 0.1 * (delta_singles/(ze - ham.diagonal()[:nv*no].reshape(nv, no))) 
                Qe_doubles[l] -= 0.1 * (delta_doubles/(ze - ham.diagonal()[nv*no:].reshape(nv*nv*no*no).reshape(nv, nv, no, no)))

                # check convergence
                norm_singles += np.linalg.norm(delta_singles)
                norm_doubles += np.linalg.norm(delta_doubles)

            if norm_singles + norm_doubles < self.tol:
                break
            #else:
        print_logging_info(f"|Delta Singles|: {norm_singles}, |Delta Doubles|: {norm_doubles}", level=2)
        # normalize the trial vectors
        for l in range(self.n_trial):
            Qe_singles[l], Qe_doubles[l] = normalize_amps(Qe_singles[l], Qe_doubles[l])
        return Qe_singles, Qe_doubles

    def construct_fake_non_sym_ham(self, nv, no):
        ham = self.construct_fake_ham(nv, no)
        # define a invertable matrix
        t_mat = np.diag(np.ones(ham.shape[0]))
        t_mat += np.random.rand(ham.shape[0], ham.shape[1])*0.01
        inv_t_mat = np.linalg.inv(t_mat)
        # similarity transformation
        ham = inv_t_mat.dot(ham).dot(t_mat)
        return ham



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
    norm_singles = np.tensordot(np.conj(u_singles), u_singles, axes=2)
    norm_doubles = np.tensordot(np.conj(u_doubles), u_doubles, axes=4)
    u_singles /= np.sqrt(norm_singles + norm_doubles)
    u_doubles /= np.sqrt(norm_singles + norm_doubles)
    return u_singles, u_doubles