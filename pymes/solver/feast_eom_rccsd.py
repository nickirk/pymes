"""
This implementation uses functions from pyscf 
to achieve the FEAST-EOM-CCSD algorithm.
"""

import numpy as np
from scipy.linalg import eig
import time
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger, module_method
from pyscf.cc import ccsd
from pyscf.cc import rintermediates as imd
from pyscf.cc.eom_rccsd import *
from pyscf import __config__

from pymes.log import print_title, print_logging_info
from pymes.solver.feast_eom_ccsd import get_gauss_legendre_quadrature

def feast(eom, nroots=1, e_r=None, e_c=None, koopmans=False, guess=None, left=False, eris=None, imds=None, **kwargs):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(eom.stdout, eom.verbose)
    if eom.verbose >= logger.WARN:
        eom.check_sanity()
    eom.dump_flags()

    if imds is None:
        imds = eom.make_imds(eris)

    matvec, diag = eom.gen_matvec(imds, left=left, **kwargs)

    size = eom.vector_size()
    nroots = min(nroots, size)
    # create initial guesses
    print_logging_info("Initialising u tensors...", level=1)
    if guess is not None:
        user_guess = True
        for g in guess:
            assert g.size == size
    else:
        user_guess = False
        guess = eom.get_init_guess(nroots, koopmans, diag)

    def precond(r, e0, x0):
        return r/(e0-diag+1e-12)

    # GHF or customized RHF/UHF may be of complex type
    real_system = (eom._cc._scf.mo_coeff[0].dtype == np.double)

    print_title("FEAST-EOM-CCSD Solver")
    time_init = time.time()

    u_vec = guess.copy()
    # gauss-legrendre quadrature
    x, w = get_gauss_legendre_quadrature(8) 
    theta = -np.pi / 2 * (x - 1)
    z = e_c + e_r * np.exp(1j * theta)

    # start iteratons
    e_norm_prev = 1e10
    for iter in range(eom.max_cycle):
        Q = [np.zeros(size, dtype=complex) for _ in range(nroots)]

        time_iter_init = time.time()

        # solve for the linear system (z-H)Q = Y at z = z_e
        for e in range(len(z)):
            print_logging_info(f"e = {e}, z = {z[e]}, theta = {theta[e]}, w = {w[e]}", level=1)
            for l in range(nroots):
                Qe = eom._gcrotmk(z[e], b=u_vec[l], diag=diag, precond=precond)

                Q[l] -= w[e]/2 * np.real(e_r * np.exp(1j * theta[e]) * Qe)
                Q[l] -= w[e]/2 * np.real(e_r * np.exp(1j * theta[e]) * Qe)
        
        # compute the projected Hamiltonian
        H_proj = np.zeros((nroots, nroots), dtype=complex)
        B = np.zeros(H_proj.shape, dtype=complex)
        Hu = [np.zeros(size) for _ in range(nroots)]
        Hu = matvec(Q)
        for i in range(nroots):
            for j in range(i):
                H_proj[j, i] = np.dot(np.conj(Q[j]), Hu[i])
                H_proj[i, j] = np.dot(np.conj(Q[i]), Hu[j]) 
                B[i, j] = np.dot(np.conj(Q[i]), Q[j])
                B[j, i] = B[i, j]
            H_proj[i, i] = np.dot(np.conj(Q[i]), Hu[i])
            B[i, i] = np.dot(np.conj(Q[i]), Q[i])
        # solve the eigenvalue problem
        eigvals, eigvecs = eig(H_proj, B)
        

        # update u_singles and u_doubles and to the trial vectors
        for l in range(nroots):
            for i in range(len(eigvals)):
                u_vec[l] += np.real(eigvecs[i, l] * Q[i])
        
        # check convergence
        e_norm = np.linalg.norm(eigvals)
        if np.abs(e_norm - e_norm_prev) < eom.conv_tol:
            break
        else:
            print_logging_info(f"Iter = {iter}, Eigenvalues: {eigvals}", level=1)
            print_logging_info(f"Norm of eigenvalues: {e_norm}, Difference: {np.abs(e_norm - e_norm_prev)}", level=1)

        e_norm_prev = e_norm


        time_end = time.time()
        print_logging_info(f"FEAST-EOM-CCSD finished in {time_end - time_init:.2f} seconds.", level=0)

    return eigvals, u_vec

    
class FEAST_EOMEESinglet(EOMEE):

    def __init__(self, cc):
        EOMEE.__init__(self, cc)
        self.ls_max_iter = 100
        self.ls_conv_tol = 1e-4
    
    kernel = feast
    matvec = eeccsd_matvec_singlet

    def get_init_guess(self, nroots=1, koopmans=False, diag=None):
        size = self.vector_size()
        guess = []
        # generate random guess
        for i in range(nroots):
            g = np.random.rand(size)-0.5
            # normalize the guess
            g = g/np.linalg.norm(g)
            guess.append(g)
        return guess

    def _gcrotmk(self, ze, b, x0=None, diag=None, precond=None, imds=None, 
                 phase=None, is_rt=False, dt=None):
        from scipy.sparse.linalg import LinearOperator, gcrotmk
        from scipy.sparse import diags

        size = self.vector_size()
        if x0 is None:
            x0 = np.zeros(size, dtype=complex)

        if phase is not None:
            b *= phase
        
        if imds is None:
            self._cc.t2 = self._cc.t2.real
            imds = self.make_imds()

        def _matvec(Qe):
            """
            Matrix-vector product for the linear system (z-H)Q = Y
            """
            # unpack the vector
            trial_vec = Qe.copy()

            delta_vec = ze * trial_vec

            if is_rt and dt is not None:
                delta_vec -= 1j*dt*self.matvec(trial_vec, imds)
            else:
                delta_vec -= self.matvec(trial_vec, imds)
            return delta_vec
        
        A = LinearOperator((size, size), matvec=_matvec)
        # construct a scipy sparse matrix M for the preconditioner using the diag_ai and diag_abij
        if diag is None:
            diag = self.get_diag()
        combined_diag = 1./(ze-diag+0.01)
        M = diags(combined_diag, offsets=0)

        Qe_vec, exit_code = gcrotmk(A, b, x0=x0, M=M, maxiter=self.ls_max_iter, tol=self.ls_conv_tol)
        print_logging_info("Linear Solver Info = ", exit_code, level=2)
        return Qe_vec


    def get_diag(self, imds=None):
        return eeccsd_diag(self, imds=None)[0]

    def gen_matvec(self, imds=None, diag=None, **kwargs):
        if imds is None: imds = self.make_imds()
        if diag is None: diag = self.get_diag(imds)
        matvec = lambda xs: [self.matvec(x, imds) for x in xs]
        return matvec, diag

    amplitudes_to_vector = staticmethod(amplitudes_to_vector_singlet)
    vector_to_amplitudes = module_method(vector_to_amplitudes_singlet,
                                         absences=['nmo', 'nocc'])
    spatial2spin = staticmethod(spatial2spin_singlet)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nov = nocc * nvir
        return nov + nov*(nov+1)//2

