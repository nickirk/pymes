"""
This implementation uses functions from pyscf 
to achieve the Real Time FEAST-EOM-CCSD algorithm.
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

def kernel(eom, dt=0.1, e_r=None, e_c=None, ngl_pts=16, koopmans=False, guess=None, left=False, eris=None, imds=None, **kwargs):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(eom.stdout, eom.verbose)
    if eom.verbose >= logger.WARN:
        eom.check_sanity()
    eom.dump_flags()
    logger.info(eom, 'RT-EOM-CCSD singlet kernel')
    logger.info(eom, 'Number of initial guesses = %d', nroots)
    logger.info(eom, 'Number of quadrature points = %d', ngl_pts)
    logger.info(eom, 'e_c = %s', e_c)
    logger.info(eom, 'e_r = %s', e_r)

    nroots = 1

    logger.info(eom, 'RT-EOM-CCSD singlet kernel')
    logger.info(eom, 'Number of initial guesses = %d', nroots)
    logger.info(eom, 'Number of quadrature points = %d', ngl_pts)
    logger.info(eom, 'e_c = %s', e_c)
    logger.info(eom, 'e_r = %s', e_r)

    if imds is None:
        imds = eom.make_imds(eris)

    matvec, diag = eom.gen_matvec(imds, left=left, **kwargs)

    size = eom.vector_size()
    # create initial guesses
    print_logging_info("Initialising u tensors...", level=1)
    if guess is not None:
        user_guess = True
        for g in guess:
            assert g.size == size
    else:
        print_logging_info("No guess provided. Generating random guess.")
        user_guess = False
        guess = eom.get_init_guess(nroots, koopmans, diag)

    def precond(r, e0, x0):
        return r/(e0-diag+1e-12)

    # GHF or customized RHF/UHF may be of complex type
    real_system = (eom._cc._scf.mo_coeff[0].dtype == np.double)

    print_title("CIFRT-EOM-CCSD Solver")
    time_init = time.time()

    u_vec = guess.copy()
    # gauss-legrendre quadrature
    x, w = get_gauss_legendre_quadrature(ngl_pts) 
    theta = -np.pi * x
    #theta = np.concatenate((-np.pi * x, np.pi * x))
    #w = np.concatenate((w, w))
    # the quadrature points
    z = (e_c*1j + e_r * np.exp(1j * theta))*dt

    # start iteratons
    e_norm_prev = 1e10
    # solve for the linear system (z-H)Q = Y at z = z_e
    Q = [np.zeros(size, dtype=complex) for _ in range(nroots)]
    for e in range(len(z)):
        print_logging_info(f"e = {e}, z = {z[e]}, theta = {theta[e]}, w = {w[e]}", level=1)
        Qe = eom._gcrotmk(z[e], b=u_vec[0], diag=diag, precond=precond, 
                          phase=np.exp(z[e]), is_rt=True, dt=dt)

        Q[0] -= w[e]/2 * (e_r * dt * np.exp(1j * theta[e]) * Qe)
        
    u_norm= np.dot(np.conj(Q[0]), Q[0])
    print_logging_info("Norm of new u vec = ", u_norm)
    #Q /= np.linalg.norm(Q)
    #u_norm= np.dot(np.conj(Q[0]), Q[0])
    #print_logging_info("Norm of new u vec after normalization = ", u_norm)
    u_vec = Q
    time_end = time.time()
    print_logging_info(f"RT-EOM-CCSD finished in {time_end - time_init:.2f} seconds.", level=0)

    return u_vec[0]

    
class CIFRT_EOMEESinglet(EOMEE):

    def __init__(self, cc):
        EOMEE.__init__(self, cc)
        self.ls_max_iter = 100
        self.ls_conv_tol = 1e-4
    
    kernel = kernel 
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

        b_ = b.copy()
        # convert b_ to complex
        if phase is not None:
            b_ = b_.astype(complex)
            b_ *= phase
        
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

        Qe_vec, exit_code = gcrotmk(A, b_, x0=x0, M=M, maxiter=self.ls_max_iter, tol=self.ls_conv_tol)
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

