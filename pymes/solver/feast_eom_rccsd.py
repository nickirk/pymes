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

def feast(eom, nroots=1, emin=None, emax=None, ngl_pts=8, koopmans=False, guess=None, left=False, eris=None, imds=None, **kwargs):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(eom.stdout, eom.verbose)
    if eom.verbose >= logger.WARN:
        eom.check_sanity()
    eom.dump_flags()

    e_r = (emax - emin)/2
    e_c = emax - e_r

    logger.info(eom, 'FEAST EOM-CCSD singlet kernel')
    logger.info(eom, 'Number of initial guesses = %d', nroots)
    logger.info(eom, 'Number of quadrature points = %d', ngl_pts)
    logger.info(eom, 'e_c = %s', e_c)
    logger.info(eom, 'e_r = %s', e_r)

    if imds is None:
        imds = eom.make_imds(eris)

    matvec, diag = eom.gen_matvec(imds, left=left, **kwargs)

    size = eom.vector_size()
    nroots = min(nroots, size)
    # create initial guesses
    logger.info(eom, "Initialising u tensors...")
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
    x, w = get_gauss_legendre_quadrature(ngl_pts) 
    theta = -np.pi / 2 * (x - 1)
    z = e_c + e_r * np.exp(1j * theta)

    # start iteratons
    e_norm_prev = 1e10
    for iter in range(eom.max_cycle):

        ntrial = len(u_vec)

        u_vec = QR(u_vec)

        Q = [np.zeros(size, dtype=complex) for _ in range(ntrial)]

        #u_vec = QR(u_vec)    
        # solve for the linear system (z-H)Q = Y at z = z_e
        #u_vec = QR(u_vec)
        for e in range(len(z)):
            logger.debug(eom, "e = %d, z = %s, theta = %s, w = %s", e, z[e], theta[e], w[e])
            for l in range(ntrial):
                Qe = eom._gcrotmk(z[e], b=u_vec[l], diag=diag, precond=precond)

                Q[l] -= w[e]/2 * np.real(e_r * np.exp(1j * theta[e]) * Qe)
        
        # compute the projected Hamiltonian
        H_proj = np.zeros((ntrial, ntrial), dtype=complex)
        B = np.zeros(H_proj.shape, dtype=complex)
        Hu = [np.zeros(size) for _ in range(ntrial)]
        Hu = matvec(Q)
        for i in range(ntrial):
            for j in range(i):
                H_proj[j, i] = np.dot(np.conj(Q[j]), Hu[i])
                H_proj[i, j] = np.dot(np.conj(Q[i]), Hu[j]) 
                B[i, j] = np.dot(np.conj(Q[i]), Q[j])
                B[j, i] = B[i, j]
            H_proj[i, i] = np.dot(np.conj(Q[i]), Hu[i])
            B[i, i] = np.dot(np.conj(Q[i]), Q[i])
        # solve the eigenvalue problem
        eigvals, eigvecs = eig(H_proj, B)
        # filter out the valid eigenvalues whose real values are within the range of [e_c - e_r, e_c + e_r]
        valid_inds = np.logical_and(np.real(eigvals) > e_c - e_r, np.real(eigvals) < e_c + e_r)
        valid_eigvals = eigvals[valid_inds].real
        valid_eigvecs = eigvecs[:, valid_inds]
        if len(valid_eigvals) == 0:
            logger.warn(eom, "No valid eigenvalues found in specified energy window.")
            break
        # get the eigenvectors corresponding to the max and min eigenvalues

        # update u_singles and u_doubles and to the trial vectors
        u_vec = [np.zeros(size) for _ in range(len(valid_eigvals))]
        for l in range(len(valid_eigvals)):
            for i in range(len(eigvals)):
                u_vec[l] += np.real(valid_eigvecs[i, l] * Q[i])
        
        # check convergence
        e_norm = np.linalg.norm(valid_eigvals)
        logger.debug(eom, "Iter = %d, all eigenvalues:", iter)
        logger.debug(eom, "%s", eigvals)
        logger.debug(eom, "Valid eigenvalues:" )
        logger.debug(eom, "%s", valid_eigvals)
        logger.info(eom, "cycle = %d, |eig| = %e, #eig = %d, delta|eig| = %e", iter, 
                    e_norm, len(valid_eigvals), np.abs(e_norm - e_norm_prev)) 
        if np.abs(e_norm - e_norm_prev) < eom.conv_tol:
            logger.info(eom, "FEAST-EOM-CCSD converged in %d iterations.", iter) 
            break
        else:
            max_valid_ind = np.argmax(valid_eigvals.real)
            min_valid_ind = np.argmin(valid_eigvals.real)
            max_ind = np.where(eigvals.real == valid_eigvals[max_valid_ind])[0][0]
            min_ind = np.where(eigvals.real == valid_eigvals[min_valid_ind])[0][0]
            max_eigval = eigvals[max_ind]
            min_eigval = eigvals[min_ind]
            max_eigvec = u_vec[max_valid_ind]
            min_eigvec = u_vec[min_valid_ind]
            # add more trial u vectors based on the max and min eigenvectors
            #u_vec.append(np.random.rand(size)-0.5)
            #u_vec.append(np.random.rand(size)-0.5)
            u_vec.append((Hu[max_ind] - max_eigval * max_eigvec)/(max_eigval - diag + 1e-10))
            if max_ind == min_ind:
                u_vec.append(np.random.rand(size)-0.5)
            else:
                u_vec.append((Hu[min_ind] - min_eigval * min_eigvec)/(min_eigval - diag + 1e-10))
            logger.debug(eom, "     # trial u vec = %d", len(u_vec))
        
        e_norm_diff = np.abs(e_norm - e_norm_prev)
        e_norm_prev = e_norm


    logger.info(eom, "All eigenvalues:" )
    logger.info(eom, "  %s", eigvals)
    logger.info(eom, "Valid eigenvalues:" )
    logger.info(eom, "  %s", np.sort(valid_eigvals))
    time_end = time.time()
    if iter == eom.max_cycle - 1 and e_norm_diff > eom.conv_tol:
        logger.warn(eom, "FEAST-EOM-CCSD not converged in %d iterations.", iter+1)
    logger.info(eom, "FEAST-EOM-CCSD finished in %s seconds.", time_end - time_init)

    return np.sort(valid_eigvals), u_vec

def QR(u):
    """
    QR decomposition of a matrix u
    """
    u = np.asarray(u).T
    Q, R = np.linalg.qr(u)
    Q = Q.T
    u = []
    for i in range(len(Q)):
        u.append(Q[i])
    return u
    
class FEAST_EOMEESinglet(EOMEE):

    def __init__(self, cc):
        EOMEE.__init__(self, cc)
        self.ls_max_iter = 100
        self.ls_conv_tol = 1e-4
        self.max_ntrial = 16
    
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
        if exit_code != 0:
            logger.warn(self, "Linear solver not converged after max %d cycles.", exit_code)
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

