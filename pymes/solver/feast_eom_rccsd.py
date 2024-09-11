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

def feast(eom, nroots=1, e_r=None, e_c=None, e_brd=1, emin=None, emax=None, ngl_pts=8,  n_aux=0, guess=None, left=False, koopmans=None, eris=None, imds=None, **kwargs):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(eom.stdout, eom.verbose)
    if eom.verbose >= logger.WARN:
        eom.check_sanity()
    eom.dump_flags()

    if emin is not None and emax is not None:
        e_r = (emax - emin)/2
        e_c = emax - e_r
    elif e_c is not None:
        user_guess = True
        e_guess = e_c
    else:
        raise ValueError("e_c or emin and emax must be specified.")


    if emin is not None and emax is not None:
        e_r = (emax - emin)/2
        e_c = emax - e_r
    elif e_c is not None:
        user_guess = True
        e_guess = e_c
    else:
        raise ValueError("e_c or emin and emax must be specified.")

    if e_r is None:
        e_r = 1    

    if emin is not None and emax is not None:
        e_r = (emax - emin)/2
        e_c = emax - e_r
    elif e_c is not None:
        user_guess = True
        e_guess = e_c
    else:
        raise ValueError("e_c or emin and emax must be specified.")

    if e_r is None:
        e_r = 1    

    if emin is not None and emax is not None:
        e_r = (emax - emin)/2
        e_c = emax - e_r
    elif e_c is not None:
        user_guess = True
        e_guess = e_c
    else:
        raise ValueError("e_c or emin and emax must be specified.")

    if e_r is None:
        e_r = 1    

    if emin is not None and emax is not None:
        e_r = (emax - emin)/2
        e_c = emax - e_r
    elif e_c is not None:
        user_guess = True
        e_guess = e_c
    else:
        raise ValueError("e_c or emin and emax must be specified.")

    if e_r is None:
        e_r = 1    

    if imds is None:
        imds = eom.make_imds(eris)

    matvec, diag = eom.gen_matvec(imds, left=left, **kwargs)

    size = eom.vector_size()
    nroots = min(nroots, size)
    # create initial guesses
    logger.info(eom, "Initialising u tensors...")
    #if n_aux is not None:
    #    n_aux = 0

    if guess is not None:
        user_guess = True
        e_guess = []
        target_u_max_loc = []
        for g in guess:
            assert g.size == size
            #g = matvec([g])[0]
            #g /= np.linalg.norm(g)
            target_u_max_loc.append(np.argmax(np.abs(g)))
            if e_c is not None:
                e_guess.append(e_c)
            else:
                e_guess.append(np.dot(g, matvec([g])[0]))
        if len(guess) < nroots:
            guess = guess + eom.get_init_guess(nroots-len(guess), koopmans, diag)
    else:
        #user_guess = False
        guess = eom.get_init_guess(nroots+n_aux, koopmans, diag)

    def precond(r, e0, x0):
        return r/(e0-diag+1e-12)

    # GHF or customized RHF/UHF may be of complex type
    real_system = (eom._cc._scf.mo_coeff[0].dtype == np.double)

    time_init = time.time()
    u_vec = guess.copy()
    # gauss-legrendre quadrature
    x, w = get_gauss_legendre_quadrature(ngl_pts) 
    theta = -np.pi / 2 * (x - 1)
    z = e_c + e_r * np.exp(1j * theta)
    print_title("FEAST-EOM-CCSD Solver")
    logger.info(eom, 'FEAST EOM-CCSD singlet kernel')
    logger.info(eom, 'Number of initial guesses = %d', nroots)
    logger.info(eom, 'Number of quadrature points = %d', ngl_pts)
    logger.info(eom, 'e_c = %s', e_c)
    logger.info(eom, 'e_r = %s', e_r)

    def prune(u_, max_iter=eom.ls_max_iter):
        from joblib import Parallel, delayed

        Q_ = [np.zeros(size, dtype=complex) for _ in range(len(u_))]

        def process_element(e):
            Q_loc = [np.zeros(size, dtype=complex) for _ in range(len(u_))]
            #if np.abs(z[e].imag) < 1e-3:
            #    ze = z[e]
            #    ze += 1j* (np.sign(z[e].imag) * 1e-3)
            #else:
            #    ze = z[e]
            #ze = z[e]
            logger.debug(eom, "e = %d, z = %s, theta = %s, w = %s", e, z[e], theta[e], w[e])
            for l in range(len(u_)):
                #logger.debug(eom, "  worker %d processing l = %d", e, l)
                Qe_ = eom._gcrotmk(z[e], b=u_[l], diag=diag, precond=precond, max_iter=max_iter)
                Q_loc[l] -= w[e]/2 * np.real(e_r * np.exp(1j * theta[e]) * Qe_)
            return Q_loc

        results = Parallel(n_jobs=-1)(delayed(process_element)(e) for e in range(len(z)))
        for result in results:
            for l in range(len(u_)):
                Q_[l] += result[l]
    
        return Q_
    # start iteratons
    e_norm_prev = 1e10
    num_eigs_prev = 0
    num_eigs = 0
    subspace_unstable = True 
    for iter in range(eom.max_cycle):

        ntrial = len(u_vec)

        #u_vec = QR(u_vec)
            # u_vec are those vectors that are within the energy window
            #u_vec = [u_vec[u] for u in valid_inds]
            #u_vec = target_u + eom.get_init_guess(nroots-1, koopmans, diag)


        Q = prune(u_vec, max_iter=eom.ls_max_iter)

        Q = QR(Q)
        
        
        # compute the projected Hamiltonian
        H_proj = np.zeros((ntrial, ntrial), dtype=complex)
        #B = np.zeros(H_proj.shape, dtype=complex)
        Hu = [np.zeros(size) for _ in range(ntrial)]
        Hu = matvec(Q)
        for i in range(ntrial):
            for j in range(i):
                H_proj[j, i] = np.dot(np.conj(Q[j]), Hu[i])
                H_proj[i, j] = np.dot(np.conj(Q[i]), Hu[j]) 
                #B[i, j] = np.dot(np.conj(Q[i]), Q[j])
                #B[j, i] = B[i, j]
            H_proj[i, i] = np.dot(np.conj(Q[i]), Hu[i])
            #B[i, i] = np.dot(np.conj(Q[i]), Q[i])
        # solve the eigenvalue problem
        eigvals, eigvecs = eig(H_proj)
        # argsort the eigenvalues in ascending order 
        all_sort_inds = np.argsort(eigvals.real)
        eigvals = eigvals[all_sort_inds]
        # filter out the valid eigenvalues whose real values are within the range of [e_c - e_r, e_c + e_r]
        valid_inds = np.where(np.logical_and(np.real(eigvals) > e_c - e_r, np.real(eigvals) < e_c + e_r))[0]
        valid_eigvals = eigvals[valid_inds].real
        num_eigs = len(valid_eigvals)
        sort_inds = np.argsort(valid_eigvals)
        e_norm = np.linalg.norm(valid_eigvals[sort_inds])
        valid_eigvals = valid_eigvals[sort_inds]

        if len(valid_eigvals) == 0:
            if not user_guess:
                logger.warn(eom, "No valid eigenvalues found in specified energy window.")
                return np.array([]), np.array([])

        # update u_singles and u_doubles and to the trial vectors
        u_vec = [np.zeros(size) for _ in range(len(eigvals))]
        for l in range(len(eigvals)):
            for i in range(len(eigvals)):
                u_vec[l] += np.real(eigvecs[i, l] * Q[i])
        
        max_comp = np.max(np.abs(np.asarray(u_vec)), axis=1)
        max_comp_loc = np.argmax(np.abs(np.asarray(u_vec)), axis=1)
        #
        e_r = np.sort(np.abs(e_c - eigvals))[::-1][n_aux] * e_brd
             
        z = e_c + e_r * np.exp(1j * theta)
        log.info("e_c = %s, e_r = %s", e_c, e_r)
        logger.debug(eom, "all max(abs(u_target)) = %s", max_comp[all_sort_inds])
        logger.debug(eom, "all argmax(abs(u_target)) = %s", max_comp_loc[all_sort_inds])
        logger.debug(eom, "valid max(abs(u_target)) = %s", max_comp[all_sort_inds][valid_inds][sort_inds])
        logger.debug(eom, "valid argmax(abs(u_target)) = %s", max_comp_loc[all_sort_inds][valid_inds][sort_inds])

        logger.info(eom, "cycle = %d, #trial = %d, |eig| = %e, #eig = %d, delta|eig| = %e", iter, 
                    len(u_vec), e_norm, len(valid_eigvals), np.abs(e_norm - e_norm_prev)) 
        logger.info(eom, "  eigvals: ")
        logger.info(eom, "      %s Ha", eigvals)
        logger.info(eom, "      %s eV", eigvals*27.2114)
        logger.info(eom, "  valid eigenvalues:" )
        logger.info(eom, "      %s Ha", valid_eigvals)
        logger.info(eom, "      %s eV", valid_eigvals*27.2114)
        if np.abs(e_norm - e_norm_prev) < eom.conv_tol:
            logger.info(eom, "FEAST-EOM-CCSD converged in %d iterations.", iter) 
            break
        else:
            if iter > 0: # and len(u_vec) <= eom.max_ntrial:
                # eigvals might contain nan
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


    time_end = time.time()
    if iter == eom.max_cycle - 1 and e_norm_diff > eom.conv_tol:
        logger.warn(eom, "FEAST-EOM-CCSD not converged in %d iterations.", iter+1)
    logger.info(eom, "  All eigenvalues: %s Ha", eigvals)
    logger.info(eom, "      %s eV", eigvals.real*27.2114)
    logger.info(eom, "  Valid eigenvalues: %s Ha", valid_eigvals)
    logger.info(eom, "      %s eV", valid_eigvals*27.2114)
    logger.info(eom, "FEAST-EOM-CCSD finished in %s seconds.", time_end - time_init)

    valid_u_vec = [u_vec[u] for u in valid_inds]
    valid_u_vec = [valid_u_vec[u] for u in sort_inds]

    return eigvals, valid_u_vec

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

    def _gcrotmk(self, ze, b, max_iter=None, x0=None, diag=None, precond=None, imds=None, 
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
        
        if max_iter is None:
            max_iter = self.ls_max_iter

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
        zero_diag_inds = np.abs(ze - diag) < 1e-5
        combined_diag = 1./(ze-diag+0.001)
        combined_diag[zero_diag_inds] = 1
        M = diags(combined_diag, offsets=0)

        Qe_vec, exit_code = gcrotmk(A, b, x0=x0, M=M, maxiter=max_iter, tol=self.ls_conv_tol)
        if exit_code != 0:
            logger.debug(self, "Linear solver not converged after max %d cycles.", exit_code)
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

