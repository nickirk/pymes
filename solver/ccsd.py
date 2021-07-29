import time
import numpy as np
import ctf
from ctf.core import *
from pymes.solver import mp2
from pymes.mixer import diis
from pymes.logging import print_logging_info

class CCSD:
    def __init__(self, is_non_canonical=False, is_dcsd=False):
        self.t_T_ai = None
        self.t_T_abij = None
        self.is_dcsd = is_dcsd

    def write_logging_info(self):
        return

    def solve(self, t_fock_pq, t_V_pqrs, amps=None):
        '''
        The ccsd algorithm, in the screened Coulomb integrals formalism
        see ref. JCP 138.14 (2013), D. Kats and F.R. Manby 
        and ref. JCP 144.4 (2016), D. Kats
        t_V_ijkl = V^{ij}_{kl}
        t_V_abij = V^{ab}_{ij}
        t_T_abij = T^{ab}_{ij}
        the upper indices refer to conjugation
        
        Parameters
        ----------
        t_fock_pq: ctf tensor, size [nq, nq]
                  The Fock matrix, the diagonal of which are the orbital energies.
                  When canonical HF orbitals are used, the matrix is diagonal.
        t_V_pqrs: ctf tensor, size [nq, nq, nq, nq]
                  The two-body integrals. In the normal CCSD, it refers
                  Coulomb integrals. In the TC framework, it refers to the
                  the sum of all the two-body integrals.
        amps: a list of two ctf tensors, size [[nv, no], [nv, nv, no, no]]
              External source of T1 and T2 amplitudes. For example from a previous
              CCSD or DCSD calculation.

        Returns
        -------
        e_ccsd: double variable
                The CCSD correlation energy.
        t_T_ai: real/complex ctf tensor, size [nv, no]
                Singles amplitude.
        t_T_abij: real/complex ctf tensor, size [nv, nv, no, no]
                Doubles amplitude.
        t_epsilon_i: real/complex ctf tensor, size [no]
                     When the Bruekner orbital is used, the orbital energies
                     are updated during the iterations.
        t_epsilon_a: real/complex ctf tensor, size [nv]
                     When the Bruekner orbital is used, the orbital energies
                     are updated during the iterations.
        dE: double variable
            The change in the correlation energy between the last and the previous
            iterations. For checking the convergence. Can be set as an attribute
            of the class?
        '''

        algo_name = "ccsd.solve"
        time_ccsd = time.time()
        world = ctf.comm()

        no = t_epsilon_i.size
        nv = t_epsilon_a.size

        # if use Bruekner method, backup the hole and particle energies
        t_epsilon_original_i = t_epsilon_i.copy()
        t_epsilon_original_a = t_epsilon_a.copy()

        # parameters
        level_shift = level_shift
        max_iter = max_iter
        #epsilon_e = 1e-8
        delta = 1.0

        # construct the needed integrals here on spot.


        t_V_iabj = t_V_pqrs[:no,no:,no:,:no]
        t_V_aijb = t_V_pqrs[no:,:no,:no,no:]
        t_V_ijab = t_V_pqrs[:no,:no,no:,no:]
        t_V_klij = t_V_pqrs[:no,:no,:no,:no]
        t_V_iajb = t_V_pqrs[:no,no:,:no,no:]
        t_V_abij = t_V_pqrs[no:,no:,:no,:no]
        t_V_abcd = t_V_pqrs[no:,no:,no:,no:]


        print_logging_info(algo_name)
        print_logging_info("Using dcsd: " , is_dcsd, level=1)
        print_logging_info("Using dr-ccsd: " , is_dr_ccsd, level=1)
        print_logging_info("Solving doubles amplitude equation", level=1)
        print_logging_info("Using data type %s" % t_V_pqrs.dtype, level=1)
        print_logging_info("Using DIIS mixer: ", is_diis, level=1)
        print_logging_info("Iteration = 0", level=1)
        e_mp2, t_T_abij = mp2.solve(t_epsilon_i,t_epsilon_a, t_V_pqrs, level_shift,\
                                    sp=sp)
        if amps is not None:
            t_T_ai, t_T_abij = amps

        t_D_abij = ctf.tensor([nv,nv,no,no],dtype=t_V_pqrs.dtype, sp=sp)
        # the following ctf expression calcs the outer sum, as wanted.
        t_D_abij.i("abij") << t_epsilon_i.i("i") + t_epsilon_i.i("j")\
                              -t_epsilon_a.i("a")-t_epsilon_a.i("b")
        #t_D_abij = ctf.tensor([no,no,nv,nv],dtype=complex, sp=1)
        t_D_abij = 1./(t_D_abij+level_shift)
        # why the ctf contraction is not used here?
        # let's see if the ctf contraction does the same job
        dE = np.abs(np.real(e_mp2))
        iteration = 0
        e_last_iter_ccsd = np.real(e_mp2)
        e_ccsd = 0.
        e_dir_ccsd = 0.
        e_ex_ccsd = 0.
        residules = []
        amps = []
        mixSize = 5
        #t_R_abij = ctf.tensor([nv,nv,no,no], dtype=complex, sp=1)
        while np.abs(dE) > epsilon_e and iteration <= max_iter:
            iteration += 1
            t_R_ai, t_R_abij = 1.0*get_residual(t_epsilon_i, t_epsilon_a, t_T_abij, \
                                            t_V_klij, t_V_ijab,\
                                            t_V_abij, t_V_iajb, t_V_iabj, t_V_abcd,\
                                            is_dcsd)


            t_delta_T_abij = ctf.einsum('abij,abij->abij', t_R_abij, t_D_abij)
            t_T_abij += delta * t_delta_T_abij

            if is_diis:
                if len(residules) == mixSize:
                    residules.pop(0)
                    amps.pop(0)
                residules.append(t_delta_T_abij.copy())
                amps.append(t_T_abij.copy())
                t_T_abij = diis.mix(residules,amps)
            # update energy and norm of amplitudes
            #if is_dr_ccsd:
            #    e_dir_ccsd, e_ex_ccsd = drccsd.get_energy(t_T_abij, t_V_ijab)
            #else:
            e_dir_ccsd, e_ex_ccsd = get_energy(t_T_abij, t_V_ijab)
            e_ccsd = np.real(e_dir_ccsd + e_ex_ccsd)
            dE = e_ccsd - e_last_iter_ccsd
            e_last_iter_ccsd = e_ccsd

            t2_l1_norm = ctf.norm(t_T_abij)
            residual_norm = ctf.norm(t_delta_T_abij)

            if iteration <= max_iter:
                print_logging_info("Iteration = ", iteration, level=1)
                print_logging_info("Correlation Energy = {:.8f}".format(e_ccsd), \
                                   level=2)
                print_logging_info("dE = {:.8e}".format(dE), level=2)
                print_logging_info("L1 Norm of T2 = {:.8f}".format(t2_l1_norm), \
                                   level=2)
                print_logging_info("Norm Residul = {:.8f}".format(residual_norm), \
                                   level=2)
            else:
                print_logging_info("A converged solution is not found!", level=1)

        print_logging_info("Direct contribution = {:.8f}".format(\
                           np.real(e_dir_ccsd)), level=1)
        print_logging_info("Exchange contribution = {:.8f}".format(\
                           np.real(e_ex_ccsd)),level=1)
        print_logging_info("ccsd correlation energy = {:.8f}".format(\
                           e_ccsd), level=1)
        print_logging_info("{:.3f} seconds spent on ccsd".format(\
                           (time.time()-time_ccsd)), level=1)

        return {"ccsd e": e_ccsd, "t2 amp": t_T_abij, "hole e": t_epsilon_i, \
                "particle e": t_epsilon_a, "dE": dE}

    def get_residual(t_epsilon_i, t_epsilon_a, t_T_ai, t_T_abij, \
                     t_V_klij, t_V_ijab, t_V_abij, t_V_iajb, t_V_iabj, \
                     t_V_abcd, is_dcsd):
        '''
        Computes the residuals for the singles and doubles amplitudes.
        '''
        algo_name = "ccsd.get_residual"
        no = t_epsilon_i.size
        nv = t_epsilon_a.size
        t_R_ai = ctf.tensor([nv,no], dtype=t_V_klij.dtype, \
                              sp=t_T_ai.sp)
        t_R_abij = ctf.tensor([nv,nv,no,no], dtype=t_V_klij.dtype, \
                              sp=t_T_abij.sp)
        return t_R_ai, t_R_abij

    def get_energy(t_T_abij, t_V_ijab):
        '''
        calculate the CCSD correlation energy
        '''
        t_dir_ccsd_e = 2. * ctf.einsum("abij, ijab ->", t_T_abij, t_V_ijab)
        t_ex_ccsd_e  = -1. * ctf.einsum("abij, ijba ->", t_T_abij, t_V_ijab)
        return [t_dir_ccsd_e, t_ex_ccsd_e]
