import time
import numpy as np
import ctf
from ctf.core import *
from pymes.solver import mp2
from pymes.solver import ccd
from pymes.mixer import diis
from pymes.logging import print_logging_info

class CCSD(ccd.CCD):
    def __init__(self, no, is_non_canonical=False, is_dcsd=False):
        self.t_T_ai = None
        self.t_T_abij = None
        self.is_dcd = is_dcsd
        # number of occupied orbitals in the reference determinant
        self.no = no
        if self.is_diis:
            self.mixer = diis.DIIS(dim_space=6)

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
        t_V_pqrs: sparse ctf tensor by default, size [nq, nq, nq, nq]
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

        no = self.no
        nv = t_fock_pq.shape[0] - self.no


        # parameters
        level_shift = level_shift
        max_iter = max_iter
        #epsilon_e = 1e-8
        delta = 1.0

        # construct the needed integrals here on spot.


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

        t_D_ai = ctf.tensor([nv,no],dtype=t_V_pqrs.dtype, sp=sp)
        t_D_abij = ctf.tensor([nv,nv,no,no],dtype=t_V_pqrs.dtype, sp=sp)
        # the following ctf expression calcs the outer sum, as wanted.
        t_epsilon_i = ctf.einsum("ii -> i", t_fock_pq[:no, :no])
        t_epsilon_a = ctf.einsum("aa -> a", t_fock_pq[no:, no:])

        t_D_abij.i("abij") << t_epsilon_i.i("i") + t_epsilon_i.i("j")\
                              -t_epsilon_a.i("a") - t_epsilon_a.i("b")
        t_D_ai.i("ai") << t_epsilon_i.i("i") - t_epsilon_a.i("a")
        #t_D_abij = ctf.tensor([no,no,nv,nv],dtype=complex, sp=1)
        t_D_abij = 1./(t_D_abij+level_shift)
        t_D_ai = 1./(t_D_ai+level_shift)
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
            t_fock_pq = self.get_T1_dressed_fock(t_fock_pq, t_T_ai, t_V_pqrs)
             
            t_V_pqrs =  self.get_T1_dressed_V(t_T_ai, t_V_pqrs)
            
            t_R_ai = self.get_singles_residual(t_fock_pq, t_T_ai, t_V_pqrs)

            t_R_abij = self.get_doubles_residual(t_fock_pq, t_T_abij, \
                                            t_V_klij, t_V_ijab,\
                                            t_V_abij, t_V_iajb, t_V_iabj, t_V_abcd,\
                                            )


            t_delta_T_ai = ctf.einsum('ai,ai->ai', t_R_ai, t_D_ai)
            t_delta_T_abij = ctf.einsum('abij,abij->abij', t_R_abij, t_D_abij)
            t_T_ai += delta * t_delta_T_ai
            t_T_abij += delta * t_delta_T_abij

            if self.is_diis:
                t_T_ai, t_T_abij = self.mixer.mix([t_delta_T_ai, \
                                     t_delta_T_abij], [t_T_ai, t_T_abij])

            # update energy and norm of amplitudes
            #if is_dr_ccsd:
            #    e_dir_ccsd, e_ex_ccsd = drccsd.get_energy(t_T_abij, t_V_ijab)
            #else:
            e_dir_ccsd, e_ex_ccsd = get_energy(t_T_ai, t_T_abij, t_V_ijab)
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


    def get_T1_dressed_fock(self, t_T_ai, t_fock_pq, t_V_pqrs):
        '''
        Using the singles similarity transformed Hamiltonian,
        the Fock matrix is dressed. The expressions are obtained from
        Ref. J. Chem. Phys. 138, 144101 (2013)
        
        Parameters:
        -----------
        t_T_ai: ctf tensor, shape [nv, no]
                nv: number of unoccupied orbitals;
                no: number of occupied orbitals;
                The singles amplitudes.
        t_fock_pq: ctf tensor, shape [nb, nb]
                   nb: number of total orbitals.
                   The Fock matrix.
                    
        t_V_pqrs: ctf tensor, shape [nb, nb, nb, nb]
                  nb: number of total orbitals.
                  The original Coulomb integrals.
        
        Returns:
        -------
        t_fock_pq: ctf tensor, shape [nb, nb]
                   nb: number of total orbitals.
                   The dressed Fock matrix, using the same memory chunk 
                   as the original.
        '''
        
        no = self.no
        nv = t_fock_pq.shape[0] - no
        t_bar_fock_pq = t_fock_pq
        t_bar_fock_pq += 2.0*ctf.einsum("piqa, ai -> pq",
                                       t_V_pqrs[:, :no, :, no:],
                                       t_T_ai)
        t_bar_fock_pq += -1.0*ctf.einsum("ipqa, ai -> pq",
                                         t_V_pqrs[:no, :, :, no:],
                                         t_T_ai)
        t_tilde_fock_pq = t_bar_fock_pq.copy()
        # tilde{f}^a_i
        t_tilde_fock_pq[no:, :no] += -1.0*ctf.einsum("ji, aj -> ai", 
                                                     t_bar_fock_pq[:no, :no),
                                                     t_T_ai)
                                     +1.0*ctf.einsum("ab, bi -> ai",
                                                     t_bar_fock_pq[no:, no:],
                                                     t_T_ai)
                                     -1.0*ctf.einsum("jb, aj, bi -> ai",
                                                     t_bar_fock_pq[:no, no:],
                                                     t_T_ai, t_T_ai)
        # tilde{f}^i_a is the same as bar{f}^i_a
        
        # tilde{f}^a_b
        t_tilde_fock_pq[no:, no:] += -1.0*ctf.einsum("jb, aj  -> ab",
                                                     t_bar_fock_pq[:no, no:],
                                                     t_T_ai)
        # tilde{f}^i_j
        t_tilde_fock_pq[:no, :no] += 1.0*ctf.einsum("ib, bj  -> ij",
                                                     t_bar_fock_pq[:no, no:],
                                                     t_T_ai)
        
        return t_tilde_fock_pq
                            


    def get_T1_dressed_V(self, t_T_ai, t_V_pqrs):
        '''
        Using the singles similarity transformed Hamiltonian,
        the Coulomb integrals are dressed. The expressions are derived from
        Ref. J. Chem. Phys. 138, 144101 (2013)
        Some of them are presented in PhD thesis by Ke Liao titled
        "Ab initio Studies of Solid Phase Diagrams with Quantum Chemical
        Theories".
        
        Parameters:
        -----------
        t_T_ai: ctf tensor, shape [nv, no]
                nv: number of unoccupied orbitals;
                no: number of occupied orbitals;
                The singles amplitudes.

        t_V_pqrs: ctf tensor, shape [nb, nb, nb, nb]
                  nb: number of total orbitals.
                  The original Coulomb integrals.
        
        Returns:
        -------
        t_V_pqrs: ctf tensor
                  The dressed Coulomb integrals, using the same storage
                  as the original Coulomb integrals.
        '''

        no = self.no
        # in order of appearance in the dressed expressions
        t_V_iajk = t_V_pqrs[:no,no:,:no,:no]
        t_V_abci = t_V_pqrs[no:,no:,no:,:no]
        t_V_iabj = t_V_pqrs[:no,no:,no:,:no]
        t_V_aijk = t_V_pqrs[no:,:no,:no,:no]
        t_V_ijkl = t_V_pqrs[:no,:no,:no,:no]
        t_V_aijb = t_V_pqrs[no:,:no,:no,no:]
        t_V_aibj = t_V_pqrs[no:,:no,no:,:no]
        t_V_ijak = t_V_pqrs[:no,:no,no:,:no]
        t_V_abic = t_V_pqrs[no:,no:,:no,no:]
        t_V_iajb = t_V_pqrs[:no,no:,:no,no:]
        t_V_abcd = t_V_pqrs[no:,no:,no:,no:]
        t_V_iabc = t_V_pqrs[:no,no:,no:,no:]
        t_V_ijab = t_V_pqrs[:no,:no,no:,no:]
        t_V_ijka = t_V_pqrs[:no,:no,:no,no:]
        t_V_abij = t_V_pqrs[no:,no:,:no,:no]

        # make sure the integrals used in contractions are the original ones
        # t_V_abij uses some parts of the Coulomb integrals that will
        # be dressed. So we should dress t_V_abij first.
        t_V_abij += - ctf.einsum("kbij, ak -> abij", t_V_iajk, t_T_ai)
                    + ctf.einsum("abcj, ci -> abij", t_V_abci, t_T_ai)
                    - ctf.einsum("kbcj, ak, ci -> abij", t_V_iabj,
                                 t_T_ai, t_T_ai)

        t_V_abij += - ctf.einsum("alij, bl -> abij", t_V_aijk, t_T_ai)
                    + ctf.einsum("klij, ak, bl -> abij", t_V_ijkl, t_T_ai,
                                 t_T_ai)
                    - ctf.einsum("alcj, ci, bl -> abij", t_V_aibj, t_T_ai,
                                 t_T_ai)
                    + ctf.einsum("klcj, ak, ci, bl -> abij", t_V_ijak, t_T_ai,
                                 t_T_ai, t_T_ai)

        t_V_abij += + ctf.einsum("abid, dj -> abij", t_V_abic, t_T_ai)
                    - ctf.einsum("kbid, ak, dj -> abij", t_V_iajb, t_T_ai,
                                t_T_ai)
                    + ctf.einsum("abcd, ci, dj -> abij", t_V_abcd, t_T_ai,
                                t_T_ai)
                    - ctf.einsum("kbcd, ak, ci, dj -> abij", t_V_iabc, t_T_ai,
                                 t_T_ai, t_T_ai)

        t_V_abij += - ctf.einsum("alid, bl, dj -> abij", t_V_aijb, t_T_ai
                                 t_T_ai)
                    + ctf.einsum("klid, ak, bl, dj -> abij", t_V_ijka, t_T_ai,
                                t_T_ai, t_T_ai)
                    - ctf.einsum("alcd, ci, bl, dj -> abij", t_V_aibc, t_T_ai,
                                t_T_ai, t_T_ai)
                    + ctf.einsum("klcd, ak, ci, bl, dj -> abij", t_V_ijab, 
                                 t_T_ai, t_T_ai, t_T_ai, t_T_ai)
        
        # t_V_ijkl
        t_V_ijkl += ctf.einsum("ijal, ak -> ijkl", t_V_ijak, t_T_ai)
                    + ctf.einsum("ijka, al -> ijkl", t_V_ijka, t_T_ai)
                    + ctf.einsum("ijab, ak, bl -> ijkl", t_V_ijab, t_T_ai, 
                                 t_T_ai)
        # t_V_ijab is unchanged

        
        # t_V_iajb
        t_V_iajb += ctf.einsum("iacb, cj -> iajb", t_V_iabc, t_T_ai)
                    - ctf.einsum("ikjb, ak -> iajb", t_V_ijka, t_T_ai)
                    - ctf.einsum("ikcb, cj, ak -> iajb", t_V_ijab, t_T_ai, 
                                 t_T_ai)

        # t_V_iabj
        t_V_iabj += - ctf.einsum("ikbj, ak -> iabj", t_V_ijak, t_T_ai)
                    + ctf.einsum("iabc, cj -> iabj", t_V_iabc, t_T_ai)
                    - ctf.einsum("ikbc, ak, cj -> iabj", t_V_ijab, t_T_ai,
                                 t_T_ai)

        # t_V_abcd 
        t_V_abcd += - ctf.einsum("jbcd, aj -> abcd", t_V_iabc, t_T_ai)
                    - ctf.einsum("aicd, bi -> abcd", t_V_aibc, t_T_ai)
                    + ctf.einsum("ijcd, aj, bi -> abcd", t_V_ijab, t_T_ai, 
                                 t_T_ai)

        return t_V_pqrs 
        
    def get_singles_residual(self, t_T_ai, t_T_abij, t_V_pqrs):
        '''
        Computes the residuals for the singles amplitudes.
        '''
        return

    def get_doubles_residual(self, t_fock_pq, t_T_abij, t_V_pqrs):
        '''
        Computes the residuals for the doubles amplitudes.
        '''
        algo_name = "ccsd.get_residual"
        # return ccd's get_residual function
        return self.get_residual(self, t_fock_pq, t_T_abij, t_V_pqrs)

    def get_energy(t_T_ai, t_T_abij, t_V_ijab):
        '''
        calculate the CCSD correlation energy
        '''
        t_T_tmp_abij = t_T_abij + ctf.einsum("ai, bj -> abij", t_T_ai, t_T_ai)
        t_dir_ccsd_e = 2. * ctf.einsum("abij, ijab ->", t_T_tmp_abij, t_V_ijab)
        t_ex_ccsd_e  = -1. * ctf.einsum("abij, ijba ->", t_T__tmp_abij, t_V_ijab)
        return [t_dir_ccsd_e, t_ex_ccsd_e]
