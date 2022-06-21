import time
import numpy as np
import ctf

from pymes.solver import mp2
from pymes.mixer import diis
from pymes.log import print_logging_info
from pymes.solver import drccd


class CCD:

    def __init__(self, no, delta_e = 1.e-8, is_dcd=False, is_diis=True, is_dr_ccd=False,
                 is_bruekner=False):
        self.is_dcd = is_dcd
        self.is_diis = is_diis
        self.is_dr_ccd = is_dr_ccd
        self.is_bruekner = is_bruekner
        self.no = no
        self.delta_e = delta_e
        self.max_iter = 50
        if self.is_diis:
            self.mixer = diis.DIIS(dim_space=6)

    def solve(self, t_fock_pq, t_V_pqrs, level_shift=0., sp=0,
              amps=None, **kwargs
              ):
        '''
        ccd algorithm
        t_V_ijkl = V^{ij}_{kl}
        t_V_abij = V^{ab}_{ij}
        t_T_abij = T^{ab}_{ij}
        the upper indices refer to conjugation
        '''
        algo_name = "ccd.solve"
        time_ccd = time.time()
        world = ctf.comm()

        no = self.no
        nv = t_fock_pq.shape[0] - no

        # if use Bruekner method, backup the hole and particle energies
        t_epsilon_i = t_fock_pq.diagonal()[:no]
        t_epsilon_a = t_fock_pq.diagonal()[no:]

        # parameters
        # level_shift = self.level_shift
        if "max_iter" in kwargs:
            max_iter = kwargs['max_iter']
        else:
            max_iter = self.max_iter
        if "delta_e" in kwargs:
            delta_e = kwargs['delta_e']
        else:
            delta_e = self.delta_e

        delta = 1.0
        # construct the needed integrals here on spot.

        t_V_iabj = t_V_pqrs[:no, no:, no:, :no]
        t_V_aijb = t_V_pqrs[no:, :no, :no, no:]
        t_V_ijab = t_V_pqrs[:no, :no, no:, no:]
        t_V_klij = t_V_pqrs[:no, :no, :no, :no]
        t_V_iajb = t_V_pqrs[:no, no:, :no, no:]
        t_V_abij = t_V_pqrs[no:, no:, :no, :no]
        t_V_abcd = t_V_pqrs[no:, no:, no:, no:]

        print_logging_info(algo_name)
        print_logging_info("Using DCD: ", self.is_dcd, level=1)
        print_logging_info("Using dr-CCD: ", self.is_dr_ccd, level=1)
        print_logging_info("Solving doubles amplitude equation", level=1)
        print_logging_info("Using data type %s" % t_V_pqrs.dtype, level=1)
        print_logging_info("Using DIIS mixer: ", self.is_diis, level=1)
        print_logging_info("Using Bruekner quasi-particle energy: ", self.is_bruekner,
                           level=1)
        print_logging_info("Iteration = 0", level=1)
        e_mp2, t_T_abij = mp2.solve(t_epsilon_i, t_epsilon_a, t_V_ijab, t_V_abij, level_shift,
                                    sp=sp)
        if amps is not None:
            t_T_abij = amps

        t_D_abij = ctf.tensor([nv, nv, no, no], dtype=t_V_pqrs.dtype, sp=sp)
        # the following ctf expression calcs the outer sum, as wanted.
        t_D_abij.i("abij") << t_epsilon_i.i("i") + t_epsilon_i.i("j") \
        - t_epsilon_a.i("a") - t_epsilon_a.i("b")

        t_D_abij = 1. / (t_D_abij + level_shift)
        # why the ctf contraction is not used here?
        # let's see if the ctf contraction does the same job
        dE = np.abs(np.real(e_mp2))
        iteration = 0
        e_last_iter_ccd = np.real(e_mp2)
        e_ccd = 0.
        e_dir_ccd = 0.
        e_ex_ccd = 0.


        while np.abs(dE) > delta_e and iteration <= max_iter:
            iteration += 1
            if self.is_dr_ccd:
                t_R_abij = drccd.get_residual(t_epsilon_i, t_epsilon_a, t_T_abij,
                                              t_V_abij, t_V_aijb, t_V_iabj,
                                              t_V_ijab)
            else:
                t_R_abij = 1.0 * self.get_residual(t_fock_pq, t_T_abij,
                                                   t_V_klij, t_V_ijab,
                                                   t_V_abij, t_V_iajb, t_V_iabj, t_V_abcd)

            if self.is_bruekner:
                # construct amp dependent quasi-particle energies
                t_tilde_T_abij = ctf.tensor([nv, nv, no, no], dtype=t_T_abij.dtype,
                                            sp=t_T_abij.sp)
                t_tilde_T_abij.set_zero()
                t_tilde_T_abij.i("abij") << 2.0 * t_T_abij.i("abij") \
                - t_T_abij.i("baij")
                t_epsilon_i = t_epsilon_i \
                              + 1. / 2 * ctf.einsum("ilcd,cdil->i", t_V_ijab,
                                                    t_tilde_T_abij)
                t_epsilon_a = t_epsilon_a \
                              - 1. / 2 * ctf.einsum("klad,adkl->a", t_V_ijab,
                                                    t_tilde_T_abij)

                # update the denominator accordingly
                t_D_abij.i("abij") << t_epsilon_i.i("i") + t_epsilon_i.i("j") \
                - t_epsilon_a.i("a") - t_epsilon_a.i("b")
                t_D_abij = 1. / (t_D_abij + level_shift)

            t_delta_T_abij = ctf.einsum('abij,abij->abij', t_R_abij, t_D_abij)
            t_T_abij += delta * t_delta_T_abij

            if self.is_diis:
                t_T_abij = self.mixer.mix([t_delta_T_abij], [t_T_abij])[0]
            # update energy and norm of amplitudes
            # if self.is_dr_ccd:
            #    e_dir_ccd, e_ex_ccd = drccd.get_energy(t_T_abij, t_V_ijab)
            # else:
            e_dir_ccd, e_ex_ccd = self.get_energy(t_T_abij, t_V_ijab)
            e_ccd = np.real(e_dir_ccd + e_ex_ccd)
            dE = e_ccd - e_last_iter_ccd
            e_last_iter_ccd = e_ccd

            t2_l1_norm = ctf.norm(t_T_abij)
            residual_norm = ctf.norm(t_delta_T_abij)

            if iteration <= max_iter:
                print_logging_info("Iteration = ", iteration, level=1)
                print_logging_info("Correlation Energy = {:.12f}".format(e_ccd),
                                   level=2)
                print_logging_info("dE = {:.12e}".format(dE), level=2)
                print_logging_info("L1 Norm of T2 = {:.12f}".format(t2_l1_norm),
                                   level=2)
                print_logging_info("Norm Residual = {:.12f}".format(residual_norm),
                                   level=2)
            else:
                print_logging_info("A converged solution is not found!", level=1)

        print_logging_info("Direct contribution = {:.12f}".format(
            np.real(e_dir_ccd)), level=1)
        print_logging_info("Exchange contribution = {:.12f}".format(
            np.real(e_ex_ccd)), level=1)
        print_logging_info("CCD correlation energy = {:.12f}".format(
            e_ccd), level=1)
        print_logging_info("{:.3f} seconds spent on CCD".format(
            (time.time() - time_ccd)), level=1)

        return {"ccd e": e_ccd, "t2 amp": t_T_abij, "hole e": t_epsilon_i,
                "particle e": t_epsilon_a, "dE": dE}

    def get_residual(self, t_fock_pq, t_T_abij, t_V_klij, t_V_ijab,
                     t_V_abij, t_V_iajb, t_V_iabj, t_V_abcd):

        algo_name = "ccd.get_residual"
        no = self.no
        nv = t_fock_pq.shape[0] - no
        t_R_abij = ctf.tensor([nv, nv, no, no], dtype=t_V_klij.dtype, sp=t_T_abij.sp)

        # t_V_ijkl and t_V_klij are not the same in transcorrelated Hamiltonian!
        t_I_klij = ctf.tensor([no, no, no, no], dtype=t_V_klij.dtype, sp=t_T_abij.sp)

        # = operatore pass the reference instead of making a copy.
        # if we want a copy, we need to specify that.
        # t_I_klij = ctf.tensor([nv,nv,no,no], dtype=t_V_klij.dtype,sp=t_V_klij.sp)
        t_I_klij += t_V_klij
        if not self.is_dcd:
            t_I_klij += ctf.einsum("klcd, cdij -> klij", t_V_ijab, t_T_abij)

        t_R_abij.i("abij") << t_V_abij.i("abij") \
                              + t_I_klij.i("klij") * t_T_abij.i("abkl")\
                              + t_V_abcd.i("abcd") * t_T_abij.i("cdij")\
    
        if not self.is_dcd:
            t_X_alcj = ctf.einsum("klcd, adkj -> alcj", t_V_ijab, t_T_abij)
            t_R_abij += ctf.einsum("alcj, cbil -> abij", t_X_alcj, t_T_abij)

        # intermediates
        # t_tilde_T_abij
        # tested using MP2 energy, the below tensor op is correct
        t_tilde_T_abij = ctf.tensor([nv, nv, no, no], dtype=t_T_abij.dtype,
                                    sp=t_T_abij.sp)
        t_tilde_T_abij.set_zero()
        t_tilde_T_abij.i("abij") << 2.0 * t_T_abij.i("abij") - t_T_abij.i("baij")

        # Xai_kbcj for the quadratic terms
        t_Xai_cbkj = ctf.einsum("klcd, dblj -> cbkj", t_V_ijab, t_tilde_T_abij)

        t_R_abij += ctf.einsum("acik, cbkj -> abij", t_tilde_T_abij, t_Xai_cbkj)

        t_fock_ab = t_fock_pq[no:, no:]
        t_fock_ij = t_fock_pq[:no, :no]

        if self.is_bruekner:
            t_X_ac = t_fock_ab
            t_X_ki = t_fock_ij
        else:
            t_X_ac = t_fock_ab - 1. / 2 * ctf.einsum("adkl, lkdc -> ac",
                                                     t_tilde_T_abij, t_V_ijab)
            t_X_ki = t_fock_ij + 1. / 2 * ctf.einsum("cdil, lkdc -> ki",
                                                     t_tilde_T_abij, t_V_ijab)

        if not self.is_dcd:
            t_X_ac -= 1. / 2. * ctf.einsum("adkl, lkdc -> ac", t_tilde_T_abij, t_V_ijab)
            t_X_ki += 1. / 2. * ctf.einsum("cdil, lkdc -> ki",
                                           t_tilde_T_abij, t_V_ijab)

        t_Ex_abij = ctf.tensor([nv, nv, no, no], dtype=t_R_abij.dtype, sp=t_R_abij.sp)
        t_Ex_baji = ctf.tensor([nv, nv, no, no], dtype=t_R_abij.dtype, sp=t_R_abij.sp)

        t_Ex_abij.i("abij") << t_X_ac.i("ac") * t_T_abij.i("cbij") \
            - t_X_ki.i("ki") * t_T_abij.i("abkj") \
            - t_V_iajb.i("kaic") * t_T_abij.i("cbkj") \
            - t_V_iajb.i("kbic") * t_T_abij.i("ackj") \
            + t_tilde_T_abij.i("acik") * t_V_iabj.i("kbcj")
        if not self.is_dcd:
            t_Xai_aibj = ctf.einsum("klcd, daki -> alci", t_V_ijab, t_T_abij)
            t_Ex_abij -= ctf.einsum("alci, cblj -> abij", t_Xai_aibj, t_T_abij)
            t_Ex_abij += ctf.einsum("alci, bclj -> abij", t_Xai_aibj, t_T_abij)

        #t_Ex_baji.i("baji") << t_Ex_abij.i("abij")


        ## !!!!!!! In TC method the following is not necessarily the same!!!!!!!!!!
        #t_Ex_baji.i("baji") << t_Ex_abij.i("abij")

        t_Ex_abij.i("abij") << t_Ex_abij.i("baji")
        # print_logging_info(test_Ex_abij - t_Ex_abij)
        #t_R_abij += t_Ex_abij + t_Ex_baji
        t_R_abij += t_Ex_abij

        return t_R_abij

    def get_energy(self, t_T_abij, t_V_ijab):
        """
        calculate the CCD energy, using the converged amplitudes
        """
        t_dir_ccd_e = 2. * ctf.einsum("abij, ijab ->", t_T_abij, t_V_ijab)
        t_ex_ccd_e = -1. * ctf.einsum("abij, ijba ->", t_T_abij, t_V_ijab)
        return t_dir_ccd_e, t_ex_ccd_e
