import time
import sys
import gc
import numpy as np
import pytblis as pytblis
from functools import partial

from pymes.util import tensors
from pymes.util.parallel_tasks import get_memory_usage 
from pymes.solver import mp2
from pymes.mixer import diis
from pymes.log import print_logging_info
from pymes.solver import drccd
from pymes.integral import eri

einsum = partial(pytblis.einsum, optimize='greedy')

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

    def solve(self, eri, level_shift=0., sp=0,
              amps=None, **kwargs
              ):
        '''
        ccd algorithm:
	    Electron Repulsion Integrals from 'eri' (ERI class).
        t_V_ijkl = V^{ij}_{kl}
        t_V_abij = V^{ab}_{ij}
        t_T_abij = T^{ab}_{ij}
        the upper indices refer to conjugation
        '''
        algo_name = "CCD.solve"

        time_ccd = time.time()
        print_logging_info(algo_name, level=0)
        mode = eri.mode
        print_logging_info("Using ERI mode: ", mode, level=1)
        if eri.mode == 'on-the-fly':
            raise NotImplementedError("CCD with on-the-fly ERI is currently not implemented.")

        no = self.no

        t_fock_pq = eri.fock
        t_V_klij  = eri.oooo
        t_V_iabj  = eri.ovvo 
        t_V_aijb  = eri.voov
        t_V_ijab  = eri.oovv
        t_V_abij  = eri.vvoo

        nv = t_fock_pq.shape[0] - no

        # If use Bruekner method, backup the hole and particle energies.
        # NOTE: t_epsilon_i and t_epsilon_a will be updated in each iteration, 
        #       if t_fock_pq.diag() is used directly, the updates will be lost <--- CHECK!
        t_epsilon_i = t_fock_pq.diagonal()[:no].copy()
        t_epsilon_a = t_fock_pq.diagonal()[no:].copy()

        # Parameters.
        if "max_iter" in kwargs:
            max_iter = kwargs['max_iter']
        else:
            max_iter = self.max_iter
        if "delta_e" in kwargs:
            delta_e = kwargs['delta_e']
        else:
            delta_e = self.delta_e

        delta = 1.0

        print_logging_info("Using DCD: ", self.is_dcd, level=1)
        print_logging_info("Using dr-CCD: ", self.is_dr_ccd, level=1)
        print_logging_info("Solving doubles amplitude equation", level=1)
        print_logging_info("Using data type %s" % t_V_klij.dtype, level=1)
        print_logging_info("Using DIIS mixer: ", self.is_diis, level=1)
        print_logging_info("Using Bruekner quasi-particle energy: ", self.is_bruekner,
                           level=1)
        print_logging_info("Using tolerance for energy convergence: {:.3e}".format(delta_e), level=1)
        print_logging_info("Initial memory usage: {:.2f} GB".format(get_memory_usage()), level=1)

        print_logging_info("Iteration = 0", level=1)
        tMP2 = mp2.MP2(no)
        mp2_results = tMP2.solve(eri, level_shift=level_shift)
        e_mp2 = mp2_results['mp2 e']
        t_T_abij = mp2_results['t2 amp']
        if amps is not None:
            t_T_abij = amps
        del tMP2

        t_D_abij = t_epsilon_i[None, None, :, None] + t_epsilon_i[None, None, None, :] - t_epsilon_a[:, None, None, None] - t_epsilon_a[None, :, None, None]

        t_D_abij = 1. / (t_D_abij + level_shift)

        dE = np.abs(np.real(e_mp2))
        iteration = 0
        e_last_iter_ccd = np.real(e_mp2)
        e_ccd = 0.
        e_dir_ccd = 0.
        e_ex_ccd = 0.

        while np.abs(dE) > delta_e and iteration <= max_iter:
            iteration += 1

            start_time_ccd_iter = time.time()
            if iteration <= max_iter:
                print_logging_info("Iteration = ", iteration, level=1)
                print_logging_info("Memory usage at iteration start: {:.2f} GB".format(get_memory_usage()), level=2)

            start_residual_time = time.time()
            if self.is_dr_ccd:
                t_R_abij = drccd.get_residual(t_epsilon_i, t_epsilon_a, t_T_abij,
                                              t_V_abij, t_V_aijb, t_V_iabj,
                                              t_V_ijab)
            else:
                t_R_abij = self.get_residual(eri, t_T_abij)
            end_residual_time = time.time()
            print_logging_info("Residual calculation time: {:.3f} seconds.".format(
                end_residual_time - start_residual_time), level=3)
            print_logging_info("Memory after residual: {:.2f} GB".format(get_memory_usage()), level=3)

            if self.is_bruekner:
                # Construct amplitude-dependent quasi-particle energies.
                t_tilde_T_abij = 2.0 * t_T_abij - einsum("baij -> abij", t_T_abij)
                # Corrections to hole and particle energies.
                t_eps_i_corr =  0.5 * einsum("ilcd,cdil->i", t_V_ijab, t_tilde_T_abij)
                t_eps_a_corr = -0.5 * einsum("klad,adkl->a", t_V_ijab, t_tilde_T_abij)
                # Update quasi-particle energies.
                t_epsilon_i = t_epsilon_i + t_eps_i_corr
                t_epsilon_a = t_epsilon_a + t_eps_a_corr
                # Recompute energy denominators.
                t_D_abij = t_epsilon_i[None, None, :, None] + t_epsilon_i[None, None, None, :] \
                                - t_epsilon_a[:, None, None, None] - t_epsilon_a[None, :, None, None]
                #t_D_abij = einsum('i, j, a, b -> abij', t_epsilon_i, t_epsilon_i, -t_epsilon_a, -t_epsilon_a)
                t_D_abij = 1. / (t_D_abij + level_shift)
                #  Delete and free memory.
                del t_tilde_T_abij, t_eps_i_corr, t_eps_a_corr

            t_delta_T_abij = t_R_abij * t_D_abij # ~ t_delta_T_abij = einsum('abij,abij->abij', t_R_abij, t_D_abij)
            t_T_abij += delta * t_delta_T_abij

            start_diis_time = time.time()
            if self.is_diis:
                t_T_abij = self.mixer.mix([t_delta_T_abij], [t_T_abij])[0]
            end_diis_time = time.time()
            print_logging_info("DIIS time: {:.3f} seconds.".format(
                end_diis_time - start_diis_time), level=3)
            
            # Update energy and norm of amplitudes.
            # if self.is_dr_ccd:
            #    e_dir_ccd, e_ex_ccd = drccd.get_energy(t_T_abij, t_V_ijab)
            # else:

            start_energy_time = time.time()
            e_dir_ccd, e_ex_ccd = self.get_energy(t_T_abij, t_V_ijab)
            e_ccd = np.real(e_dir_ccd + e_ex_ccd)
            dE = e_ccd - e_last_iter_ccd
            e_last_iter_ccd = e_ccd
            end_energy_time = time.time()
            print_logging_info("Energy calculation time: {:.3f} seconds.".format(
                end_energy_time - start_energy_time), level=3)

            t2_l1_norm = np.linalg.norm(t_T_abij)
            residual_norm = np.linalg.norm(t_delta_T_abij)

            del t_delta_T_abij, t_R_abij

            end_time_ccd_iter = time.time()
            print_logging_info("Iteration time = {:.3f} seconds.".format(
                end_time_ccd_iter - start_time_ccd_iter), level=2)
            print_logging_info("Memory at iteration end: {:.2f} GB".format(get_memory_usage()), level=2)

            if iteration <= max_iter:
                print_logging_info("Correlation Energy = {:.12f}".format(e_ccd),
                                   level=2)
                print_logging_info("dE = {:.12e}".format(dE), level=2)
                print_logging_info("L1 Norm of T2 = {:.12f}".format(t2_l1_norm),
                                   level=2)
                print_logging_info("Norm Residual = {:.12f}".format(residual_norm),
                                   level=2)
                sys.stdout.flush()
            else:
                print_logging_info("A converged solution is not found!", level=1)
                sys.stdout.flush()

        print_logging_info("Direct contribution = {:.12f}".format(
            np.real(e_dir_ccd)), level=1)
        print_logging_info("Exchange contribution = {:.12f}".format(
            np.real(e_ex_ccd)), level=1)
        print_logging_info("CCD correlation energy = {:.12f}".format(
            e_ccd), level=1)
        print_logging_info("Final memory usage: {:.2f} GB".format(get_memory_usage()), level=1) 
        print_logging_info("{:.3f} seconds spent on CCD".format(
            (time.time() - time_ccd)), level=1)

        return {"ccd e": e_ccd, "t2 amp": t_T_abij, "hole e": t_epsilon_i,
                "particle e": t_epsilon_a, "dE": dE}

    def get_residual(self, eri, t_T_abij):

        algo_name = "CCD.get_residual"

        print_logging_info(algo_name + ": Calculating R_abij residual ...", level=2)
        start_initial_residual_time = time.time()

        no = self.no

        t_fock_pq = eri.fock 
        t_V_iabj  = eri.ovvo
        t_V_ijab  = eri.oovv
        t_V_klij  = eri.oooo
        t_V_iajb  = eri.ovov
        t_V_abij  = eri.vvoo

        nv = t_fock_pq.shape[0] - no

        # NOTE: t_V_ijkl and t_V_klij are not the same in transcorrelated Hamiltonian!
        # t_I_klij = np.zeros([no, no, no, no], dtype=t_V_klij.dtype)
        # t_I_klij += t_V_klij
        t_I_klij = t_V_klij.copy()
        if not self.is_dcd:
            t_I_klij += einsum("klcd, cdij -> klij", t_V_ijab, t_T_abij)

        # Residual tensor R_abij.
        #t_R_abij = np.zeros([nv, nv, no, no], dtype=t_V_klij.dtype)
        #t_R_abij += t_V_abij 
        t_R_abij = t_V_abij.copy()
        t_R_abij += einsum("klij, abkl -> abij", t_I_klij, t_T_abij)

        end_initial_residual_time = time.time()
        print_logging_info(" Elapsed initial part of residual time: {:.3f} seconds.".format(
            end_initial_residual_time - start_initial_residual_time), level=2)
        start_vvvv_residual_time = time.time()

        # Calculate block size dynamically to optimize memory usage.
        element_size = t_T_abij.dtype.itemsize  # Size of one element in bytes
        block_size = tensors.calculate_block_size(0, tuple((nv, nv, nv, nv)), element_size,
                                                    memory_fraction=0.55, is_shared_memory=False)
        
        print_logging_info("Using block size of {} for 'vvvv'-contribution.".format(block_size), level=3)
        print_logging_info("Memory per block: {:.2f} GB".format(
            block_size * nv * nv * nv * element_size / (1024 ** 3)), level=3)

        # Process tensor 'vvvv'-contribution in blocks.
        for block_start in range(0, nv, block_size):
            start_vvvv_time = time.time()
            block_end = min(block_start + block_size, nv)
            print_logging_info(" Calculating block: {} to {}.".format(block_start, block_end), level=3)
            indx = tuple((block_start, block_end, 0, nv, 0, nv, 0, nv))
            t_V_xbcd = eri.get_pqrs('vvvv', idx=indx)
            end_vvvv_time = time.time()
            print_logging_info(" Elapsed vvvv integral time: {:.3f} seconds.".format(
                end_vvvv_time - start_vvvv_time), level=3)
            print_logging_info(" Memory after loading vvvv block: {:.2f} GB".format(get_memory_usage()), level=3)
            start_block_time = time.time()
            t_R_abij[block_start:block_end, :, :, :] += einsum("xbcd, cdij -> xbij", t_V_xbcd, t_T_abij) # += t_R_xbij
            end_block_time = time.time()
            print_logging_info(" Elapsed block contr. time: {:.3f} seconds.".format(
                end_block_time - start_block_time), level=3)
            del t_V_xbcd
            gc.collect()
            print_logging_info(" Memory after block cleanup: {:.2f} GB".format(get_memory_usage()), level=3)
            sys.stdout.flush()

        end_vvvv_residual_time = time.time()
        print_logging_info(" Elapsed vvvv part of residual time: {:.3f} seconds.".format(
            end_vvvv_residual_time - start_vvvv_residual_time), level=2)
        start_final_residual_time = time.time()
        if not self.is_dcd:
            t_X_alcj = einsum("klcd, adkj -> alcj", t_V_ijab, t_T_abij)
            t_R_abij += einsum("alcj, cbil -> abij", t_X_alcj, t_T_abij)
            del t_X_alcj

        # Intermediates for CCD residual.
        # t_tilde_T_abij ...
        # t_tilde_T_abij = np.zeros([nv, nv, no, no], dtype=t_T_abij.dtype)
        t_tilde_T_abij = 2.0 * t_T_abij - einsum("baij -> abij", t_T_abij)
        # Xai_kbcj for the quadratic terms ...
        t_Xai_cbkj = einsum("klcd, dblj -> cbkj", t_V_ijab, t_tilde_T_abij)

        t_R_abij += einsum("acik, cbkj -> abij", t_tilde_T_abij, t_Xai_cbkj)

        # Fock matrix contributions ...
        t_fock_ab = t_fock_pq[no:, no:]
        t_fock_ij = t_fock_pq[:no, :no]

        if self.is_bruekner:
            t_X_ac = t_fock_ab
            t_X_ki = t_fock_ij
        else:
            t_X_ac = t_fock_ab - 1. / 2 * einsum("adkl, lkdc -> ac",
                                                     t_tilde_T_abij, t_V_ijab)
            t_X_ki = t_fock_ij + 1. / 2 * einsum("cdil, lkdc -> ki",
                                                     t_tilde_T_abij, t_V_ijab)

        if not self.is_dcd:
            t_X_ac -= 1. / 2. * einsum("adkl, lkdc -> ac", t_tilde_T_abij, t_V_ijab)
            t_X_ki += 1. / 2. * einsum("cdil, lkdc -> ki",
                                           t_tilde_T_abij, t_V_ijab)

        # Exchange part of the residual ...
        # t_Ex_abij  = np.zeros([nv, nv, no, no], dtype=t_R_abij.dtype)
        # t_Ex_abij += einsum("ac, cbij -> abij", t_X_ac, t_T_abij)
        t_Ex_abij  = einsum("ac, cbij -> abij", t_X_ac, t_T_abij)
        t_Ex_abij -= einsum("ki, abkj -> abij", t_X_ki, t_T_abij)
        t_Ex_abij -= einsum("kaic, cbkj -> abij", t_V_iajb, t_T_abij)
        t_Ex_abij -= einsum("kbic, ackj -> abij", t_V_iajb, t_T_abij)
        t_Ex_abij += einsum("acik, kbcj -> abij", t_tilde_T_abij, t_V_iabj)

        if not self.is_dcd:
            t_Xai_aibj = einsum("klcd, daki -> alci", t_V_ijab, t_T_abij)
            t_Ex_abij -= einsum("alci, cblj -> abij", t_Xai_aibj, t_T_abij)
            t_Ex_abij += einsum("alci, bclj -> abij", t_Xai_aibj, t_T_abij)

        t_Ex_abij += t_Ex_abij.transpose(1, 0, 3, 2)
        t_R_abij += t_Ex_abij

        del t_I_klij, t_tilde_T_abij, t_Xai_cbkj, t_X_ac, t_X_ki, t_Ex_abij
        #gc.collect()

        end_final_residual_time = time.time()
        print_logging_info(" Elapsed final part of residual time: {:.3f} seconds.".format(
            end_final_residual_time - start_final_residual_time), level=2)

        return t_R_abij

    def get_energy(self, t_T_abij, t_V_ijab):
        """
        calculate the CCD energy, using the converged amplitudes
        """
        t_dir_ccd_e = 2. * einsum("abij, ijab ->", t_T_abij, t_V_ijab)
        t_ex_ccd_e = -1. * einsum("abij, ijba ->", t_T_abij, t_V_ijab)
        return t_dir_ccd_e, t_ex_ccd_e
