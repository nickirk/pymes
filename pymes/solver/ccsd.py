import time
import numpy as np
import ctf

from pymes.solver import mp2, ccd
from pymes.mixer import diis
from pymes.log import print_logging_info
from pymes.integral.partition import part_2_body_int

'''
CCSD implementation using the T1-similarity transformed Hamiltonian.
In this formalism, CCSD class can inherit some functionalities from
the CCD class. E.g. the doubles amplitudes update.

Author: Ke Liao <ke.liao.whu@gmail.com>
'''


class CCSD(ccd.CCD):
    def __init__(self, no, is_diis=True, delta_e=1.e-8, is_non_canonical=False, is_dcsd=False):
        self.t_T_ai = None
        self.t_T_abij = None
        self.is_dcd = is_dcsd
        self.is_diis = is_diis
        # temperary control parameters for compatibility with CCD, 
        # might be needed in the future
        self.is_bruekner = False
        # number of occupied orbitals in the reference determinant
        self.no = no

        # parameters
        self.max_iter = 50
        self.delta = 1.0
        # convergence criterion for energy difference
        self.delta_e = delta_e

        if self.is_diis:
            self.mixer = diis.DIIS(dim_space=6)

    def write_logging_info(self):
        return

    def solve(self, t_fock_pq, t_V_pqrs, level_shift=0., amps=None, sp=0, **kwargs):
        """
        The ccsd algorithm, in the screened Coulomb integrals formalism
        see ref. JCP 138.14 (2013), D. Kats and F.R. Manby
        and ref. JCP 144.4 (2016), D. Kats
        t_V_klij = V^{ij}_{kl}
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
        level_shift: double
                    by default set to self.level_shift which is 0.
        amps: a list of two ctf tensors, size [[nv, no], [nv, nv, no, no]]
              External source of T1 and T2 amplitudes. For example from a previous
              CCSD or DCSD calculation.
        sp: 0 or 1
            0 for dense tensor
            1 for sparse tensor

        Returns
        -------
        e_ccsd: double
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
        """

        algo_name = "ccsd.solve"
        time_ccsd = time.time()
        world = ctf.comm()

        no = self.no
        nv = t_fock_pq.shape[0] - self.no

        delta = self.delta

        if "max_iter" in kwargs:
            max_iter = kwargs['max_iter']
        else:
            max_iter = self.max_iter
        if "delta_e" in kwargs:
            delta_e = kwargs['delta_e']
        else:
            delta_e = self.delta_e

        # construct the needed integrals here on spot.
        t_epsilon_i = t_fock_pq.diagonal()[:no]
        t_epsilon_a = t_fock_pq.diagonal()[no:]

        # try to reduce the number of times partitioning big tensors, 
        # it is very time consuming.
        dict_t_V = part_2_body_int(no, t_V_pqrs)

        print_logging_info(algo_name)
        print_logging_info("Using dcsd: ", self.is_dcd, level=1)
        # print_logging_info("Using dr-ccsd: " , is_dr_ccsd, level=1)
        print_logging_info("Solving doubles amplitude equation", level=1)
        print_logging_info("Using data type %s" % t_V_pqrs.dtype, level=1)
        print_logging_info("Using DIIS mixer: ", self.is_diis, level=1)
        print_logging_info("Iteration = 0", level=1)

        e_mp2, t_T_abij = mp2.solve(t_epsilon_i, t_epsilon_a, dict_t_V["ijab"], dict_t_V["abij"], level_shift,
                                    sp=sp)
        t_T_ai = ctf.tensor([nv, no], dtype=t_T_abij.dtype, sp=sp)
        t_T_ai.set_zero()
        if amps is not None:
            t_T_ai, t_T_abij = amps

        t_D_ai = ctf.tensor([nv, no], dtype=t_V_pqrs.dtype, sp=sp)
        t_D_abij = ctf.tensor([nv, nv, no, no], dtype=t_V_pqrs.dtype, sp=sp)

        dE = np.abs(np.real(e_mp2))
        iteration = 0
        e_last_iter_ccsd = np.real(e_mp2)
        e_ccsd = 0.
        e_dir_ccsd = 0.
        e_ex_ccsd = 0.

        # before computing dressed fock and V, make a copy of the original
        # data. In density fitting, this will be the density-fitted quantities

        t_fock_pq_orig = t_fock_pq.copy()

        t_epsilon_i = ctf.einsum("ii -> i", t_fock_pq_orig[:no, :no])
        t_epsilon_a = ctf.einsum("aa -> a", t_fock_pq_orig[no:, no:])

        t_D_abij.i("abij") << t_epsilon_i.i("i") + t_epsilon_i.i("j") \
        - t_epsilon_a.i("a") - t_epsilon_a.i("b")
        t_D_ai.i("ai") << t_epsilon_i.i("i") - t_epsilon_a.i("a")
        # t_D_abij = ctf.tensor([no,no,nv,nv],dtype=complex, sp=1)
        t_D_abij = 1. / (t_D_abij + level_shift)
        t_D_ai = 1. / (t_D_ai + level_shift)

        # t_R_abij = ctf.tensor([nv,nv,no,no], dtype=complex, sp=1)
        while np.abs(dE) > delta_e and iteration <= max_iter:
            iteration += 1
            t_fock_pq = t_fock_pq_orig.copy()

            t_fock_pq = self.get_T1_dressed_fock(t_fock_pq, t_T_ai, dict_t_V)

            dict_t_V_dressed = self.get_T1_dressed_V(t_T_ai, dict_t_V)

            t_R_ai = self.get_singles_residual(t_fock_pq, t_T_ai, t_T_abij,
                                               dict_t_V)

            t_V_klij_dressed = dict_t_V_dressed['klij']
            t_V_ijab_dressed = dict_t_V_dressed['ijab']
            t_V_abij_dressed = dict_t_V_dressed['abij']
            t_V_iajb_dressed = dict_t_V_dressed['iajb']
            t_V_iabj_dressed = dict_t_V_dressed['iabj']
            t_V_abcd_dressed = dict_t_V_dressed['abcd']

            t_R_abij = self.get_doubles_residual(
                t_fock_pq, t_T_abij,
                t_V_klij_dressed, t_V_ijab_dressed,
                t_V_abij_dressed, t_V_iajb_dressed,
                t_V_iabj_dressed, t_V_abcd_dressed
            )

            t_delta_T_ai = ctf.einsum('ai,ai->ai', t_R_ai, t_D_ai)
            t_delta_T_abij = ctf.einsum('abij,abij->abij', t_R_abij, t_D_abij)
            t_T_ai += delta * t_delta_T_ai
            t_T_abij += delta * t_delta_T_abij

            if self.is_diis:
                t_T_ai, t_T_abij = self.mixer.mix([t_delta_T_ai,
                                                   t_delta_T_abij], [t_T_ai, t_T_abij])

            # update energy and norm of amplitudes
            # if is_dr_ccsd:
            #    e_dir_ccsd, e_ex_ccsd = drccsd.get_energy(t_T_abij, t_V_ijab)
            # else:
            e_1b_ccsd, e_dir_ccsd, e_ex_ccsd = self.get_energy(
                t_fock_pq_orig[:no, no:], t_T_ai,
                t_T_abij, dict_t_V["ijab"])
            e_ccsd = np.real(e_1b_ccsd + e_dir_ccsd + e_ex_ccsd)
            dE = e_ccsd - e_last_iter_ccsd
            e_last_iter_ccsd = e_ccsd

            t2_l1_norm = ctf.norm(t_T_abij)
            residual_norm = ctf.norm(t_delta_T_abij)

            if iteration <= max_iter:
                print_logging_info("Iteration = ", iteration, level=1)
                print_logging_info("Correlation Energy = {:.14f}".format(e_ccsd),
                                   level=2)
                print_logging_info("dE = {:.12e}".format(dE), level=2)
                print_logging_info("L1 Norm of T2 = {:.14f}".format(t2_l1_norm),
                                   level=2)
                print_logging_info("Norm Residual = {:.14f}".format(residual_norm),
                                   level=2)
            else:
                print_logging_info("A converged solution is not found!", level=1)

        print_logging_info("Fock contribution = {:.12f}".format(
            np.real(e_1b_ccsd)), level=1)
        print_logging_info("Direct contribution = {:.12f}".format(
            np.real(e_dir_ccsd)), level=1)
        print_logging_info("Exchange contribution = {:.12f}".format(
            np.real(e_ex_ccsd)), level=1)
        print_logging_info("CCSD correlation energy = {:.12f}".format(
            e_ccsd), level=1)
        print_logging_info("{:.3f} seconds spent on ccsd".format(
            (time.time() - time_ccsd)), level=1)
        self.t_T_ai = t_T_ai
        self.t_T_abij = t_T_abij
        return {"ccsd e": e_ccsd, "t1": t_T_ai, "t2": t_T_abij, "hole e": t_epsilon_i,
                "particle e": t_epsilon_a, "dE": dE}

    def get_T1_dressed_fock(self, t_fock_pq, t_T_ai, dict_t_V):
        """
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

        dict_t_V: dictionary of ctf tensors

        Returns:
        -------
        t_tilde_fock_pq: ctf tensor, shape [nb, nb]
                   nb: number of total orbitals.
                   The dressed Fock matrix, using the same memory chunk
                   as the original.
        """

        no = self.no

        t_tilde_fock_pq = t_fock_pq.copy()
        # needed undressed t_V tensors
        # t_V_iabj, t_V_ijab
        # t_V_ijak, t_V_iabc
        # t_V_iajb, t_V_ijka 
        #
        # dressed f^i_a block
        t_tilde_fock_pq[:no, no:] += 2.0 * ctf.einsum("bj, jabi->ia", t_T_ai, dict_t_V['iabj'])
        t_tilde_fock_pq[:no, no:] += -1.0 * ctf.einsum("bj, jiab->ia", t_T_ai, dict_t_V['ijab'])
        # dressed f^a_i
        t_tilde_fock_pq[no:, :no] += -1.0 * ctf.einsum("ji, aj->ai", t_fock_pq[:no, :no], t_T_ai)
        t_tilde_fock_pq[no:, :no] += 1.0 * ctf.einsum("ab, bi->ai", t_fock_pq[no:, no:], t_T_ai)
        t_tilde_fock_pq[no:, :no] += -1.0 * ctf.einsum("jb, bi, aj->ai", t_fock_pq[:no, no:], t_T_ai, t_T_ai)

        t_tilde_fock_pq[no:, :no] += 2.0 * ctf.einsum("bj, jabi->ai", t_T_ai, dict_t_V['iabj'])
        t_tilde_fock_pq[no:, :no] += -2.0 * ctf.einsum("bj, jkbi, ak->ai", t_T_ai, dict_t_V['ijak'], t_T_ai)
        t_tilde_fock_pq[no:, :no] += 2.0 * ctf.einsum("bj, jabc, ci->ai", t_T_ai, dict_t_V['iabc'], t_T_ai)
        t_tilde_fock_pq[no:, :no] += -2.0 * ctf.einsum("bj, jkbc, ci, ak->ai", t_T_ai, dict_t_V['ijab'], t_T_ai, t_T_ai)

        t_tilde_fock_pq[no:, :no] -= 1.0 * ctf.einsum("bj, jaib->ai", t_T_ai, dict_t_V['iajb'])
        t_tilde_fock_pq[no:, :no] -= -1.0 * ctf.einsum("bj, jkib, ak->ai", t_T_ai, dict_t_V['ijka'], t_T_ai)
        t_tilde_fock_pq[no:, :no] -= 1.0 * ctf.einsum("bj, jacb, ci->ai", t_T_ai, dict_t_V['iabc'], t_T_ai)
        t_tilde_fock_pq[no:, :no] -= -1.0 * ctf.einsum("bj, jkcb, ci, ak->ai", t_T_ai, dict_t_V['ijab'], t_T_ai, t_T_ai)

        # dressed f^i_j block
        t_tilde_fock_pq[:no, :no] += 2.0 * ctf.einsum("ck, kicj->ij", t_T_ai, dict_t_V['ijak'])
        t_tilde_fock_pq[:no, :no] += -1.0 * ctf.einsum("ck, kijc->ij", t_T_ai, dict_t_V['ijka'])
        t_tilde_fock_pq[:no, :no] += 1.0 * ctf.einsum("ib, bj->ij", t_fock_pq[:no, no:], t_T_ai)
        t_tilde_fock_pq[:no, :no] += 2.0 * ctf.einsum("ck, kicb, bj->ij", t_T_ai, dict_t_V['ijab'], t_T_ai)
        t_tilde_fock_pq[:no, :no] += -1.0 * ctf.einsum("ck, kibc, bj->ij", t_T_ai, dict_t_V['ijab'], t_T_ai)

        # dressed f^a_b block
        t_tilde_fock_pq[no:, no:] += 2.0 * ctf.einsum("ci, iacb->ab", t_T_ai, dict_t_V['iabc'])
        t_tilde_fock_pq[no:, no:] += -1.0 * ctf.einsum("ci, iabc->ab", t_T_ai, dict_t_V['iabc'])
        t_tilde_fock_pq[no:, no:] += -1.0 * ctf.einsum("ib, ai->ab", t_fock_pq[:no, no:], t_T_ai)
        t_tilde_fock_pq[no:, no:] += -2.0 * ctf.einsum("ck, klcb, al->ab", t_T_ai, dict_t_V['ijab'], t_T_ai)
        t_tilde_fock_pq[no:, no:] += 1.0 * ctf.einsum("ck, kibc, ai->ab", t_T_ai, dict_t_V['ijab'], t_T_ai)

        return t_tilde_fock_pq

    def get_T1_dressed_V(self, t_T_ai, dict_t_V, dict_t_V_dressed=None):
        """
        Using the singles similarity transformed Hamiltonian,
        the Coulomb integrals are dressed. The expressions are derived from
        Ref. J. Chem. Phys. 138, 144101 (2013)
        Some of them are presented in PhD thesis by Ke Liao titled
        "Ab initio Studies of Solid Phase Diagrams with Quantum Chemical
        Theories".

        Parameters:
        ----------
            t_T_ai: ctf tensor, shape [nv, no]
                nv: number of unoccupied orbitals;
                no: number of occupied orbitals;
                The singles amplitudes.
            dict_t_V: dictionary, containing partitions of original t_V_pqrs tensor
            dict_t_V_dressed: dictionary, containing the keys of requested partitions.
                If no keys present or set to None, the keys from dict_t_V will be used.

        Returns:
        -------
            dict_t_V_dressed: ctf tensor
                  The dressed Coulomb integrals, using the same storage
                  as the original Coulomb integrals.
        """

        if dict_t_V_dressed is None or len(dict_t_V_dressed) == 0:
            dict_t_V_dressed= {}.fromkeys(dict_t_V, None)
        # make sure the integrals used in contractions are the original ones
        # t_V_abij uses some parts of the Coulomb integrals that will
        # be dressed. So we should dress t_V_abij first.

        if "abij" in dict_t_V_dressed:
            t_V_abij_dressed = dict_t_V['abij'].copy()
            t_V_abij_dressed += -1.0 * ctf.einsum("kbij, ak -> abij", dict_t_V['iajk'], t_T_ai)
            t_V_abij_dressed += ctf.einsum("abcj, ci -> abij", dict_t_V['abci'], t_T_ai)
            t_V_abij_dressed += -1.0 * ctf.einsum("kbcj, ak, ci -> abij", dict_t_V['iabj'], t_T_ai, t_T_ai)
            t_V_abij_dressed += -1.0 * ctf.einsum("alij, bl -> abij", dict_t_V['aijk'], t_T_ai)
            t_V_abij_dressed += ctf.einsum("klij, ak, bl -> abij", dict_t_V['klij'], t_T_ai, t_T_ai)
            t_V_abij_dressed += -1.0 * ctf.einsum("alcj, ci, bl -> abij", dict_t_V['aibj'], t_T_ai, t_T_ai)
            t_V_abij_dressed += ctf.einsum("klcj, ak, ci, bl -> abij", dict_t_V['ijak'], t_T_ai, t_T_ai, t_T_ai)

            t_V_abij_dressed += ctf.einsum("abid, dj -> abij", dict_t_V['abic'], t_T_ai)
            t_V_abij_dressed += -1.0 * ctf.einsum("kbid, ak, dj -> abij", dict_t_V['iajb'], t_T_ai, t_T_ai)
            t_V_abij_dressed += ctf.einsum("abcd, ci, dj -> abij", dict_t_V['abcd'], t_T_ai, t_T_ai)
            t_V_abij_dressed += -1.0 * ctf.einsum("kbcd, ak, ci, dj -> abij", dict_t_V['iabc'], t_T_ai, t_T_ai, t_T_ai)

            t_V_abij_dressed += -1.0 * ctf.einsum("alid, bl, dj -> abij", dict_t_V['aijb'], t_T_ai, t_T_ai)
            t_V_abij_dressed += ctf.einsum("klid, ak, bl, dj -> abij", dict_t_V['ijka'], t_T_ai, t_T_ai, t_T_ai)
            t_V_abij_dressed += - 1.0 * ctf.einsum("alcd, ci, bl, dj -> abij", dict_t_V['aibc'], t_T_ai, t_T_ai, t_T_ai)
            t_V_abij_dressed += ctf.einsum("klcd, ak, ci, bl, dj -> abij", dict_t_V['ijab'],
                                           t_T_ai, t_T_ai, t_T_ai, t_T_ai)

            dict_t_V_dressed["abij"] = t_V_abij_dressed

        # t_V_klij
        if "klij" in dict_t_V_dressed:
            t_V_klij_dressed = dict_t_V['klij'].copy()
            t_V_klij_dressed += ctf.einsum("klaj, ai -> klij", dict_t_V['ijak'], t_T_ai)
            t_V_klij_dressed += ctf.einsum("klib, bj -> klij", dict_t_V['ijka'], t_T_ai)
            t_V_klij_dressed += ctf.einsum("klab, ai, bj -> klij", dict_t_V['ijab'], t_T_ai, t_T_ai)

            dict_t_V_dressed["klij"] = t_V_klij_dressed

        # t_V_ijab is unchanged
        if "ijab" in dict_t_V_dressed:
            t_V_ijab_dressed = dict_t_V['ijab'].copy()
            dict_t_V_dressed["ijab"] = t_V_ijab_dressed

        if "ijka" in dict_t_V_dressed:
            t_V_ijka_dressed = dict_t_V["ijka"].copy()
            t_V_ijka_dressed += ctf.einsum("ijba, bk -> ijka", dict_t_V["ijab"], t_T_ai)
            dict_t_V_dressed["ijka"] = t_V_ijka_dressed

        if "ijak" in dict_t_V_dressed:
            t_V_ijak_dressed = dict_t_V["ijak"].copy()
            t_V_ijak_dressed += ctf.einsum("ijab, bk -> ijak", dict_t_V["ijab"], t_T_ai)
            dict_t_V_dressed["ijak"] = t_V_ijak_dressed

        # t_V_iajb
        if "iajb" in dict_t_V_dressed:
            t_V_iajb_dressed = dict_t_V['iajb'].copy()
            t_V_iajb_dressed += ctf.einsum("iacb, cj -> iajb", dict_t_V['iabc'], t_T_ai)
            t_V_iajb_dressed += -1.0 * ctf.einsum("ikjb, ak -> iajb", dict_t_V['ijka'], t_T_ai)
            t_V_iajb_dressed += -1.0 * ctf.einsum("ikcb, cj, ak -> iajb", dict_t_V['ijab'], t_T_ai, t_T_ai)
            dict_t_V_dressed["iajb"] = t_V_iajb_dressed

        # t_V_iabj
        if "iabj" in dict_t_V_dressed:
            t_V_iabj_dressed = dict_t_V['iabj'].copy()
            t_V_iabj_dressed += -1.0 * ctf.einsum("ikbj, ak -> iabj", dict_t_V['ijak'], t_T_ai)
            t_V_iabj_dressed += ctf.einsum("iabc, cj -> iabj", dict_t_V['iabc'], t_T_ai)
            t_V_iabj_dressed += -1.0 * ctf.einsum("ikbc, ak, cj -> iabj", dict_t_V['ijab'], t_T_ai, t_T_ai)
            dict_t_V_dressed["iabj"] = t_V_iabj_dressed

        if "iabc" in dict_t_V_dressed:
            t_V_iabc_dressed = dict_t_V["iabc"].copy()
            t_V_iabc_dressed += -1.0 * ctf.einsum("ijbc, aj -> iabc", dict_t_V["ijab"], t_T_ai)
            dict_t_V_dressed["iabc"] = t_V_iabc_dressed

        if "abic" in dict_t_V_dressed:
            t_V_abic_dressed = dict_t_V["abic"].copy()
            t_V_abic_dressed += -1.0 * ctf.einsum("jbic, aj -> abic", dict_t_V["iajb"], t_T_ai)
            t_V_abic_dressed += ctf.einsum("abdc, di -> abic", dict_t_V["abcd"], t_T_ai)
            t_V_abic_dressed += -1.0 * ctf.einsum("jbdc, aj, di -> abic", dict_t_V["iabc"], t_T_ai, t_T_ai)
            t_V_abic_dressed += -1.0 * ctf.einsum("ajic, bj -> abic", dict_t_V["aijb"], t_T_ai)
            t_V_abic_dressed += ctf.einsum("kjic, ak, bj -> abic", dict_t_V["ijka"], t_T_ai, t_T_ai)
            t_V_abic_dressed += -1.0 * ctf.einsum("ajdc, di, bj -> abic", dict_t_V["aibc"], t_T_ai, t_T_ai)
            t_V_abic_dressed += ctf.einsum("kjdc, ak, di, bj -> abic", dict_t_V["ijab"], t_T_ai, t_T_ai, t_T_ai)
            dict_t_V_dressed["abic"] = t_V_abic_dressed

        # t_V_iajk or t_V_aijk
        if "iajk" in dict_t_V_dressed:
            t_V_iajk_dressed = dict_t_V["iajk"].copy()
            t_V_iajk_dressed += -1.0 * ctf.einsum("iljk, al -> iajk", dict_t_V["klij"], t_T_ai)
            t_V_iajk_dressed += ctf.einsum("iajb, bk -> iajk", dict_t_V["iajb"], t_T_ai)
            t_V_iajk_dressed += -1.0 * ctf.einsum("iljb, al, bk -> iajk", dict_t_V["ijka"], t_T_ai, t_T_ai)
            t_V_iajk_dressed += ctf.einsum("iabk, bj -> iajk", dict_t_V["iabj"], t_T_ai)
            t_V_iajk_dressed += -1.0 * ctf.einsum("ilbk, bj, al -> iajk", dict_t_V["ijak"], t_T_ai, t_T_ai)
            t_V_iajk_dressed += ctf.einsum("iabc, bj, ck -> iajk", dict_t_V["iabc"], t_T_ai, t_T_ai)
            t_V_iajk_dressed += -1.0 * ctf.einsum("ilbc, bj, al, ck -> iajk", dict_t_V["ijab"], t_T_ai, t_T_ai, t_T_ai)
            dict_t_V_dressed["iajk"] = t_V_iajk_dressed

        # t_V_abcd
        if "abcd" in dict_t_V_dressed:
            t_V_abcd_dressed = dict_t_V["abcd"].copy()
            t_V_abcd_dressed += -1.0 * ctf.einsum("jbcd, aj -> abcd", dict_t_V['iabc'], t_T_ai)
            t_V_abcd_dressed += -1.0 * ctf.einsum("aicd, bi -> abcd", dict_t_V['aibc'], t_T_ai)
            t_V_abcd_dressed += 1.0 * ctf.einsum("jicd, aj, bi -> abcd", dict_t_V['ijab'], t_T_ai, t_T_ai)
            dict_t_V_dressed["abcd"] = t_V_abcd_dressed

        return dict_t_V_dressed

    def get_singles_residual(self, t_fock_pq, t_T_ai, t_T_abij, dict_t_V):
        """
        Computes the residuals for the singles amplitudes.
        """
        no = self.no
        t_tilde_T_abij = ctf.tensor(t_T_abij.shape, dtype=t_T_abij.dtype, sp=t_T_abij.sp)
        t_tilde_T_abij.i("abij") << 2.0 * t_T_abij.i("abij") - 1.0 * t_T_abij.i("baij")
        t_R_ai = t_fock_pq[no:, :no].copy()
        t_R_ai += ctf.einsum("jb, abij -> ai", t_fock_pq[:no, no:], t_tilde_T_abij)
        t_R_ai += ctf.einsum("ajbc, bcij -> ai", dict_t_V['aibc'], t_tilde_T_abij)
        t_R_ai += -1.0 * ctf.einsum("kjbc, ak, bcij -> ai", dict_t_V['ijab'], t_T_ai, t_tilde_T_abij)
        t_R_ai += -1.0 * ctf.einsum("jkib, abjk -> ai", dict_t_V['ijka'], t_tilde_T_abij)
        t_R_ai += -1.0 * ctf.einsum("jkcb, ci, abjk -> ai", dict_t_V['ijab'], t_T_ai, t_tilde_T_abij)

        return t_R_ai

    def get_doubles_residual(self,
                             t_fock_pq, t_T_abij,
                             t_V_klij, t_V_ijab,
                             t_V_abij, t_V_iajb, t_V_iabj, t_V_abcd
                             ):
        """
        Computes the residuals for the doubles amplitudes.
        """
        algo_name = "ccsd.get_doubles_residual"
        # return ccd's get_residual function
        return self.get_residual(t_fock_pq, t_T_abij, t_V_klij,
                                 t_V_ijab, t_V_abij, t_V_iajb, t_V_iabj, t_V_abcd)

    def get_energy(self, t_fock_ia, t_T_ai, t_T_abij, t_V_ijab):
        '''
        calculate the CCSD correlation energy
        '''
        t_T_tmp_abij = t_T_abij + ctf.einsum("ai, bj -> abij", t_T_ai, t_T_ai)
        t_dir_ccsd_e = 2. * ctf.einsum("abij, ijab ->", t_T_tmp_abij, t_V_ijab)
        t_ex_ccsd_e = -1. * ctf.einsum("abij, ijba ->", t_T_tmp_abij, t_V_ijab)
        t_1b_e = 2.0 * ctf.einsum("ia, ai ->", t_fock_ia, t_T_ai)
        return [t_1b_e, t_dir_ccsd_e, t_ex_ccsd_e]

