import time
import numpy as np
import ctf
from ctf.core import *
from pymes.solver import ccsd
from pymes.mixer import diis
from pymes.log import print_logging_info

'''
EOM-CCSD implementation
EOM-CCSD is used to calculate excitation energies of a system.
This implementation will take the transcorrelation into account,
i.e. the lower symmetries in the tc integrals than in the normal Coulomb
integrals. 

EOM-CCSD needs the amplitudes from a prior CCSD calculation and the integrals.
The main algorithm consists of the multiplication of the similarity-transformed
Hamiltonian and the u vectors, where the elements of the u vectors are the 
coefficients of the linear ansatz for excited states.

This implementation is for closed-shell systems. And pertinent equations are obtained
by quantwo software developed by Daniel Kats. The pdf version can be found in the doc 
directory of pymes.

Author: Ke Liao <ke.liao.whu@gmail.com>
'''


class EOM_CCSD:
    def __init__(self, ccsd, n_excit=3):
        """
        EOM_CCSD takes in a CCSD object, because the T1, T2 and dressed
        integrals are needed.
        """
        self.algo_name = "EOM-CCSD"
        self.ccsd = ccsd
        self.n_excit = n_excit
        self.u_singles = []
        self.u_doubles = []
        self.e_excit = np.zeros(n_excit)
        self.max_dim = n_excit * 4
        self.e_epsilon = 1.e-5

    def write_logging_info(self):
        return

    def solve(self, t_fock_pq, t_V_pqrs):
        """
        Solve for the requested number (n_excit) of excited states vectors and
        energies.

        """
        time_init = time.time()
        dict_t_V = self.ccsd.partition_V(t_V_pqrs)
        t_fock_dressed_pq = self.ccsd.get_T1_dressed_fock(t_fock_pq,
                                                          self.ccsd.t_T_ai,
                                                          dict_t_V)
        dict_t_V_dressed = self.ccsd.get_T1_dressed_V(self.ccsd.t_T_ai,
                                                      dict_t_V)

        # build guesses
        no = self.ccsd.no
        t_epsilon_i = t_fock_pq.diagonal()[:no]
        t_epsilon_a = t_fock_pq.diagonal()[no:]
        t_D_ai = ctf.tensor(self.ccsd.t_T_ai.shape)
        t_D_abij = ctf.tensor(self.ccsd.t_T_abij.shape)
        t_D_ai.i("ai") << t_epsilon_i.i("i") - t_epsilon_a.i("a")
        t_D_abij.i("abij") << t_epsilon_i.i("i") + t_epsilon_i.i("j") \
                              - t_epsilon_a.i("a") - t_epsilon_a.i("b")
        D_ai = -t_D_ai.to_nparray().ravel()
        lowest_ex_ind = np.argsort(D_ai)[:self.n_excit]

        for i in range(self.n_excit):
            A = np.zeros(t_D_ai.shape).ravel()
            A[lowest_ex_ind] = 1.
            A = A.reshape(-1, no)
            self.u_singles.append(ctf.astensor(A))
            self.u_doubles.append(ctf.tensor(t_D_abij.shape))

        # start iterative solver, arnoldi or davidson
        # need QR decomposition of the matrix made up of the states to ensure the orthogonality among them~~
        # u_singles, u_doubles = QR(u_singles, u_doubles)
        # in a first attempt, we don't need the QR decomposition
        for i in range(self.max_it):
            time_iter_init = time.time()
            subspace_dim = len(self.u_singles)
            w_singles = [ctf.tensor(A.shape, dtype=A.dtype, sp=A.sp)] * subspace_dim
            w_doubles = [ctf.tensor(t_D_abij.shape, dtype=t_D_abij.dtype, sp=t_D_abij.sp)] * subspace_dim
            B = np.zeros([subspace_dim, subspace_dim])
            for l in range(subspace_dim):
                w_singles[l] += self.update_singles(t_fock_dressed_pq,
                                                      dict_t_V_dressed, self.u_singles[l],
                                                      self.u_doubles[l])
                w_doubles[l] += self.update_doubles(t_fock_dressed_pq,
                                                      dict_t_V_dressed, self.u_singles[l],
                                                      self.u_doubles[l])
                # build Hamiltonian inside subspace
                for j in range(l):
                    B[j, l] = ctf.einsum("ai, ai->", self.u_singles[j], w_singles[l]) \
                             + ctf.einsum("abij, abij->", self.u_doubles[j], w_doubles[l])
                    B[l, j] = ctf.einsum("ai, ai->", self.u_singles[l], w_singles[j]) \
                              + ctf.einsum("abij, abij->", self.u_doubles[l], w_doubles[j])
                B[l, l] = ctf.einsum("ai, ai->", self.u_singles[l], w_singles[l]) \
                          + ctf.einsum("abij, abij->", self.u_doubles[l], w_doubles[l])

           # diagnolise matrix B, find the lowest energies
            e, v = np.linalg.eig(B)
            lowest_ex_ind = e.argsort()[:self.n_excit]
            e = e[lowest_ex_ind]
            v = v[:, lowest_ex_ind]

            # construct residuals
            y_singles = ctf.tensor(w_singles[l].shape, dtype=w_singles[l].dtype, sp=w_singles.sp)
            y_doubles = ctf.tensor(w_doubles[l].shape, dtype=w_doubles[l].dtype, sp=w_doubles.sp)
            if subspace_dim >= self.max_dim:
                for n in range(self.n_excit):
                    y_singles.set_zero()
                    y_doubles.set_zero()
                    for l in range(subspace_dim):
                        y_singles += self.u_singles[l] * v[l, n]
                        y_doubles += self.u_doubles[l] * v[l, n]
                    self.u_singles[n] = y_singles
                    self.u_doubles[n] = y_doubles
                self.u_singles = self.u_singles[:self.n_excit]
                self.u_doubles = self.u_doubles[:self.n_excit]
            else:
                for n in range(self.n_excit):
                    for l in range(subspace_dim):
                        y_singles += w_singles[l] * v[l, n]
                        y_doubles += w_doubles[l] * v[l, n]
                        y_singles -= e[n] * self.u_singles[l] * v[l, n]
                        y_doubles -= e[n] * self.u_doubles[l] * v[l, n]
                    self.u_singles.append(y_singles/(e[n]-D_ai[n]))
                    self.u_doubles.append(y_doubles/(e[n]-D_ai[n]))

            diff_e_norm = np.linalg.norm(self.e_excit - e)
            if diff_e_norm < self.e_epsilon:
                print_logging_info("Iterative solver converged.", level=1)
                break
            else:
                print_logging_info("Iteration = ", i, level = 2)
                print_logging_info("Norm of energy difference = ", diff_e_norm, level = 2)
                print_logging_info("Excited states energies = ", e, level = 2)
                print_logging_info("Took {.3f} seconds ".format(time.time()-time_iter_init), level=2)
        print_logging_info("EOM-CCSD finished in {.3f} seconds".format(time.time()-time_init), level=1)
        print_logging_info("Converged excited states energies = ", e, level=1)
        self.e_excit = e

        return self.e_excit

    def update_singles(self, t_fock_pq, dict_t_V, u_singles_n, u_doubles_n):
        """
        Calculate the matrix-vector product between similarity-transformed H and u vector for the singles
        block.

        Parameters:
        -----------
        t_fock_pq: ctf tensor, fock matrix
        dict_t_V: dictionary of V blocks, which are ctf tensors
        u_singles_n: ctf tensor, the singles part of the nth state
        u_doubles_n: ctf tensor, the doubles part of the nth state
        Returns:
        --------
        t_delta_singles: ctf tensor, the change of the singles block of u for the nth state
        """

        no = self.ccsd.no
        t_delta_singles = ctf.tensor(u_singles_n.shape,
                                     dtype=u_singles_n.dtype,
                                     sp=u_singles_n.sp)

        # fock matrix contribution
        t_delta_singles += 2. * ctf.einsum("jb, baji->ai", t_fock_pq[:no, no:],
                                           u_doubles_n) \
                           - ctf.einsum("ij, aj", t_fock_pq[:no, :no],
                                        u_singles_n) \
                           - ctf.einsum("bj, abji->ai", t_fock_pq[:no, no:],
                                        u_doubles_n) \
                           - ctf.einsum("ba, bi->ai", t_fock_pq[no:, no:],
                                        u_singles_n)
        # integral and u_singles products
        t_delta_singles += 2. * ctf.einsum("jabi, bj->ai", dict_t_V["iabj"],
                                           u_singles_n) \
                           - ctf.einsum("jaib, bj->ai", dict_t_V["iajb"],
                                        u_singles_n)
        # integral and u_doubles products
        t_delta_singles += -2. * ctf.einsum("jkib, abjk->ai", dict_t_V["ijka"],
                                            u_doubles_n) \
                           + 2. * ctf.einsum("jabc, bcji->ai", dict_t_V["iabc"],
                                             u_doubles_n) \
                           + ctf.einsum("jkib, bajk->ai", dict_t_V["ijka"],
                                        u_doubles_n) \
                           - ctf.einsum("jacb, bcji->ai", dict_t_V["iabc"],
                                        u_doubles_n)
        # integral, T and u_singles products
        t_delta_singles += 4. * ctf.einsum("jkbc, baji, ck->ai", dict_t_V["ijab"],
                                           self.ccsd.t_T_abij, u_singles_n) \
                           - 2. * ctf.einsum("jkbc, bajk, ci->ai", dict_t_V["ijab"],
                                             self.ccsd.t_T_abij, u_singles_n) \
                           - 2. * ctf.einsum("jkbc, bcji, ak->ai", dict_t_V["ijab"],
                                             self.ccsd.t_T_abij, u_singles_n) \
                           - 2. * ctf.einsum("jkbc, abji, ak->ai", dict_t_V["ijab"],
                                             self.ccsd.t_T_abij, u_singles_n) \
                           - 2. * ctf.einsum("jkcb, baji, ck->ai", dict_t_V["ijab"],
                                             self.ccsd.t_T_abij, u_singles_n) \
                           + ctf.einsum("jkbc, abjk, ci->ai", dict_t_V["ijab"],
                                        self.ccsd.t_T_abij, u_singles_n) \
                           + ctf.einsum("jkcb, bcji, ak->ai", dict_t_V["ijab"],
                                        self.ccsd.t_T_abij, u_singles_n) \
                           + ctf.einsum("jkcb, abji, ck->ai", dict_t_V["ijab"],
                                        self.ccsd.t_T_abij, u_singles_n)
        return t_delta_singles

    def update_doubles(self, t_fock_pq, dict_t_V, u_singles_n, u_doubles_n):
        no = self.ccsd.no
        t_delta_doubles = ctf.tensor(u_doubles_n.shape,
                                     dtype=u_doubles_n.dtype,
                                     sp=u_doubles_n.sp)

        # add those involving P(ijab,jiba) and from u_singles, in total 18 terms
        t_delta_doubles += - 2. * ctf.einsum("klid, abkj, dl -> abij", dict_t_V["ijka"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           - 2. * ctf.einsum("klci, cbkj, al -> abij", dict_t_V["ijak"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           + 2. * ctf.einsum("kacd, cbkj, di -> abij", dict_t_V["iabc"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           + 2. * ctf.einsum("ladc, cbij, dl -> abij", dict_t_V["iabc"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           - 1. * ctf.einsum("kd, abkj, di -> abij", t_fock_pq[:no, no:], dict_t_V["abij"],
                                             u_singles_n) \
                           - 1. * ctf.einsum("lc, cbij, al -> abij", t_fock_pq[:no, no:], dict_t_V["abij"],
                                             u_singles_n) \
                           + 1. * ctf.einsum("klid, abkl, dj -> abij", dict_t_V["ijka"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           + 1. * ctf.einsum("klic, cbkj, al -> abij", dict_t_V["ijka"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           + 1. * ctf.einsum("klid, adkj, bl -> abij", dict_t_V["ijka"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           - 1. * ctf.einsum("kbij, ak -> abij", dict_t_V["iajk"], u_singles_n) \
                           + 1. * ctf.einsum("kldi, bdkj, al -> abij", dict_t_V["ijak"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           - 1. * ctf.einsum("kacd, bckj, di -> abij", dict_t_V["iabc"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           + 1. * ctf.einsum("kldi, abkj, dl -> abij", dict_t_V["ijak"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           - 1. * ctf.einsum("kadc, cbkj, di -> abij", dict_t_V["iabc"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           - 1. * ctf.einsum("kadc, bcki, dj -> abij", dict_t_V["iabc"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           - 1. * ctf.einsum("lacd, cdji, bl -> abij", dict_t_V["iabc"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           - 1. * ctf.einsum("lacd, cbij, dl -> abij", dict_t_V["iabc"], self.ccsd.t_T_abij,
                                             u_singles_n) \
                           + 1. * ctf.einsum("abic, cj -> abij", dict_t_V["abic"], u_singles_n)

        # add those involving P(ijab,jiba) and from u_doubles, in total 22 terms
        t_delta_doubles += + 4. * ctf.einsum("klcd, caki, dblj -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           - 2. * ctf.einsum("klcd, cakl, dbij -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           - 2. * ctf.einsum("klcd, cdki, ablj -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           - 2. * ctf.einsum("klcd, caki, bdlj -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           + 2. * ctf.einsum("kaci, cbkj -> abij", dict_t_V["iabj"],
                                             u_doubles_n) \
                           - 2. * ctf.einsum("klcd, acki, dblj -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           - 2. * ctf.einsum("kldc, caki, dblj -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           - 2. * ctf.einsum("klcd, caki, dblj -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           - 2. * ctf.einsum("lkcd, cbij, adlk -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           - 1. * ctf.einsum("ki, abkj -> abij", t_fock_pq[:no, no:],
                                             u_doubles_n) \
                           + 1. * ctf.einsum("ac, cbij -> abij", t_fock_pq[no:, no:],
                                             u_doubles_n) \
                           - 1. * ctf.einsum("kaic, cbkj -> abij", dict_t_V["iajb"],
                                             u_doubles_n) \
                           - 1. * ctf.einsum("kbic, ackj -> abij", dict_t_V["iajb"],
                                             u_doubles_n) \
                           + 1. * ctf.einsum("klcd, ackl, dbij -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           + 1. * ctf.einsum("kldc, dcki, ablj -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           + 1. * ctf.einsum("klcd, acki, bdlj -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           - 1. * ctf.einsum("kaci, bckj -> abij", dict_t_V["iabj"],
                                             u_doubles_n) \
                           + 1. * ctf.einsum("kldc, acki, dblj -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           + 1. * ctf.einsum("kldc, abkj, dcli -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           + 1. * ctf.einsum("kldc, caki, dbjl -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           + 1. * ctf.einsum("kldc, ackj, dbil -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n) \
                           + 1. * ctf.einsum("lkcd, cbij, dalk -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                             u_doubles_n)

        # add exchange contributions
        t_delta_doubles.i("abij") << t_delta_doubles.i("baji")
        # after adding exchanging indices contribution from P(ijab, jiba),
        # now add all terms that don't involve P(ijab,jiba)
        t_delta_doubles += + ctf.einsum("klij, abkl -> abij", dict_t_V["ijkl"], u_doubles_n) \
                           + ctf.einsum("kldc, abkl, dcij -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                        u_doubles_n) \
                           + ctf.einsum("lkcd, cdij, ablk -> abij", dict_t_V["ijab"], self.ccsd.t_T_abij,
                                        u_doubles_n) \
                           + ctf.einsum("abcd, cdij -> abij", dict_t_V["abcd"], u_doubles_n)

        return t_delta_doubles

    def QR(self, u_singles, u_doubles):
        """
        This QR algorithm is designed to orthogonalize the states, in consideration of the ctf date structure
        and aiming to minimize
        the memory footprint. Each state consists of a singles and doubles block.
        Parameters:
        -----------
        u_singles: list of ctf tensors, list of singles coefficients
        u_doubles: list of ctf tensors, list of doubles coefficients

        Returns:
        --------
        u_singles_: list of ctf tensors, which are now orthogonalised
        u_doubles_: list of ctf tensors, which are now orthogonalised
        """

        return u_singles_, u_doubles_
