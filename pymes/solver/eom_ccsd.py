import time
import numpy as np
import ctf

from pymes.solver import ccsd
from pymes.mixer import diis
from pymes.log import print_logging_info, print_title
from pymes.integral.partition import part_2_body_int


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
    def __init__(self, no, n_excit=3):
        '''
        EOM_CCSD takes in a CCSD object, because the T1, T2 and dressed
        integrals are needed.
        '''
        self.algo_name = "EOM-CCSD"
        # the ground state CCSD solver instance, which shall not be changed in this class and
        # its member functions
        self.no = no
        self.n_excit = n_excit
        self.u_singles = None
        self.u_doubles = None
        self.u_vecs = [self.u_singles, self.u_doubles]
        self.e_excit = np.zeros(n_excit)

        self.max_iter = 50

    def write_logging_info(self):
        return

    def solve(self, t_fock_pq, dict_t_V, t_T_abij):
        """
        Solve for the requested number (n_excit) of excited states vectors and
        energies.

        Params:
        -----------
            t_fock_pq: ctf tensor, (singles dressed) fock matrix
            dict_t_V: dict of ctf tensors, (singles dressed) two-body integrals
            t_T_abij: ctf tensor, the doubles amplitudes from a ground state CCSD calculation
                For EOM-CCSD, t_T_abij should be singles dressed.
                For EOM-MP2, t_T_abij is the original doubles amplitude
        Returns:
            e_exit: numpy array of size n_excit
            u_vecs: list of singles and doubles coefficients (list of tensors)
        """
        print_title("EOM-CCSD Solver", )

        print_logging_info("Initialising u tensors...", level=1)
        if self.u_singles is None:
            nv = t_T_abij.shape[0]
            self.u_singles = [ctf.tensor([nv, self.no], dtype=t_T_abij.dtype, sp=t_T_abij.sp)] * self.n_excit
        if self.u_doubles is None:
            self.u_doubles = [ctf.tensor(t_T_abij.shape, dtype=t_T_abij.dtype, sp=t_T_abij.sp)] * self.n_excit
        # start iterative solver, arnoldi or davidson
        is_converged = False
        print_logging_info("Starting iterative solver.", level=1)
        for i in range(self.max_iter):
            print_logging_info("Iter: ", i, level=2)
            if not is_converged:
                for n in range(self.n_excit):
                    print_logging_info("Updating the ", n, "-th state", level=3)
                    self.u_singles[n] += self.update_singles(t_fock_pq, dict_t_V, self.u_singles[n],
                                                       self.u_doubles[n], t_T_abij)
                    self.u_doubles[n] += self.update_doubles(t_fock_pq, dict_t_V, self.u_singles[n],
                                                          self.u_doubles[n], t_T_abij)


        return self.e_excit, self.u_vecs

    def update_singles(self, t_fock_pq, dict_t_V, t_u_ai, t_u_abij, t_T_abij):
        """
        Calculate the matrix-vector product between similarity-transformed H and u vector for the singles
        block.

        Parameters:
        -----------
        t_fock_pq: ctf tensor, fock matrix
        dict_t_V: dictionary of V blocks, which are ctf tensors
        t_u_ai: ctf tensor, the singles coefficients for the EOM-CCSD ansatz, to be updated.
        t_u_abij: ctf tensor, the doubles coefficients for EOM-CCSD ansatz (which shall not be changed in this step)
        t_T_abij: ctf tensor, the doubles amplitudes from ground state CCSD calculation
        Returns:
        --------
        t_delta_u: ctf tensor, the change of the singles block of u
        """

        no = self.no
        t_delta_singles = ctf.tensor(t_u_ai.shape,
                                     dtype=t_u_ai.dtype,
                                     sp=t_u_ai.sp)

        # fock matrix contribution
        t_delta_singles += 2. * ctf.einsum("jb, baji->ai", t_fock_pq[:no, no:],
                                           t_u_abij) \
                           - ctf.einsum("ji, aj", t_fock_pq[:no, :no],
                                        t_u_ai) \
                           - ctf.einsum("jb, abji->ai", t_fock_pq[:no, no:],
                                        t_u_abij) \
                           - ctf.einsum("ab, bi->ai", t_fock_pq[no:, no:],
                                        t_u_ai)
        # integral and t_u_ai products
        t_delta_singles += 2. * ctf.einsum("jabi, bj->ai", dict_t_V["iabj"],
                                           t_u_ai) \
                           - ctf.einsum("jaib, bj->ai", dict_t_V["iajb"],
                                        t_u_ai)
        # integral and t_u_abij products
        t_delta_singles += - 2. * ctf.einsum("jkib, abjk->ai", dict_t_V["ijka"],
                                             t_u_abij) \
                           + 2. * ctf.einsum("jabc, bcji->ai", dict_t_V["iabc"],
                                             t_u_abij) \
                           + ctf.einsum("jkib, bajk->ai", dict_t_V["ijka"],
                                        t_u_abij) \
                           - ctf.einsum("jacb, bcji->ai", dict_t_V["iabc"],
                                        t_u_abij)
        # integral, T and t_u_ai products
        t_delta_singles += 4. * ctf.einsum("jkbc, baji, ck->ai", dict_t_V["ijab"],
                                           t_T_abij, t_u_ai) \
                           - 2. * ctf.einsum("jkbc, bajk, ci->ai", dict_t_V["ijab"],
                                             t_T_abij, t_u_ai) \
                           - 2. * ctf.einsum("jkbc, bcji, ak->ai", dict_t_V["ijab"],
                                             t_T_abij, t_u_ai) \
                           - 2. * ctf.einsum("jkbc, abji, ck->ai", dict_t_V["ijab"],
                                             t_T_abij, t_u_ai) \
                           - 2. * ctf.einsum("jkcb, baji, ck->ai", dict_t_V["ijab"],
                                             t_T_abij, t_u_ai) \
                           + ctf.einsum("jkbc, abjk, ci->ai", dict_t_V["ijab"],
                                        t_T_abij, t_u_ai) \
                           + ctf.einsum("jkcb, bcji, ak->ai", dict_t_V["ijab"],
                                        t_T_abij, t_u_ai) \
                           + ctf.einsum("jkcb, abji, ck->ai", dict_t_V["ijab"],
                                        t_T_abij, t_u_ai)

        return t_delta_singles

    def update_doubles(self, t_fock_pq, dict_t_V, t_u_ai, t_u_abij, t_T_abij):
        """
        Calculate the matrix-vector product between similarity-transformed H and u vector for the singles
        block.

        Parameters:
        -----------
        t_fock_pq: ctf tensor, fock matrix
        dict_t_V: dictionary of V blocks, which are ctf tensors
        t_u_ai: ctf tensor, the singles coefficients.
        t_u_abij: ctf tensor, the doubles coefficients to be updated.
        t_T_abij: ctf tensor, the doubles amplitudes from a ground state calculation.
        Returns:
        --------
        t_delta_doubles: ctf tensor, the change of the doubles block of u
        """
        no = self.no
        t_delta_doubles = ctf.tensor(t_u_abij.shape,
                                     dtype=t_u_abij.dtype,
                                     sp=t_u_abij.sp)

        # add those involving P(ijab,jiba) and from t_u_ai, in total 18 terms
        t_delta_doubles += - 2. * ctf.einsum("klid, abkj, dl -> abij", dict_t_V["ijka"], t_T_abij,
                                             t_u_ai) \
                           - 2. * ctf.einsum("klci, cbkj, al -> abij", dict_t_V["ijak"], t_T_abij,
                                             t_u_ai) \
                           + 2. * ctf.einsum("kacd, cbkj, di -> abij", dict_t_V["iabc"], t_T_abij,
                                             t_u_ai) \
                           + 2. * ctf.einsum("ladc, cbij, dl -> abij", dict_t_V["iabc"], t_T_abij,
                                             t_u_ai) \
                           - 1. * ctf.einsum("kd, abkj, di -> abij", t_fock_pq[:no, no:], dict_t_V["abij"],
                                             t_u_ai) \
                           - 1. * ctf.einsum("lc, cbij, al -> abij", t_fock_pq[:no, no:], dict_t_V["abij"],
                                             t_u_ai) \
                           + 1. * ctf.einsum("klid, abkl, dj -> abij", dict_t_V["ijka"], t_T_abij,
                                             t_u_ai) \
                           + 1. * ctf.einsum("klic, cbkj, al -> abij", dict_t_V["ijka"], t_T_abij,
                                             t_u_ai) \
                           + 1. * ctf.einsum("klid, adkj, bl -> abij", dict_t_V["ijka"], t_T_abij,
                                             t_u_ai) \
                           - 1. * ctf.einsum("kbij, ak -> abij", dict_t_V["iajk"], t_u_ai) \
                           + 1. * ctf.einsum("kldi, bdkj, al -> abij", dict_t_V["ijak"], t_T_abij,
                                             t_u_ai) \
                           - 1. * ctf.einsum("kacd, bckj, di -> abij", dict_t_V["iabc"], t_T_abij,
                                             t_u_ai) \
                           + 1. * ctf.einsum("kldi, abkj, dl -> abij", dict_t_V["ijak"], t_T_abij,
                                             t_u_ai) \
                           - 1. * ctf.einsum("kadc, cbkj, di -> abij", dict_t_V["iabc"], t_T_abij,
                                             t_u_ai) \
                           - 1. * ctf.einsum("kadc, bcki, dj -> abij", dict_t_V["iabc"], t_T_abij,
                                             t_u_ai) \
                           - 1. * ctf.einsum("lacd, cdji, bl -> abij", dict_t_V["iabc"], t_T_abij,
                                             t_u_ai) \
                           - 1. * ctf.einsum("lacd, cbij, dl -> abij", dict_t_V["iabc"], t_T_abij,
                                             t_u_ai) \
                           + 1. * ctf.einsum("abic, cj -> abij", dict_t_V["abic"], t_u_ai)

        # add those involving P(ijab,jiba) and from t_u_abij, in total 22 terms
        t_delta_doubles += + 4. * ctf.einsum("klcd, caki, dblj -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           - 2. * ctf.einsum("klcd, cakl, dbij -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           - 2. * ctf.einsum("klcd, cdki, ablj -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           - 2. * ctf.einsum("klcd, caki, bdlj -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           + 2. * ctf.einsum("kaci, cbkj -> abij", dict_t_V["iabj"],
                                             t_u_abij) \
                           - 2. * ctf.einsum("klcd, acki, dblj -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           - 2. * ctf.einsum("kldc, caki, dblj -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           - 2. * ctf.einsum("klcd, caki, dblj -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           - 2. * ctf.einsum("lkcd, cbij, adlk -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           - 1. * ctf.einsum("ki, abkj -> abij", t_fock_pq[:no, no:],
                                             t_u_abij) \
                           + 1. * ctf.einsum("ac, cbij -> abij", t_fock_pq[no:, no:],
                                             t_u_abij) \
                           - 1. * ctf.einsum("kaic, cbkj -> abij", dict_t_V["iajb"],
                                             t_u_abij) \
                           - 1. * ctf.einsum("kbic, ackj -> abij", dict_t_V["iajb"],
                                             t_u_abij) \
                           + 1. * ctf.einsum("klcd, ackl, dbij -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           + 1. * ctf.einsum("kldc, dcki, ablj -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           + 1. * ctf.einsum("klcd, acki, bdlj -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           - 1. * ctf.einsum("kaci, bckj -> abij", dict_t_V["iabj"],
                                             t_u_abij) \
                           + 1. * ctf.einsum("kldc, acki, dblj -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           + 1. * ctf.einsum("kldc, abkj, dcli -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           + 1. * ctf.einsum("kldc, caki, dbjl -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           + 1. * ctf.einsum("kldc, ackj, dbil -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij) \
                           + 1. * ctf.einsum("lkcd, cbij, dalk -> abij", dict_t_V["ijab"], t_T_abij,
                                             t_u_abij)

        # add exchange contributions
        t_delta_doubles.i("abij") << t_delta_doubles.i("baji")
        # after adding exchanging indices contribution from P(ijab, jiba),
        # now add all terms that don't involve P(ijab,jiba)
        t_delta_doubles +=  ctf.einsum("klij, abkl -> abij", dict_t_V["klij"], t_u_abij) \
                           + ctf.einsum("kldc, abkl, dcij -> abij", dict_t_V["ijab"], t_T_abij,
                                        t_u_abij) \
                           + ctf.einsum("lkcd, cdij, ablk -> abij", dict_t_V["ijab"], t_T_abij,
                                        t_u_abij) \
                           + ctf.einsum("abcd, cdij -> abij", dict_t_V["abcd"], t_u_abij)

        return t_delta_doubles
