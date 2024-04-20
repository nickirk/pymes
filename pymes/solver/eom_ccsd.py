import time
import numpy as np
import ctf

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
        self.algo_name = "EOM-CCSD"
        self.no = no
        self.n_excit = n_excit
        self.u_singles = []
        self.u_doubles = []
        self.e_excit = np.zeros(n_excit)
        self.max_dim = n_excit * 8
        self.e_epsilon = 1.e-8

        self.max_iter = 200

    def write_logging_info(self):
        return

    def solve(self, t_fock_dressed_pq, dict_t_V_dressed, t_T_abij):
        """
        Solve for the requested number (n_excit) of excited states vectors and
        energies.

        Params:
        -----------
            t_fock_pq: ctf tensor, (singles dressed) fock matrix
            dict_t_V: dict of ctf tensors, (singles dressed) two-body integrals
            t_T_abij: ctf tensor, the doubles amplitudes from a ground state CCSD calculation
                For EOM-CCSD, t_T_abij should be singles dressed. ??
                For EOM-MP2, t_T_abij is the original doubles amplitude
        Returns:
            e_exit: numpy array of size n_excit
            u_vecs: list of singles and doubles coefficients (list of tensors)
        """
        print_title("EOM-CCSD Solver", )
        time_init = time.time()
        # build guesses
        no = self.no
        t_epsilon_i = t_fock_dressed_pq.diagonal()[:no]
        t_epsilon_a = t_fock_dressed_pq.diagonal()[no:]
        nv = t_epsilon_a.shape[0]
        t_D_ai = ctf.tensor([nv, no])
        t_D_abij = ctf.tensor(t_T_abij.shape)
        t_D_ai.i("ai") << t_epsilon_i.i("i") - t_epsilon_a.i("a")
        t_D_abij.i("abij") << t_epsilon_i.i("i") + t_epsilon_i.i("j") \
                              - t_epsilon_a.i("a") - t_epsilon_a.i("b")
        D_ai = -t_D_ai.to_nparray().ravel()
        lowest_ex_ind_init = np.argsort(D_ai)[:self.n_excit]

        print_logging_info("Initialising u tensors...", level=1)
        for i in range(self.n_excit):
            A = np.zeros(t_D_ai.shape).ravel()
            A[lowest_ex_ind_init[i]] = 1.
            A = A.reshape(-1, no)
            self.u_singles.append(ctf.astensor(A))
            self.u_doubles.append(ctf.tensor(t_D_abij.shape))

        # start iterative solver, arnoldi or davidson
        # need QR decomposition of the matrix made up of the states to ensure the orthogonality among them~~
        # u_singles, u_doubles = QR(u_singles, u_doubles)
        # in a first attempt, we don't need the QR decomposition
        for i in range(self.max_iter):
            time_iter_init = time.time()
            subspace_dim = len(self.u_singles)
            self.u_singles, self.u_doubles = self.QR(self.u_singles, self.u_doubles)
            w_singles = [ctf.tensor(t_D_ai.shape, dtype=t_D_ai.dtype, sp=t_D_ai.sp) for _ in range(subspace_dim)]
            w_doubles = [ctf.tensor(t_D_abij.shape, dtype=t_D_abij.dtype, sp=t_D_abij.sp) for _ in range(subspace_dim)]
            B = np.zeros([subspace_dim, subspace_dim])
            for l in range(subspace_dim):
                w_singles[l] = self.update_singles(t_fock_dressed_pq,
                                                   dict_t_V_dressed, self.u_singles[l],
                                                   self.u_doubles[l], t_T_abij)
                w_doubles[l] = self.update_doubles(t_fock_dressed_pq,
                                                   dict_t_V_dressed, self.u_singles[l],
                                                   self.u_doubles[l], t_T_abij)
                # build Hamiltonian inside subspace
                for j in range(l):
                    B[j, l] = ctf.einsum("ai, ai->", self.u_singles[j], w_singles[l]) \
                              + ctf.einsum("abij, abij->", self.u_doubles[j], w_doubles[l])
                    B[l, j] = ctf.einsum("ai, ai->", self.u_singles[l], w_singles[j]) \
                              + ctf.einsum("abij, abij->", self.u_doubles[l], w_doubles[j])
                B[l, l] = ctf.einsum("ai, ai->", self.u_singles[l], w_singles[l]) \
                          + ctf.einsum("abij, abij->", self.u_doubles[l], w_doubles[l])
            # diagnolise matrix B, find the lowest energies
            e_old = self.e_excit
            e, v = np.linalg.eig(B)
            lowest_ex_ind = e.argsort()[:self.n_excit]
            e_imag = np.imag(e[lowest_ex_ind])
            e = np.real(e[lowest_ex_ind])
            v_imag = np.imag(v[:, lowest_ex_ind])
            v = np.real(v[:, lowest_ex_ind])

            # construct residuals
            u_singles_tmp = []
            u_doubles_tmp = []
            if subspace_dim >= self.max_dim:
                for n in range(self.n_excit):
                    y_singles = ctf.tensor(w_singles[-1].shape, dtype=w_singles[-1].dtype, sp=w_singles[-1].sp)
                    y_doubles = ctf.tensor(w_doubles[-1].shape, dtype=w_doubles[-1].dtype, sp=w_doubles[-1].sp)
                    for l in range(subspace_dim):
                        y_singles += self.u_singles[l] * v[l, n]
                        y_doubles += self.u_doubles[l] * v[l, n]
                    u_singles_tmp.append(y_singles)
                    u_doubles_tmp.append(y_doubles)
                self.u_singles = u_singles_tmp
                self.u_doubles = u_doubles_tmp
                self.e_excit = e_old
            else:
                for n in range(self.n_excit):
                    y_singles = ctf.tensor(w_singles[-1].shape, dtype=w_singles[-1].dtype, sp=w_singles[-1].sp)
                    y_doubles = ctf.tensor(w_doubles[-1].shape, dtype=w_doubles[-1].dtype, sp=w_doubles[-1].sp)
                    for l in range(subspace_dim):
                        y_singles += w_singles[l] * v[l, n]
                        y_doubles += w_doubles[l] * v[l, n]
                        y_singles -= e[n] * self.u_singles[l] * v[l, n]
                        y_doubles -= e[n] * self.u_doubles[l] * v[l, n]
                    self.u_singles.append(y_singles / (e[n] - D_ai[lowest_ex_ind_init[n]] + 1e-5))
                    self.u_doubles.append(y_doubles / (e[n] - D_ai[lowest_ex_ind_init[n]] + 1e-5))
                e_old = self.e_excit
                diff_e_norm = np.linalg.norm(self.e_excit - e)
                self.e_excit = e
            if diff_e_norm < self.e_epsilon:
                print_logging_info("Iterative solver converged.", level=1)
                print_logging_info("Norm of energy difference = {:.12f}".format(diff_e_norm), level=2)
                for r in range(self.n_excit):
                    print_logging_info("Excited state {:d} energy = {:.12f}".format(r, e[r]), level=2)
                print_logging_info("Excited states energies imaginary part = ", e_imag, level=2)
                break
            else:
                print_logging_info("Iteration = ", i, level=1)
                print_logging_info("Norm of energy difference = ", diff_e_norm, level=2)
                for r in range(self.n_excit):
                    print_logging_info("Excited state {:d} energy = {:.12f}".format(r, e[r]), level=2)
                print_logging_info("Excited states energies imaginary part = ", e_imag, level=2)
                print_logging_info("Took {:.3f} seconds ".format(time.time() - time_iter_init), level=2)
        print_logging_info("EOM-CCSD finished in {:.3f} seconds".format(time.time() - time_init), level=1)
        print_logging_info("Converged excited states energies:", level=1)
        for r in range(self.n_excit):
            print_logging_info("Excited state {:d} energy = {:.12f}".format(r, e[r]), level=2)

        return self.e_excit

    def get_diag_singles(self, t_fock_pq, dict_t_V, t_T_abij):
        """
        Get the diagonal elements of the singles block of the similarity-transformed Hamiltonian.

        """
        no = self.no
        nv = t_fock_pq.shape[0] - no
        diag_singles = ctf.tensor([nv, no], dtype=t_fock_pq.dtype, sp=t_fock_pq.sp)

        # integral and t_u_ai products
        diag_singles.i("ai") << -1. * t_fock_pq[:no, :no].i("ii") + 1. * t_fock_pq[no:, no:].i("aa")
        diag_singles.i("ai") << 2. * dict_t_V["iabj"].i("iaai") - 1. * dict_t_V["iajb"].i("iaia")
        # integral, T and t_u_ai products
        diag_singles += 4. * ctf.einsum("jiba, baji->ai", dict_t_V["ijab"], t_T_abij)
        diag_singles.i("ai") << -2. * ctf.einsum("jkba, bajk->a", dict_t_V["ijab"], t_T_abij).i("a")
        diag_singles.i("ai") << -2. * ctf.einsum("jibc, bcji->i", dict_t_V["ijab"], t_T_abij).i("i")
        diag_singles += -2. * ctf.einsum("jiba, abji->ai", dict_t_V["ijab"], t_T_abij)
        diag_singles += -2. * ctf.einsum("jiab, baji->ai", dict_t_V["ijab"], t_T_abij)
        diag_singles.i("ai") << +1. * ctf.einsum("jkba, abjk->a", dict_t_V["ijab"], t_T_abij).i("a")
        diag_singles.i("ai") << +1. * ctf.einsum("jicb, bcji->i", dict_t_V["ijab"], t_T_abij).i("i")
        diag_singles += +1. * ctf.einsum("jiab, abji->ai", dict_t_V["ijab"], t_T_abij)

        return diag_singles
    
    def get_diag_doubles(self, t_fock_pq, dict_t_V, t_T_abij):
        """
        Get the diagonal elements of the doubles block of the similarity-transformed Hamiltonian.
        """
        no = self.no
        nv = t_fock_pq.shape[0] - no
        diag_doubles = ctf.tensor([nv, nv, no, no], dtype=t_fock_pq.dtype, sp=t_fock_pq.sp)

        # add those involving P(ijab,jiba) and from t_u_abij, in total 22 terms
        diag_doubles.i("abij") << +4. * ctf.einsum("kica, caki -> ai", dict_t_V["ijab"], t_T_abij).i("ai")
        diag_doubles.i("abij") << -2. * ctf.einsum("klca, cakl  -> a", dict_t_V["ijab"], t_T_abij).i("a")
        diag_doubles.i("abij") << -2. * ctf.einsum("kicd, cdki -> i", dict_t_V["ijab"], t_T_abij).i("i")
        diag_doubles.i("abij") << -2. * ctf.einsum("kica, caki  -> ai", dict_t_V["ijab"], t_T_abij).i("ai")
        diag_doubles.i("abij") << +2. * dict_t_V["iabj"].i("iaai")
        diag_doubles.i("abij") << -2. * ctf.einsum("kica, acki  -> ai", dict_t_V["ijab"], t_T_abij).i("ai")
        diag_doubles.i("abij") << -2. * ctf.einsum("kiac, caki -> ai", dict_t_V["ijab"], t_T_abij).i("ai")
        diag_doubles.i("abij") << -2. * ctf.einsum("kjab, abkj -> abj", dict_t_V["ijab"], t_T_abij).i("abj")
        diag_doubles.i("abij") << -2. * ctf.einsum("ijcb, cbij  -> ij", dict_t_V["ijab"], t_T_abij).i("ij")
        diag_doubles.i("abij") << -1. * t_fock_pq[:no, :no].i("ii") + 1. * t_fock_pq[no:, no:].i("aa")
        diag_doubles.i("abij") << -1. * ctf.einsum("iaia -> ai", dict_t_V["iajb"]).i("ai")
        diag_doubles.i("abij") << -1. * ctf.einsum("ibib -> bi", dict_t_V["iajb"]).i("bi")
        diag_doubles.i("abij") << +1. * ctf.einsum("klca, ackl  -> a", dict_t_V["ijab"], t_T_abij).i("a")
        diag_doubles.i("abij") << +1. * ctf.einsum("kidc, cdki -> i", dict_t_V["ijab"], t_T_abij).i("i")
        diag_doubles.i("abij") << +1. * ctf.einsum("kicb, acki -> ai", dict_t_V["ijab"], t_T_abij).i("ai")
        diag_doubles.i("abij") << -1. * dict_t_V["iabj"].i("iaai")
        diag_doubles.i("abij") << +1. * ctf.einsum("kiac, acki -> ai", dict_t_V["ijab"], t_T_abij).i("ai")
        diag_doubles.i("abij") << +1. * ctf.einsum("kiab, abkj -> abij", dict_t_V["ijab"], t_T_abij).i("abij")
        diag_doubles.i("abij") << +1. * ctf.einsum("kjac, caki -> aij", dict_t_V["ijab"], t_T_abij).i("aij")
        diag_doubles.i("abij") << +1. * ctf.einsum("kjac, ackj -> aj", dict_t_V["ijab"], t_T_abij).i("aj")
        diag_doubles.i("abij") << +1. * ctf.einsum("ijca, cbij -> abij", dict_t_V["ijab"], t_T_abij).i("abij")

        # add exchange contributions
        diag_doubles.i("abij") << diag_doubles.i("baji")
        # after adding exchanging indices contribution from P(ijab, jiba),
        # now add all terms that don't involve P(ijab,jiba)
        diag_doubles.i("abij") << ctf.einsum("ijij-> ij", dict_t_V["klij"]).i("ij")
        diag_doubles.i("abij") << ctf.einsum("klab, abkl->ab", dict_t_V["ijab"], t_T_abij).i("ab")
        diag_doubles.i("abij") << ctf.einsum("ijcd, cdij->ij", dict_t_V["ijab"], t_T_abij).i("ij")
        diag_doubles.i("abij") << dict_t_V["abcd"].i("abab")

        return diag_doubles

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
        t_delta_singles: ctf tensor, the change of the singles block of u for the nth state
        """

        no = self.no
        t_delta_singles = ctf.tensor(t_u_ai.shape, dtype=t_u_ai.dtype, sp=t_u_ai.sp)

        t_delta_singles += 2. * ctf.einsum("jb, baji->ai", t_fock_pq[:no, no:], t_u_abij)
        t_delta_singles += -1. * ctf.einsum("ji, aj -> ai", t_fock_pq[:no, :no], t_u_ai)
        t_delta_singles += -1. * ctf.einsum("jb, abji->ai", t_fock_pq[:no, no:], t_u_abij)
        t_delta_singles += 1. * ctf.einsum("ab, bi->ai", t_fock_pq[no:, no:], t_u_ai)
        # integral and t_u_ai products
        t_delta_singles += 2. * ctf.einsum("jabi, bj->ai", dict_t_V["iabj"], t_u_ai)
        t_delta_singles += -1. * ctf.einsum("jaib, bj->ai", dict_t_V["iajb"], t_u_ai)
        # integral and t_u_abij products
        t_delta_singles += -2. * ctf.einsum("jkib, abjk->ai", dict_t_V["ijka"], t_u_abij)
        t_delta_singles += 2. * ctf.einsum("jabc, bcji->ai", dict_t_V["iabc"], t_u_abij)
        t_delta_singles += ctf.einsum("jkib, bajk->ai", dict_t_V["ijka"], t_u_abij)
        t_delta_singles += -1. * ctf.einsum("jacb, bcji->ai", dict_t_V["iabc"], t_u_abij)
        # integral, T and t_u_ai products
        t_delta_singles += 4. * ctf.einsum("jkbc, baji, ck->ai", dict_t_V["ijab"], t_T_abij, t_u_ai)
        t_delta_singles += -2. * ctf.einsum("jkbc, bajk, ci->ai", dict_t_V["ijab"], t_T_abij, t_u_ai)
        t_delta_singles += -2. * ctf.einsum("jkbc, bcji, ak->ai", dict_t_V["ijab"], t_T_abij, t_u_ai)
        t_delta_singles += -2. * ctf.einsum("jkbc, abji, ck->ai", dict_t_V["ijab"], t_T_abij, t_u_ai)
        t_delta_singles += -2. * ctf.einsum("jkcb, baji, ck->ai", dict_t_V["ijab"], t_T_abij, t_u_ai)
        t_delta_singles += +1. * ctf.einsum("jkbc, abjk, ci->ai", dict_t_V["ijab"], t_T_abij, t_u_ai)
        t_delta_singles += +1. * ctf.einsum("jkcb, bcji, ak->ai", dict_t_V["ijab"], t_T_abij, t_u_ai)
        t_delta_singles += +1. * ctf.einsum("jkcb, abji, ck->ai", dict_t_V["ijab"], t_T_abij, t_u_ai)

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
        t_delta_doubles = ctf.tensor(t_u_abij.shape, dtype=t_u_abij.dtype, sp=t_u_abij.sp)

        # add those involving P(ijab,jiba) and from t_u_ai, in total 18 terms
        t_delta_doubles += - 2. * ctf.einsum("klid, abkj, dl -> abij", dict_t_V["ijka"], t_T_abij, t_u_ai)
        t_delta_doubles += - 2. * ctf.einsum("klci, cbkj, al -> abij", dict_t_V["ijak"], t_T_abij, t_u_ai)
        t_delta_doubles += + 2. * ctf.einsum("kacd, cbkj, di -> abij", dict_t_V["iabc"], t_T_abij, t_u_ai)
        t_delta_doubles += + 2. * ctf.einsum("ladc, cbij, dl -> abij", dict_t_V["iabc"], t_T_abij, t_u_ai)
        t_delta_doubles += - 1. * ctf.einsum("kd, abkj, di -> abij", t_fock_pq[:no, no:], t_T_abij, t_u_ai)
        t_delta_doubles += - 1. * ctf.einsum("lc, cbij, al -> abij", t_fock_pq[:no, no:], t_T_abij, t_u_ai)
        t_delta_doubles += + 1. * ctf.einsum("klid, abkl, dj -> abij", dict_t_V["ijka"], t_T_abij, t_u_ai)
        t_delta_doubles += + 1. * ctf.einsum("klic, cbkj, al -> abij", dict_t_V["ijka"], t_T_abij, t_u_ai)
        t_delta_doubles += + 1. * ctf.einsum("klid, adkj, bl -> abij", dict_t_V["ijka"], t_T_abij, t_u_ai)
        t_delta_doubles += - 1. * ctf.einsum("kbij, ak -> abij", dict_t_V["iajk"], t_u_ai)
        t_delta_doubles += + 1. * ctf.einsum("kldi, bdkj, al -> abij", dict_t_V["ijak"], t_T_abij, t_u_ai)
        t_delta_doubles += - 1. * ctf.einsum("kacd, bckj, di -> abij", dict_t_V["iabc"], t_T_abij, t_u_ai)
        t_delta_doubles += + 1. * ctf.einsum("kldi, abkj, dl -> abij", dict_t_V["ijak"], t_T_abij, t_u_ai)
        t_delta_doubles += - 1. * ctf.einsum("kadc, cbkj, di -> abij", dict_t_V["iabc"], t_T_abij, t_u_ai)
        t_delta_doubles += - 1. * ctf.einsum("kadc, bcki, dj -> abij", dict_t_V["iabc"], t_T_abij, t_u_ai)
        t_delta_doubles += - 1. * ctf.einsum("lacd, cdji, bl -> abij", dict_t_V["iabc"], t_T_abij, t_u_ai)
        t_delta_doubles += - 1. * ctf.einsum("lacd, cbij, dl -> abij", dict_t_V["iabc"], t_T_abij, t_u_ai)
        t_delta_doubles += + 1. * ctf.einsum("abic, cj -> abij", dict_t_V["abic"], t_u_ai)

        # add those involving P(ijab,jiba) and from t_u_abij, in total 22 terms
        t_delta_doubles += +4. * ctf.einsum("klcd, caki, dblj -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += -2. * ctf.einsum("klcd, cakl, dbij -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += -2. * ctf.einsum("klcd, cdki, ablj -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += -2. * ctf.einsum("klcd, caki, bdlj -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += +2. * ctf.einsum("kaci, cbkj -> abij", dict_t_V["iabj"], t_u_abij)
        t_delta_doubles += -2. * ctf.einsum("klcd, acki, dblj -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += -2. * ctf.einsum("kldc, caki, dblj -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += -2. * ctf.einsum("kldc, abkj, dcil -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += -2. * ctf.einsum("lkcd, cbij, adlk -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += -1. * ctf.einsum("ki, abkj -> abij", t_fock_pq[:no, :no], t_u_abij)
        t_delta_doubles += +1. * ctf.einsum("ac, cbij -> abij", t_fock_pq[no:, no:], t_u_abij)
        t_delta_doubles += -1. * ctf.einsum("kaic, cbkj -> abij", dict_t_V["iajb"], t_u_abij)
        t_delta_doubles += -1. * ctf.einsum("kbic, ackj -> abij", dict_t_V["iajb"], t_u_abij)
        t_delta_doubles += +1. * ctf.einsum("klcd, ackl, dbij -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += +1. * ctf.einsum("kldc, cdki, ablj -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += +1. * ctf.einsum("klcd, acki, bdlj -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += -1. * ctf.einsum("kaci, bckj -> abij", dict_t_V["iabj"], t_u_abij)
        t_delta_doubles += +1. * ctf.einsum("kldc, acki, dblj -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += +1. * ctf.einsum("kldc, abkj, dcli -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += +1. * ctf.einsum("kldc, caki, dbjl -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += +1. * ctf.einsum("kldc, ackj, dbil -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += +1. * ctf.einsum("lkcd, cbij, dalk -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)

        # add exchange contributions
        t_delta_doubles.i("abij") << t_delta_doubles.i("baji")
        # after adding exchanging indices contribution from P(ijab, jiba),
        # now add all terms that don't involve P(ijab,jiba)
        t_delta_doubles += ctf.einsum("klij, abkl -> abij", dict_t_V["klij"], t_u_abij)
        t_delta_doubles += ctf.einsum("kldc, abkl, dcij -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += ctf.einsum("lkcd, cdij, ablk -> abij", dict_t_V["ijab"], t_T_abij, t_u_abij)
        t_delta_doubles += ctf.einsum("abcd, cdij -> abij", dict_t_V["abcd"], t_u_abij)

        return t_delta_doubles

    def update_singles_test(self, fake_ham, t_u_ai, t_u_abij):
        no = self.no
        nv = t_u_ai.shape[0]
        u_ai = t_u_ai.to_nparray().ravel()
        u_abij = t_u_abij.to_nparray().ravel()
        u_vec = np.concatenate((u_ai, u_abij), axis=None)
        delta_singles = np.dot(fake_ham, u_vec)[:no*nv]
        delta_singles = delta_singles.reshape(nv, no)
        return ctf.astensor(delta_singles)

    def update_doubles_test(self, fake_ham, t_u_ai, t_u_abij):
        no = self.no
        nv = t_u_ai.shape[0]
        u_ai = t_u_ai.to_nparray().ravel()
        u_abij = t_u_abij.to_nparray().ravel()
        u_vec = np.concatenate((u_ai, u_abij), axis=None)
        delta_doubles = np.dot(fake_ham, u_vec)[no*nv:]
        delta_doubles = delta_doubles.reshape(nv, nv, no, no)
        return ctf.astensor(delta_doubles)

    def construct_fake_ham(self, nv, no):
        dim = nv*no + nv**2*no**2
        fake_ham = np.diag(np.arange(dim)*4.)
        fake_ham += (np.random.random([dim, dim])-0.5)*0.1
        fake_ham += fake_ham.T
        fake_ham /= 2
        return fake_ham
    

    def test_davidson(self):
        nv = 5
        no = self.no
        time_init = time.time()
        ham = self.construct_fake_ham(nv, no)
        e_target, v_target = np.linalg.eig(ham)

        lowest_ex_ind_target = e_target.argsort()[:self.n_excit]
        e_target = e_target[lowest_ex_ind_target]
        v_target = v_target[lowest_ex_ind_target]

        print_logging_info("Initialising u tensors...", level=1)
        for i in range(self.n_excit):
            A = np.zeros(nv*no).ravel()
            A[i] = 1.
            A = A.reshape(-1, no)
            self.u_singles.append(ctf.astensor(A))
            self.u_doubles.append(ctf.tensor([nv, nv, no, no]))

        diff_e_norm = np.inf
        for i in range(self.max_iter):
            time_iter_init = time.time()
            subspace_dim = len(self.u_singles)
            self.u_singles, self.u_doubles = self.QR(self.u_singles, self.u_doubles)
            w_singles = [ctf.tensor([nv, no])] * subspace_dim
            w_doubles = [ctf.tensor([nv, nv, no, no])] * subspace_dim
            B = np.zeros([subspace_dim, subspace_dim])
            for l in range(subspace_dim):
                w_singles[l] = self.update_singles_test(ham,
                                                    self.u_singles[l],
                                                    self.u_doubles[l])
                w_doubles[l] = self.update_doubles_test(ham,
                                                    self.u_singles[l],
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
            e_imag = np.imag(e[lowest_ex_ind])
            e = np.real(e[lowest_ex_ind])
            v_imag = np.imag(v[:, lowest_ex_ind])
            v = np.real(v[:, lowest_ex_ind])

            # construct residuals
            u_singles_tmp = []
            u_doubles_tmp = []
            if subspace_dim >= self.max_dim:
                for n in range(self.n_excit):
                    y_singles = ctf.tensor(w_singles[0].shape, dtype=w_singles[0].dtype, sp=w_singles[0].sp)
                    y_doubles = ctf.tensor(w_doubles[0].shape, dtype=w_doubles[0].dtype, sp=w_doubles[0].sp)
                    for l in range(subspace_dim):
                        y_singles += self.u_singles[l] * v[l, n]
                        y_doubles += self.u_doubles[l] * v[l, n]
                    u_singles_tmp.append(y_singles)
                    u_doubles_tmp.append(y_doubles)
                self.u_singles = u_singles_tmp
                self.u_doubles = u_doubles_tmp
                self.e_excit = e_old
            else:
                for n in range(self.n_excit):
                    y_singles = ctf.tensor(w_singles[0].shape, dtype=w_singles[0].dtype, sp=w_singles[0].sp)
                    y_doubles = ctf.tensor(w_doubles[0].shape, dtype=w_doubles[0].dtype, sp=w_doubles[0].sp)
                    for l in range(subspace_dim):
                        y_singles += w_singles[l] * v[l, n]
                        y_doubles += w_doubles[l] * v[l, n]
                        y_singles -= e[n] * self.u_singles[l] * v[l, n]
                        y_doubles -= e[n] * self.u_doubles[l] * v[l, n]
                    diag_e_ind = lowest_ex_ind_target[n]
                    diag_e_ind = n
                    self.u_singles.append(y_singles / (e[n] - ham[diag_e_ind, diag_e_ind]))
                    self.u_doubles.append(y_doubles / (e[n] - ham[diag_e_ind, diag_e_ind]))
                    e_old = self.e_excit
                diff_e_norm = np.linalg.norm(np.abs(self.e_excit - e))
                self.e_excit = e
            if diff_e_norm < self.e_epsilon:
                print_logging_info("Iterative solver converged.", level=1)
                break
            else:
                print_logging_info("Iteration = ", i, level=1)
                print_logging_info("Norm of energy difference = ", diff_e_norm, level=2)
                print_logging_info("Excited states energies real part = ", e, level=2)
                print_logging_info("Excited states energies imaginary part = ", e_imag, level=2)
                print_logging_info("Target energies = ", e_target, level=2)
                print_logging_info("Took {:.3f} seconds ".format(time.time() - time_iter_init), level=2)
        print_logging_info("EOM-CCSD finished in {:.3f} seconds".format(time.time() - time_init), level=1)
        print_logging_info("Converged excited states energies = ", e, level=1)

        assert np.allclose(e, e_target, atol=1e-6)

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
        no = self.no
        nv = self.u_singles[0].shape[0]
        subspace_len = len(u_singles)
        subspace_matrix = ctf.tensor([no*nv+nv**2*no**2, subspace_len])
        for i in range(subspace_len):
            subspace_matrix[:no*nv, i] = u_singles[i].to_nparray().ravel()
            subspace_matrix[no*nv:, i] = u_doubles[i].to_nparray().ravel()
        Q, R = np.linalg.qr(subspace_matrix.to_nparray())
        Q = ctf.astensor(Q)

        for i in range(subspace_len):
            u_singles[i] = Q[:no*nv, i]
            u_singles[i] = u_singles[i].reshape(nv, no)
            u_doubles[i] = Q[no*nv:, i]
            u_doubles[i] = u_doubles[i].reshape(nv, nv, no, no)
        return u_singles, u_doubles
