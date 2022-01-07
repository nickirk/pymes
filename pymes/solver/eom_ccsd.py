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
        '''
        EOM_CCSD takes in a CCSD object, because the T1, T2 and dressed
        integrals are needed.
        '''
        self.algo_name = "EOM-CCSD"
        self.ccsd = ccsd
        self.n_excit = n_excit
        self.u_singles = [ctf.tensor(ccsd.t_T1_pq.shape, 
                          dtype=ccsd.t_T1_pq.dtype, 
                          sp=ccsd.t_T1_pq.sp),] * n_excit
        self.u_doubles = [ctf.tensor(ccsd.t_T2_pqrs.shape, 
                          dtype=ccsd.t_T2_pqrs.dtype, 
                          sp=ccsd.t_T2_pqrs.sp),] * n_excit
        self.u_vecs = [self.u_singles, self.u_doubles]
        self.e_excit = np.zeros(n_excit)
        
    def write_logging_info(self):
        return

    def solve(self, t_fock_pq, t_V_pqrs):
        '''
        Solve for the requested number (n_excit) of excited states vectors and
        energies. 

        '''
        dict_t_V = self.ccsd.partition_V(t_V_pqrs)
        t_fock_dressed_pq = self.ccsd.get_T1_dressed_fock(t_fock_pq, 
                                                  self.ccsd.t_T_ai, 
                                                  dict_t_V)
        dict_t_V_dressed =  self.ccsd.get_T1_dressed_V(self.ccsd.t_T_ai, 
                                                       dict_t_V)

        # start iterative solver, arnoldi or davidson
        for i in range(self.max_iter):
            if not is_converged:
                self.u_singles += self.update_singles(t_fock_dressed_pq, 
                                                      dict_t_V_dressed) 
                self.u_doubles += self.update_doubles(t_fock_dressed_pq,
                                                      dict_t_V_dressed)


        return self.e_excit, self.u_vecs

    def update_singles(self, t_fock_pq, dict_t_V):
        no = self.ccsd.no
        t_delta_singles = ctf.tensor(self.u_singles.shape, 
                                     dtype=self.u_singles.dtype, 
                                     sp=self.u_singles.sp)
        
        # fock matrix contribution
        t_delta_singles += 2.*ctf.einsum("bj, baji->ai", t_fock_pq[:no,no:],
                                         self.u_doubles)\
                            - ctf.einsum("ij, aj", t_fock_pq[:no,:no],
                                           self.u_singles)\
                            - ctf.einsum("bj, abji->ai",t_fock_pq[:no,no:],
                                         self.u_doubles)\
                            - ctf.einsum("ba, bi->ai", t_fock_pq[no:,no:],
                                         self.u_singles)
        # integral and u_singles products
        t_delta_singles += 2.*ctf.einsum("jabi, bj->ai", dict_t_V["iabj"],
                                         self.u_singles)\
                            - ctf.einsum("jaib, bj->ai", dict_t_V["iajb"],
                                         self.u_singles)
        # integral and u_doubles products
        t_delta_singles += -2.*ctf.einsum("jkib, abjk->ai", dict_t_V["ijka"],
                                          self.u_doubles)\
                            + 2.*ctf.einsum("jabc, bcji->ai", dict_t_V["iabc"],
                                          self.u_doubles)\
                            + ctf.einsum("jkib, bajk->ai", dict_t_V["ijka"],
                                          self.u_doubles)\
                            - ctf.einsum("jacb, bcji->ai", dict_t_V["iabc"],
                                          self.u_doubles)
        # integral, T and u_singles products
        t_delta_singles += 4.*ctf.einsum("jkbc, baji, ck->ai", dict_t_V["ijab"],
                                         self.ccsd.t_T_abij, self.u_singles)\
                           -2.*ctf.einsum("jkbc, bajk, ci->ai", dict_t_V["ijab"],
                                          self.ccsd.t_T_abij, self.u_singles)\
                           -2.*ctf.einsum("jkbc, bcji, ak->ai", dict_t_V["ijab"],
                                          self.ccsd.t_T_abij, self.u_singles)\
                           -2.*ctf.einsum("jkbc, abji, ak->ai", dict_t_V["ijab"],
                                          self.ccsd.t_T_abij, self.u_singles)\
                           -2.*ctf.einsum("jkcb, baji, ck->ai", dict_t_V["ijab"],
                                          self.ccsd.t_T_abij, self.u_singles)\
                            + ctf.einsum("jkbc, abjk, ci->ai", dict_t_V["ijab"],
                                          self.ccsd.t_T_abij, self.u_singles)\
                            + ctf.einsum("jkcb, bcji, ak->ai", dict_t_V["ijab"],
                                          self.ccsd.t_T_abij, self.u_singles)\
                            + ctf.einsum("jkcb, abji, ck->ai", dict_t_V["ijab"],
                                          self.ccsd.t_T_abij, self.u_singles)\
                            
        return t_delta_singles
    
    def update_doubles(self, t_fock_pq, dict_t_V):
        no = self.ccsd.no
        t_delta_doubles = ctf.tensor(self.u_doubles.shape, 
                                     dtype=self.u_doubles.dtype, 
                                     sp=self.u_doubles.sp)
        return t_delta_doubles
    
