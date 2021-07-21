import numpy as np

import ctf
from ctf.core import *

import pymes
from pymes.logging import print_title, print_logging_info

class Transcorrelation:
    '''This class contains functions for computing transcorrelated integrals
    for solids, interfacing to GPAW for getting density-fitted Fourier 
    Transformed integrals, such the ft_pair_density and ft_pair_momentum.

    Attributes
    ----------
    G_grid: ndarray, [3, n]
            This is the plane-wave grid for representing the correlation
            function, Fourier transformed pair density, pair momentum.
    '''
    def __init__(self):
        self.correlator = None
        self.generic_3b_opqrst = None
        self.laplace_2b_pqrs = None
        self.non_hermi_2b_pqrs = None
        self.covol_2b_pqrs = None
        self.single_contr_3b_pqrs = None
        self.double_contr_3b_pq = None
        self.triple_contr_3b = None
        self.ft_pair_density = None
        self.V_pqrs = None
        self.correlator = None
        
        
   
    def set_correlator(self, G_grid):
        '''Member function of class Transcorrelation. This function initialise
        the variables and arrays used by the correlator. The correlation function
        has values on the G_grid.

        Parameters
        ----------
        G_grid: ndarray, [3, n]

        Returns
        -------
        correlator: ndarray, real, [nG]
        
        '''
        return self.correlator

    def eval_laplace_2b(self, ft_pair_density): 
        '''Member function of class Transcorrelation. This function evaluates
        the 2-body integrals arising from the similarity transformation on the
        Hamiltonian. The Laplace term refers to the addition 2-body interaction
        in terms of mathematical expression: 
        1/2(<pq|\nabla_{r_i}^2 u(r_i,r_j)|rs> + <pq|\nabla_{r_j}^2 u(r_i,r_j)|rs>)
        Since normally the correlation function u(r_i, r_j) is symmetric
        with respect to exchanging r_i and r_j, we only calculate one term and
        return the integrals as such. 

        Parameter
        --------
        ft_pair_density: real or complex ndarray/ctf tensor, [nb, nb, nG]

        Returns
        -------
        laplace_2b_pqrs: real/complex ctf tensor, [nb, nb, nG]
                         If the normal Coulomb integrals are not None, 
                         the laplace_2b will be added to them. Otherwise,
                         pure laplace_2b integrals will be returned.
        '''
    
        # transposed conjugation of ft_pair_density_pqG
        tran_conj_ft_pair_density = ctf.tensor(ft_pair_density.shape, 
                                               sp=ft_pair_density.sp)
        tran_conj_ft_pair_density.i("qpG") << ctf.conj(ft_pair_density).i("pqG")
        
        if self.V_pqrs is not None:
            self.laplace_2b = self.V_pqrs

        self.laplace_2b += ctf.einsum("prG, G, qsG -> pqrs", 
                                      tran_conj_ft_pair_density, self.correlator
                                      ft_pair_density)
        
        
        return self.laplace_2b

    def eval_non_hermi_2b(self, ft_pair_density, ft_pair_momentum):
        '''
        '''
        return

    def eval_covol_2b(self, ft_pair_density):
        '''
        '''
        return
