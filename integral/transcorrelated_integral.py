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
        self.generic_3b = None
        self.laplace_2b = None
        self.non_hermi_2b = None
        self.covol_2b = None
        self.single_contr_3b = None
        self.double_contr_3b = None
        self.triple_contr_3b = None
        
   
    def init_correlator(self, G_grid):
    '''Member function of class Transcorrelation. This function initialise
    the variables and arrays used by the correlator. The correlation function
    has values on the G_grid.

    Parameters
    ----------
    G_grid: ndarray, [3, n]
    
    '''
    return

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
    '''
    
    return

    def eval_non_hermi_2b(self, ft_pair_density, ft_pair_momentum):
    '''
    '''
    return

    def eval_covol_2b(self, ft_pair_density):
    '''
    '''
    return
