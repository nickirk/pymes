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
    def __init__(self, corr_func, G_grid):
        self.correlator = None
        self.generic_3b_opqrst = None
        self.laplace_2b_pqrs = None
        self.non_hermi_2b_pqrs = None
        self.convol_2b_pqrs = None
        self.single_contr_3b_pqrs = None
        self.double_contr_3b_pq = None
        self.triple_contr_3b = None
        self.ft_pair_density = None
        self.V_pqrs = None
        self.correlator = Correlator(corr_func)
        self.G_grid = G_grid
        
        
   

    def eval_laplace_2b(self, ft_pair_density_pqG): 
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
        ft_pair_density_pqG: real or complex ndarray/ctf tensor, [nb, nb, nG]

        Returns
        -------
        laplace_2b_pqrs: real/complex ctf tensor, [nb, nb, nG]
                         If the normal Coulomb integrals are not None, 
                         the laplace_2b will be added to them. Otherwise,
                         pure laplace_2b integrals will be returned.
        '''
    
        # transposed conjugation of ft_pair_density_pqG
        trconj_ft_pair_density_pqG = ctf.tensor(ft_pair_density_pqG.shape, 
                                               sp=ft_pair_density_pqG.sp)
        trconj_ft_pair_density_pqG.i("qpG") << \\
            ctf.conj(ft_pair_density_pqG).i("pqG")
        
        if self.V_pqrs is not None:
            self.laplace_2b_pqrs = self.V_pqrs

        corr_G = self.correlator.set_correlator_on_grid(self.G_grid)
        self.laplace_2b_pqrs += ctf.einsum("prG, G, qsG -> pqrs", 
                                      trconj_ft_pair_density_pqG, corr_G
                                      ft_pair_density_pqG)
        
        
        return self.laplace_2b_pqrs

    def eval_non_hermi_2b(self, ft_pair_density_pqG, ft_pair_momentum_pqvG):
        '''Member function of class Transcorrelation. This function computes
        the non-hermitian 2-body transcorrelated integrals in the 
        TC-Hamiltonian. The mathematical expression for this kind of integrals
        is: 
        <pq|\nabla_i u(r_i,r_j)\cdot \nabla_i|rs>
        = \int \mathrm d{\bf G} \tilde{u}({\bf G}) (-i{\bf G})\cdot \int d{\bf r_i}
        \phi_p^*({\bf r_i})\nabla_i\phi_r({\bf r_i})e^{-i{\bf G}\cdot {\bf r_i}}
        \int d{\bf r_j}\phi_q^*({\bf r_j})\phi_s({\bf r_j})e^{i{\bf G \cdot r_j}}
        '''
        return

    def eval_convol_2b(self, ft_pair_density_pqG):
        '''Member function of class Transcorrelation. This function computes
        the convolutional 2-body integrals in the transcorrelated Hamiltonian.
        The mathematical expression is
        <pq|(\nabla_i u(r_i,r_j))^2|rs> = \sum_{\bf G,G'}\tilde{u}({\bf G-G'}
        \tilde{u}({\bf G'})({\bf G}\cdot {\bf G'}-{\bf G'}\cdot {\bf G'})
        C^{r*}_p({\bf G})C^q_s({\bf G}).
        In principle, there is also  <pq|(\nabla_j u(r_i,r_j))^2|rs>, it is
        the same as <pq|(\nabla_i u(r_i,r_j))^2|rs>, as long as 
        u(r_i, r_j) = u(r_j, r_i). So in this implementation, a factor of 2 will
        be multiplied to include this. 
        TODO: check if this is true.

        Parameter
        ---------
        ft_pair_density_pqG: real/complex ndarray/ctf tensor, [nb, nb, nG]
        
        Returns
        -------
        convol_2b_pqrs: real/complex ctf tensor, [nb, nb, nb, nb] 
        '''
        nG = G_grid.shape[1]
        tilde_u_G = self.correlator(G_grid)
        
        # tensor {\bf G-G'}
        G_m_g_Gg = np.einsum("vG, vg -> vGg", G_grid, -G_grid)
        tilde_u_G_m_g_Gg  = self.correlator(G_m_g_Gg.reshape(3, nG**2))
        tilde_u_G_m_g_Gg = tilde_u_G_m_g_Gg.reshape(3, nG, nG) 
        
        # construct G\dot g - g\dot g
        G_dot_g_m_g_dot_g_Gg = np.einsum("vG, vg -> Gg", G_grid, G_grid)
        g_dot_g = np.einsum("vg, vg -> g", G_grid, G_grid)
        G_dot_g_m_g_dot_g_Gg -= np.diag(g_dot_g)
        
        # do we need the transposed conjugation of ft_pair_density_pqG?
        convol_2b_pqrs = 2.0*ctf.einsum("Gg, g, Gg, rpG, qsG -> pqrs",
                                        tilde_u_G_m_g_Gg, tilde_u_G, 
                                        G_dot_g_m_g_dot_g_Gg,
                                        ctf.conj(ft_pair_density_pqG),
                                        ft_pair_density_pqG)

        return convol_2b_pqrs

    def eval_single_contr_3b_pqrs(self):
    return

    def eval_double_contr_3b_pq(self):
    return

    def eval_triple_contr_3b(self):
    return
    
    def eval_generic_3b_opqrst(self):
    return


class Correlator:

    def __init__(self, corr_func, **kwargs):
        self.corr_func = corr_func
        if 'G_grid' in kwargs:
            return self.set_correlator_on_grid(kwargs.get('G_grid'))

    def set_correlator_on_grid(self, G_grid):
        '''Member function of class Correlator. This function initialise
        the variables and arrays used by the correlator. 
        The correlation function has values on the G_grid.

        Parameters
        ----------
        corr_func: function, that takes the length of G as input
        G_grid: ndarray, [3, n]

        Returns
        -------
        correlator: ndarray, real, [nG]
        
        '''
        
        G_l1_norm = np.einsum("iG, iG -> G", G_grid, G_grid)
        G_l1_norm = np.sqrt(G_l1_norm)
        correlator = corr_func(G_l1_norm)

        return correlator
