#!/usr/bin/python3

import numpy as np
import ctf
import scipy
import string
from pymes.logging import print_logging_info

class DIIS:
    def __init__(self, dim_space=5):
        self.dim_space = dim_space
        self.L = np.zeros((1,1))
        self.error_list = []
        self.amplitude_list = []
    
    def mix(self, error, amplitude):
        '''
        Mix the amplitudes from n iterations to minimize the errors in residuals.
        
        Parameters
        ----------
        error: list of ctf tensors, [size of amplitudes]
                The changes of amplitudes in the nth iteration.
        amplitude: ctf tensors, size [size of amplitudes]
                The amplitudes from the nth iteration. Amplitudes refer
                to the doubles amplitudes in CCD/DCD, and to the 
                singles and doubles amplitudes in CCSD/DCSD. Or in general,
                it can be a list of tensors that need to be updated in the 
                current iteration.
    
        Returns 
        -------
        opt_amp: list of ctf tensor, size [size of amplitudes]
                The optimized amplitudes.
        '''
        algoName="diis.mix"
    
        world = ctf.comm()
    
        # if the list of error and amplitude has fewer items than
        # the dim_space, it means we need all these new tensors to 
        # construct matrix L.
        # Or else, we are adding a new tensor. We remove the 0th row and
        # column, the move the rest of the L matrix to the left upper corner
        # we update the (dim_space-2)th row and column only.

        assert(len(self.error_list) == len(self.amplitude_list))
        running_dim_space = len(self.error_list)

        if running_dim_space == self.dim_space:
            self.error_list.pop(0)
            self.amplitude_list.pop(0)
        self.error_list.append(error)
        self.amplitude_list.append(amplitude)

        L_tmp = np.zeros((len(self.error_list)+1, len(self.error_list)+1))
        L_tmp[-1, :-1] = -1.
        L_tmp[:-1, -1] = -1.

        # TODO: explain what is happening here
        if running_dim_space == self.dim_space:
            L_tmp[:-3, :-3] = self.L[1:-2, 1:-2]
        else:
            L_tmp[:-2, :-2] = self.L[:-1, :-1]

        # TODO: explain what is happening here
        # i loop over saved error list
        for i in range(len(self.error_list)):
          # only need the last list of new errors, because we update only 
          # one row and column of the L matrix
          # nt loops over the types of amplitudes, ie singles and doubles
          
          for nt in range(len(self.error_list[-1])):
              # get the shape of the tensor
              indices = string.ascii_lowercase[:len(\
                          self.error_list[-1][nt].shape)] 
              L_tmp[i,-2] += np.real(\
                               ctf.einsum(\
                               indices+","+indices+"->", \
                               self.error_list[i][nt], \
                               self.error_list[-1][nt]))

        L_tmp[-2, :] = L_tmp[:,-2]
        self.L = L_tmp.copy()
    
    
        unitVec = np.zeros(len(self.error_list)+1)
        unitVec[-1] = -1.
        eigen_values, eigen_vectors = scipy.linalg.eigh(self.L)
    
        if np.any(np.abs(eigen_values) <= 1e-14):
            print_logging_info("Linear dependence found in DIIS subspace.", \
                               level=2)
            valid_indices = np.abs(eigen_values) > 1e-14
            c = np.dot(eigen_vectors[:,valid_indices]\
                         *(1./eigen_values[valid_indices]),\
                       np.dot(eigen_vectors[:,valid_indices].T.conj(), unitVec))
        else:
            c = np.linalg.inv(self.L).dot(unitVec)
    
    
        
        optAmp = [ctf.tensor(self.amplitude_list[0][i].shape, \
                          dtype=self.amplitude_list[0][i].dtype, \
                          sp=self.amplitude_list[0][i].sp)\
                  for i in range(len(self.amplitude_list[0]))]
    
    
        for a in range(0,len(self.error_list)):
            for i in range(len(self.amplitude_list[0])):
                optAmp[i] += self.amplitude_list[a][i]*c[a]
    
        print_logging_info(algoName, level=2)
        print_logging_info("Coefficients for combining amplitudes=",level=3)
        print_logging_info(c[:-1],level=3)
        print_logging_info("Sum of coefficients = {:.8f}".format(np.sum(c[:-1])),\
                           level=3)
        print_logging_info("Langrangian multiplier = {:.8f}".format(c[-1]), level=3)
    
        return optAmp
