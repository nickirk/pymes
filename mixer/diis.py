#!/usr/bin/python3

import numpy as np
import ctf
import scipy
import string
from pymes.logging import print_logging_info

def mix(error_list, amplitude_list):
    '''
    Mix the amplitudes from n iterations to minimize the errors in residuals.
    
    Parameters
    ----------
    error_list: list of ctf tensors, size [[size of amplitudes], n] 
            The changes of amplitudes in n consective iterations.
    amplitude_list: list of ctf tensors, size [[size of amplitudes], n]
            The amplitudes from the last n iterations. Amplitudes refer
            to the doubles amplitudes in CCD/DCD, and to the singles and doubles
            amplitudes in CCSD/DCSD.

    Returns 
    -------
    opt_amp: list of ctf tensor, size [size of amplitudes]
            The optimized amplitudes.
    '''
    algoName="diis.mix"

    world = ctf.comm()

    # construct the Lagrangian
    # TODO
    # no need to construct the whole matrix in every iteration,
    # only need to update one row and one column.
    assert(len(error_list) == len(amplitude_list))

    L = np.zeros((len(error_list)+1, len(error_list)+1))
    L[-1, :-1] = -1.
    L[:-1, -1] = -1.

    for i in range(len(error_list)):
        for j in range(i,len(error_list)):
            for nt in range(len(error_list[j])):
                # get the shape of the tensor
                indices = string.ascii_lowercase[:len(error_list[j][nt].shape)] 
                L[i,j] += np.real(ctf.einsum(indices+","+indices+"->", \
                                 error_list[i][nt], error_list[j][nt]))
            L[j,i] = L[i,j]

    unitVec = np.zeros(len(error_list)+1)
    unitVec[-1] = -1.
    eigen_values, eigen_vectors = scipy.linalg.eigh(L)

    if np.any(np.abs(eigen_values) <= 1e-14):
        print_logging_info("Linear dependence found in DIIS subspace.",level=2)
        valid_indices = np.abs(eigen_values) > 1e-14
        c = np.dot(eigen_vectors[:,valid_indices]*(1./eigen_values[valid_indices]),\
                np.dot(eigen_vectors[:,valid_indices].T.conj(), unitVec))
    else:
        c = np.linalg.inv(L).dot(unitVec)


    
    optAmp = [ctf.tensor(amplitude_list[0][i].shape, \
                      dtype=amplitude_list[0][i].dtype, \
                      sp=amplitude_list[0][i].sp)\
              for i in range(len(amplitude_list[0]))]


    for a in range(0,len(error_list)):
        for i in range(len(amplitude_list[0])):
            optAmp[i] += amplitude_list[a][i]*c[a]

    print_logging_info(algoName, level=2)
    print_logging_info("Coefficients for combining amplitudes=",level=3)
    print_logging_info(c[:-1],level=3)
    print_logging_info("Sum of coefficients = {:.8f}".format(np.sum(c[:-1])),\
                       level=3)
    print_logging_info("Langrangian multiplier = {:.8f}".format(c[-1]), level=3)

    return optAmp
