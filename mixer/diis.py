#!/usr/bin/python3

import numpy as np
import ctf
import scipy
from pymes.logging import print_logging_info

def mix(errors, amplitudes):
    '''
    errors are a list of residules in ctf tensor format
    '''
    algoName="diis.mix"

    world = ctf.comm()

    # construct the Lagrangian
    # TODO
    # no need to construct the whole matrix in every iteration,
    # only need to update one row and one column.
    assert(len(errors) == len(amplitudes))

    L = np.zeros((len(errors)+1, len(errors)+1))
    L[-1, :-1] = -1.
    L[:-1, -1] = -1.

    for i in range(len(errors)):
        for j in range(i,len(errors)):
            L[i,j] = np.real(ctf.einsum("abij,abij->", errors[i], errors[j]))
            L[j,i] = L[i,j]

    unitVec = np.zeros(len(errors)+1)
    unitVec[-1] = -1.
    eigen_values, eigen_vectors = scipy.linalg.eigh(L)

    if np.any(np.abs(eigen_values) <= 1e-13):
        print_logging_info("Linear dependence found in DIIS subspace.",level=2)
        valid_indices = np.abs(eigen_values) > 1e-13
        c = np.dot(eigen_vectors[:,valid_indices]*(1./eigen_values[valid_indices]),\
                np.dot(eigen_vectors[:,valid_indices].T.conj(), unitVec))
    else:
        c = np.linalg.inv(L).dot(unitVec)


    optAmp = ctf.tensor(amplitudes[0].shape, dtype=amplitudes[0].dtype, sp=amplitudes[0].sp)


    for a in range(0,len(errors)):
        optAmp += amplitudes[a]*c[a]

    print_logging_info(algoName, level=2)
    print_logging_info("Coefficients for combining amplitudes=",level=3)
    print_logging_info(c[:-1],level=3)
    print_logging_info("Sum of coefficients = {:.8f}".format(np.sum(c[:-1])),\
                       level=3)
    print_logging_info("Langrangian multiplier = {:.8f}".format(c[-1]), level=3)

    return optAmp
