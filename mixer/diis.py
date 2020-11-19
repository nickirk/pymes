#!/usr/bin/python3

import numpy as np
import ctf

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
    c = np.linalg.inv(L).dot(unitVec)

    optAmp = ctf.tensor(amplitudes[0].shape, dtype=amplitudes[0].dtype, sp=amplitudes[0].sp)


    if world.rank() == 0:
        print("\t\t\t"+algoName+": coefficients for combining amplitudes=")
        print("\t\t\t",c[:-1])
        print("\t\t\tSum of coefficients=",np.sum(c[:-1]))
        print("\t\t\tLangrangian multiplier = ",c[-1])
    for a in range(0,len(errors)):
        optAmp += amplitudes[a]*c[a]

    return optAmp
