#!/usr/bin/python3

import numpy as np
import ctf

def optimize(errors, amplitudes):
    '''
    errors are a list of residules in ctf tensor format
    '''

    # construct the Lagrangian
    assert(len(errors) == len(amplitudes))

    L = np.zeros((len(errors)+1, len(errors)+1))
    L[-1, :-1] = 1.
    L[:-1, -1] = 1.
    
    for i in range(len(errors)):
        for j in range(i,len(errors)):
            overlap = ctf.tensor([1],dtype=complex,sp=1)
            overlap = ctf.einsum("abij, abij->", errors[i], errors[j])
            print(overlap)
            L[i,j] = np.real(np.einsum("abij,abij->", errors[i].to_nparray(), errors[j].to_nparray()))
            L[j,i] = L[i,j]

    unitVec = np.zeros(len(errors)+1)
    unitVec[-1] = 1.
    c = np.linalg.inv(L).dot(unitVec.T)

    optAmp =ctf.tensor(amplitudes[0].shape, dtype=complex)


    for a in range(0,len(errors)):
        optAmp += amplitudes[a]*c[a]

    return optAmp
