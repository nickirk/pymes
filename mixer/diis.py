#!/usr/bin/python3

import numpy as np
import ctf

def mix(errors, amplitudes):
    '''
    errors are a list of residules in ctf tensor format
    '''
    algoName="diis.mix"

    if ctf.comm().rank() == 0:
        print("\t\tUsing "+algoName)

    # construct the Lagrangian
    assert(len(errors) == len(amplitudes))

    L = np.zeros((len(errors)+1, len(errors)+1))
    L[-1, :-1] = 1.
    L[:-1, -1] = 1.
    
    for i in range(len(errors)):
        for j in range(i,len(errors)):
            L[i,j] = np.real(ctf.einsum("abij,abij->", errors[i], errors[j]))
            L[j,i] = L[i,j]

    unitVec = np.zeros(len(errors)+1)
    unitVec[-1] = 1.
    c = np.linalg.inv(L).dot(unitVec)

    optAmp = ctf.tensor(amplitudes[0].shape, dtype=complex)


    for a in range(0,len(errors)):
        optAmp += amplitudes[a]*c[a]

    return optAmp
