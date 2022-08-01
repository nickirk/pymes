#!/usr/bin/python3 -u
import ctf
import numpy as np

'''
This module is used for computing the transition structure factor in reciprocal space
defined in 
ref. J. Chem. Phys. 145, 141102 (2016) and PHYSICAL REVIEW X 8, 021043 (2018)
and in real space defined in 
ref. PHYSICAL REVIEW LETTERS 123, 156401 (2019)
'''


def calcReciprocalSpaceStructureFactor(tAmps_abij, pwBasis):
    '''
    tAmps_abij: amplitudes computed by solvers, eg CCD, DCD, FCIQMC
        it should be in the ctf tensor format: tT_abij of shape [nv,nv,no,no]
    pwBasis: planewave basis used to expand the orbitals or overlap 
    '''



def calcRealSpaceStructureFactor(tAmps_abij, pwBasis, r):
    '''
    tAmps_abij: amplitudes computed by solvers, eg CCD, DCD, FCIQMC
        it should be in the ctf tensor format: tT_abij of shape [nv,nv,no,no]
    pwBasis: planewave basis used to expand the orbitals or overlap 
    r : distance between two electrons, should be a 3d vector, or an array of
        3d vectors, shape [3,n]
    '''
    # form the exponential tensor exp^{ii({\bf k}_i-{\bf k}_a)\cdot{\bf r}}
    no = tAmps_abij.shape[3]
    nv = tAmps_abij.shape[0]

    deltaK = np.zeros((3,no,nv))

    for i in range(no):
        for a in range(nv):
            deltaK[:,i,a] = pwBasis[i*2].kp - pwBasis[(a+no)*2].kp
    deltaKDotR = np.einsum("ria,rn->nia",deltaK,r)

    phaseFactor = np.exp(-1j*deltaKDotR)
    tPhaseFactor_nia = ctf.astensor(phaseFactor, dtype=complex)
    tTildeT_abij = ctf.tensor(tAmps_abij.shape)
    tTildeT_abij.i("abij") << 2*tAmps_abij.i("abij") - 1*tAmps_abij.i("abji")
    tTildeT_abij = tTildeT_abij
    g_at_r = ctf.einsum("nia, abij->n", tPhaseFactor_nia, tTildeT_abij)
    g_at_r = g_at_r.to_nparray()
    g_at_r = np.real(g_at_r * 1./(2*np.pi)**3)
    g_at_r =  1 + g_at_r

    return g_at_r




