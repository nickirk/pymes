#!/usr/bin/python3

import time
import numpy as np
import sys

sys.path.append("/home/liao/Work/Research/TCSolids/scripts/")

import ctf
from ctf.core import *

import pymes
from pymes.solver import mp2
from pymes.model import ueg
from pymes.solver import dcd
from pymes.solver import ccd
from pymes.util.interface2Cc4s import write2Cc4sTensor
from pymes.util.fcidump import write2Fcidump

## Script for generating plane wave basis
## hamiltonian integrals for unifrom electron gas

##############################################################################
#   1. ctf tensors starts with a lower case t
#   2. indices follow each variable after a _
#   3. numpy's nparray runs fastest with the right most index
#   4. for small tensors, use nparrays, only when large contractions are needed
#      then use ctf tensors. In case in the future some other tensor engines 
#      might be used
##############################################################################

def print_title(title, under='='):
    '''Print the underlined title.

:param string title: section title to print out
:param string under: single character used to underline the title
'''
    print('# %s\n# %s\n#' % (title, under*len(title)))




def calcGamma(sys, basis, overlap_basis, nP):
    '''
    FTOD : Fourier Transformed Overlap Density
    C^p_q({\bf G}) = \int\mathrm d{\bf r} \phi^*_p({\bf r}\phi_q({\bf r})e^{i{\bf G\cdot r}} 
    '''
    nG = int(len(overlap_basis)/2)
    gamma_pqG = np.zeros((nP,nP,nG))

    for p in range(0,nP,1):
        for q in range(0,nP,1):
            for g in range(0,nG,1):
                if ((basis[2*p].k-basis[2*q].k) == overlap_basis[2*g].k).all():
                    GSquare = overlap_basis[2*g].kp.dot(overlap_basis[2*g].kp)
                    if np.abs(GSquare) > 1e-12 :
                        gamma_pqG[p,q,g] = np.sqrt(4.*np.pi/GSquare/sys.Omega)
    return gamma_pqG

def calcOccupiedOrbE(kinetic_G, tV_ijkl, no):
    e = ctf.astensor(kinetic_G[0:no], dtype = complex)
    #tConjGamma_jiG = ctf.einsum("ijG->jiG", ctf.conj(tGamma_ijG))
    #coul = ctf.einsum('ikG,jlG->ijkl', tConjGamma_jiG, tGamma_ijG)
    #exCoul = ctf.einsum('ikG,ljG->ilkj', tConjGamma_jiG, tGamma_ijG)
    dirE = 2.* ctf.einsum('ijij->i', tV_ijkl)
    exE = - 1.* ctf.einsum('ijji->i', tV_ijkl)
    e = e + dirE 
    e = e + exE
    return e

def calcVirtualOrbE(kinetic_G, tV_aibj, tV_aijb, no, nv):
    algoName = "calcVirtualOrbE"
    e = ctf.astensor(kinetic_G[no:], dtype = complex)
    #tConjGamma_aiG = ctf.einsum("iaG -> aiG", ctf.conj(tGamma_iaG))
    #dirCoul_aibj =  ctf.einsum('aiG,bjG->aibj',tConjGamma_aiG, tGamma_aiG)
    #exCoul_aijb = ctf.einsum('ajG,ibG->aijb',tConjGamma_aiG, tGamma_iaG)
    dirE = ctf.tensor([nv], dtype=complex, sp=1)
    dirE.i("a") << 2. * tV_aibj.i("aiai")
    exE = ctf.tensor([nv], dtype=complex, sp=1)
    exE.i("a") << -1. * tV_aijb.i("aiia")

    e = e + dirE
    e = e + exE
    return e

def genGCoeff(nG, nP):
    c=np.zeros((nG,nP))
    np.fill_diagonal(c,1.)
    return c


def evalCoulIntegrals(basis_fns,sys):
    world = ctf.comm()
    algoName = "evalCoulIntegrals"

    nP = int(len(basis_fns)/2)
    tV_pqrs = ctf.tensor([nP,nP,nP,nP], dtype=complex, sp=1)
    indices = []
    values = []
    
    basisKp = [ basis_fns[2*i].kp.tolist() for i in range(nP)]
    

    rank = world.rank()
    for p in range(nP):
          for r in range(nP):
              dk = basis_fns[p*2].kp-basis_fns[r*2].kp
              for q in range(nP):
                  if (p+r+q) % world.np() == rank:
                    s = -1
                    ks = basis_fns[q*2].kp + dk
                    ks = ks.tolist()
                    if ks in basisKp:
                      s = basisKp.index(ks)
                    dkSquare = dk.dot(dk)
                    if np.abs(dkSquare) > 0. and s >= 0:
                        indices.append(nP**3*p + nP**2*q + nP*r + s)
                        values.append(4.*np.pi/dkSquare/sys.Omega)
                        #tV_pqrs[p,q,r,s] = 4.*np.pi/dkSquare/sys.Omega
    tV_pqrs.write(indices,values)
    return tV_pqrs

def findKVec(k, basis_fns):
    for i in range(int(len(basis_fns)/2)):
        if (np.abs(basis_fns[2*i].kp - k) == [0.,0.,0.]).all():
            return i
    return -1

#def evalTransCorr2BodyIntegrals(tV_pqrs, basis_fns, correlator):
#
#
#    return tTC_pqrs
#
#def evalTransCorrThreeBodyIntegrals(tV_pqrs, basis_fns, correlator):
#
#
#    return tTC_opqrst


def main():
    world=ctf.comm()
    nel = 54
    no = int(nel/2)
    nalpha = 27
    nbeta = 27
    rs = 0.5

    # Cutoff for the single-particle basis set.
    cutoff = 6.

    # correspond to cell parameter in neci
    nMax = 2

    # Symmetry of the many-particle wavefunction: consider gamma-point only.
    gamma = np.zeros(3)
    timeSys = time.time()
    sys = ueg.UEG(nel, nalpha, nbeta, rs)
    if world.rank() == 0:
        print("%i electrons" % nel)
        print("rs=", rs)
        print("Volume of the box : %f " % sys.Omega)
        print("Length of the box : %f " % sys.L)
        print("%f.3 seconds spent on setting up system" % (time.time()-timeSys))


    timeBasis = time.time()
    basis_fns = sys.init_single_basis(nMax, cutoff, gamma)
    if world.rank() == 0:
        print_title('Basis set', '-')
        print('# %i spin-orbitals\n' % (len(basis_fns)))
        print("%f.3 seconds spent on generating basis." % (time.time()-timeBasis))


    timeCoulInt = time.time()
    tV_pqrs = evalCoulIntegrals(basis_fns,sys)
    if world.rank() == 0:
        print("%f.3 seconds spent on evaluating Coulomb integrals" % (time.time()-timeCoulInt))
    
    nSpatialOrb = int(len(basis_fns)/2)
    nP = nSpatialOrb
    nGOrb = nSpatialOrb

    nv = nP - no
    G = []
    kinetic_G = []
    for i in range(0,len(basis_fns),2):
        G.append(basis_fns[i].k)
        kinetic_G.append(basis_fns[i].kinetic)
    kinetic_G = np.asarray(kinetic_G)

    
    G = np.asarray(G)

    tKinetic_G = ctf.astensor(kinetic_G)

    tV_ijkl = tV_pqrs[:no,:no,:no,:no]
    tEpsilon_i = calcOccupiedOrbE(kinetic_G, tV_ijkl, no)
    holeEnergy = np.real(tEpsilon_i.to_nparray())

    tV_aibj = tV_pqrs[no:,:no,no:,:no]
    tV_aijb = tV_pqrs[no:,:no,:no,no:]
    tEpsilon_a = calcVirtualOrbE(kinetic_G, tV_aibj, tV_aijb, no, nv)
    particleEnergy = np.real(tEpsilon_a.to_nparray())


    ### calculate HF energy: E_{HF} = \sum_i epsilon_i +\sum_ij (2*V_{ijij}-V_{ijji})
    tEHF = 2*ctf.einsum('i->',tEpsilon_i)
    tV_klij = tV_pqrs[:no,:no,:no,:no]
    ## !!!!! The following code is buggy when more than 1 cores are used!!!!
    ## !!!!! Report to ctf lib
    #tDirHFE_i = 2. * ctf.einsum('jiji->i',tV_klij)
    #dirHFE = ctf.einsum('i->', tDirHFE_i)
    #excHFE_i = -1. * ctf.einsum('ijji->i',tV_klij)
    #excHFE = ctf.einsum('i->', excHFE_i)

    ### for now I will use einsum from numpy
    dirHFE = 2. * ctf.einsum('jiji->',tV_klij.to_nparray())
    excHFE = -1. * ctf.einsum('ijji->',tV_klij.to_nparray())

    tEHF = tEHF-(dirHFE + excHFE)
    if world.rank() == 0:
        print("Direct =", dirHFE)
        print("Exchange =", excHFE)
        print("HF energy=", tEHF)





    #tV_abij = tV_pqrs[no:,no:,:no,:no]

    #mp2E, mp2Amp = mp2.solve(tV_abij, tEpsilon_i, tEpsilon_a)
    ##mp2E, mp2Amp = mp2.solve(tV_abij, tEpsilon_i, tEpsilon_a)
    ccdE, dcdAmp = ccd.solve(tV_pqrs, tEpsilon_i, tEpsilon_a)


if __name__ == '__main__':
    main()
    ctf.MPI_Stop()
