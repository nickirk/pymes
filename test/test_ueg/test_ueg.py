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

def calcOccupiedOrbE(kinetic_G, gamma_ijG, no):
    e = ctf.astensor(kinetic_G[0:no], dtype = complex)
    coul = ctf.einsum('kiG,jlG->ijkl', np.conj(gamma_ijG), gamma_ijG)
    exCoul = ctf.einsum('kiG,ljG->ilkj', np.conj(gamma_ijG), gamma_ijG)
    dirE = 2.* ctf.einsum('ijij->i', coul)
    exE = - 1.* ctf.einsum('ijji->i', coul)
    print("Direct=", np.sum(dirE))
    print("Exe=", np.sum(exE))
    e = e + dirE 
    e = e + exE
    return e

def calcVirtualOrbE(kinetic_G, gamma_ijG, gamma_iaG, gamma_aiG, gamma_abG, no, nv):
    algoName = "calcVirtualOrbE"
    print(algoName+": evaluating virtual orbital energies.")
    e = ctf.astensor(kinetic_G[no:], dtype = complex)
    print(algoName+": evaluating direct integrals")
    dirCoul_aibj =  ctf.einsum('iaG,bjG->aibj',np.conj(gamma_iaG), gamma_aiG)
    print(algoName+": evaluating exchange integrals")
    exCoul_aijb = ctf.einsum('jaG,ibG->aijb',np.conj(gamma_iaG), gamma_iaG)
    print(algoName+": evaluating direct contribution")
    dirE = ctf.tensor([nv], dtype=complex, sp=1)
    dirE.i("a") << 2. * dirCoul_aibj.i("aiai")
    print(algoName+": evaluating exchange contribution")
    exE = ctf.tensor([nv], dtype=complex, sp=1)
    exE.i("a") << -1. * exCoul_aijb.i("aiia")

    print(algoName+": adding direct contribution")
    e = e + dirE
    print(algoName+": adding exchange contribution")
    e = e + exE
    return e

def genGCoeff(nG, nP):
    c=np.zeros((nG,nP))
    np.fill_diagonal(c,1.)
    return c


def evalCoulIntegrals(basis_fns):

    nP = int(len(basis_fns)/2)
    V_pq = np.zeros((nP,nP))
    
    for p in range(0,nP,1):
        for q in range(0,nP,1):
           G = basis_fns[2*p].kp-basis_fns[2*q].kp
           GSquare = G.dot(G)
           if np.abs(GSquare) > 1e-12 :
               V_pq[p,q] = 4.*np.pi/GSquare/sys.Omega
    return V_pq


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
    # 4 electron UEG system at r_s=1 a.u.
    nel = 14
    no = int(nel/2)
    nalpha = 7
    nbeta = 7
    rs = 1.

    # Cutoff for the single-particle basis set.
    cutoff = 4.

    # correspond to cell parameter in neci
    nMax = 3

    # Symmetry of the many-particle wavefunction: consider gamma-point only.
    gamma = np.zeros(3)
    sys = ueg.UEG(nel, nalpha, nbeta, rs)
    print("%i electrons" % nel)
    print("rs=", rs)
    print("Volume of the box : %f " % sys.Omega)
    print("Length of the box : %f " % sys.L)


    timeBasis = time.time()
    print_title('Basis set', '-')
    basis_fns = sys.init_single_basis(nMax, cutoff, gamma)
    print(basis_fns[4*2], basis_fns[3*2])
    overlap_basis_fns = sys.init_single_basis(2*nMax, 10*cutoff, gamma)
    
    print('# %i spin-orbitals\n' % (len(basis_fns)))
    print('# %i G vectors for overlap integrals\n' % (len(overlap_basis_fns)/2))
    nSpatialOrb = int(len(basis_fns)/2)
    nP = nSpatialOrb
    nGOrb = nSpatialOrb
    nGOverlap = int(len(overlap_basis_fns)/2)

    nv = nP - no
    G = []
    kinetic_G = []
    for i in range(0,len(basis_fns),2):
        G.append(basis_fns[i].k)
        kinetic_G.append(basis_fns[i].kinetic)
    kinetic_G = np.asarray(kinetic_G)

    GOverlap = []
    for i in range(0,len(overlap_basis_fns),2):
        GOverlap.append(overlap_basis_fns[i].k)


    print("%f.3 seconds spent on generating basis." % (time.time()-timeBasis))

    # calculate CoulombVertex, $\Gamma_{pq}(G)=\sqrt{\frac{4\pi}{G^2}}C^*_p(G)C_q(G)$
    timeCoulombVertex = time.time()
    gamma_pqG = calcGamma(sys, basis_fns, overlap_basis_fns, nP)

    # see if this changes the ordering of the tensor elements?

    dim = [3,len(gamma_pqG[0,0,:]), nP, nP]
    gamma_pqG = gamma_pqG.astype(complex)
    write2Cc4sTensor(gamma_pqG,dim,"CoulombVertex","c")

    tGamma_pqG = ctf.astensor(gamma_pqG)

    
    G = np.asarray(G)
    GOverlap = np.asarray(GOverlap)
    print("G size=",G.shape)
    write2Cc4sTensor(G, [2,3,nGOrb], "OrbitalMomenta")
    write2Cc4sTensor(GOverlap, [2,3,nGOverlap], "Momenta")

    tKinetic_G = ctf.astensor(kinetic_G)
    print("Type of kinetic_G=", type(kinetic_G))

    gamma_ijG = gamma_pqG[:no,:no,:]
    tGamma_ijG = ctf.astensor(gamma_ijG)

    tEpsilon_i = calcOccupiedOrbE(kinetic_G, gamma_ijG, no)
    #tEpsilon_i = ctf.astensor(holeEnergy,dtype=complex)
    holeEnergy = np.real(tEpsilon_i.to_nparray())

    write2Cc4sTensor(holeEnergy, [1,no], "HoleEigenEnergies")
    gamma_iaG = gamma_pqG[:no,no:,:]
    gamma_aiG = gamma_pqG[no:,:no,:]
    gamma_abG = gamma_pqG[no:,no:,:]
    tEpsilon_a = calcVirtualOrbE(kinetic_G, gamma_ijG, gamma_iaG, gamma_aiG,gamma_abG, no, nv)
    particleEnergy = np.real(tEpsilon_a.to_nparray())
    #tEpsilon_a = particleEnergy
    write2Cc4sTensor(particleEnergy, [1,nv], "ParticleEigenEnergies")

    print("%f.3 seconds spent on CoulombVertex." % (time.time()-timeCoulombVertex))


    ## calculate HF energy: E_{HF} = \sum_i epsilon_i +\sum_ij (2*V_{ijij}-V_{ijji})
    tEHF = 2*ctf.einsum('i->',tEpsilon_i)
    print("Sum of Orbital energies=", tEHF)
    tV_klij = ctf.einsum('ikG,ljG->klij',ctf.conj(tGamma_ijG), tGamma_ijG)
    Vklij = tV_klij.to_nparray()
    write2Cc4sTensor(Vklij,[4,no,no,no,no],"Vklij")
    #print(tV_ijkl.read_local_nnz())
    dirHFE = 2. * ctf.einsum('jiji->',tV_klij)
    excHFE = -1. * ctf.einsum('ijji->',tV_klij)
    print("Direct =", dirHFE)
    print("Exchange =", excHFE)
    tEHF = tEHF-(dirHFE + excHFE)
    print("HF energy=", tEHF)

    tGamma_iaG = ctf.astensor(gamma_iaG)
    tGamma_aiG = ctf.astensor(gamma_aiG)
    tV_abij = ctf.tensor([nv,nv,no,no], dtype=complex, sp=1)
    tV_abij = ctf.einsum('iaG,bjG->abij',ctf.conj(tGamma_iaG),tGamma_aiG)
    write2Cc4sTensor(tV_abij.to_nparray(),[4,nv,nv,no,no],"Vabij")





    mp2E, mp2Amp = mp2.solve(tV_abij, tEpsilon_i, tEpsilon_a)
    dcdE, dcdAmp = dcd.solve(tGamma_pqG, tEpsilon_i, tEpsilon_a)

    tConjGamma_qpG = ctf.einsum("pqG->qpG", ctf.conj(tGamma_pqG))
    tV_pqrs = ctf.einsum("rpG, qsG -> pqrs", tConjGamma_qpG, tGamma_pqG)

    
    
    V_pqrs = np.real(tV_pqrs.to_nparray())

    write2Fcidump(V_pqrs,kinetic_G, no)
    write2Cc4sTensor(V_pqrs, [4, nP, nP, nP, nP], "V_pqrs")


    #tV_pq = ctf.astensor(evalCoulIntegrals(basis_fns),dtype=complex)

    #eDir = 2*ctf.eisum('ia,ia->', tV_pq[:no,no:], tV_pq[:no.no:])

if __name__ == '__main__':
    main()
    ctf.MPI_Stop()
