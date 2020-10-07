#!/usr/bin/python3 -u

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
from pymes.util.tcdump import write2Tcdump

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





def calcOccupiedOrbE(kinetic_G, tV_ijkl, no):
    dtype = tV_ijkl.dtype
    e = ctf.astensor(kinetic_G[0:no], dtype = dtype)
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
    e = ctf.astensor(kinetic_G[no:], dtype = tV_aijb.dtype)
    #tConjGamma_aiG = ctf.einsum("iaG -> aiG", ctf.conj(tGamma_iaG))
    #dirCoul_aibj =  ctf.einsum('aiG,bjG->aibj',tConjGamma_aiG, tGamma_aiG)
    #exCoul_aijb = ctf.einsum('ajG,ibG->aijb',tConjGamma_aiG, tGamma_iaG)
    dirE = ctf.tensor([nv], dtype=tV_aijb.dtype, sp=0)
    dirE.i("a") << 2. * tV_aibj.i("aiai")
    exE = ctf.tensor([nv], dtype=tV_aibj.dtype, sp=0)
    exE.i("a") << -1. * tV_aijb.i("aiia")

    e = e + dirE
    e = e + exE
    return e

def genGCoeff(nG, nP):
    c=np.zeros((nG,nP))
    np.fill_diagonal(c,1.)
    return c





def main(nel, cutoff,rs, gamma, kc):
    world=ctf.comm()
    nel = nel
    no = int(nel/2)
    nalpha = int(nel/2)
    nbeta = int(nel/2)
    rs = rs

    # Cutoff for the single-particle basis set.
    cutoff = cutoff

    # correspond to cell parameter in neci
    nMax = 4

    # Symmetry of the many-particle wavefunction: consider gamma-point only.
    timeSys = time.time()
    sys = ueg.UEG(nel, nalpha, nbeta, rs)
    if world.rank() == 0:
        print("%i electrons" % nel)
        print("rs=", rs)
        print("Volume of the box : %f " % sys.Omega)
        print("Length of the box : %f " % sys.L)
        print("%f.3 seconds spent on setting up system" % (time.time()-timeSys))


    timeBasis = time.time()
    sys.init_single_basis(cutoff,nMax)

    nSpatialOrb = int(len(sys.basis_fns)/2)
    nP = nSpatialOrb
    nGOrb = nSpatialOrb

    nv = nP - no
    if world.rank() == 0:
        print_title('Basis set', '-')
        print('# %i Spin Orbitals\n' % int(len(sys.basis_fns)))
        print('# %i Spatial Orbitals\n' % nSpatialOrb)
        print("%f.3 seconds spent on generating basis." % (time.time()-timeBasis))


    timeCoulInt = time.time()
    sys.gamma = gamma
    print("Gamma=",gamma)
    #tV_pqrs = sys.eval2BodyIntegrals(correlator=sys.coulomb,sp=1)
    #sys.kCutoff = kc
    #tV_pqrs = sys.eval2BodyIntegrals(correlator=sys.trunc,rpaApprox=True, sp=1)
    #tV_pqrs = sys.eval2BodyIntegrals(correlator=sys.trunc,rpaApprox=False, effective2Body=True,sp=1)
    tV_opqrst = sys.eval3BodyIntegrals(correlator=sys.trunc)
    #write2Tcdump(tV_opqrst)

    write2Tcdump(tV_opqrst)
    inds, vals = tV_opqrst.read_local_nnz()
    print("Nonzero terms:", len(vals))
    #tV_pqrs = sys.eval2BodyIntegrals()

    if world.rank() == 0:
        print("%f.3 seconds spent on evaluating 3-body integrals" % (time.time()-timeCoulInt))
    
    if world.rank() == 0:
        print("Kinetic energy")
    G = []
    kinetic_G = []
    for i in range(nSpatialOrb):
        G.append(sys.basis_fns[2*i].k)
        kinetic_G.append(sys.basis_fns[2*i].kinetic)
    kinetic_G = np.asarray(kinetic_G)

    
    #G = np.asarray(G)

    #tKinetic_G = ctf.astensor(kinetic_G)

    #if world.rank() == 0:
    #    print("Partitioning V_pqrs")
    #tV_ijkl = tV_pqrs[:no,:no,:no,:no]

    #if world.rank() == 0:
    #    print("Calculating hole energies")
    #tEpsilon_i = calcOccupiedOrbE(kinetic_G, tV_ijkl, no)
    #holeEnergy = np.real(tEpsilon_i.to_nparray())

    #tV_aibj = tV_pqrs[no:,:no,no:,:no]
    #tV_aijb = tV_pqrs[no:,:no,:no,no:]
    #if world.rank() == 0:
    #    print("Calculating particle energies")
    #tEpsilon_a = calcVirtualOrbE(kinetic_G, tV_aibj, tV_aijb, no, nv)
    #particleEnergy = np.real(tEpsilon_a.to_nparray())


    #### calculate HF energy: E_{HF} = \sum_i epsilon_i +\sum_ij (2*V_{ijij}-V_{ijji})
    #if world.rank() == 0:
    #    print("Calculating HF energy")
    #tEHF = 2*ctf.einsum('i->',tEpsilon_i)
    #tV_klij = tV_pqrs[:no,:no,:no,:no]
    ### !!!!! The following code is buggy when more than 1 cores are used!!!!
    ### !!!!! Report to ctf lib
    ##tDirHFE_i = 2. * ctf.einsum('jiji->i',tV_klij)
    ##dirHFE = ctf.einsum('i->', tDirHFE_i)
    ##excHFE_i = -1. * ctf.einsum('ijji->i',tV_klij)
    ##excHFE = ctf.einsum('i->', excHFE_i)

    #### for now I will use einsum from numpy
    #if world.rank() == 0:
    #    print("Calculating dir and exc HF energy")
    #dirHFE = 2. * ctf.einsum('jiji->',tV_klij.to_nparray())
    #excHFE = -1. * ctf.einsum('ijji->',tV_klij.to_nparray())

    #if world.rank() == 0:
    #    print("Summing dir and exc HF energy")
    #tEHF = tEHF-(dirHFE + excHFE)
    #if world.rank() == 0:
    #    print("Direct =", dirHFE)
    #    print("Exchange =", excHFE)
    #    print("HF energy=", tEHF)





    #if world.rank() == 0:
    #    print("Starting MP2")

    #mp2E, mp2Amp = mp2.solve(tEpsilon_i, tEpsilon_a, tV_pqrs)
    #ccdE = 0.
    #dcdE = 0.
    ###mp2E, mp2Amp = mp2.solve(tV_abij, tEpsilon_i, tEpsilon_a)
    ##ccdE, dcdAmp = ccd.solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=0., sp=0, fDiis=True)
    #ccdE, ccdAmp = ccd.solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=0., sp=0, fDiis=True)
    #dcdE, dcdAmp = dcd.solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=0., sp=0, fDiis=True)
    ##ccdE, dcdAmp = ccd.solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=0., sp=1, fDiis=False, stoch=True, thresh=thresh)
    ##ccdEApprox, ccdAmp = ccd.solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=0., sp=0, fDiis=False, stoch=True)

    #if world.rank() == 0:
    #    f = open("tcE_numOrb_rs"+str(rs)+"_"+str(sys.correlator.__name__)+".dat", "a")
    #    print("#spin orb={}, rs={}, kc={}, HF E={}, ccd E={}, dcd E={}".format(len(sys.basis_fns),rs,kc, tEHF, ccdE, dcdE))
    #    print("Total ccd E={}, Total dcd E={}".format(tEHF+ccdE,tEHF+dcdE))
    #    f.write(str(len(sys.basis_fns))+"  "+str(kc)+"  "+str(tEHF)+"  "+str(mp2E)+"  "+str(ccdE)+"  "+str(dcdE)+"\n")

if __name__ == '__main__':
  #for gamma in None:
  gamma = None
  for rs in [0.5]:
    for cutoff in [1]:
      #for kc in range(2,cutoff,2):
      for kc in [4]:
        main(14,cutoff,rs, gamma, kc)
  ctf.MPI_Stop()
