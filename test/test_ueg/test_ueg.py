#!/usr/bin/python3 -u

import time

import ctf
import numpy as np
from pymes.mean_field import hf
from pymes.model import ueg
from pymes.solver import ccd, mp2
from pymes.log import print_title

##############################################################################
#   1. ctf tensors starts with a lower case t
#   2. indices follow each variable after a _
#   3. numpy's nparray runs fastest with the right most index
#   4. for small tensors, use nparrays, only when large contractions are needed
#      then use ctf tensors. In case in the future some other tensor engines
#      might be used
##############################################################################

def main(nel, cutoff,rs, gamma, kc, rpa,efftive2b):
    world=ctf.comm()
    nel = nel
    no = int(nel/2)
    nalpha = int(nel/2)
    nbeta = int(nel/2)
    rs = rs

    # Cutoff for the single-particle basis set.
    cutoff = cutoff

    # correspond to cell parameter in neci
    nMax = 5

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
    sys.init_single_basis(cutoff)

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
    sys.kCutoff = sys.L/(2*np.pi)*2.3225029893472993/rs

    if world.rank() == 0:
        print("kCutoff=",sys.kCutoff)
    if rpa:
        tV_pqrs = sys.eval2BodyIntegrals(correlator=sys.trunc,rpaApprox=True,only2Body=False, sp=1)
    elif efftive2b:
        tV_pqrs = sys.eval2BodyIntegrals(correlator=sys.trunc,rpaApprox=False,effective2Body=True, sp=1)
    else:
        tV_pqrs = sys.eval2BodyIntegrals(correlator=sys.trunc,rpaApprox=False,only2Body=True, sp=1)

    if world.rank() == 0:
        print("%f.3 seconds spent on evaluating Coulomb integrals" % (time.time()-timeCoulInt))

    if world.rank() == 0:
        print("Kinetic energy")
    G = []
    kinetic_G = []
    for i in range(nSpatialOrb):
        G.append(sys.basis_fns[2*i].k)
        kinetic_G.append(sys.basis_fns[2*i].kinetic)
    kinetic_G = np.asarray(kinetic_G)


    G = np.asarray(G)

    tKinetic_G = ctf.astensor(kinetic_G)

    if world.rank() == 0:
        print("Partitioning V_pqrs")
    tV_ijkl = tV_pqrs[:no,:no,:no,:no]

    if world.rank() == 0:
        print("Calculating hole energies")
    tEpsilon_i = hf.calcOccupiedOrbE(kinetic_G, tV_ijkl, no)
    holeEnergy = np.real(tEpsilon_i.to_nparray())

    tV_aibj = tV_pqrs[no:,:no,no:,:no]
    tV_aijb = tV_pqrs[no:,:no,:no,no:]
    if world.rank() == 0:
        print("Calculating particle energies")
    tEpsilon_a = hf.calcVirtualOrbE(kinetic_G, tV_aibj, tV_aijb, no, nv)
    particleEnergy = np.real(tEpsilon_a.to_nparray())
    # the correction to the one particle energies from doubly contracted 3-body
    # integrals
    contr_from_doubly_contra_3b = sys.double_contractions_in_3_body()
    contr_from_triply_contra_3b = sys.triple_contractions_in_3_body()
    print("HF orbital energies:")
    print(tEpsilon_i)
    print(tEpsilon_a)
    print("contributions from 3 body to total energy:")
    print(contr_from_triply_contra_3b)
    print("contributions from 3body to 1 particle energies:")
    print(contr_from_doubly_contra_3b)


    ### calculate HF energy: E_{HF} = \sum_i epsilon_i +\sum_ij (2*V_{ijij}-V_{ijji})
    if world.rank() == 0:
        print("Calculating HF energy")
    tEHF = 2*ctf.einsum('i->',tEpsilon_i)
    tV_klij = tV_pqrs[:no,:no,:no,:no]

    if world.rank() == 0:
        print("Calculating dir and exc HF energy")
    dirHFE = 2. * ctf.einsum('jiji->',tV_klij.to_nparray())
    excHFE = -1. * ctf.einsum('ijji->',tV_klij.to_nparray())

    if world.rank() == 0:
        print("Summing dir and exc HF energy")
    tEHF = tEHF-(dirHFE + excHFE)
    if world.rank() == 0:
        print("Direct =", dirHFE)
        print("Exchange =", excHFE)
        print("HF energy=", tEHF)





    if world.rank() == 0:
        print("Starting MP2")

    mp2E, mp2Amp = mp2.solve(tEpsilon_i, tEpsilon_a, tV_pqrs)
    dcdE = 0.

    ccdE, ccdAmp, hole, particle = ccd.solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=-1., sp=0, maxIter=60, fDiis=True)

    if world.rank() == 0:
        if rpa:
          f = open("tcE_"+str(nel)+"e_rs"+str(rs)+"_"+str(sys.correlator.__name__)+".rpa.optKc.dat", "a")
        else:
          f = open("tcE_"+str(nel)+"e_rs"+str(rs)+"_"+str(sys.correlator.__name__)+".optKc.dat", "a")
        print("#spin orb={}, rs={}, kCutoff={}, HF E={}, ccd E={}, dcd E={}".format(len(sys.basis_fns),rs,sys.kCutoff, tEHF, ccdE, dcdE))
        print("Total ccd E={}, Total dcd E={}".format(tEHF+ccdE,tEHF+dcdE))
        f.write(str(len(sys.basis_fns))+"  "+str(sys.kCutoff)+"  "+str(tEHF)+"  "+str(mp2E)+"  "+str(ccdE)+"  "+str(dcdE)+"\n")

if __name__ == '__main__':
  gamma = None
  nel = 14
  rpa=False
  for rs in [0.5]:
    for cutoff in [2]:
      kCutoffFraction = None
      for efftive2b in [True]:
        main(nel,cutoff,rs, gamma, kCutoffFraction,rpa,efftive2b)
  ctf.MPI_Stop()
