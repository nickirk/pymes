#!/usr/bin/python3 -u

import time
import numpy as np
import sys

sys.path.append("/home/liao/Work/Research/TCSolids/scripts/")

import ctf
from ctf.core import *

import pymes
from pymes.model import ueg

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


def main(nel, cutoff,rs, gamma, kc, tc):
    world=ctf.comm()
    nel = nel
    no = int(nel/2)
    nalpha = int(nel/2)
    nbeta = int(nel/2)
    rs = rs

    # Cutoff for the single-particle basis set.
    cutoff = cutoff

    # Symmetry of the many-particle wavefunction: consider gamma-point only.
    timeSys = time.time()
    ueg_model = ueg.UEG(nel, nalpha, nbeta, rs)
    if world.rank() == 0:
        print("%i electrons" % nel)
        print("rs=", rs)
        print("Volume of the box : %f " % ueg_model.Omega)
        print("Length of the box : %f " % ueg_model.L)
        print("%f.3 seconds spent on setting up system" % (time.time()-timeSys))


    timeBasis = time.time()
    ueg_model.init_single_basis(cutoff)

    nSpatialOrb = int(len(ueg_model.basis_fns)/2)
    nP = nSpatialOrb
    nGOrb = nSpatialOrb

    nv = nP - no
    if world.rank() == 0:
        print('# %i Spin Orbitals\n' % int(len(ueg_model.basis_fns)))
        print('# %i Spatial Orbitals\n' % nSpatialOrb)
        print("%f.3 seconds spent on generating basis." % (time.time()-timeBasis))


    timeCoulInt = time.time()
    ueg_model.gamma = gamma

    ueg_model.kCutoff = ueg_model.L/(2*np.pi)*2.3225029893472993/rs
    if world.rank() == 0:
        print("kCutoff=",ueg_model.kCutoff)

    # Now add the contributions from the 3-body integrals into the diagonal and
    # two body operators, also to the total energy, corresponding to 3 orders
    # of contractions

    tV_opqrst = ueg_model.eval3BodyIntegrals(correlator=ueg_model.trunc,sp=1)
    # left perl diagram: 2 hole lines, 1 loop, factor 2 from spin
    left_perl = (-1)**3*2*ctf.einsum("pjkjpk->p",tV_opqrst[:,:no,:no,:no,:,:no])
    double_contractions = left_perl
    print("Step 1, left perl=", left_perl)
    # right perl diagram: 2 hole lines, 1 loop,
    right_perl = (-1)**3*2*ctf.einsum("jpkpjk->p",tV_opqrst[:no,:,:no,:,:no,:no])
    double_contractions += right_perl
    print("Step 2, right perl=", right_perl)

    # left wave diagram: 2 hole lines, 0 loops
    left_wave = (-1)**2*ctf.einsum("pkiipk->p",tV_opqrst[:,:no,:no,:no,:,:no])
    double_contractions += left_wave
    print("Step 3, left wave=", left_wave)

    # right wave diagram: 2 hole lines, 0 loops
    right_wave = (-1)**2*ctf.einsum("ipkpki->p",tV_opqrst[:no,:,:no,:,:no,:no])
    double_contractions += right_wave
    print("Step 4, right wave=", right_wave)

    # left frog diagram: 2 hole lines, 0 loops. Mirror symmetry factor of 2
    left_frog = (-1)**2*2*ctf.einsum("jpiijp->p",tV_opqrst[:no,:,:no,:no,:no,:])
    double_contractions += left_frog
    print("Step 5, left frog=", left_frog)

    # right frog diagram: 2 hole lines, 0 loops. Mirror symmetry factor of 2
    right_frog = (-1)**2*2*ctf.einsum("ijpjpi->p",tV_opqrst[:no,:no,:,:no,:,:no])
    double_contractions += right_frog
    print("Step 6, right frog=", right_frog)

    # shield diagram: 2 hole lines, 1 loops
    shield = (-1)**3*ctf.einsum("jipijp->p",tV_opqrst[:no,:no,:,:no,:no,:])
    double_contractions += shield
    print("Step 7, shield=", shield)

    # seesaw diagram: 2 hole lines, 2 loops
    seesaw = (-1)**4*ctf.einsum("ijpijp->p",tV_opqrst[:no,:no,:,:no,:no,:])
    double_contractions += seesaw
    print("Step 8, seesaw=", seesaw)

    # left pan diagram: 2 hole lines, 1 loops.  Mirrow symm factor 2
    left_pan = (-1)**3*2*ctf.einsum("ijpipj->p",tV_opqrst[:no,:no,:,:no,:,:no])
    double_contractions += left_pan
    print("Step 9, left_pan=", left_pan)

    # right pan diagram: 2 hole lines, 1 loops.  Mirrow symm factor 2
    right_pan = (-1)**3*2*ctf.einsum("ipjijp->p",tV_opqrst[:no,:,:no,:no,:no,:])
    double_contractions += right_pan
    print("Step 10, right_pan=", right_pan)

    print("Final doubly contracted contribution=", double_contractions*2)

    contr_from_doubly_contra_3b = ueg_model.double_contractions_in_3_body()
    contr_from_triply_contra_3b = ueg_model.triple_contractions_in_3_body()
    if world.rank() == 0:
        print("contributions from 3 body to total energy:")
        print(contr_from_triply_contra_3b)
        print("contributions from 3body to 1 particle energies:")
        print(contr_from_doubly_contra_3b)


if __name__ == '__main__':
    #for gamma in None:
    gamma = None
    nel = 14
    for rs in [0.5]:
        for cutoff in [1]:
            kCutoffFraction = None
            for tc in [True]:
                main(nel,cutoff,rs, gamma, kCutoffFraction,tc)
    ctf.MPI_Stop()
