#!/usr/bin/python3 -u

import time
import numpy as np
import sys

sys.path.append("/home/liao/Work/Research/TCSolids/scripts/")

import ctf
from ctf.core import *

import pymes
from pymes.model import ueg


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
    tV_ijklmn = tV_opqrst[:no,:no,:no,:no,:no,:no]
    triple_contractions = 0
    # 3 hole lines, 3 loops, factor 8 from 3 spin summations
    triple_contractions += 8*ctf.einsum("ijkijk->",tV_ijklmn)
    print("Step 1, all direct contraction=", triple_contractions)
    # 3 hole lines, 2 loops, factor 4 from 2 spin summations
    pokemon_ball = -2*2*ctf.einsum("ijkjik->",tV_ijklmn)
    triple_contractions +=  pokemon_ball
    print("Step 2, pokemon ball diagram=", pokemon_ball)
    # 3 hole lines, 1 loop, factor 2 from spin summation
    # left UFO
    left_ufo = 2*ctf.einsum("ijkkij->",tV_ijklmn)
    triple_contractions += left_ufo
    print("Step 3, left ufo diagram=", left_ufo)
    # 3 hole lines, 1 loop
    right_ufo = 2*ctf.einsum("ijkjki->",tV_ijklmn)
    triple_contractions +=  right_ufo
    print("Step 4, right ufo diagram=", right_ufo)
    # 3 hole lines, 2 loops
    left_racket = 2*ctf.einsum("ijkikj->",tV_ijklmn)
    triple_contractions +=  left_racket
    print("Step 5, left bat diagram=", left_racket)
    # 3 hole lines, 2 loops
    right_racket = 2*ctf.einsum("ijkkji->",tV_ijklmn)
    triple_contractions +=  right_racket
    print("Step 6, left bat diagram=", right_racket)
    print("Final triply contracted contribution=", triple_contractions)

    #contr_from_doubly_contra_3b = ueg_model.double_contractions_in_3_body()
    contr_from_triply_contra_3b = ueg_model.triple_contractions_in_3_body()
    if world.rank() == 0:
        print("contributions from 3 body to total energy:")
        print(contr_from_triply_contra_3b)

    if np.abs(np.sum(triple_contractions-contr_from_triply_contra_3b))<1e-10:
        print("3 body triple contractions test: past")
    else:
        print("3 body triple contractions test: failed")

    return False


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
