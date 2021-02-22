#!/usr/bin/python3 -u

import time
import numpy as np
import sys

sys.path.append("/home/liao/Work/Research/TCSolids/scripts/")

import ctf
from ctf.core import *

import pymes
from pymes.model import ueg
from pymes.util import tcdump

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

    # writing in chemists form the 3-body integral is
    # (or|ps|qt)

    tV_opqrst = ueg_model.eval3BodyIntegrals(correlator=ueg_model.trunc,sp=1)
    # symmetrize the 3-body integrals
    tV_sym_opqrst = ctf.tensor([nSpatialOrb,nSpatialOrb,nSpatialOrb,nSpatialOrb,\
                                nSpatialOrb,nSpatialOrb], sp=tV_opqrst.sp)
    print("symmetrizing 3-body integral")
    print("size of 3-body integral=", tV_sym_opqrst.size)
    tV_sym_opqrst.i("opqrst") <<  1./3 * (tV_opqrst.i("opqrst") +tV_opqrst.i("oqprts")\
                                     +tV_opqrst.i("qpotsr"))

    # check symmetry
    # first and second pair exchange
    t_diff_1_2 = ctf.tensor(tV_sym_opqrst.shape)
    t_diff_1_2.i("opqrst") << tV_sym_opqrst.i("opqrst") - tV_sym_opqrst.i("poqsrt")
    if ctf.norm(t_diff_1_2)<1e-10:
        print("Successfully asserted the 1st and 2nd pair of indices are symmetric")

    # first and third pair exchange
    t_diff_1_3 = ctf.tensor(tV_sym_opqrst.shape)
    t_diff_1_3.i("opqrst") << tV_sym_opqrst.i("opqrst") - tV_sym_opqrst.i("qpotsr")
    if ctf.norm(t_diff_1_3)<1e-10:
        print("Successfully asserted the 1st and 3rd pair of indices are symmetric")

    # second and third pair exchange
    t_diff_2_3 = ctf.tensor(tV_sym_opqrst.shape)
    t_diff_2_3.i("opqrst") << tV_sym_opqrst.i("opqrst") - tV_sym_opqrst.i("oqprts")
    if ctf.norm(t_diff_2_3)<1e-10:
        print("Successfully asserted the 2nd and 3rd pair of indices are symmetric")

    print("Writing 3-body integrals into TCDUMP")
    tcdump.write2Tcdump(tV_sym_opqrst)



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
