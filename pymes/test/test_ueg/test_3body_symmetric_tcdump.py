#!/usr/bin/python3 -u

import time
import numpy as np

from pymes.model import ueg
from pymes.util import tcdump


def main(nel, cutoff,rs, gamma, kc, tc):
    print("TCDUMP util is not maintained, use at your own risk")
    return
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
    print('# %i Spin Orbitals\n' % int(len(ueg_model.basis_fns)))
    print('# %i Spatial Orbitals\n' % nSpatialOrb)
    print("%f.3 seconds spent on generating basis." % (time.time()-timeBasis))


    timeCoulInt = time.time()
    ueg_model.gamma = gamma

    ueg_model.kCutoff = ueg_model.L/(2*np.pi)*2.3225029893472993/rs
    print("kCutoff=",ueg_model.kCutoff)

    # writing in chemists form the 3-body integral is
    # (or|ps|qt)

    tV_opqrst = ueg_model.eval3BodyIntegrals(correlator=ueg_model.trunc,sp=1)
    # symmetrize the 3-body integrals
    tV_sym_opqrst = np.zeros([nSpatialOrb,nSpatialOrb,nSpatialOrb,nSpatialOrb,\
                                nSpatialOrb,nSpatialOrb])
    print("symmetrizing 3-body integral")
    print("size of 3-body integral=", tV_sym_opqrst.size)
    tV_sym_opqrst = 1./3 * (tV_opqrst + tV_opqrst.transpose((0,2,1,3,5,4)) +tV_opqrst.transpose((2,1,0,5,4,3)))

    # check symmetry
    # first and second pair exchange
    t_diff_1_2 = np.zeros(tV_sym_opqrst.shape)
    t_diff_1_2 = tV_sym_opqrst - tV_sym_opqrst.transpose((1,0,2,4,3,5))
    if np.linalg.norm(t_diff_1_2)<1e-10:
        print("Successfully asserted the 1st and 2nd pair of indices are symmetric")

    # first and third pair exchange
    t_diff_1_3 = np.zeros(tV_sym_opqrst.shape)
    t_diff_1_3 = tV_sym_opqrst - tV_sym_opqrst.transpose((2,1,0,5,4,3))
    if np.linalg.norm(t_diff_1_3)<1e-10:
        print("Successfully asserted the 1st and 3rd pair of indices are symmetric")

    # second and third pair exchange
    t_diff_2_3 = np.zeros(tV_sym_opqrst.shape)
    t_diff_2_3 = tV_sym_opqrst - tV_sym_opqrst.transpose((0,2,1,3,5,4))
    if np.linalg.norm(t_diff_2_3)<1e-10:
        print("Successfully asserted the 2nd and 3rd pair of indices are symmetric")

    print("Writing 3-body integrals into TCDUMP")
    tcdump.write(tV_sym_opqrst)



if __name__ == '__main__':
    #for gamma in None:
    gamma = None
    nel = 14
    for rs in [0.5]:
        for cutoff in [1]:
            kCutoffFraction = None
            for tc in [True]:
                main(nel,cutoff,rs, gamma, kCutoffFraction,tc)
