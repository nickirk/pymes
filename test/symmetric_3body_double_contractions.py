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
    tV_sym_opqrst = ctf.tensor([nSpatialOrb,nSpatialOrb,nSpatialOrb,nSpatialOrb,\
                                nSpatialOrb,nSpatialOrb], sp=tV_opqrst.sp)
    print("symmetrizing 3-body integral")
    print("size of 3-body integral=", tV_sym_opqrst.size)

    # a factor of 2 just to be consistent as in molpro
    tV_sym_opqrst.i("opqrst") <<  2*1./3 * (tV_opqrst.i("opqrst") +tV_opqrst.i("oqprts")\
                                     +tV_opqrst.i("qpotsr"))

    double_contractions = ctf.tensor(nSpatialOrb)
    # factor of 2**2 comes from two spin summations, (pp|ii|jj)
    first_term = -2**2*0.5*ctf.einsum("pijpij->p", tV_sym_opqrst[:,:no,:no,:,:no,:no])
    double_contractions += first_term
    print("First term=",first_term)

    # factor of 2 comes from one spin summation on i,j, (pp|ij|ji)
    second_term = 2*0.5*ctf.einsum("pijpji->p", tV_sym_opqrst[:,:no,:no,:,:no,:no])
    double_contractions += second_term
    print("Second term=",second_term)

    # factor of 2 comes from one spin summation on j, (jj|pi|ip)
    third_term = 2*0.5*ctf.einsum("jpijip->p", tV_sym_opqrst[:no,:,:no,:no,:no,:])
    double_contractions += third_term
    print("Third term=",third_term)

    # factor of 2 comes from one spin summation on i, (pj|jp|ii)
    fourth_term = 2*0.5*ctf.einsum("pjijpi->p", tV_sym_opqrst[:,:no,:no,:no,:,:no])
    double_contractions += fourth_term
    print("Fourth term=",fourth_term)

    # (pi|ij|jp)
    fifth_term = -0.5*ctf.einsum("pijijp->p", tV_sym_opqrst[:,:no,:no,:no,:no,:])
    double_contractions += fifth_term
    print("Fifth term=",fifth_term)

    # (jp|pi|ij)
    sixth_term = -0.5*ctf.einsum("jpipij->p", tV_sym_opqrst[:no,:,:no,:,:no,:no])
    double_contractions += sixth_term
    print("Sixth term=",sixth_term)

    # 2 subsequent single contractions:
    tV_pkql = ctf.tensor([nSpatialOrb, nSpatialOrb, no,no])
    # (pq|kl|mm)
    tV_pkql = -ctf.einsum("pkmqlm->pkql",tV_sym_opqrst[:,:no,:no,:,:no,:no])
    # (pq|km|ml)
    tV_pkql += ctf.einsum("pkmqml->pkql",tV_sym_opqrst[:,:no,:no,:,:no,:no])
    # (kl|pm|mq)
    tV_pkql += ctf.einsum("kpmlmq->pkql",tV_sym_opqrst[:no,:,:no,:no,:no,:])
    tV_pkql = 1/2*tV_pkql
    f_ij = ctf.einsum("ikjk->ij", tV_pkql[:no,:,:no,:])
    f_ij += -ctf.einsum("ikkj->ij", tV_pkql[:no,:,:no,:])
    print("Two subsequent single contractions:")
    print("    Occupied block:")
    print(f_ij.to_nparray().diagonal())

    # 2 subsequent single contractions:
    # still buggy
    #tV_aklb = ctf.tensor([nv, no, no,nv])
    ## (pq|kl|mm)
    #tV_aklb = -ctf.einsum("akmlbm->aklb",tV_sym_opqrst[no:,:no,:no,:no,no:,:no])
    ## (pq|km|ml)
    #tV_aklb += ctf.einsum("akmlmb->aklb",tV_sym_opqrst[no:,:no,:no,:no,:no,no:])
    ## (kl|pm|mq)
    #tV_aklb += ctf.einsum("kambml->aklb",tV_sym_opqrst[:no,no:,:no,no:,:no,:no])
    #tV_aklb = 1/2*tV_aklb
    #f_ab = ctf.einsum("akbk->ab", tV_pkql[no:,:no,no:,:no])
    #print("shape of f_ab")
    #f_ab += -ctf.einsum("akkb->ab", tV_aklb[no:,:no,:no,no:])
    #print("Two subsequent single contractions:")
    #print("    Virtual block:")
    #print(f_ab.to_nparray().diagonal())

    contr_from_doubly_contra_3b = ueg_model.double_contractions_in_3_body()
    if world.rank() == 0:
        print("contributions from asymetric 3 body to 1 particle energies:")
        print(contr_from_doubly_contra_3b)
        print("contributions from symetric 3 body to 1 particle energies:")
        print(double_contractions)
    assert(np.abs(np.sum(double_contractions-contr_from_doubly_contra_3b)) < 1e-10)


if __name__ == '__main__':
    #for gamma in None:
    gamma = None
    nel = 14
    for rs in [0.5]:
        for cutoff in [2]:
            kCutoffFraction = None
            for tc in [True]:
                main(nel,cutoff,rs, gamma, kCutoffFraction,tc)
    ctf.MPI_Stop()
