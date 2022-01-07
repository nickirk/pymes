#!/usr/bin/python3 -u

import time
import numpy as np
import sys


import ctf
from ctf.core import *

import pymes
from pymes.model import ueg
from pymes.log import print_logging_info

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

    #ueg_model.kCutoff = ueg_model.L/(2*np.pi)*2.3225029893472993/rs
    ueg_model.kCutoff = 0.
    if world.rank() == 0:
        print("kCutoff=",ueg_model.kCutoff)

    # generate the full 3-body
    tV_opqrst = ueg_model.eval3BodyIntegrals(correlator=ueg_model.trunc,sp=1)

    # doing contractions.
    # RPA, factor 2 for spin summation
    tomega_rpa_pqrs = (nel-2)/nel*2*ctf.einsum("opqrsq-> oprs", tV_opqrst[:,:,:no,:,:,:no])

    # evaluating RPA using expressions
    tV_pqrs =  ueg_model.eval2BodyIntegrals(correlator=ueg_model.trunc, only2Body=True, sp=1)
    tV_rpa_pqrs =  ueg_model.eval2BodyIntegrals(correlator=ueg_model.trunc, rpaApprox=True, sp=1)
    tomega_rpa_an_pqrs = 1./2*(tV_rpa_pqrs - tV_pqrs)

    # check if they are the same
    print_logging_info("omega_rpa from 3-body = \n", tomega_rpa_pqrs.read_all_nnz())
    print_logging_info("omega_rpa from analytical = \n", tomega_rpa_an_pqrs.read_all_nnz())
    diff_norm = ctf.norm(tomega_rpa_an_pqrs-tomega_rpa_pqrs)
    print_logging_info("diff rpa norm = ", diff_norm)

    # evaluating 3 exchange contractions using 3-body
    # 1. creation with 3. annihilation
    # factor 2 from mirror symmetry
    tomega_ex1_pqrs = -2*ctf.einsum("opqrso->qprs", tV_opqrst[:no,:,:,:,:,:no])

    # ensure exchange pairs of indices are equivalent. 

    #exchange_indices_norm = ctf.tensor(tomega_ex1_pqrs.shape)
    #exchange_indices_norm.i("pqrs") << tomega_ex1_pqrs.i("pqrs") - tomega_ex1_pqrs.i("rspq")
    #print_logging_info("exchange of indices norm check:",ctf.norm(exchange_indices_norm))

    # all effective 2-body from analytical expressions, here the RPA does not
    # have factor (n-2)/n
    tomega_ex1_an_pqrs = 1./2*ueg_model.eval2BodyIntegrals(correlator=ueg_model.trunc, \
                                               exchange1=True, sp=1)


    # check if they are the same
    print_logging_info("length omega_ex1 from 3-body = \n", len(tomega_ex1_pqrs.read_all_nnz()[0]))
    print_logging_info("length omega_ex1 from analytical = \n", len(tomega_ex1_an_pqrs.read_all_nnz()[0]))
    print_logging_info("omega_ex1 from 3-body = \n", tomega_ex1_pqrs.read_all_nnz())
    print_logging_info("omega_ex1 from analytical = \n", tomega_ex1_an_pqrs.read_all_nnz())
    ex1_diff_norm = ctf.norm(tomega_ex1_an_pqrs-tomega_ex1_pqrs)
    print_logging_info("diff ex1 norm = ", ex1_diff_norm)


    # 3. creation with 1. annihilation
    # factor 2 from mirror symmetry
    tomega_ex2_pqrs = -2*ctf.einsum("opqqst->opts", tV_opqrst[:,:,:no,:no,:,:])

    tomega_ex2_an_pqrs = 1./2*ueg_model.eval2BodyIntegrals(correlator=ueg_model.trunc, \
                                               exchange2=True, sp=1)


    # check if they are the same
    print_logging_info("length omega_ex2 from 3-body = \n", len(tomega_ex2_pqrs.read_all_nnz()[0]))
    print_logging_info("length omega_ex2 from analytical = \n", len(tomega_ex2_an_pqrs.read_all_nnz()[0]))
    print_logging_info("omega_ex2 from 3-body = \n", tomega_ex2_pqrs.read_all_nnz())
    print_logging_info("omega_ex2 from analytical = \n", tomega_ex2_an_pqrs.read_all_nnz())
    ex2_diff_norm = ctf.norm(tomega_ex2_an_pqrs-tomega_ex2_pqrs)
    print_logging_info("diff ex2 norm = ", ex2_diff_norm)

    # 2. creation with 1. annihilation
    # factor 2 from mirror symmetry
    tomega_ex3_pqrs = -2*ctf.einsum("opqpst->oqst", tV_opqrst[:,:no,:,:no,:,:])

    tomega_ex3_an_pqrs = 1./2*ueg_model.eval2BodyIntegrals(correlator=ueg_model.trunc, \
                                               exchange3=True, sp=1)


    # check if they are the same
    print_logging_info("length omega_ex3 from 3-body = \n", len(tomega_ex3_pqrs.read_all_nnz()[0]))
    print_logging_info("length omega_ex3 from analytical = \n", len(tomega_ex3_an_pqrs.read_all_nnz()[0]))
    print_logging_info("omega_ex3 from 3-body = \n", tomega_ex3_pqrs.read_all_nnz())
    print_logging_info("omega_ex3 from analytical = \n", tomega_ex3_an_pqrs.read_all_nnz())
    ex3_diff_norm = ctf.norm(tomega_ex3_an_pqrs-tomega_ex3_pqrs)
    print_logging_info("diff ex3 norm = ", ex3_diff_norm)
    #tomega_ex_pqrs += -2*ctf.einsum("opqqst-> opst", tV_opqrst[:,:,:no,:no,:,:])
    # 1. creation with 2. annihilation
    # factor 2 from mirror symmetry
    #tomega_ex_pqrs += -2*ctf.einsum("opqrot-> pqrt", tV_opqrst[:no,:,:,:,:no,:])


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
