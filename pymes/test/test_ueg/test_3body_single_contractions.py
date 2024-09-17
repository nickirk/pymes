#!/usr/bin/python3 -u

import time
import numpy as np



from pymes.model import ueg
from pymes.log import print_logging_info



def test_single_contraction(nel=2, cutoff=1., rs=0.5, gamma=None, kc=None, tc=True):
    
    nel = nel
    no = int(nel/2)
    nalpha = int(nel/2)
    nbeta = int(nel/2)
    

    # Cutoff for the single-particle basis set.
    

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

    ueg_model.k_cutoff = ueg_model.L / (2 * np.pi) * 2.3225029893472993 / rs
    
    
    print("kCutoff=", ueg_model.k_cutoff)

    # generate the full 3-body
    t_V_opqrst = ueg_model.eval_3b_integrals(correlator=ueg_model.trunc,sp=0)

    # doing contractions.
    # RPA, factor 2 for spin summation
    tomega_rpa_pqrs = (nel-2)/nel*2*np.einsum("opqrsq-> oprs", t_V_opqrst[:,:,:no,:,:,:no])

    # evaluating RPA using expressions
    tV_pqrs =  ueg_model.eval_2b_integrals(correlator=ueg_model.trunc, is_only_2b=True, sp=0)
    tV_rpa_pqrs =  ueg_model.eval_2b_integrals(correlator=ueg_model.trunc, is_rpa_approx=True, sp=0)
    tomega_rpa_an_pqrs = 1./2*(tV_rpa_pqrs - tV_pqrs)

    # check if they are the same
    #print_logging_info("omega_rpa from 3-body = \n", tomega_rpa_pqrs)
    #print_logging_info("omega_rpa from analytical = \n", tomega_rpa_an_pqrs)
    diff_norm = np.linalg.norm(tomega_rpa_an_pqrs-tomega_rpa_pqrs)
    print_logging_info("diff rpa norm = ", diff_norm)
    assert diff_norm < 1.e-10

    # evaluating 3 exchange contractions using 3-body
    # 1. creation with 3. annihilation
    # factor 2 from mirror symmetry
    tomega_ex1_pqrs = -2*np.einsum("opqrso->qprs", t_V_opqrst[:no,:,:,:,:,:no])

    # ensure exchange pairs of indices are equivalent. 


    # all effective 2-body from analytical expressions, here the RPA does not
    # have factor (n-2)/n
    tomega_ex1_an_pqrs = 1./2*ueg_model.eval_2b_integrals(correlator=ueg_model.trunc, \
                                               is_exchange_1=True, sp=0)


    # check if they are the same
    print_logging_info("length omega_ex1 from 3-body = \n", len(tomega_ex1_pqrs[0]))
    print_logging_info("length omega_ex1 from analytical = \n", len(tomega_ex1_an_pqrs[0]))
    print_logging_info("omega_ex1 from 3-body = \n", tomega_ex1_pqrs)
    print_logging_info("omega_ex1 from analytical = \n", tomega_ex1_an_pqrs)
    ex1_diff_norm = np.linalg.norm(tomega_ex1_an_pqrs-tomega_ex1_pqrs)
    print_logging_info("diff ex1 norm = ", ex1_diff_norm)
    assert ex1_diff_norm < 1.e-10

    # 3. creation with 1. annihilation
    # factor 2 from mirror symmetry
    tomega_ex2_pqrs = -2*np.einsum("opqqst->opts", t_V_opqrst[:,:,:no,:no,:,:])

    tomega_ex2_an_pqrs = 1./2*ueg_model.eval_2b_integrals(correlator=ueg_model.trunc, \
                                               is_exchange_2=True, sp=0)


    # check if they are the same
    print_logging_info("length omega_ex2 from 3-body = \n", len(tomega_ex2_pqrs[0]))
    print_logging_info("length omega_ex2 from analytical = \n", len(tomega_ex2_an_pqrs[0]))
    print_logging_info("omega_ex2 from 3-body = \n", tomega_ex2_pqrs)
    print_logging_info("omega_ex2 from analytical = \n", tomega_ex2_an_pqrs)
    ex2_diff_norm = np.linalg.norm(tomega_ex2_an_pqrs-tomega_ex2_pqrs)
    print_logging_info("diff ex2 norm = ", ex2_diff_norm)
    assert ex2_diff_norm < 1.e-10
    # 2. creation with 1. annihilation
    # factor 2 from mirror symmetry
    tomega_ex3_pqrs = -2*np.einsum("opqpst->oqst", t_V_opqrst[:,:no,:,:no,:,:])

    tomega_ex3_an_pqrs = 1./2*ueg_model.eval_2b_integrals(correlator=ueg_model.trunc, \
                                               is_exchange_3=True, sp=0)


    # check if they are the same
    print_logging_info("length omega_ex3 from 3-body = \n", len(tomega_ex3_pqrs[0]))
    print_logging_info("length omega_ex3 from analytical = \n", len(tomega_ex3_an_pqrs[0]))
    print_logging_info("omega_ex3 from 3-body = \n", tomega_ex3_pqrs)
    print_logging_info("omega_ex3 from analytical = \n", tomega_ex3_an_pqrs)
    ex3_diff_norm = np.linalg.norm(tomega_ex3_an_pqrs-tomega_ex3_pqrs)
    print_logging_info("diff ex3 norm = ", ex3_diff_norm)
    assert ex3_diff_norm < 1.e-10
    #tomega_ex_pqrs += -2*np.einsum("opqqst-> opst", t_V_opqrst[:,:,:no,:no,:,:])
    # 1. creation with 2. annihilation
    # factor 2 from mirror symmetry
    #tomega_ex_pqrs += -2*np.einsum("opqrot-> pqrt", t_V_opqrst[:no,:,:,:,:no,:])
   
if __name__ == '__main__':
    test_single_contraction()
    print("Test on single contractions successful!")