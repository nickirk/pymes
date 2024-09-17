#!/usr/bin/python3 -u

import time
import numpy as np


from pymes.model import ueg



def test_double_contraction(nel=2, cutoff=2, rs=0.5, gamma=None, kc=None, tc=True):
    print("Test starts...")
    nel = nel
    no = int(nel/2)
    nalpha = int(nel/2)
    nbeta = int(nel/2)
    rs = rs

    # Cutoff for the single-particle basis set.
    cutoff = cutoff

    # Symmetry of the many-particle wavefunction: consider gamma-point only.
    timeSys = time.time()
    print("Setting up model")
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

    # Now add the contributions from the 3-body integrals into the diagonal and
    # two body operators, also to the total energy, corresponding to 3 orders
    # of contractions

    t_V_opqrst = ueg_model.eval_3b_integrals(correlator=ueg_model.trunc,sp=0)
    # left perl diagram: 2 hole lines, 1 loop, factor 2 from spin in k index
    # (pj|kk|jp)
    left_perl = (-1)**3*2*np.einsum("pjkjpk->p",t_V_opqrst[:,:no,:no,:no,:,:no])
    double_contractions = left_perl
    print("Step 1, left perl (pj|jp|ii)=", left_perl)
    # right perl diagram: 2 hole lines, 1 loop,
    right_perl = (-1)**3*2*np.einsum("jpkpjk->p",t_V_opqrst[:no,:,:no,:,:no,:no])
    double_contractions += right_perl
    print("Step 2, right perl (jp|pj|ii)=", right_perl)

    # left wave diagram: 2 hole lines, 0 loops
    left_wave = (-1)**2*np.einsum("pkiipk->p",t_V_opqrst[:,:no,:no,:no,:,:no])
    double_contractions += left_wave
    print("Step 3, left wave (pi|jp|ij)=", left_wave)

    # right wave diagram: 2 hole lines, 0 loops
    right_wave = (-1)**2*np.einsum("ipkpki->p",t_V_opqrst[:no,:,:no,:,:no,:no])
    double_contractions += right_wave
    print("Step 4, right wave (ip|pj|ji)=", right_wave)

    # left frog diagram: 2 hole lines, 0 loops. Mirror symmetry factor of 2
    left_frog = (-1)**2*2*np.einsum("jpiijp->p",t_V_opqrst[:no,:,:no,:no,:no,:])
    double_contractions += left_frog
    print("Step 5, left frog (ji|ip|pj)=", left_frog)

    # right frog diagram: 2 hole lines, 0 loops. Mirror symmetry factor of 2
    right_frog = (-1)**2*2*np.einsum("ijpjpi->p",t_V_opqrst[:no,:no,:,:no,:,:no])
    double_contractions += right_frog
    print("Step 6, right frog (ij|pi|jp)=", right_frog)

    # shield diagram: 2 hole lines, 1 loops, a factor of 2 from spin in i,j
    shield = (-1)**3*2*np.einsum("jipijp->p",t_V_opqrst[:no,:no,:,:no,:no,:])
    double_contractions += shield
    print("Step 7, shield (ji|ij|pp)=", shield)

    # seesaw diagram: 2 hole lines, 2 loops
    seesaw = (-1)**4*np.einsum("ijpijp->p",t_V_opqrst[:no,:no,:,:no,:no,:])
    double_contractions += seesaw
    print("Step 8, seesaw (ii|jj|pp)=", seesaw)

    # left pan diagram: 2 hole lines, 1 loops.  Mirrow symm factor 2
    left_pan = (-1)**3*2*np.einsum("ijpipj->p",t_V_opqrst[:no,:no,:,:no,:,:no])
    double_contractions += left_pan
    print("Step 9, left_pan (pp|ij|ji)=", left_pan)

    # right pan diagram: 2 hole lines, 1 loops.  Mirrow symm factor 2
    right_pan = (-1)**3*2*np.einsum("ipjijp->p",t_V_opqrst[:no,:,:no,:no,:no,:])
    double_contractions += right_pan
    print("Step 10, right_pan (ij|pp|ji)=", right_pan)

    print("Final doubly contracted contribution=", double_contractions)

    contr_from_doubly_contra_3b = ueg_model.double_contractions_in_3_body()
    contr_from_triply_contra_3b = ueg_model.triple_contractions_in_3_body()
    print("contributions from 3 body to total energy:")
    print(contr_from_triply_contra_3b)
    print("analytical contributions from 3body to 1 particle energies:")
    print(contr_from_doubly_contra_3b)
    print("direct summation from 3b tensor to 1 particle energies:")
    print(double_contractions)
    assert(np.allclose(double_contractions, contr_from_doubly_contra_3b))

if __name__ == "__main__":
    test_double_contraction()
    print("Test passed!")