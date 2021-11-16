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
from pymes.mean_field import hf
from pymes.logging import print_title, print_logging_info


##############################################################################
#   1. ctf tensors starts with a lower case t
#   2. indices follow each variable after a _
#   3. numpy's nparray runs fastest with the right most index
#   4. for small tensors, use nparrays, only when large contractions are needed
#      then use ctf tensors. In case in the future some other tensor engines
#      might be used
##############################################################################



def compute_kinetic_energy(ueg):
    """
    input: ueg, an UEG class;
    return: an array of size of basis_fns/2
    """
    num_spatial_orb = int(len(ueg.basis_fns)/2)
    G = []
    kinetic_G = []
    for i in range(num_spatial_orb):
        G.append(ueg.basis_fns[2*i].k)
        kinetic_G.append(ueg.basis_fns[2*i].kinetic)
    kinetic_G = np.asarray(kinetic_G)
    return kinetic_G


def main(nel, cutoff,rs, gamma, kc, amps):
    world=ctf.comm()
    no = int(nel/2)
    nalpha = int(nel/2)
    nbeta = int(nel/2)
    rs = rs

    # Cutoff for the single-particle basis set.
    cutoff = cutoff

    # Symmetry of the many-particle wavefunction: consider gamma-point only.
    time_set_sys = time.time()
    ueg_model = ueg.UEG(nel, nalpha, nbeta, rs)
    print_title("System Information Summary",'=')
    print_logging_info("Number of electrons = {}".format(nel))
    print_logging_info("rs = {}".format(rs))
    print_logging_info("Volume of the box = {}".format(ueg_model.Omega))
    print_logging_info("Length of the box = {}".format(ueg_model.L))
    print_logging_info("{:.3f} seconds spent on setting up model"\
                       .format((time.time()-time_set_sys)))


    time_init_basis = time.time()
    ueg_model.init_single_basis(cutoff)

    num_spatial_orb = int(len(ueg_model.basis_fns)/2)
    nP = num_spatial_orb
    nGOrb = num_spatial_orb

    nv = nP - no
    print_title('Basis set', '=')
    print_logging_info('Number of spin orbitals = {}'\
                       .format(int(len(ueg_model.basis_fns))))
    print_logging_info('Number of spatial orbitals (plane waves) = {}'\
                       .format(num_spatial_orb))
    print_logging_info("{:.3f} seconds spent on generating basis."\
                       .format((time.time()-time_init_basis)))


    time_pure_2_body_int = time.time()
    ueg_model.gamma = gamma
    # specify the k_c in the correlator
    #ueg_model.kCutoff = ueg_model.L/(2*np.pi)*2.3225029893472993/rs
    ueg_model.kCutoff = kc

    print_title('Evaluating pure 2-body integrals','=')

    # consider only true two body operators (excluding the singly contracted
    # 3-body integrals). This integral will be used to compute the HF energy
    tV_pqrs = ueg_model.eval2BodyIntegrals(correlator=ueg_model.trunc,\
                                     only2Body=True,sp=1)

    print_logging_info("{:.3f} seconds spent on evaluating pure 2-body integrals"\
                       .format((time.time()-time_pure_2_body_int)))

    print_title('Evaluating HF energy','=')
    # Getting the kinetic energy array (1-body operator)
    kinetic_G = compute_kinetic_energy(ueg_model)

    time_hf = time.time()

    print_logging_info("Partitioning V_pqrs")
    tV_ijkl = tV_pqrs[:no,:no,:no,:no]

    print_logging_info("Calculating hole and particle energies")
    tEpsilon_i = hf.calcOccupiedOrbE(kinetic_G, tV_ijkl, no)
    holeEnergy = np.real(tEpsilon_i.to_nparray())

    tV_aibj = tV_pqrs[no:,:no,no:,:no]
    tV_aijb = tV_pqrs[no:,:no,:no,no:]
    tEpsilon_a = hf.calcVirtualOrbE(kinetic_G, tV_aibj, tV_aijb, no, nv)
    particleEnergy = np.real(tEpsilon_a.to_nparray())

    print_logging_info("HF orbital energies:")
    print_logging_info(tEpsilon_i.to_nparray())
    print_logging_info(tEpsilon_a.to_nparray())


    ### calculate HF energy: E_{HF} = \sum_i epsilon_i +\sum_ij (2*V_{ijij}-V_{ijji})
    print_logging_info("Calculating HF energy")
    tEHF = 2*ctf.einsum('i->',tEpsilon_i)
    tV_klij = tV_pqrs[:no,:no,:no,:no]

    print_logging_info("Calculating dir and exc HF energy")

    dirHFE = 2. * ctf.einsum('jiji->',tV_klij.to_nparray())
    excHFE = -1. * ctf.einsum('ijji->',tV_klij.to_nparray())

    print_logging_info("Summing dir and exc HF energy")

    tEHF = tEHF-(dirHFE + excHFE)
    print_logging_info("Direct = {}".format(dirHFE))
    print_logging_info("Exchange = {}".format(excHFE))
    print_logging_info("HF energy = {}".format(tEHF))

    print_logging_info("{:.3f} seconds spent on evaluating HF energy"\
                       .format((time.time()-time_hf)))

    # Now add the contributions from the 3-body integrals into the diagonal and
    # two body operators, also to the total energy, corresponding to 3 orders
    # of contractions
    print_title('Evaluating effective 2-body integrals','=')
    time_eff_2_body = time.time()
    # before calculating new integrals, delete the old one to release memory
    del tV_pqrs
    tV_pqrs = ueg_model.eval2BodyIntegrals(correlator=ueg_model.trunc,\
                                     effective2Body=True,sp=1)
    print_logging_info("{:.3f} seconds spent on evaluating effective 2-body integrals"\
                       .format((time.time()-time_eff_2_body)))


    orbital_energy_correction = True
    # the correction to the one particle energies from doubly contracted 3-body
    # integrals
    contr_from_doubly_contra_3b = ueg_model.double_contractions_in_3_body()
    contr_from_triply_contra_3b = ueg_model.triple_contractions_in_3_body()
    print_logging_info("Mean field contributions from 3 body to total energy:")
    print_logging_info(contr_from_triply_contra_3b)
    print_logging_info("Contributions from 3 body to 1 particle energies:")
    print_logging_info(contr_from_doubly_contra_3b)

    if orbital_energy_correction:
        tEpsilon_i += contr_from_doubly_contra_3b[:no]
        tEpsilon_a += contr_from_doubly_contra_3b[no:]



    print_logging_info("Starting MP2")

    mp2_e, mp2Amp = mp2.solve(tEpsilon_i, tEpsilon_a, tV_pqrs)
    ccd_e = 0.
    dcd_e = 0.

    ls = -(np.log(rs)*0.8+1.0)
    print_logging_info("Starting CCD")
    ccd_results = ccd.solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=ls, \
                            sp=0, maxIter=70, fDiis=True, amps=amps)
    # unpacking
    ccd_e = ccd_results["ccd e"]
    ccd_amp = ccd_results["t2 amp"]
    ccd_dE = ccd_results["dE"]

    print_logging_info("Starting DCD")
    dcd_results = dcd.solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=ls,\
                            sp=0, maxIter=70, fDiis=True,amps=ccd_amp)
    dcd_e = dcd_results["ccd e"]
    dcd_amp = dcd_results["t2 amp"]
    dcd_dE = dcd_results["dE"]


    print_title("Summary of results","=")
    print_logging_info("Num spin orb={}, rs={}, kCutoff={}".format(len(ueg_model.basis_fns),rs,\
                        ueg_model.kCutoff))
    print_logging_info("HF E = {:.8f}".format(tEHF))
    print_logging_info("CCD correlation E = {:.8f}".format(ccd_e))
    print_logging_info("DCD correlation E = {:.8f}".format(dcd_e))
    print_logging_info("3-body mean-field E = {:.8f}"\
                       .format(contr_from_triply_contra_3b))
    print_logging_info("Total CCD E = {:.8f}".format(tEHF+ccd_e+contr_from_triply_contra_3b))
    print_logging_info("Total DCD E = {:.8f}".format(tEHF+dcd_e+contr_from_triply_contra_3b))

    ccd_t2_norm = ctf.norm(ccd_amp)
    dcd_t2_norm = ctf.norm(dcd_amp)

    if world.rank() == 0:
        f = open("tcE_"+str(nel)+"e_rs"+str(rs)+"_"+str(ueg_model.correlator.__name__)+".scan.kc.dat", "a")
        f.write(str(len(ueg_model.basis_fns))+"  "+str(ueg_model.kCutoff)+"  "+str(tEHF)\
                +"  "+str(contr_from_triply_contra_3b)+"  "+str(mp2_e)+"  "\
                +str(ccd_e)+"  "+str(dcd_e)+" "+str(ccd_t2_norm)+" "+str(dcd_t2_norm)+\
                " "+str(ccd_dE)+" "+str(dcd_dE)+"\n")
    return dcd_amp

if __name__ == '__main__':
    #for gamma in None:
    gamma = None
    nel = 54
    opt_kc = np.loadtxt("rs_opt_kc.dat")
    for cutoff in [36]:
        amps = None
        n = 0
        for rs in [0.5,1.0,2.0,5.0,10.0,20.0,50.0]:
            print_logging_info("rs = ", rs, "rs from file = ", opt_kc[n,0])
            kCutoffFraction = opt_kc[n,1]
            amps = main(nel, cutoff, rs, gamma, kCutoffFraction, amps)
            n = n+1
    ctf.MPI_Stop()
