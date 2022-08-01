#!/usr/bin/python3 -u

import time
import numpy as np
import sys

import ctf
from ctf.core import *

import pymes
from pymes.solver import mp2
from pymes.model import ueg
from pymes.solver import dcd
from pymes.solver import ccd
from pymes.mean_field import hf
from pymes.log import print_title, print_logging_info


##############################################################################
#   1. ctf tensors starts with a lower case t
#   2. indices follow each variable after a _
#   3. numpy's nparray runs fastest with the right most index
#   4. for small tensors, use nparrays, only when large contractions are needed
#      then use ctf tensors. In case in the future some other tensor engines
#      might be used
##############################################################################



def compute_kinetic_energy(ueg):
    """ Function computing kinetic energy in 3D UEG
    Parameters
    ----------
    ueg: an UEG class

    Returns
    -------
    kinetic_G: nparray
        an array of kinetic energies of size of basis_fns/2
    """
    num_spatial_orb = int(len(ueg.basis_fns)/2)
    G = []
    kinetic_G = []
    for i in range(num_spatial_orb):
        G.append(ueg.basis_fns[2*i].k)
        kinetic_G.append(ueg.basis_fns[2*i].kinetic)
    kinetic_G = np.asarray(kinetic_G)
    return kinetic_G


def main(nel, cutoff, rs, gamma, kc):
    """
    Parameters 
    ----------
    nel: integer
        number of electrons 
    cutoff: float
        kinetic energy cutoff for plane wave basis set
    rs: float 
        $r_s$ density parameter for 3D UEG
    gamma: float
        a parameter inside correlators for fine tuning, see specific 
        correlators for details
    kc: float
        a paramter in the correlator that used in Ref. 10.1021/acs.jctc.7b01257
        for truncating the correlator to be 0, when $k > kc$.
        Also used in Ref. https://arxiv.org/pdf/2103.03176
    """

    world=ctf.comm()
    no = int(nel/2)
    n_alpha = int(nel/2)
    n_beta = int(nel/2)
    rs = rs

    # Cutoff for the single-particle basis set.
    cutoff = cutoff

    # Symmetry of the many-particle wavefunction: consider gamma-point only.
    time_set_sys = time.time()
    ueg_model = ueg.UEG(nel, n_alpha, n_beta, rs)
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
    n_p = num_spatial_orb
    nGOrb = num_spatial_orb

    nv = n_p - no
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
    ueg_model.k_cutoff = ueg_model.L/(2*np.pi)*2.3225029893472993/rs

    print_title('Evaluating pure 2-body integrals','=')
    print_logging_info("k_cutoff = {}".format(ueg_model.k_cutoff))

    # consider only true two body operators (excluding the singly contracted
    # 3-body integrals). This integral will be used to compute the HF energy
    t_V_pqrs = ueg_model.eval_2b_integrals(sp=1)

    print_logging_info("{:.3f} seconds spent on evaluating pure 2-body integrals"\
                       .format((time.time()-time_pure_2_body_int)))

    print_title('Evaluating HF energy','=')
    # Getting the kinetic energy array (1-body operator)
    kinetic_G = compute_kinetic_energy(ueg_model)

    time_hf = time.time()

    print_logging_info("Partitioning V_pqrs")
    t_V_ijkl = t_V_pqrs[:no,:no,:no,:no]

    print_logging_info("Calculating hole and particle energies")
    t_epsilon_i = hf.calcOccupiedOrbE(kinetic_G, t_V_ijkl, no)
    hole_e = np.real(t_epsilon_i.to_nparray())

    t_V_aibj = t_V_pqrs[no:,:no,no:,:no]
    t_V_aijb = t_V_pqrs[no:,:no,:no,no:]
    t_epsilon_a = hf.calcVirtualOrbE(kinetic_G, t_V_aibj, t_V_aijb, no, nv)
    particle_e = np.real(t_epsilon_a.to_nparray())

    print_logging_info("HF orbital energies:")
    print_logging_info(t_epsilon_i.to_nparray())
    print_logging_info(t_epsilon_a.to_nparray())


    ### calculate HF energy: E_{HF} = \sum_i epsilon_i +\sum_ij (2*V_{ijij}-V_{ijji})
    print_logging_info("Calculating HF energy")
    t_e_hf = 2*ctf.einsum('i->',t_epsilon_i)
    t_V_klij = t_V_pqrs[:no,:no,:no,:no]

    print_logging_info("Calculating dir and exc HF energy")

    dirHFE = 2. * ctf.einsum('jiji->',t_V_klij.to_nparray())
    excHFE = -1. * ctf.einsum('ijji->',t_V_klij.to_nparray())

    dir_orb_e = 2. * ctf.einsum('jiji->i',t_V_klij.to_nparray())
    exc_orb_e= -1. * ctf.einsum('ijji->i',t_V_klij.to_nparray())
    
    print("Direct correction to orbital energies:\n", dir_orb_e)
    print("Exchange correction to orbital energies:\n", exc_orb_e)

    print_logging_info("Summing dir and exc HF energy")

    t_e_hf = t_e_hf-(dirHFE + excHFE)
    print_logging_info("Direct = {}".format(dirHFE))
    print_logging_info("Exchange = {}".format(excHFE))
    print_logging_info("HF energy = {}".format(t_e_hf))

    print_logging_info("{:.3f} seconds spent on evaluating HF energy"\
                       .format((time.time()-time_hf)))

    mp2_e, mp2Amp = mp2.solve(t_epsilon_i, t_epsilon_a, t_V_pqrs)
    ccd_e = 0.
    dcd_e = 0.

    print_logging_info("Starting CCD")
    t_h_pq = ctf.astensor(np.diag(kinetic_G))
    t_fock_pq = hf.construct_hf_matrix(t_h_pq, t_V_pqrs, no)
    solver = ccd.CCD(no,is_diis=True)
    ccd_results  = solver.solve(t_fock_pq, t_V_pqrs, \
                                             level_shift=-1., sp=0, \
                                             max_iter=60)
    # unpacking ccd results
    ccd_e = ccd_results["ccd e"]
    ccd_amp = ccd_results["t2 amp"]

    print_logging_info("Starting DCD")
    solver = ccd.CCD(is_dcd=True, is_diis=True)
    dcd_results = solver.solve(t_epsilon_i, t_epsilon_a, t_V_pqrs, \
                                             level_shift=-1., sp=0, \
                                             max_iter=60, \
                                             amps=ccd_amp)
    # unpacking dcd results
    dcd_e = dcd_results["ccd e"]
    dcd_amps = dcd_results["t2 amp"]

    print_title("Summary of results","=")
    print_logging_info("Num spin orb={}, rs={}, k_cutoff={}".format(\
                       len(ueg_model.basis_fns), rs, ueg_model.k_cutoff))
    print_logging_info("HF E = {:.8f}".format(t_e_hf))
    print_logging_info("CCD correlation E = {:.8f}".format(ccd_e))
    print_logging_info("DCD correlation E = {:.8f}".format(dcd_e))
    print_logging_info("Total CCD E = {:.8f}".format(t_e_hf+ccd_e))
    print_logging_info("Total DCD E = {:.8f}".format(t_e_hf+dcd_e))

    if world.rank() == 0:
        f = open("E_"+str(nel)+"e_rs"+str(rs)+".dat", "a")
        f.write(str(len(ueg_model.basis_fns))+"  "+str(t_e_hf)\
                +"  "+str(mp2_e)+"  "+str(ccd_e)+"  "+str(dcd_e)+"\n")
    return (ccd_e, dcd_e)

if __name__ == '__main__':
  #for gamma in None:
  gamma = None
  nel = 14
  for rs in [0.5]:
    for cutoff in [10]:
      kc = None
      ccd_e, dcd_e = main(nel, cutoff, rs, gamma, kc)
    assert(np.abs(ccd_e - -0.31611540) < 1e-8)
    assert(np.abs(dcd_e - -0.31810448) < 1e-8)
  ctf.MPI_Stop()
