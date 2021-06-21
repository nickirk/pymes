#!/usr/bin/python3 -u
import time
import numpy as np
import sys
import warnings


import ctf
from ctf.core import *

import pymes
from pymes.logging import print_title, print_logging_info
from pymes.integral import coulomb_int

# dependencies for gpaw
from ase import Atoms
from ase.parallel import paropen
from ase.units import Hartree
from gpaw import GPAW, PW
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw.hybrids.coulomb import coulomb_interaction
from gpaw.xc import exx


def main():

    print_title("Test HF energy convergence with box size in "+ \
                "GPAW and Pymes",'=')

    print_logging_info("Setting up system")
    #mole = molecule('Li2')
    #mole = Atoms('He')
    sys_name = 'H2'


    print_logging_info("Initialising calculator")

    logging_file = 'gpaw.log'

    smearing = {'name': 'fermi-dirac', 'width': 0.0000001}


    # box length in Angstrom
    box_min = 4
    box_max = 14

    mean_field_exx = []
    recalc_exx = []
    bond_lengths = []
    boxes = []

    wstc = True

    bond_length = 0.8
    #wf_file = 'Li2.gpw'


    for box in np.arange(box_min, box_max, 2):
        print_title("System Information Summary",'=')
        print_logging_info('Units: Length [Angstrom], Energy [eV]', level=1)
        print_logging_info("Box size = ", box, " x ", box, " x ", box, level=1)
        print_logging_info('Bond length = %f' % bond_length, level=1)
        print_logging_info("System: "+sys_name, level=1)
        wf_file = sys_name+'_'+str(box)+'.gpw'

        calc = GPAW(mode=PW(200, force_complex_dtype=True),
                    nbands=16,
                    maxiter=300,
                    xc='PBE',
                    hund=False,
                    occupations=smearing,
                    txt=logging_file,
                    parallel={'domain': 1},
                    convergence={'density': 1.e-6})

        mole = Atoms(sys_name,
                     positions=[(box/2. - bond_length/2., box/2., box/2.),
                                (box/2. + bond_length/2., box/2., box/2.)
                                ],
                     cell = [box, box, box],
                     pbc = True)

        mole.calc = calc


        E1_pbe = mole.get_potential_energy()

        print_logging_info("E_pbe = ", E1_pbe, level=1)

        print_logging_info("Writing wavefunction to file "+wf_file, level=1)
        print_logging_info("See GPAW txt files for logs", level=1)
        mole.calc.write(wf_file, mode='all')

        print_logging_info("Reading wavefunction from "+wf_file, level=1)
        print_logging_info("Starting nonselfconsistent one-shot ", \
                           "HF calculation", level=1)
        E1_hf = nsc_energy(wf_file, 'EXX')
        print_logging_info("E_hf components = \n", E1_hf, level=1)
        print_logging_info("E_exvv component = ", E1_hf[-1], level=1)
        print_logging_info("E_hf sum = ", E1_hf.sum(), level=1)

        print_title('Testing EXX module from GPAW', '=')
        print_logging_info('HF from exx module', level=1)
        exx_calc = exx.EXX(mole.calc)
        exx_calc.calculate()

        print_title('Testing Pair Density in Pymes', '=')
        print_logging_info("Using Wigner-Seitz truncation: ", wstc, level=1)

        no = int(mole.calc.get_occupation_numbers().sum()/2)
        print_logging_info('Number of occupied bands = %d' % no, level=1)

        ftpd_nnG = coulomb_int.calc_ft_overlap_density(wf_file=wf_file, \
                                           wigner_seitz_trunc=wstc, \
                                           nb=16, ecut=200)


        V_ijkl = np.einsum("jiG, klG -> ikjl", np.conj(ftpd_nnG[:no,:no,:]), \
                ftpd_nnG[:no,:no,:])

        E_exx = -np.real(np.einsum("ijji ->", V_ijkl)) * Hartree
        print_logging_info("Exx from mean field: %f" % E1_hf[-1], level=1)
        print_logging_info("Exx from Coulomb integrals summed: %f" % E_exx,\
                           level=1)
        mean_field_exx.append(E1_hf[-1])
        recalc_exx.append(E_exx)
        bond_lengths.append(bond_length)
        boxes.append(box)

        if wstc:
            file_name = 'Exx_wstc_'+sys_name+'.txt'
        else:
            file_name = 'Exx_noG0_'+sys_name+'.txt'

    mean_field_exx = np.array(mean_field_exx)
    recalc_exx = np.array(recalc_exx)
    bond_lengths = np.array(bond_lengths)
    boxes = np.array(boxes)

    np.savetxt(file_name, np.column_stack([boxes, mean_field_exx, \
               recalc_exx]))
    # linear extrapolation to infinite box size
    coef_mean = np.polyfit((1./boxes)[-3:], mean_field_exx[-3:], 1)
    coef_recalc = np.polyfit((1./boxes)[-3:], recalc_exx[-3:], 1)

    mean_fn = np.poly1d(coef_mean)
    recalc_fn = np.poly1d(coef_recalc)

    err_at_inf_box_size = np.abs(mean_fn(0.)-mean_field_exx[0]\
                                 -(recalc_fn(0.)-recalc_exx[0]))
    try:
        assert(err_at_inf_box_size < 0.001)
    except AssertionError:
        print_logging_info("HF energy at infinite box size do not agree"\
                           +" between GPAW and Pymes!")
        print_logging_info("HF at infinite box size GPAW = ", mean_fn(0.))
        print_logging_info("HF at infinite box size Pymes = ", recalc_fn(0.))
        print_logging_info("Error at infinite box size = ", err_at_inf_box_size)


if __name__ == '__main__':
    main()
