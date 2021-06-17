#!/usr/bin/python3 -u
import time
import numpy as np
import sys
import warnings


import ctf
from ctf.core import *

import pymes
from pymes.solver import mp2
from pymes.model import ueg
from pymes.solver import dcd
from pymes.solver import ccd
from pymes.mean_field import hf
from pymes.logging import print_title, print_logging_info
from pymes.test import test_coulomb_int
#from pymes.integral import coulomb_int

# dependencies for gpaw
from ase import Atoms
from ase.parallel import paropen
from ase.units import Hartree
from gpaw import GPAW, PW
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw.hybrids.coulomb import coulomb_interaction
from gpaw.xc import exx

import matplotlib.pyplot as plt


def main():

    print_title("Test: HF disscociation curve from GPAW and Pymes",'=')

    print_logging_info("Setting up system")
    #mole = molecule('Li2')
    #mole = Atoms('He')

    print_title("System Information Summary",'=')

    print_logging_info("Initialising calculator")

    logging_file = 'gpaw.log'

    smearing = {'name': 'fermi-dirac', 'width': 0.0000001}


    # bond length in Angstrom
    bond_min = 2.0
    bond_max = 2.8

    mean_field_exx = []
    recalc_exx = []
    bond_lengths = []
    boxes = []

    wstc = True

    box = 8.
    sys_name = 'CO'
    #wf_file = 'Li2.gpw'


    for bond_length in np.arange(bond_min, bond_max, 0.2):
        wf_file = sys_name+'_'+str(box)+'.gpw'

        print_logging_info('Bond length = %f' % bond_length)
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
        #mole = Atoms(sys_name,
        #             positions=[(box/2. - bond_length/2., box/2., box/2.),
        #                        (box/2. + bond_length/2., box/2., box/2.)],
        #             cell = [box, box, box],
        #             pbc = True)

        mole.calc = calc


        E1_pbe = mole.get_potential_energy()

        print_logging_info("E_pbe = ", E1_pbe)

        print_logging_info("Writing wavefunction to file "+wf_file)
        print_logging_info("See GPAW txt files for logs")
        mole.calc.write(wf_file, mode='all')

        print_logging_info("Reading wavefunction from "+wf_file)
        print_logging_info("Starting nonselfconsistent one-shot HF calculation")
        E1_hf = nsc_energy(wf_file, 'EXX')
        print_logging_info("E_hf components = ", E1_hf)
        print_logging_info("E_exvv components = ", E1_hf[-1])
        print_logging_info("E_hf sum = ", E1_hf.sum())

        print_title('Testing EXX module from GPAW', '=')
        print_logging_info('HF from exx module')
        exx_calc = exx.EXX(mole.calc)
        exx_calc.calculate()

        print_title('Testing Pair Density in Pymes', '=')
        print_logging_info("Using Wigner-Seitz truncation: ", wstc, level=1)
        no = int(mole.calc.get_occupation_numbers().sum()/2)
        print_logging_info('Number of occupied bands = %d' % no)
        ftpd_nnG = test_coulomb_int.calc_ft_overlap_density(wf_file=wf_file, \
                                           wigner_seitz_trunc=wstc, \
                                           nb=16, ecut=200)
        # use the pd.integrate function to compute V_ijkl
        print_title('Using GPAW pd.integrate to compute Coulomb '+ \
                            'integrals', '=')


        weight_G = 1. / (mole.calc.wfs.pd.gd.N_c.prod() * mole.calc.wfs.pd.gd.dv)
        print_logging_info('integration weight of each G point: %f' % weight_G)
        V_ijkl = np.einsum("jiG, klG -> ikjl", np.conj(ftpd_nnG[:no,:no,:]), \
                ftpd_nnG[:no,:no,:]) * weight_G

        E_exx = -np.real(np.einsum("ijji ->", V_ijkl)) * Hartree
        print_logging_info("Exx from mean field: %f" % E1_hf[-1])
        print_logging_info("Exx from Coulomb integrals summed: %f" % E_exx)
        mean_field_exx.append(E1_hf[-1])
        recalc_exx.append(E_exx)
        bond_lengths.append(bond_length)
        boxes.append(box)

        if wstc:
            file_name = 'Exx_wstc_'+sys_name+'.txt'
        else:
            file_name = 'Exx_noG0_'+sys_name+'.txt'

        np.savetxt(file_name, np.column_stack([bond_lengths, mean_field_exx, \
                   recalc_exx]))


if __name__ == '__main__':
    main()
