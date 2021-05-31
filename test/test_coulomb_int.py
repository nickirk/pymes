#!/usr/bin/python3 -u 
import time
import numpy as np
import sys

# the following way of adding pymes to PYTHONPATH is recommended
# echo "export PYTHONPATH=$PYTHONPATH:/path/to/the/parent/directory/to/pymes"
# for example if /home/liao/Work/Research/TCSolids/scripts/pymes is where my
# pymes is, then I should
# echo "export
# PYTHONPATH=$PYTHONPATH:/home/liao/Work/Research/TCSolids/scripts/" >>
# ~/.bashrc
# source ~/.bashrc
sys.path.append("/Users/keliao/Work/Research/gpaw")

import ctf
from ctf.core import *

import pymes
from pymes.solver import mp2
from pymes.model import ueg
from pymes.solver import dcd
from pymes.solver import ccd
from pymes.mean_field import hf
from pymes.logging import print_title, print_logging_info
from pymes.integral import coulomb_int

# dependencies for gpaw
from ase.optimize import BFGS
from ase.build import molecule
from ase.parallel import paropen
from ase.parallel import paropen
from ase.units import Hartree
from ase.utils.timing import timer, Timer
from gpaw import GPAW, PW
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw.xc.rpa import RPACorrelation
from gpaw.response.pair import PairDensity
from gpaw.response.chi0 import Chi0
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.pw import PWDescriptor

#@timer('mean_field')
def mean_field(wf_file='N2.gpw', logging_file='H2.txt'):
    print_logging_info("Setting up system")
    N = molecule('N2')
    N.cell = (4, 4, 4)
    N.center()
    print_logging_info("Initialising calculator")
    calc = GPAW(mode=PW(400, force_complex_dtype=True),
                nbands=16,
                maxiter=300,
                xc='PBE',
                hund=False,
                txt=logging_file,
                parallel={'domain': 1},
                convergence={'density': 1.e-6})
    
    N.calc = calc
    E1_pbe = N.get_potential_energy()

    print_logging_info("E_pbe = ", E1_pbe)
    
    print_logging_info("Writing wavefunction to file "+wf_file)
    print_logging_info("See GPAW txt files for logs")
    calc.write(wf_file, mode='all')
    
    print_logging_info("Reading wavefunction from "+wf_file)
    print_logging_info("Starting nonselfconsistent one-shot HF calculation")
    E1_hf = nsc_energy(wf_file, 'EXX')
    print_logging_info("E_hf = ", E1_hf)
    print_logging_info("Starting full diagonalisation of mean field", \
            " Hamiltonian")
    calc.diagonalize_full_hamiltonian(nbands=600)
    print_logging_info("Writing wavefunction to file "+wf_file)
    calc.write(wf_file, mode='all')
    
    return calc

#@timer('RPA')
def rpa(wf_file='N2.gpw', logging_file='N_rpa.txt'):
    rpa = RPACorrelation(wf_file, nblocks=1, nfrequencies=10, 
                         truncation='wigner-seitz',
                         txt=logging_file)
    E1_i = rpa.calculate(ecut=200)
    print_logging_info("E_rpa = ", E1_i)


def get_ft_pair_density(wf_file, ecut=100):
    q_c = np.asarray([0., 0., 0.], dtype=float)
    chi0 = Chi0(wf_file)
    pair = PairDensity(wf_file, ecut=ecut, response='density')
    qd = KPointDescriptor([q_c])
    pd = PWDescriptor(pair.ecut, pair.calc.wfs.gd, complex, qd, gammacentered=True)
    bzk_kv, PWSA = chi0.get_kpoints(pd)
    ftpd = chi0.get_matrix_element(q_c, 0, 0, 1, 0, 1, pd = pd, kd = qd, symmetry=PWSA)
    print("Cia size = ", ftpd.shape)
    print(ftpd)
    #pair.generate_pair_densities(pd, 2, 4, [0, 1])

def main():
    print_title("System Information Summary",'=')
    print_title("Testing coulomb integrals from gpaw",'-')
    time_set_sys = time.time()
    timer = Timer()
    print_logging_info("Starting DFT-PBE, one-shot hf and full diagonalisation", level=2)
    #mean_calc = mean_field(wf_file='N2.gpw',logging_file='N2.txt')
    print_logging_info("Testing pair density class", level=2)
    get_ft_pair_density(wf_file='N2.gpw')
    #print_logging_info("Starting RPA calculation")
    #rpa()


if __name__ == '__main__':
    main()
