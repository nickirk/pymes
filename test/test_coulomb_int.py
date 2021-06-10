#!/usr/bin/python3 -u
import time
import numpy as np
import sys
import warnings

# the following way of adding pymes to PYTHONPATH is recommended
# echo "export PYTHONPATH=$PYTHONPATH:/path/to/the/parent/directory/to/pymes"
# for example if /home/liao/Work/Research/TCSolids/scripts/pymes is where my
# pymes is, then I should
# echo "export
# PYTHONPATH=$PYTHONPATH:/home/liao/Work/Research/TCSolids/scripts/" >>
# ~/.bashrc
# source ~/.bashrc

# working from Mac
#sys.path.append("/Users/keliao/Work/Research/gpaw")

# working on office PC
# sys.path.append("/home/liao/Work/Research/gpaw")

import ctf
from ctf.core import *

import pymes
from pymes.solver import mp2
from pymes.model import ueg
from pymes.solver import dcd
from pymes.solver import ccd
from pymes.mean_field import hf
from pymes.logging import print_title, print_logging_info
#from pymes.integral import coulomb_int

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
from gpaw.hybrids.coulomb import coulomb_interaction

import matplotlib.pyplot as plt

def mean_field(box = 5, wf_file='N2.gpw', logging_file='H2.txt'):
    print_logging_info("Setting up system")
    #N = molecule('N2')
    N = molecule('H2')
    N.cell = (box, box, box)
    N.center()
    print_logging_info("Initialising calculator")
    calc = GPAW(mode=PW(200, force_complex_dtype=True),
                nbands=16,
                maxiter=300,
                xc='LDA',
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
    print_logging_info("E_hf components = ", E1_hf)
    print_logging_info("E_exvv components = ", E1_hf[-1])
    print_logging_info("E_hf sum = ", E1_hf.sum())
    print_logging_info("Starting full diagonalisation of mean field",
            " Hamiltonian")
    #calc.diagonalize_full_hamiltonian(nbands=100)
    print_logging_info("Writing wavefunction to file "+wf_file)
    calc.write(wf_file, mode='all')

    return {"calc": calc, "exx_vv": E1_hf[-1]}

def rpa(wf_file='N2.gpw', logging_file='N_rpa.txt'):
    rpa = RPACorrelation(wf_file, nblocks=1, nfrequencies=10,
                         truncation='wigner-seitz',
                         txt=logging_file)
    E1_i = rpa.calculate(ecut=200)
    print_logging_info("E_rpa = ", E1_i)


def calc_ft_overlap_density(wf_file, nb=100, ecut=400):
    """ This function computes the Fourier transformed
    overlap (coulomb) density (ftod), \gamma^p_q(G) = C^p_q(G) \sqrt(4\pi/G^2),
    where C^p_q(G) is the fourier transformed pair density (ftpd),
    C^p_q(G) = <\phi_p(r)|e^{i(G+k_q-k_p)r}|\phi_q(r)>

    Parameters
    ----------
    wf_file: string
        File name of the mean field wavefunction.
    nb: int
        Number of bands (orbitals)
    ecut: float
        Energy cutoff for the plane waves used for density fitting the pair
        density, should not be larger than ecut of the single particle wavefunc.

    Returns
    -------
    gamma_pqG: ndarray, [nb, nb, nG], float or complex
        Fourier transformed overlap (Coulomb) density.
    """
    func_name = "calc_ft_overlap_density"

    print_logging_info(func_name, level=1)
    print_logging_info("ecut = ", ecut, level=1)
    q_c = np.asarray([0., 0., 0.], dtype=float)
    #chi0 = Chi0(wf_file)
    pair = PairDensity(wf_file, ecut=ecut, response='density')
    if pair.ecut > pair.calc.wfs.pd.ecut:
        print_logging_info("Energy cutoff for density fitting is larger ", \
                           "than for wavefunction! ", \
                           "Redundant plane waves included.", level=2)

    qd = KPointDescriptor([q_c])
    pd = PWDescriptor(pair.ecut, pair.calc.wfs.gd, complex, qd, \
                      gammacentered=True)
    s = 0
    kpt_pair = pair.get_kpoint_pair(pd, s, q_c, 0, nb, 0, nb)
    n_n = np.arange(0, nb)
    m_m = np.arange(0, nb)
    # I am not so sure about the usage of extend_head here. If set to True,
    # the third dimension of the returned ndarray is nG+2. Here we set it to
    # False, in order to ensure that the 3rd dimension of ftpd_nmG is of
    # the same lenght as the nG.
    C_pqG = pair.get_pair_density(pd, kpt_pair, n_n, m_m, extend_head=False)
    print_logging_info("Shape of FTPD: ", C_pqG.shape, level = 2)
    vol = np.abs(np.linalg.det(pair.calc.wfs.gd.cell_cv))
    omega = 0.0 # not short range, but regularised coulomb interaction
    coulomb = coulomb_interaction(omega, pair.calc.wfs.gd, pair.calc.wfs.kd)
    v_G = coulomb.get_potential(pd)
    print("length of v_G =\n", len(v_G))

    #gamma_pqG = mult_coulomb_kernel(C_pqG, pd)
    print_logging_info("Assertion of G vector length ", v_G.shape[0], " ", \
                       C_pqG.shape[2], level = 2)
    gamma_pqG = np.einsum("pqG, G -> pqG", C_pqG, np.sqrt(v_G/vol))

    return gamma_pqG

def mult_coulomb_kernel(C_pqG, pd):
    """Function to multiply the sqrt(4pi/G^2) with the Fourier transformed
    pair density, C^p_q(G) = <\phi_p(r)|e^{i(G+k_q-k_p)r}|\phi_q(r)>

    Parameters
    ----------
    C_pqG: nparray, float/complex
        Fourier transformed pair density. Size nb x nb x nG, where nb is the
        number of bands and nG is the number of plane waves.
    pd: GPAW plane wave descriptor object
        It contains the plane wave basis on which the
        pair density is expanded. It should be the same one used in
        the KPointPair object in GPAW. Pay attention to that Q_G might
        be different than the plane wave basis used for expansion of
        orbitals.

    Returns
    -------
    gamma_pqG: nparray, float/complex
        C_pqG*sqrt(4pi/G^2), where G=0 is set to 0.
        Named as fourier transformed overlap (Coulomb) density (ftod),
        in cc4s it is called ftod.
    """
    # get_reciprocal_vectors return reciprocal lattice vectors plus q, G + q
    # in xyz coordinates.
    G = pd.get_reciprocal_vectors()
    print_logging_info("Assertion of G vector length ", G.shape[0], " ", \
                       C_pqG.shape[2], level = 2)
    assert( G.shape[0] == C_pqG.shape[2])
    G_square = np.einsum("Gi, Gi -> G", G, G)
    inv_G_square = np.divide(4*np.pi, G_square, \
            out = np.zeros_like(G_square), where=np.abs(G_square) > 1e-12)
    gamma_pqG = np.einsum("pqG, G -> pqG", C_pqG, np.sqrt(inv_G_square))
    return gamma_pqG

def main():
    print_title("System Information Summary",'=')
    print_title("Testing coulomb integrals from gpaw",'-')
    time_set_sys = time.time()
    timer = Timer()
    mean_field_exx = []
    recalc_exx = []
    box_min = 3
    box_max = 10
    box_size = []
    for box in range(box_min, box_max):
        print_logging_info(" box siz = ", box)
        print_logging_info("Starting DFT-PBE, one-shot hf and full ", \
                           "diagonalisation", level=0)
        mean_calc = mean_field(box = box, wf_file='N2.gpw', \
                               logging_file='N2.txt')
        print_logging_info("Testing pair density class", level=0)
        ftpd_nnG = calc_ft_overlap_density(wf_file='N2.gpw', nb=16, ecut=200)
        no = 1
        V_ijkl = np.einsum("ijG, klG -> ikjl", np.conj(ftpd_nnG[:no,:no,:]), \
                ftpd_nnG[:no,:no,:])
        E_exx = -np.real(np.einsum("ijji ->", V_ijkl)) * Hartree
        print_logging_info("Exx from mean field", mean_calc["exx_vv"])
        print_logging_info("Exx from Coulomb integrals", E_exx)
        mean_field_exx.append(mean_calc["exx_vv"])
        recalc_exx.append(E_exx)
        box_size.append(box)

    np.savetxt('data.txt', np.column_stack([box_size, mean_field_exx, recalc_exx]))
    plt.plot(1./np.arange(box_min, box_max), mean_field_exx, \
             label="Exx from Mean Field")
    plt.plot(1./np.arange(box_min, box_max), recalc_exx, label="Exx from Integrals")
    plt.xlabel("Box Length [A]")
    plt.ylabel("HF energy [eV]")
    plt.tight_layout()
    plt.show()

    #


if __name__ == '__main__':
    main()
