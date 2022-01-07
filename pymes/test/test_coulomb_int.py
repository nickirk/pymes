#!/usr/bin/python3 -u
import time
import numpy as np
import sys
import warnings


import ctf
from ctf.core import *

import pymes
from pymes.log import print_title, print_logging_info
from pymes.integral import pair_coulomb_integral

# dependencies for gpaw
from ase import Atoms
from ase.parallel import paropen
from ase.units import Hartree
from ase.utils.timing import timer, Timer

# modules needed from gpaw
from gpaw import GPAW, PW
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw.hybrids.coulomb import coulomb_interaction
from gpaw.xc import exx


def main():

    print_title("Test: HF disscociation curve from GPAW and Pymes",'=')

    print_logging_info("Setting up system")
    #mole = molecule('Li2')
    #mole = Atoms('He')
    box = 8.
    sys_name = 'CO'


    print_logging_info("Initialising calculator")
    calc = GPAW(mode=PW(200, force_complex_dtype=True),
                nbands=16,
                maxiter=300,
                xc='LDA',
                hund=False,
                setups='ae',
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
    calc.diagonalize_full_hamiltonian(nbands=100)
    print_logging_info("Writing wavefunction to file "+wf_file)
    calc.write(wf_file, mode='all')
    
    return calc

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
    #bzk_kv, PWSA = chi0.get_kpoints(pd)
    #ftpd = chi0.get_matrix_element(q_c, 0, 0, 1, 0, 1, pd = pd, kd = qd, \
    #                               symmetry=PWSA)
    #generate_pair_densities returns a generator. How to use it?
    #ftpd = pair.generate_pair_densities(pd, 2, 4, [0, 1])
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

    # Theoretically not clear yet if we can use the truncated coulomb
    # interaction for post-HF calculations and if that will increase
    # convergence with system size. But worth trying and figuring out
    # the details. For now, simply ignore the G=0 component.
    #omega = 0.0 # not short range, but regularised coulomb interaction
    #coulomb = coulomb_interaction(omega, pair.calc.wfs.gd, pair.calc.wfs.kd)
    #v_G = coulomb.get_potential(pd)
    #print("length of v_G =\n", len(v_G))

    gamma_pqG = mult_coulomb_kernel(C_pqG, pd)
    #print_logging_info("Assertion of G vector length ", v_G.shape[0], " ", \
    #                   C_pqG.shape[2], level = 2)
    #gamma_pqG = np.einsum("pqG, G -> pqG", C_pqG, np.sqrt(v_G/vol))
    gamma_pqG /= np.sqrt(vol)

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

    logging_file = 'gpaw.log'

    smearing = {'name': 'fermi-dirac', 'width': 0.0000001}


    # bond length in Angstrom
    bond_min = 1.8
    bond_max = 2.6

    mean_field_exx = []
    recalc_exx = []
    bond_lengths = []
    boxes = []

    wstc = True

    #wf_file = 'Li2.gpw'


    for bond_length in np.arange(bond_min, bond_max, 0.2):
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

        ftpd_nnG = pair_coulomb_integral.calc_ft_pair_coulomb_density(\
                                           wf_file=wf_file, \
                                           wigner_seitz_trunc=wstc, \
                                           nb=16, ecut=200)
        print_logging_info("Data type of the FTPD:", ftpd_nnG.dtype, level=2)


        V_ijkl = np.einsum("kiG, jlG -> ijkl", np.conj(ftpd_nnG[:no,:no,:]), \
                ftpd_nnG[:no,:no,:])
        E_exx = -np.real(np.einsum("ijji ->", V_ijkl)) * Hartree
        E_exx += 2*np.real(np.einsum("iijj ->", V_ijkl)) * Hartree
        print_logging_info("Recalculted E_exx from Coulomb integrals: ", E_exx)

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

    np.savetxt(file_name, np.column_stack([bond_lengths, mean_field_exx, \
               recalc_exx]))
    energy_diff_err_per_ele = np.abs(np.abs(mean_field_exx-mean_field_exx[0]) \
                                     - np.abs(recalc_exx-recalc_exx[0]))
    energy_diff_err_per_ele /= no*2.

    try:
        # assert the energy difference error is smaller than 1 meV/electron
        assert( (energy_diff_err_per_ele <= 0.001).all())
    except AssertionError:
        print_logging_info("Hartree-Fock energy difference error per electron:\
                           \n", energy_diff_err_per_ele)


if __name__ == '__main__':
    main()
