import numpy as np

import pymes
from pymes.logging import print_title, print_logging_info

from gpaw import GPAW, PW
from gpaw.response.pair import PairDensity
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.hybrids.coulomb import coulomb_interaction

def calc_ft_overlap_density(wf_file, wigner_seitz_trunc=True, nb=100, ecut=400):
    """ This function computes the Fourier transformed
    overlap (coulomb) density (ftod), \gamma^p_q(G) = C^p_q(G) \sqrt(4\pi/G^2),
    where C^p_q(G) is the fourier transformed pair density (ftpd),
    C^p_q(G) = <\phi_p(r)|e^{i(G+k_q-k_p)r}|\phi_q(r)>

    Parameters
    ----------
    wf_file: string
        File name of the mean field wavefunction.
    wigner_seitz_trunc: bool
        Use Wigner-Seitz truncated Coulomb potential or not. Default yes.
        If False, using the Coulomb potential 4pi/G^2, with G=0 excluded.
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

    qd = KPointDescriptor([q_c])
    pd = PWDescriptor(pair.ecut, pair.calc.wfs.gd, complex, qd, \
                      gammacentered=True)
    s = 0
    kpt_pair = pair.get_kpoint_pair(pd, s, q_c, 0, nb, 0, nb)
    n_n = np.arange(0, nb)
    m_m = np.arange(0, nb)

    # I am not so sure about the usage of extend_head here. If set to True,
    # the third dimension of the returned ndarray is nG+2. Here we set it to
    # False, in order to ensure that the 3rd dimension of C_pqG is of
    # the same lenght as the nG.
    C_pqG = pair.get_pair_density(pd, kpt_pair, n_n, m_m, \
                                  extend_head=False)
    print_logging_info("Shape of Fourier transformed pair density: ", \
                       C_pqG.shape, level = 2)
    # volume of the supercell for weighting
    vol = np.abs(np.linalg.det(pair.calc.wfs.gd.cell_cv))

    if wigner_seitz_trunc:
        # using the Wigner-Seitz truncated potential
        omega = 0.0 # not short range, but regularised coulomb interaction
        coulomb = coulomb_interaction(omega, pair.calc.wfs.gd, pair.calc.wfs.kd)
        v_G = coulomb.get_potential(pd)

        print_logging_info("Assertion of G vector length ", v_G.shape[0], " ", \
                           C_pqG.shape[2], level = 2)
        assert(v_G.shape[0] == C_pqG.shape[2])
        gamma_pqG = np.einsum("pqG, G -> pqG", C_pqG, np.sqrt(v_G/vol))
    else:
        # using real Coulomb potential without the G=0 component
        gamma_pqG = mult_coulomb_kernel(C_pqG, pd)
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
