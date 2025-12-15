import time
import warnings
import sys

import numpy as np
import pytblis as pytblis

from pymes.basis_set import planewave
from pymes.log import print_logging_info
from pymes.mean_field import hf
from pymes.model.ueg_helper import _get_2b_int
from pymes.util.tensors import get_block_index
from pymes.util.parallel_tasks import det_num_threads
from scipy import special
from functools import partial

einsum = partial(pytblis.einsum, optimize='greedy')

class UEG:
    """ This class defines a model system of 3d uniform electron gas
    """

    def __init__(self, n_ele, n_alpha, n_beta, rs, is_tc=False):
        """
        Parameters
        ----------
        n_ele: int
            number of electrons
        n_alpha: int
            number of spin up electrons
        n_beta: int
            number of spin down electrons
        rs: float
            density parameter
        is_tc: bool
            parameter which determines whether transcorrelated framework is
            active or not for the calculation of the integrals.

        Attributes
        ----------
        basis_indices_map: nparray of int dtype
            an array to store indices of basis functions (plane waves) for
            later lookup. Size Nx*Ny*Nz, Nx, Ny, Nz are the k-vector points
            in x, y, z directions.

        basis_fns: tuple
            of relevant class PlaneWaveFn objects,
            i.e. the single-particle basis set.

        correlator: function
            correlator used in transcorrelation scheme.

        cutoff: float
            plane wave vector cutoff for determining the plane wave basis
            functions in units of `1/2*(2pi/L)^2`.

        k_cutoff: float
            plane wave vector cutoff inside the correlaor function trunc.
        """

        if (n_ele) % 2 != 0:
            warnings.warn("The number of electrons is not even, currently only\
                          closed shell systems are supported!")
        self.n_ele = int(n_ele)
        #: number of alpha (spin-up) electrons
        self.n_alpha = int(n_alpha)
        #: number of beta (spin-down) electrons
        self.n_beta = int(n_beta)
        if self.n_alpha != self.n_beta:
            warnings.warn("The number of electrons is not even, currently only\
                          closed shell systems are supported!")
        #: number of electrons for closed shell systems
        if ( not planewave.is_closed_shell(self.n_ele) ):
            raise ValueError("The number of electrons is not a closed shell system, currently only\
                          closed shell systems are supported!")
        #: electronic density.
        self.rs = rs
        #: length of the cubic simulation cell containing n_ele electrons
        #: at the density of rs.
        self.L = self.rs * ((4 * np.pi * self.n_ele) / 3) ** (1.0 / 3.0)
        #: volume of the cubic simulation cell containing n_ele electrons at the density.
        #: of rs.
        self.Omega = self.L ** 3

        self.basis_fns = None

        self.imax = 0

        self.cutoff = 0.

        self.basis_indices_map = None

        #: k-mesh (k-prime) for the discrete 
        #: Convolution Theorem (2-body TC integrals).
        self.kmesh_fac = None

        self.kmesh_cutoff = None

        self.kmesh_imax = None

        self.kPrime = None

        self.correlator = None

        self.k_cutoff = None

        self.gamma = None

        self.is_tc = is_tc

        #: Correlator types and index mapping.
        self.CORRELATOR_NONE = 0
        self.CORRELATOR_TRUNC = 1
        self.CORRELATOR_COULOMB = 2
        self.CORRELATOR_YUKAWA = 3
        self.CORRELATOR_YUKAWA_COULOMB = 4
        self.CORRELATOR_GASKELL = 5
        self.CORRELATOR_GASKELL_MODIFIED = 6
        self.CORRELATOR_SMOOTH = 7
        self.CORRELATOR_RPA = 8

    def is_k_in_basis(self, ke):
        """
        Checks if the input k-vector is inside of the basis set
        defined by the kinetic energy cutoff or not.

        Parameters
        ----------
        ke: float, kinetic energy of this plane wave function, possibly shifted

        Returns:
        --------
        a bool
        """
        kinetic_cutoff = self.cutoff * (2 * np.pi / self.L) ** 2 / 2.
        if ke <= kinetic_cutoff:
            return True
        return False

    def init_basis_indices_map(self):
        """ Member function of class UEG.
            Initialising a map between indices and the k-vectors.
            The map is stored in self.basis_indices_map.
            This map will be used for fast lookup and manipulation of
            the k-vectors, for example in momentum conservation checkups.

        Modifies
        ----------------
           self.basis_indices_map: nparray,
                dtype = int,
                size = product of numbers of k-points in x, y, z directions
        """

        num_k_in_each_dir = self.imax * 2 + 1
        self.basis_indices_map = -1 * np.ones(num_k_in_each_dir ** 3).astype(int)
        for i in range(int(len(self.basis_fns) / 2)):
            s = num_k_in_each_dir ** 2 * (self.basis_fns[i * 2].k[0] + self.imax) + \
                num_k_in_each_dir * (self.basis_fns[i * 2].k[1] + self.imax) + \
                self.basis_fns[i * 2].k[2] + self.imax
            self.basis_indices_map[s] = i

    # --- Basis set ---
    def init_single_basis(self, cutoff, k_shift=[0., 0., 0.]):
        """Create single-particle basis. Member function of class UEG

        Parameters
        ----------
        cutoff: float,
            energy cutoff, in units of `2*(2pi/L)^2`, defining the
            single-particle basis.
            Only single-particle basis functions with a kinetic energy equal
            to or less than the cutoff are included as basis functions.
        k_shift: 1D float array or list, a shift added to plane waves, in
                 unit of 2pi/L

        Returns
        -------
        basis_fns: tuple
            of relevant class PlaneWaveFn objects,
            i.e. the single-particle basis set.

    """

        # Single particle basis within the desired energy cutoff.
        # cutoff = cutoff*(2*np.pi/self.L)**2
        k_shift = np.array(k_shift)
        kp_shift = k_shift * 2 * np.pi / self.L
        imax = int(np.ceil(np.sqrt(cutoff + k_shift.dot(k_shift)))) + 1
        self.cutoff = cutoff
        self.imax = imax
        basis_fns = []

        for i in range(-imax, imax + 1):
            for j in range(-imax, imax + 1):
                for k in range(-imax, imax + 1):
                    bfn = planewave.BasisFunc(i, j, k, self.L, 1, k_shift)
                    if self.is_k_in_basis(bfn.kinetic):
                        basis_fns.append(planewave.BasisFunc(i, j, k, self.L, 1, k_shift))
                        basis_fns.append(planewave.BasisFunc(i, j, k, self.L, -1, k_shift))

        basis_fns.sort()
        basis_fns = tuple(basis_fns)
        self.basis_fns = basis_fns

        self.init_basis_indices_map()

        return basis_fns

    def compute_kinetic_energy(self,
                                dtype=np.float64):
        """ Member function of class UEG to compute the kinetic component
        of the orbital energies.
        Returns:
        -----------
        kinetic_G: array object
            Kinetic energies, dimension [n_p] (number of spatial orbitals).
        """
        algo_name = "UEG.compute_kinetic_energy"
        if self.basis_fns is None:
            raise ValueError(algo_name, "Basis functions not initialized!")
        n_p = int(len(self.basis_fns)/2)
        kinetic_G = np.zeros([n_p], dtype=dtype)
        for i in range(n_p):
            kinetic_G[i] = self.basis_fns[2*i].kinetic
        return kinetic_G
    
    def get_fock(self,
                 dtype=np.float64):
        """ Member function of class UEG to compute the Hartree Fock Energy,
                    the Fock matrix and the 'OOOO', 'VOVO', 'VOOV' block of 
                    the Coulomb tensor.
        Returns:
        -----------
        EHF: scalar object
            Hartree Fock energy.
        Epsilon_i: array object
            Orbital energies of the occupied orbitals, dimension [n_occ]
        Epsilon_a: array object
            Orbital energies of the virtual orbitals, dimension [n_virt]
        fock_pq: array object
            Fock matrix, dimension [n_p, n_p] (number of spatial orbitals)
        V_oooo: array object
            'OOOO' block of the Coulomb tensor, dimension [n_occ, n_occ, n_occ, n_occ]
        V_vovo: array object
            'VOVO' block of the Coulomb tensor, dimension [n_virt, n_occ, n_virt, n_occ]
        V_voov: array object
            'VOOV' block of the Coulomb tensor, dimension [n_virt, n_occ, n_occ, n_virt]
        """
        algo_name = "UEG.get_fock"
        print_logging_info(algo_name, ": calculating the Fock Matrix and the Hatree-Fock energy", level=0)
        start_time = time.time()
        if self.basis_fns is None:
            raise ValueError(algo_name, "Basis functions not initialized!")
        if self.is_tc:
            print_logging_info("Using TC method", level=1)
            if self.correlator is None:
                raise ValueError("Correlator for the transcorrelated framework not initialized!")
            else:
                print_logging_info("Using correlator: ", self.correlator.__name__, level=1)
            if self.k_cutoff is None:
                raise ValueError("K-cutoff for the transcorrelated framework not initialized!")
            else:
                print_logging_info("K-cutoff in correlator: {:.8f} [2π/L]".format(self.k_cutoff), level=1)
            if self.kmesh_fac is None:
                print_logging_info("K'-mesh scaling factor: 1.0 [L'/L] (default)", level=1)
            else:
                print_logging_info("K'-mesh scaling factor: {:.8f} [L'/L]".format(self.kmesh_fac), level=1)
            if self.kmesh_cutoff is None:
                print_logging_info("K'-mesh cutoff: consistent with energy cutoff of the simulation box.", level=1)
            else:
                print_logging_info("K'-mesh cutoff: {:.8f} ½[2π/L']^2".format(self.kmesh_cutoff), level=1)
            #if self.gamma is None:
            #    raise ValueError("Gamma not initialized!")
        else:
            print_logging_info("Using non-TC method", level=1)
        
        nP = int(len(self.basis_fns) / 2)
        no = int(self.n_ele / 2)
        nv = nP - no
        # initialize the fock matrix.
        fock_pq = np.zeros([nP, nP], dtype=dtype)
        # initialize the Coulomb tensor.
        V_oooo = np.zeros([no, no, no, no], dtype=dtype)
        V_vovo = np.zeros([nv, no, nv, no], dtype=dtype)
        V_voov = np.zeros([nv, no, no, nv], dtype=dtype)
        # initialize the Hartree Fock energy.
        EHF = 0.0

        # get the components of the Coulomb tensor.
        start_time_coulomb = time.time()
        if self.is_tc:
            print_logging_info("Calculating the Coulomb tensor and pure TC 2-body integrals [oooo][vovo][voov]", level=1)
            print_logging_info("Calculating [oooo] block", level=2)
            idx    = get_block_index( 'oooo', nP, no)
            V_oooo = self.get_2b_int( idx, \
                                 is_only_2b=True)
            print_logging_info("Calculating [vovo] block", level=2)
            idx    = get_block_index( 'vovo', nP, no)
            V_vovo = self.get_2b_int( idx, \
                                 is_only_2b=True)
            print_logging_info("Calculating [voov] block", level=2)
            idx    = get_block_index( 'voov', nP, no)
            V_voov = self.get_2b_int( idx, \
                                 is_only_2b=True)
        else:
            print_logging_info("Calculating the Coulomb tensor [oooo][vovo][voov]", level=1)
            print_logging_info("Calculating [oooo] block", level=2)
            idx    = get_block_index( 'oooo', nP, no)
            V_oooo = self.get_2b_int( idx )
            print_logging_info("Calculating [vovo] block", level=2)
            idx    = get_block_index( 'vovo', nP, no)
            V_vovo = self.get_2b_int( idx )
            print_logging_info("Calculating [voov] block", level=2)
            idx    = get_block_index( 'voov', nP, no)
            V_voov = self.get_2b_int( idx )
        end_time_coulomb = time.time()
        print_logging_info("Elapsed time = {:.3f} s: ".format(end_time_coulomb - start_time_coulomb) +
                            "calculating the Coulomb tensor.", level=1)

        # get the kinetic energies of the basis functions.
        kinetic_G = self.compute_kinetic_energy()

        # get the orbital energies (with/without pure 2b int. from transcorrelation).
        Epsilon_i = hf.calcOccupiedOrbE(kinetic_G, V_oooo, no)
        Epsilon_a = hf.calcVirtualOrbE(kinetic_G, V_vovo, V_voov, no, nv)

        # get the Hartree Fock energy.
        print_logging_info("Calculating the Hartree Fock energy", level=1)
        EHF = hf.calc_hf_e_part(Epsilon_i, V_oooo)
        print_logging_info("HF E = {:.8f}".format(EHF), level=2)

        # get the singly contractions (effective 2-body integrals) from the 3-body integrals.
        if self.is_tc:
            start_time_effect_2b = time.time()
            print_logging_info("Calculating the effective 2-body integrals [oooo][vovo][voov]", level=1)
            print_logging_info("Calculating [oooo] block", level=2)
            idx    = get_block_index( 'oooo', nP, no)
            V_oooo += self.get_2b_int( idx, \
                                    is_effect_2b=True)
            print_logging_info("Calculating [vovo] block", level=2)
            idx    = get_block_index( 'vovo', nP, no)
            V_vovo += self.get_2b_int( idx, \
                                    is_effect_2b=True)
            print_logging_info("Calculating [voov] block", level=2)
            idx    = get_block_index( 'voov', nP, no)
            V_voov += self.get_2b_int( idx, \
                                    is_effect_2b=True)
            end_time_effect_2b = time.time()
            print_logging_info("Elapsed time = {:.3f} s: ".format(end_time_effect_2b - start_time_effect_2b) +
                                "calculating the effective 2-body integrals.", level=1)
            
        # get doubly and tryply contractions of the 3-body integrals,
        #     correct orbital energies,
        #     and add the mean field contribution from the 3-body integrals 
        #     to the Hartree Fock energy.
        if self.is_tc:
            start_time_3b = time.time()
            print_logging_info("Calculating the doubly and triply contractions of the 3-body integrals", level=1)

            contr_from_doubly_contra_3b = self.double_contractions_in_3_body()
            contr_from_triply_contra_3b = self.triple_contractions_in_3_body()

            Epsilon_i += contr_from_doubly_contra_3b[:no]
            Epsilon_a += contr_from_doubly_contra_3b[no:]

            print_logging_info("3-body mean-field E = {:.8f}".format(contr_from_triply_contra_3b), level=1)
            EHF += contr_from_triply_contra_3b
            end_time_3b = time.time()
            print_logging_info("Elapsed time = {:.3f} s: ".format(end_time_3b - start_time_3b) +
                                "calculating the doubly and triply contractions of the 3-body integrals.", level=1)

        # get the Hartree Fock matrix.
        print_logging_info("Calculating the Fock matrix", level=1)
        fock_pq = hf.construct_hf_matrix_part(no, np.diag(kinetic_G), V_oooo, V_vovo, V_voov)

        print_logging_info("Total HF E = {:.8f}".format(EHF), level=1)

        print_logging_info("Elapsed time = {:.3f} s: ".format(time.time() - start_time) +   
                           "calculating the Hartree Fock energy and the Fock matrix.", level=1)

        return EHF, Epsilon_i, Epsilon_a, fock_pq, V_oooo, V_vovo, V_voov
    
    def get_2b_int(self, idx, 
                   is_only_2b=False, 
                   is_effect_2b=False,
                   dtype=np.float64):
        """ Member function of class UEG to compute the 2-body integrals
        (Coulomb integrals) and the additional 2-body integrals from
        the transcorrelated method: pure 2-body integrals, effective 2-body
        integrals from the singly contracted 3-body integrals.
        Parameters
        ----------
        idx: tuple of int
            indices of the block of the Coulomb tensor to be computed.
        is_only_2b: bool
            parameter which determines to include only the additional
            pure 2-body tc integrals, besides the Coulomb integrals.
        is_effect_2b: bool
            parameter which determines to include the effective 2-body integrals 
            as a result of single contractions from the 3-body integrals. 
            There are four types of single contractions in the 3-body integrals: 
            RPA type and 3 exchange types.
        
        Returns
        -------
        V_pqrs: tensor object (tensor by default)
            of size [ idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7] ], np array.
        """
        algo_name = "UEG.get_2b_int"
        #start_time = time.time()
        if self.basis_fns is None:
            raise ValueError(algo_name, "Basis functions not initialized!")
        if self.is_tc:
            if self.correlator is None:
                raise ValueError("Correlator for the transcorrelated framework not initialized!")
            if self.k_cutoff is None:
                raise ValueError("K-Cutoff for the transcorrelated framework not initialized!")
            #if self.gamma is None:
            #    raise ValueError("Gamma not initialized!")

        nP = int(len(self.basis_fns) / 2)
        no = int(self.n_ele / 2)
        nv = nP - no

        #print_logging_info(algo_name + ": calculating 2-body integrals for block: {}".format(idx), level=2)

        # Using Numba JIT-compiled version for better performance.
        #print_logging_info(algo_name + ": Using Numba JIT-compiled version for better performance.", level=2)
        # 1. Unpack data into Numba-compatible structures (NumPy arrays).
        if self.is_tc:
            correlator_idx = self.get_correlator_idx()
        else:
            correlator_idx = 0  # None
        k_cutoff = self.k_cutoff if self.k_cutoff is not None else  int(np.ceil(np.sqrt(self.cutoff)))
        gamma  = self.gamma if self.gamma is not None else 1.0
        if self.kPrime is None:
            raise ValueError("kPrime not initialized!")
        else:
            kPrime = self.kPrime.astype(np.float64) * (2.0 * np.pi / (self.L * self.kmesh_fac))
        OmegaPrime = (self.L * self.kmesh_fac) ** 3
        basis_occ_Kp = np.array([self.basis_fns[i * 2].kp for i in range(no)], dtype=np.float64)
        basis_Kvec = np.array([self.basis_fns[i * 2].k for i in range(nP)], dtype=np.int32)
        basis_Kp = np.array([self.basis_fns[i * 2].kp for i in range(nP)], dtype=np.float64)
        # 2. Prepare for parallel execution.
        #p_idx_range = idx[1] - idx[0]
        #det_num_threads(p_idx_range)
        # 3. Compute the integrals.
        V_pqrs=  _get_2b_int( idx, self.n_ele, self.Omega, self.L, self.imax, 
                                k_cutoff, gamma, OmegaPrime,
                                kPrime, self.basis_indices_map,
                                basis_occ_Kp, basis_Kvec, basis_Kp,
                                is_only_2b, is_effect_2b, self.is_tc,
                                correlator_idx, multiply_by_k_square=False,
                                dtype=dtype)
        return V_pqrs
    
    def eval_3b_integrals(self, correlator=None, dtype=np.float64, sp=1):
        """ Member function of class UEG to evaluate the full 3-body integrals
        within the transcorrelation framework.

        Parameters:
        -----------
        correlator: function
            for example the function member function trunc
        dtype: data types in which the integrals are stored
            for example, by default dtype = np.float64
        sp: int
            specify whether the integrals are stored in sparse format or not
            1: sparse format (default)
            0: dense format

        Returns:
        -------
        V_opqrst: tensor object (tensor by default)
            The full 3-body integrals, dimension [nP, nP, nP, nP, nP, nP],
            where nP: int, is the number of spatial orbitals.
        """

        algo_name = "UEG.eval_3b_integrals"
        print_logging_info(algo_name, level=0)
        start_time = time.time()


        if self.basis_fns is None:
            raise ValueError(algo_name, "Basis functions not initialized!")
        if correlator is None:
            self.correlator = self.trunc
            print_logging_info("No correlator given.", level=1)
            print_logging_info("Using the default correlator: " \
                               + self.correlator.__name__, level=1)
        else:
            self.correlator = correlator
        if self.basis_indices_map is None:
            raise ValueError(algo_name, "Basis indices map not initialized!")

        print_logging_info(algo_name)
        print_logging_info("Using TC method", level=1)
        print_logging_info("Using correlator:", correlator.__name__, level=1)
        print_logging_info("k_cutoff in correlator:", self.k_cutoff, level=1)

        nP = int(len(self.basis_fns) / 2)
        V_opqrst = np.zeros([nP, nP, nP, nP, nP, nP], dtype=dtype)
        # due to the momentum conservation, only 5 indices are free.
        # implementation follow closely the get_lmat_ueg in NECI
        num_k_in_each_dir = self.imax * 2 + 1

        for o in range(nP):
            print_logging_info("Elapsed time = {:.3f} s: "
                               .format(time.time() - start_time) +
                               "calculating the {} out of {} orbitals".format(o, nP), level=1)
            for r in range(nP):
                k_int_vec1 = self.basis_fns[2 * r].k - self.basis_fns[2 * o].k
                for p in range(nP):
                    for s in range(nP):
                        k_int_vec2 = self.basis_fns[2 * p].k - self.basis_fns[2 * s].k
                        for q in range(nP):
                            t_int_vec = -k_int_vec1 + k_int_vec2 + self.basis_fns[2 * q].k
                            locT = num_k_in_each_dir ** 2 * (t_int_vec[0] + self.imax) + \
                                   num_k_in_each_dir * (t_int_vec[1] + self.imax) + \
                                   t_int_vec[2] + self.imax
                            if len(self.basis_indices_map) > locT >= 0:
                                t = int(self.basis_indices_map[locT])
                                if t < 0:
                                    continue
                            else:
                                continue

                            k_vec1 = 2.0 * np.pi / self.L * k_int_vec1
                            k_vec2 = 2.0 * np.pi / self.L * k_int_vec2

                            w12 = self.correlator(k_vec1.dot(k_vec1)) \
                                  * self.correlator(k_vec2.dot(k_vec2)) \
                                  * k_vec1.dot(k_vec2)
                            w = -(w12) / 2. / self.Omega ** 2
                            index = o * nP ** 5 + p * nP ** 4 + q * nP ** 3 + r * nP ** 2 + s * nP + t

                            if index >= nP ** 6:
                                raise "Index exceeds size of the tensor"

                            V_opqrst[o, p, q, r, s, t] = w


        print_logging_info("{:.3f} s spent on ".format(time.time() - start_time) + __name__, \
                           level=1)

        return V_opqrst

    def eval_2b_integrals(self, correlator=None,
                          is_rpa_approx=False,
                          is_only_2b=False,
                          is_only_non_hermi_2b=False,
                          is_only_hermi_2b=False,
                          is_effect_2b=False,
                          is_exchange_1=False,
                          is_exchange_2=False,
                          is_exchange_3=False,
                          dtype=np.float64,
                          sp=1):
        """Member function of class UEG. This function evaluates 2-body
        integrals within the transcorrelation framework. The 2-body integrals
        that can be computed by this function are:
        1. normal Coulomb repulsion integrals, V_pqrs
        2. additional 2-body terms from transcorrection
        3. singly contracted 3-body terms (effective 2-body integrals)

        Parameters
        ----------
        correlator: function
            for example the member function UEG.trunc
        is_rpa_approx: bool
            parameter which determines whether to include only the RPA type
            contractions from the 3-body integrals into the final 2-body
            integrals, besides the normal Coulomb repulsion integrals and
            the additional 2-body tc integrals.
        is_only_2b: bool
            parameter which determines whether to include only the additional
            2-body tc integrals, besides the Coulomb integrals.
        is_only_non_hermi_2b: bool
            parameter which determines whether to include only the nonhermitian
            2-body tc integrals, besides the Coulomb integrals.
        is_only_hermi_2b: bool
            parameter which determines whether to inclulde only the hermitian
            2-body tc terms/integrals (excluding the nonhermitian terms),
            besides the Coulomb integrals.
        is_effect_2b: bool
            parameter which determines to include all 2-body integrals,
            including the normal Coulomb and 2-body tc integrals, plus
            the effective 2-body integrals as a result of single contractions
            from the 3-body integrals. There are four types of single
            contractions in the 3-body integrals: RPA type and 3 exchange types.
        is_exchange_1: bool
            generates only the 1st type of single contraction in the 3-body
            integrals. For test purpose only.
        is_exchange_2: bool
            generates only the 2nd type of single contraction in the 3-body
            integrals. For test purpose only.
        is_exchange_3: bool
            generates only the 3rd type of single contraction in the 3-body
            integrals. For test purpose only.
        dtype: data type
            for the returning integral elements. By default, np.float64 is used.
        sp: int
            Sparse format for the returned integrals.
            1: sparse (default)
            0: dense

        Returns
        -------
        V_pqrs: tensor
            of size [n_p, n_p, n_p, n_p], by default, np array
        """

        start_time = time.time()

        print_logging_info(__name__, level=0)

        if self.basis_fns is None:
            raise ValueError("Basis functions not initialized!")

        if correlator is not None:
            self.correlator = correlator
            print_logging_info("Using TC method", level=1)
            print_logging_info("Using correlator: ", correlator.__name__,
                               level=1)
            if self.k_cutoff is not None:
                print_logging_info("k_cutoff in correlator = {:.8f}".format(
                    self.k_cutoff), level=1)
            if self.gamma is not None:
                print_logging_info("Gamma in correlator = {:.8f}".format(
                    self.gamma), level=1)

            if is_only_2b:
                print_logging_info("Including only pure 2-body terms: ",
                                   is_only_2b, level=1)
            if is_rpa_approx:
                print_logging_info("Including only RPA approximation for ",
                                   "3-body: ", is_rpa_approx, level=1)
            if is_effect_2b:
                print_logging_info("Including effective 2-body terms from ",
                                   "3-body: ", is_effect_2b, level=1)
            if is_exchange_1:
                print_logging_info("Using only 1. exchange type 2-body from ",
                                   "3-body: ", is_exchange_1, level=1)
            if is_exchange_2:
                print_logging_info("Using only 2. exchange type 2-body from ",
                                   "3-body: ", is_exchange_2, level=1)

            if is_exchange_3:
                print_logging_info("Using only 3. exchange type 2-body from ",
                                   "3-body: ", is_exchange_3, level=1)
            # Just for testing, not used in production
            if is_only_non_hermi_2b:
                print_logging_info("Including only non-hermitian 2-body ",
                                   "terms: ", is_only_non_hermi_2b, level=1)

            if is_only_hermi_2b:
                print_logging_info("Including only hermitian 2-body terms: ",
                                   is_only_non_hermi_2b, level=1)

        n_p = int(len(self.basis_fns) / 2)
        V_pqrs = np.zeros([n_p, n_p, n_p, n_p], dtype=dtype)
        indices = []
        values = []

        num_k_in_each_dir = self.imax * 2 + 1

        for p in range(n_p):
            print_logging_info("Elapsed time = {:.3f} s: calculating "
                               .format(time.time() - start_time)
                               + "the {} of {} orbitals"
                               .format(p,n_p), level=1)
            for r in range(n_p):
                d_int_k = self.basis_fns[r * 2].k - self.basis_fns[p * 2].k
                d_k_vec = self.basis_fns[r * 2].kp - self.basis_fns[p * 2].kp
                u_mat = 0.
                if correlator is not None:
                    u_mat = self.sumNablaUSquare(d_k_vec)

                for q in range(n_p):
                    int_ks = self.basis_fns[q * 2].k - d_int_k
                    # if self.is_k_in_basis(int_ks):
                    loc_s = num_k_in_each_dir ** 2 * (int_ks[0] + self.imax) + \
                            num_k_in_each_dir * (int_ks[1] + self.imax) + \
                            int_ks[2] + self.imax
                    if len(self.basis_indices_map) > loc_s >= 0:
                        s = int(self.basis_indices_map[loc_s])
                        if s < 0 or s >= n_p:
                            continue
                    else:
                        continue

                    dk_square = d_k_vec.dot(d_k_vec)
                    w = 0.
                    if correlator is None:
                        if np.abs(dk_square) > 0.:
                            w = 4. * np.pi / dk_square / self.Omega
                            #indices.append(n_p ** 3 * p + n_p ** 2 * q + n_p * r + s)
                            #values.append(w)
                    elif is_rpa_approx:
                        # tc
                        if np.abs(dk_square) > 0.:
                            w = - (self.n_ele) * dk_square \
                                * correlator(dk_square) ** 2 / self.Omega
                            w = w / self.Omega
                        else:
                            w = 0.
                        #indices.append(n_p ** 3 * p + n_p ** 2 * q + n_p * r + s)
                        #values.append(w)
                    elif is_only_2b:
                        if np.abs(dk_square) > 0.:
                            rs_dk = self.basis_fns[r * 2].kp \
                                    - self.basis_fns[s * 2].kp
                            w = 4. * np.pi / dk_square \
                                + u_mat \
                                + dk_square * correlator(dk_square) \
                                - (rs_dk.dot(d_k_vec)) \
                                * correlator(dk_square)
                            w = w / self.Omega
                        else:
                            w = u_mat / self.Omega
                        #indices.append(n_p ** 3 * p + n_p ** 2 * q + n_p * r + s)
                        #values.append(w)
                    elif is_only_hermi_2b:
                        if np.abs(dk_square) > 0.:
                            w = 4. * np.pi / dk_square \
                                + u_mat \
                                + dk_square * correlator(dk_square)
                            w = w / self.Omega
                        else:
                            w = u_mat / self.Omega
                        #indices.append(n_p ** 3 * p + n_p ** 2 * q + n_p * r + s)
                        #values.append(w)
                    elif is_only_non_hermi_2b:
                        w = 0.
                        if np.abs(dk_square) > 0.:
                            rs_dk = self.basis_fns[r * 2].kp \
                                    - self.basis_fns[s * 2].kp
                            w = 4. * np.pi / dk_square \
                                - (rs_dk.dot(d_k_vec)) * correlator(dk_square)
                            w = w / self.Omega

                        #indices.append(n_p ** 3 * p + n_p ** 2 * q + n_p * r + s)
                        #values.append(w)
                    elif is_effect_2b:
                        if np.abs(dk_square) > 0.:
                            w = - (self.n_ele) * dk_square \
                                * correlator(dk_square) ** 2 / self.Omega \
                                + 2. * self.contract_exchange_3_body(
                                self.basis_fns[2 * r].kp, d_k_vec) \
                                - 2. * self.contract_exchange_3_body(
                                self.basis_fns[2 * p].kp, d_k_vec) \
                                + 2. * self.contractP_KWithQ(
                                self.basis_fns[2 * r].kp, d_k_vec)
                        else:
                            w = (2. * self.contractP_KWithQ(
                                self.basis_fns[2 * r].kp, d_k_vec))
                        w = w / self.Omega
                        #indices.append(n_p ** 3 * p + n_p ** 2 * q + n_p * r + s)
                        #values.append(w)
                    # exchange 1-3 are for test purpose only.
                    elif is_exchange_1:
                        if np.abs(dk_square) > 0.:
                            w = + 2. * self.contract_exchange_3_body(
                                self.basis_fns[2 * r].kp, d_k_vec)
                            w = w / self.Omega
                        else:
                            w = 0.
                        #indices.append(n_p ** 3 * p + n_p ** 2 * q + n_p * r + s)
                        #values.append(w)
                    elif is_exchange_2:
                        if np.abs(dk_square) > 0.:
                            w = - 2. * self.contract_exchange_3_body(
                                self.basis_fns[2 * p].kp, d_k_vec)
                            w = w / self.Omega
                        else:
                            w = 0.
                        #indices.append(n_p ** 3 * p + n_p ** 2 * q + n_p * r + s)
                        #values.append(w)
                    elif is_exchange_3:
                        if np.abs(dk_square) > 0.:
                            w = + 2. * self.contractP_KWithQ(
                                self.basis_fns[2 * r].kp, d_k_vec)
                            w = w / self.Omega
                        else:
                            w = + 2. * self.contractP_KWithQ(
                                self.basis_fns[2 * r].kp, d_k_vec)
                            w = w / self.Omega
                        #indices.append(n_p ** 3 * p + n_p ** 2 * q + n_p * r + s)
                        #values.append(w)
                    V_pqrs[p,q,r,s] = w

        if is_effect_2b:
            # symmetrize the integral with respect to electron 1 and 2
            V_sym_pqrs = np.zeros(V_pqrs.shape)
            V_sym_pqrs += 0.5 * (V_pqrs + V_pqrs.transpose((1,0,3,2)))
            V_pqrs = V_sym_pqrs

        print_logging_info("{:.3f} s spent on ".format(time.time() - start_time)+__name__, level=1)
        return V_pqrs

    def contract_exchange_3_body(self, p_vec, kVec):
        """ Member function of class UEG. Calculate the 1st and 2nd types of
        single exchange type contractions in the 3-body integrals.

        Parameters
        ----------
        p_vec: nparray of float dtype, size 3
        kVec: nparray of float dtype, size 3
            momentum transfer

        Returns:
        --------
        result: float
        """

        p_prim = np.array([self.basis_fns[i * 2].kp for i in \
                          range(int(self.n_ele / 2))])
        p_vec = p_vec - p_prim
        kVecSquare  = np.einsum("i,i->", kVec, kVec,      optimize=True)
        pVecSquare  = np.einsum("ni,ni->n", p_vec, p_vec, optimize=True)
        pVecDotKVec = np.einsum("ni,i->n", p_vec, kVec,   optimize=True)
        #kVecSquare  = np.sum( (kVec * kVec), axis=-1)
        #pVecSquare  = np.sum( (p_vec * p_vec), axis=-1)
        #pVecDotKVec = np.sum( (p_vec * kVec), axis=-1) 
        result = pVecDotKVec * self.correlator(kVecSquare) \
                 * self.correlator(pVecSquare)
        result = np.einsum("n->", result, optimize=True) / self.Omega

        return result

    def contractP_KWithQ(self, pVec, kVec):
        """ Member function of class UEG. Calculate the 3rd type of
        single exchange type contractions in the 3-body integrals.

        Parameters
        ----------
        pVec: nparray of float dtype, size 3
        kVec: nparray of float dtype, size 3
            momentum transfer

        Returns:
        --------
        result: float
        """

        pPrim = np.array([self.basis_fns[i * 2].kp for i in \
                          range(int(self.n_ele / 2))])
        vec1 = pVec - kVec - pPrim
        vec2 = pVec - pPrim

        dotProduct = np.einsum("ni,ni->n", vec1, vec2, optimize=True)
        vec1Square = np.einsum("ni,ni->n", vec1, vec1, optimize=True)
        vec2Square = np.einsum("ni,ni->n", vec2, vec2, optimize=True)
        #dotProduct = np.sum((vec1 * vec2), axis=-1)
        #vec1Square = np.sum((vec1 * vec1), axis=-1)
        #vec2Square = np.sum((vec2 * vec2), axis=-1)
        result = dotProduct * self.correlator(vec1Square) \
                 * self.correlator(vec2Square)

        result = np.einsum("n->", result, optimize=True) / self.Omega

        return result

    def contract3BodyIntegralsTo2Body(self, integrals):
        # factor 2 for the spin degeneracy
        fac = 2
        RPA2Body = fac * einsum("opqrsq->oprs", integrals)
        return RPA2Body

    def sumNablaUSquare(self, k):

        if self.kPrime is None:
            raise ValueError("kPrime not initialized!")
        
        k1 = 2 * np.pi * self.kPrime / ( self.L * self.kmesh_fac )
        k2 = k - k1

        k1Square = np.einsum("ni,ni->n", k1, k1, optimize=True)
        k2Square = np.einsum("ni,ni->n", k2, k2, optimize=True)
        k1DotK2  = np.einsum("ni,ni->n", k1, k2, optimize=True)
        result = k1DotK2 * self.correlator(k1Square) * self.correlator(k2Square)
        result = np.einsum("n->", result, optimize=True) / self.Omega

        return result
    
    def init_kPrime(self):
        """
        Member function of class UEG
        This function generates the k' vectors for the transcorrelated
        integrals. The k' vectors are generated in the range of -ipmax to 
        ipmax (defined by the energy cutoff in the k'-mesh self.kmesh_cutoff) 
        in each direction. The k' vectors are stored in the class
        variable kPrime.

        Note: need to test convergence of this k'-mesh cutoff.

        Returns
        -------
        kPrime: nparray of int dtype, size (3*ipmax+1, 3)
        """

        if self.kmesh_fac is None:
            self.kmesh_fac = 1.
        elif self.kmesh_fac < 1:
            raise ValueError("kmesh_fac should be >= 1!")
        
        if self.kmesh_cutoff is None:
            self.kmesh_cutoff = self.cutoff * (self.kmesh_fac**2)
        elif self.kmesh_cutoff < self.cutoff:
            raise ValueError("kmesh_cutoff should be >= cutoff!")

        ipmax = int(np.ceil(np.sqrt(self.kmesh_cutoff))) + 1

        kPrime = np.array([[i, j, k] for i in range(-ipmax, ipmax + 1) \
                           for j in range(-ipmax, ipmax + 1) for k in \
                            range(-ipmax, ipmax + 1)])

        self.kmesh_imax = ipmax
        self.kPrime = kPrime

    def triple_contractions_in_3_body(self):
        """
        This function computes the triply contracted 3-body interactions.
        Return: a scalar (float) which should be added to the total energy
        """
        algo_name = "UEG.triple_contractions_in_3_body"
        print_logging_info(algo_name, level=1)

        p_pi = np.array([self.basis_fns[i * 2].kp for i in range(int(self.n_ele / 2))])
        q_qi = np.array([self.basis_fns[i * 2].kp for i in range(int(self.n_ele / 2))])
        p_q_pqi = np.zeros([len(p_pi), len(q_qi), 3])
        p_q_pqi = p_pi[:, None, :] - q_qi[None, :, :]
        p_qSquare_pq = einsum("pqi,pqi->pq", p_q_pqi, p_q_pqi)

        p_qSquare = p_qSquare_pq
        up_q_pq = self.correlator(p_qSquare)

        dirE = up_q_pq ** 2 * p_qSquare
        # factor 2 comes from sum over spins
        dirE = sum(sum(dirE)) * self.n_ele / 2 / self.Omega ** 2 * 2

        # exchange type
        p_o_dot_p_q = einsum("poi,pqi->pqo", p_q_pqi, p_q_pqi)

        u_pq_u_po = einsum("pq,po->pqo", up_q_pq, up_q_pq)
        #u_pq_u_po = up_q_pq[:, :, None] + up_q_pq[:, None, :]
        # factor 2 from sum over spin, another factor of 2 from mirror symmetry
        excE = -2 * 2 * einsum("pqo,pqo->", p_o_dot_p_q, u_pq_u_po) / 2. / self.Omega ** 2
        result = dirE + excE
        print_logging_info("Direct E = {:.8f}".format(dirE), level=2)
        print_logging_info("Exchange E = {:.8f}".format(excE), level=2)

        return result

    def double_contractions_in_3_body(self):
        """
            This function makes two contractions in the 3-body integrals,
            resulting in one body energies, which should be added to the
            original one-particle energies from the HF theory

            return: a numpy array of size equal to the number of plane waves
        """
        algo_name = "UEG.double_contractions_in_3_body"
        print_logging_info(algo_name, level=1)
        # some constants

        num_o = int(self.n_ele / 2)
        num_v = int(len(self.basis_fns) / 2) - num_o
        num_p = num_o + num_v

        # initialize the one_particle_energies
        one_particle_energies = np.zeros(num_p)
        e_perl = np.zeros(num_p)

        # generate p vectors
        k_vec_p = np.array([self.basis_fns[i * 2].kp for i in \
                            range(int(len(self.basis_fns) / 2))])
        k_vec_i = np.array([self.basis_fns[i * 2].kp for i in \
                            range(int(self.n_ele / 2))])

        # the perl shape diagram
        for orb_p in range(num_p):
            k_vec_p_minus_i = k_vec_p[orb_p] - k_vec_i
            k_vec_p_minus_i_square = einsum("ij, ij-> i", k_vec_p_minus_i, \
                                               k_vec_p_minus_i)
            e_perl[orb_p] = np.sum(self.correlator(k_vec_p_minus_i_square) ** 2 \
                                   * k_vec_p_minus_i_square)

        e_perl = 2.0 * self.n_ele / self.Omega ** 2 / 2 * e_perl

        one_particle_energies += e_perl

        # wave diagram
        e_wave = np.zeros(num_p)
        diff_vec_pi_pij = np.zeros([num_p, num_o, 3])
        #diff_vec_pi_pij.i("pij") << ctf.astensor(k_vec_p).i("pj") \
        #- ctf.astensor(k_vec_i).i("ij")
        # diff_vec_pi_pij = einsum("pj, ij -> pij", k_vec_p,-k_vec_i)
        diff_vec_pi_pij = k_vec_p[:, None, :]-k_vec_i[None, :, :]
        diff_vec_pi_square_pi = einsum("pij,pij -> pi", \
                                             diff_vec_pi_pij, \
                                             diff_vec_pi_pij)
        diff_pi_dot_diff_pj_pij = einsum("pik, pjk -> pij", \
                                               diff_vec_pi_pij, \
                                               diff_vec_pi_pij)
        diff_pi_square = diff_vec_pi_square_pi
        u_diff_pi = self.correlator(diff_pi_square)
        t_u_diff_pi_multiply_u_diff_pj_pij = einsum("pi,pj->pij", \
                                                        u_diff_pi, u_diff_pi)
        #t_u_diff_pi_multiply_u_diff_pj_pij = u_diff_pi[:, :, None] + u_diff_pi[:, None, :]
        e_wave = einsum("pij,pij->p", diff_pi_dot_diff_pj_pij, \
                            t_u_diff_pi_multiply_u_diff_pj_pij)

        e_wave = -e_wave * 2 / self.Omega ** 2 / 2

        one_particle_energies += e_wave

        # shield diagram, which is independent of vector p. So initialize as
        # ones

        e_shield = np.ones(num_p)
        diff_vec_ij_ijk = np.zeros([num_o, num_o, 3])
        #diff_vec_ij_ijk.i("ijk") << ctf.astensor(k_vec_i).i("ik") \
        #- ctf.astensor(k_vec_i).i("jk")
        #diff_vec_ij_ijk = einsum("ik, jk -> ijk", k_vec_i, -k_vec_i)
        diff_vec_ij_ijk = k_vec_i[:, None, :] - k_vec_i[None, :, :]
        diff_vec_ij_square_ij = einsum("ijk,ijk -> ij", diff_vec_ij_ijk, diff_vec_ij_ijk)
        diff_ij_square = diff_vec_ij_square_ij
        u_diff_ij = self.correlator(diff_ij_square)
        u_diff_ij_square = u_diff_ij ** 2
        e_shield = e_shield * einsum("ij,ij->", u_diff_ij_square, diff_ij_square)
        # bug, missing a factor of 2 from spin degree of freedom
        e_shield = 2 * e_shield / 2 / self.Omega ** 2

        one_particle_energies += e_shield

        # frog diagram (there are two types which turn out to be the same, so
        # a factor of 4 will be multiplied in the end)
        e_frog = np.zeros(num_p)
        # Using -(p-i) as vector (i-p), pay attention to the exchange of p and
        # i indices
        diff_ij_dot_diff_ip_ijp = einsum("ijk, pik -> ijp", \
                                               diff_vec_ij_ijk, \
                                               -diff_vec_pi_pij)

        t_u_diff_ij_multiply_u_diff_ip_ijp = einsum("ij,pi->ijp", \
                                                        u_diff_ij, \
                                                        u_diff_pi)
        e_frog = einsum("ijp, ijp->p", diff_ij_dot_diff_ip_ijp, \
                            t_u_diff_ij_multiply_u_diff_ip_ijp)

        e_frog = -e_frog * 4 / self.Omega ** 2 / 2

        one_particle_energies += e_frog

        return one_particle_energies

    # CORRELATORS -----------------------------------------------------
    # Collection of correlators, should them be collected into a class?
    # Each correlator has some default parameters that are dependent on
    # the system and they are specific to UEG, so they should be part of
    # the UEG class.

    def get_correlator_idx(self):
        """ Member function of class UEG.
        Returns an integer index for the correlator function being used.

        Returns
        -------
        idx: int
            index of the correlator function
            0: None,
            1: trunc,
            2: coulomb,
            3: yukawa,
            4: yukawa-coulomb,
            5: gaskell,
            6: gaskell-modified,
            7: smooth.
        """
        if self.correlator is None:
            raise ValueError("Correlator function not initialized!")
        elif self.correlator == self.trunc:
            idx = self.CORRELATOR_TRUNC
        elif self.correlator == self.coulomb:
            idx = self.CORRELATOR_COULOMB
        elif self.correlator == self.yukawa:
            idx = self.CORRELATOR_YUKAWA
        elif self.correlator == self.yukawa_coulomb:
            idx = self.CORRELATOR_YUKAWA_COULOMB
        elif self.correlator == self.gaskell:
            idx = self.CORRELATOR_GASKELL
            raise NotImplementedError("Gaskell correlator not implemented yet!")
        elif self.correlator == self.gaskell_modified:
            idx = self.CORRELATOR_GASKELL_MODIFIED
            raise NotImplementedError("Modified Gaskell correlator not implemented yet!")
        elif self.correlator == self.smooth:
            idx = self.CORRELATOR_SMOOTH
            raise NotImplementedError("Smooth correlator not implemented yet!")
        elif self.correlator == self.RPA:
            idx = self.CORRELATOR_RPA
            raise NotImplementedError("RPA correlator not implemented yet!")
        else:
            raise ValueError("Correlator function not recognized!")
        return idx

    def yukawa(self, kSquare, multiply_by_k_square=False):
        '''
        The G=0 terms need more consideration
        '''
        rho = self.n_ele / self.Omega
        gamma_0 = np.sqrt(rho / 4. * np.pi)
        if self.gamma is None:
            # gamma = np.sqrt(4.*(3.*rho/np.pi)**(1/3.))
            gamma = gamma_0
        else:
            gamma = self.gamma * gamma_0
        # has to be - and divided by gamm to satisfy the cusp condition
        a = -4. * np.pi
        if self.k_cutoff is not None:
            k_cutoffSquare = (self.k_cutoff * (2 * np.pi / self.L)) ** 2
            # k_cutoffDenom = k_cutoffSquare*(k_cutoffSquare + gamma**2)
            k_cutoffDenom = (k_cutoffSquare + gamma)
        else:
            k_cutoffDenom = 1e-12
        if not multiply_by_k_square:
            # b = kSquare*(kSquare+gamma**2)
            b = (kSquare + gamma)
            result = np.divide(a, b, out=np.zeros_like(b), \
                               where=np.abs(b) > k_cutoffDenom)
        else:
            if kSquare > k_cutoffSquare:
                result = a / (kSquare + gamma) * kSquare
            else:
                result = 0.

        return result

    def trunc(self, kSquare):
        """ Member function of class UEG. A correlator function. Defined as
        -4pi/k^4 (k>kc), 0 (k<=kc).

        Parameters
        ----------
        kSquare: float or nparray of float
            the k-vector squared ($k^2$).

        Returns
        -------
        result: float or nparray of float
        """
        if self.k_cutoff is None:
            self.k_cutoff = int(np.ceil(np.sqrt(self.cutoff)))

        if self.gamma is None:
            self.gamma = 1.0

        k_cutoffSquare = (self.k_cutoff * (2 * np.pi / self.L)) ** 2

        if not isinstance(kSquare, np.ndarray):
            if kSquare <= k_cutoffSquare * (1 + 0.00001):
                kSquare = 0.
        else:
            kSquare[kSquare <= k_cutoffSquare * (1 + 0.00001)] = 0.
        result = np.divide(-4. * np.pi, kSquare ** 2, out=np.zeros_like(kSquare), \
                           where=(kSquare > 1e-12))
        return result * self.gamma

    def gaskell_modified(self, kSquare, multiply_by_k_square=False):
        '''
        input: G^2, will be scaled by k_fermi as beta^2=G^2/k_f^2
        output: \mu/beta^2, beta<2; 4\mu/beta^4, beta>2
        '''
        # define the parameter mu in gaskell correlator
        # this calculation will be done multipule times, it is not optimal to
        # recalculate it everytime. After refactoring all the correlators into
        # classes, this problem can be solved by using it as parameter of the
        # gaskall correlator class and only initialize it once. For now I will
        # keep it here.
        if self.k_cutoff is not None:
            k_cutoffSquare = (self.k_cutoff * (2 * np.pi / self.L)) ** 2
        else:
            k_cutoffSquare = 2
        mu = np.pi
        # k_fermi.dot(k_fermi)

        if not isinstance(kSquare, np.ndarray):
            result = 0.
            if kSquare < k_cutoffSquare and kSquare > 1e-12:
                # result = 4*mu/kSquare
                result = 0.
            else:
                result = 4 * mu / kSquare ** 2
        else:
            result = np.divide(0. * mu, kSquare, out=np.zeros_like(kSquare), \
                               where=(kSquare > 1e-12))
            result[kSquare >= k_cutoffSquare] = 0.
            result += np.divide(4 * mu, kSquare ** 2, out=np.zeros_like(kSquare), \
                                where=(kSquare >= k_cutoffSquare))
        # there should be an overall - sign
        return -result

    def gaskell(self, kSquare, multiply_by_k_square=False):
        '''
        input: G^2, will be scaled by k_fermi as beta^2=G^2/k_f^2
        output: \mu/beta^2, beta<2; 4\mu/beta^4, beta>2
        '''
        # define the parameter mu in gaskell correlator
        # this calculation will be done multipule times, it is not optimal to
        # recalculate it everytime. After refactoring all the correlators into
        # classes, this problem can be solved by using it as parameter of the
        # gaskall correlator class and only initialize it once. For now I will
        # keep it here.
        rho = self.n_ele / self.Omega
        mu = np.sqrt(4. * np.pi / rho)
        k_fermi = self.basis_fns[int(self.n_ele / 2) * 2].kp
        k_fermi_square = k_fermi.dot(k_fermi)
        # delta_k_square = (2.*np.pi/self.L)**2
        delta_k_square = k_fermi_square
        # int_k_fermi = self.basis_fns[int(self.n_ele / 2) * 2].k
        # beta_square = kSquare / (k_fermi.dot(k_fermi))

        if self.gamma is not None:
            gamma = self.gamma
        else:
            gamma = 1.

        mu *= gamma

        if self.k_cutoff is not None:
            k_cutoffSquare = self.k_cutoff ** 2 * delta_k_square
        else:
            k_cutoffSquare = 4. * delta_k_square

        if not isinstance(kSquare, np.ndarray):
            result = 0.
            if kSquare < k_cutoffSquare and kSquare > 1e-12:
                result = mu / kSquare
            else:
                # result = 4 * mu / kSquare ** 2
                result = 0.
        else:
            result = np.divide(mu, kSquare, out=np.zeros_like(kSquare),
                               where=(kSquare > 1e-12))
            result[kSquare > k_cutoffSquare] = 0.
            # result += np.divide(4 * mu, kSquare ** 2,
            #                    out=np.zeros_like(kSquare),
            #                    where=(kSquare >= k_cutoffSquare))
        # there should be an overall - sign
        return -result

    def smooth(self, kSquare, multiply_by_k_square=False):
        '''
        The G=0 terms need more consideration
        '''
        if self.k_cutoff is None:
            self.k_cutoff = int(np.ceil(np.sqrt(self.cutoff)))

        if self.gamma is None:
            self.gamma = 0.01

        k_cutoffSquare = (self.k_cutoff * (2 * np.pi / self.L)) ** 2

        kc = np.sqrt(k_cutoffSquare)
        k = np.sqrt(kSquare)
        result = np.divide(-4. * np.pi * (1. + special.erf((k - kc) \
                                                           / (kc * self.gamma))) / 2., kSquare ** 2, \
                           out=np.zeros_like(kSquare), \
                           where=kSquare > (kc * self.gamma) ** 2)
        return result

    def coulomb(self, kSquare, multiply_by_k_square=False):
        '''
        The G=0 terms need more consideration
        '''
        if self.gamma is None:
            gamma = 1.
        else:
            gamma = self.gamma
        result = np.divide(-4. * np.pi * gamma, kSquare, \
                           out=np.zeros_like(kSquare), where=kSquare > 1e-12)
        return result

    def stg(self, kSquare, multiply_by_k_square=False):
        if self.gamma is None:
            rho = self.n_ele / self.Omega
            gamma = np.sqrt(4. * np.pi * rho)
            # gamma = np.sqrt(4.*(3.*rho/np.pi)**(1/3.))
        else:
            gamma = self.gamma
        a = -4. * np.pi / gamma
        if self.k_cutoff is not None:
            k_cutoffSquare = (self.k_cutoff * (2 * np.pi / self.L)) ** 2
            k_cutoffDenom = (k_cutoffSquare + gamma ** 2) ** 2
        else:
            k_cutoffDenom = 1e-12
        if not multiply_by_k_square:
            b = (kSquare + gamma ** 2) ** 2
            result = np.divide(a, b, out=np.zeros_like(b), \
                               where=np.abs(b) > k_cutoffDenom)

        return result

    def yukawa_coulomb(self, kSquare, multiply_by_k_square=False):
        '''
        The G=0 terms need more consideration
        '''
        gamma_0 = 1.5
        if self.gamma is None:
            # gamma = np.sqrt(4.*(3.*rho/np.pi)**(1/3.))
            gamma = gamma_0
        else:
            gamma = self.gamma
        # A corresponds to 1/gamma**2 in Gruneis paper
        A = np.sqrt(self.Omega / (4.0 * np.pi * self.n_ele))
        A = 1. / A * gamma
        # has to be - and divided by gamm to satisfy the cusp condition
        a = -4. * np.pi
        if self.k_cutoff is not None:
            k_cutoffSquare = (self.k_cutoff * (2 * np.pi / self.L)) ** 2
            # k_cutoffDenom = k_cutoffSquare*(k_cutoffSquare + gamma**2)
            k_cutoffDenom = (k_cutoffSquare + A)
        else:
            k_cutoffDenom = 1e-12
        if not multiply_by_k_square:
            # b = kSquare*(kSquare+gamma**2)
            b = (kSquare + A) * kSquare
            result = np.divide(a, b, out=np.zeros_like(b), where=np.abs(b) > k_cutoffDenom)
        else:
            if kSquare > k_cutoffSquare:
                result = a / (kSquare + A)
            else:
                result = 0.

        return result
    
    def RPA(self, kSquare):
        '''
        J. Chem. Phys. 157, 074105 (2022); https://doi.org/10.1063/5.0101776
        '''
        rho = self.n_ele / self.Omega

        if self.k_cutoff is not None:
            k_cutoffSquare = (self.k_cutoff * (2 * np.pi / self.L)) ** 2
        else:
            k_cutoffSquare = 1e-12

        if self.gamma is None:
            self.gamma = 1.

        if not isinstance(kSquare, np.ndarray):
            kVec = np.sqrt(kSquare)
            kFermi = (3.0 * np.pi**2 * rho) ** (1.0 / 3.0)
            if kVec > (2*kFermi):
                T2 = 1.0
            elif kVec <= (2*kFermi):
                T2 = (3./4.)*(kVec/kFermi) - (1./16.)*(kVec/kFermi)**3
            a = 2. * rho * T2
            b = np.sqrt((kSquare ** 2) + 16. * np.pi * rho * (T2**2))
            k_cutoffDenom = k_cutoffSquare * a
            if np.abs(a) >= k_cutoffSquare:
                A = 1.0 / a
            else:
                A = 0.0
            if np.abs(a * kSquare) >= k_cutoffDenom:
                B = b / (kSquare * a)
            else:
                B = 0.0

            result = A - B
        else:
            kVec = np.sqrt(kSquare)
            kFermi = (3.0 * np.pi**2 * rho) ** (1.0 / 3.0)
            T2 = np.where(kVec > (2*kFermi), 
                          1.0,
                          (3./4.)*(kVec/kFermi) - (1./16.)*(kVec/kFermi)**3)
            a = 2. * rho * T2
            b = np.sqrt((kSquare ** 2) + 16. * np.pi * rho * (T2**2))
            k_cutoffDenom = k_cutoffSquare * a
            A = np.where(np.abs(a) >= k_cutoffSquare, 
                         1.0 / a, 
                         0.0)
            B = np.where(np.abs(a * kSquare) >= k_cutoffDenom,
                         b / (kSquare * a),
                         0.0)
            result = A - B
    
        return result * self.gamma

    def calcGamma(self, overlap_basis, nP):
        """
        Interface to CC4S, for test purpose only, not essential

        Parameters
        ----------
        overlap_basis: tuple
            of PlaneWaveFn objects. It serves the role of plane waves used for
            density fitting in real solids.
        nP: int
            number of spatial orbitals to consider.

        Returns
        -------
        gamma_pqG: nparray of dtype float
            of size nP*nP*nG, where nG is the length of overlap_basis.
            Meaning Fourier transformed overlap densities/pair densities (FTOD)
            $C^p_q({\bf G}) = \int\mathrm d{\bf r}
             \phi^*_p({\bf r}\phi_q({\bf r})e^{i{\bf G\cdot r}}$
        """

        if self.basis_fns == None:
            raise ValueError("Basis functions not initialized!")

        nG = int(len(overlap_basis) / 2)
        gamma_pqG = np.zeros((nP, nP, nG))

        for p in range(0, nP, 1):
            for q in range(0, nP, 1):
                for g in range(0, nG, 1):
                    if ((self.basis[2 * p].k - self.basis[2 * q].k) == overlap_basis[2 * g].k).all():
                        GSquare = overlap_basis[2 * g].kp.dot(overlap_basis[2 * g].kp)
                        if np.abs(GSquare) > 1e-12:
                            gamma_pqG[p, q, g] = np.sqrt(4. * np.pi / GSquare / self.Omega)
        return gamma_pqG

    def madelung(self, rs, nel):
        """
        Madelung Constant for the UEG: correction of the interaction of the electrons
        with themselves (periodic images).
        Notes:
            # there is still uncertainty in the factor of 2 divided ##
            # but using it seems to make the tc-dcd agree with BF-DMC

        Args:
            rs: float. Wigner-Seitz radius for controlling the density
            nel: int. Number of electrons.

        Returns:
            float
        """
        return -1.760118928190842*rs**(-1)*nel**(-1./3)/2
