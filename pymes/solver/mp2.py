import gc
import sys
import time
import numpy as np
import pytblis as pytblis
from functools import partial
from pymes.log import print_logging_info
from pymes.util import tensors
from pymes.util.parallel_tasks import get_memory_usage

einsum = partial(pytblis.einsum, optimize='greedy')

class MP2:

    def __init__(self, no, **kwargs):
        self.no = no

    def solve(self, eri, level_shift=0., sp=0, nv_part_size=None, **kwargs):
        """ Member function of MP2 class. Wrapper function for MP2 solver
        assuming a closed-shell reference and diagonal Fock matrix.
        Selects the appropiate MP2 algorithm based on the sparsity of the input tensors.
        Parameters:
            eri: ERI class object. 
                    The ERI object containing the necessary tensors for MP2 calculation.
            level_shift: float.
                    The level shift to be added to the energy denominator in MP2 amplitude calculation, to avoid divergence.
            sp: 0 or 1.
                    The sparsity of the input tensors. 0 for dense tensors, 1 for sparse tensors.
            nv_part_size: integer.
                    The partition size of the virtual index in calculating the MP2 energies, to save memory.
                    The default value is None, which means no partition is used. It will be set to nv in the algorithm.
            kwargs: other keyword arguments for MP2 solver, e.g. debug_level for logging.
        Returns:
            e_mp2: float.
                    The calculated MP2 correlation energy.
            T_abij: np array.
                    The calculated MP2 amplitudes.
        """
        algo_name = "mp2.solve"
        print_logging_info(algo_name, level=0)
        mode = eri.mode
        print_logging_info("Using ERI mode: ", mode, level=1)
        if mode == 'incore' or mode == 'semi-incore':
            if sp == 0:
                print_logging_info("Using incore-dense MP2 algorithm", level=1)
                e_mp2, T_abij = self._solve_incore_dense(eri, level_shift=level_shift, **kwargs)
            elif sp == 1:
                print_logging_info("Using incore-sparse MP2 algorithm", level=1)
                e_mp2, T_abij = self._solve_incore_sp(eri, level_shift=level_shift, sp=sp, nv_part_size=nv_part_size, **kwargs)
            else:
                raise ValueError(algo_name, "Invalid value for sp: ", sp)
        elif mode == 'on-the-fly':
            print_logging_info("Using on-the-fly MP2 algorithm", level=1)
            e_mp2, T_abij = self._solve_on_the_fly(eri, level_shift=level_shift, **kwargs)
        return {"mp2 e": e_mp2, "t2 amp": T_abij}

    def _solve_incore_dense(self, eri, level_shift=0., **kwargs):
        """
        dense mp2 algorithm
        Note that t_V_ijab and t_V_abij are not necessarily
        the same, e.g. in transcorrelated Hamiltonian.
        -------------
        Parameters:
            eri: ERI class object. 
                The ERI object containing the necessary tensors for MP2 calculation.
            level_shift: float.
                The level shift to be added to the energy denominator in MP2 amplitude calculation, to avoid divergence.
        kwargs: other keyword arguments for MP2 solver, e.g. debug_level for logging.

        """
        algo_name = "mp2._solve_incore_dense"
        print_logging_info(algo_name, level=1)

        t_epsilon_i = eri.eps_occ
        t_epsilon_a = eri.eps_virt
        t_V_ijab = eri.get_pqrs('oovv')
        t_V_abij = eri.get_pqrs('vvoo')

        start_time = time.time()
        t_T_abij = t_V_abij.copy()
        t_D_abij = t_epsilon_i[None, None, :, None] + t_epsilon_i[None, None, None, :] - t_epsilon_a[:, None, None, None] - t_epsilon_a[None, :, None, None]
        t_T_abij /= (t_D_abij + level_shift)
        e_dir_mp2 = 2.0*einsum('abij, ijab->',t_T_abij, t_V_ijab)
        e_exc_mp2 = -1.0*einsum('abij, jiab->',t_T_abij, t_V_ijab)
        e_total_mp2 = e_dir_mp2 + e_exc_mp2
        end_time = time.time()

        print_logging_info("Direct contribution = {:.12f}".format(np.real(e_dir_mp2)),\
                        level=2)
        print_logging_info("Exchange contribution = {:.12f}".format(np.real(e_exc_mp2)),\
                        level=2)
        print_logging_info("MP2 correlation energy = {:.12f}".format(np.real(e_total_mp2)), level=2)
        print_logging_info("{:.3f} seconds spent on MP2".format((end_time-start_time)), level=2)
        
        return [e_total_mp2, t_T_abij]

    def _solve_incore_sp(self, eri, level_shift=0., sp=0, nv_part_size=None, **kwargs):
        """
        sparse mp2 algorithm: not debugged yet
        Note that t_V_ijab and t_V_abij are not necessarily the same, e.g. in transcorrelated Hamiltonian.
        -------------
        Parameters:
            eri: ERI class object.
                The ERI object containing the necessary tensors for MP2 calculation, and the tensors are in sparse format.
            level_shift: float.
                The level shift to be added to the energy denominator in MP2 amplitude calculation, to avoid divergence.
            sp: 0 or 1. 
                Sparsity of np arrays
            nv_part_size: integer. 
                The partition size of the virtual index in calculating the MP2 energies, to save memory.
                The default value is 0, which means no partition is used. It will be set to nv in the algorithm.
        """

        algo_name = "mp2._solve_incore_sp"
        start_time = time.time()
        print_logging_info(algo_name,level=1)

        if "debug_level" in kwargs:
            debug_level = kwargs["debug_level"]
        else:
            debug_level = 3

        t_epsilon_i = eri.eps_occ
        t_epsilon_a = eri.eps_virt
        t_V_ijab = eri.get_pqrs('oovv')
        t_V_abij = eri.get_pqrs('vvoo')

        no = t_epsilon_i.size
        nv = t_epsilon_a.size

        # the following ctf expression calcs the outer sum, as wanted.
        print_logging_info("Creating D_abij", level = 2, debug_level=debug_level)

        # memory efficient implementation for sparse V_abij, validity for dense still need to be tested.
        print_logging_info("Calculating T_abij", level = 2, debug_level=debug_level)
        inds, vals = t_V_abij.read_local_nnz()
        del t_V_abij
        epsilon_i = t_epsilon_i.to_nparray()
        epsilon_a = t_epsilon_a.to_nparray()

        print_logging_info("Looping through nnz in t_V_abij", level = 2)
        print_logging_info("Total nnz entries on rank 0 = ", len(inds), level = 2)

        for ind in range(len(inds)):
            if ind % num_proc == 0:
                print_logging_info("Completed {:.2f} percent...".format(ind/len(inds)*100), level=3, debug_level=debug_level)
            global_ind = inds[ind]
            [a, b, i, j] = self.get_orb_inds(global_ind, [nv, nv, no, no])
            vals[ind] /= (epsilon_i[i] + epsilon_i[j] - epsilon_a[a] - epsilon_a[b] + level_shift)

        del epsilon_i, t_epsilon_i
        del epsilon_a, t_epsilon_a

        t_T_abij = np.zeros([nv,nv,no,no], dtype=t_V_ijab.dtype)
        t_T_abij.write(inds, vals)

        if nv_part_size is None:
            n_part = 1
            nv_part_size = nv
        else:
            n_part = int(nv//nv_part_size) + 1
        e_dir_mp2 = 0.
        e_exc_mp2 = 0.
        print_logging_info("Summing direct and exchange contributions", level=2)
        print_logging_info("Partitioning T and V tensors", level=2)
        for n in range(n_part):
            n_lower = n * nv_part_size
            if n_lower >= nv:
                break
            n_higher = (n + 1) * nv_part_size
            if n_higher > nv:
                n_higher = nv
            print_logging_info("n_lower = ", n_lower, ", n_higher = ", n_higher, level = 3)
            t_T_nmij = t_T_abij[n_lower:n_higher, :, :, :]
            t_V_ijnm = t_V_ijab[:, :, n_lower:n_higher, :]

            e_dir_mp2 += 2.0*einsum('abij, ijab->',t_T_nmij, t_V_ijnm)
            e_exc_mp2 += -1.0*einsum('abij, jiab->',t_T_nmij, t_V_ijnm)
            e_total_mp2 = e_dir_mp2 + e_exc_mp2
        end_time = time.time()

        print_logging_info("Direct contribution = {:.12f}".format(np.real(e_dir_mp2)),\
                        level=2)
        print_logging_info("Exchange contribution = {:.12f}".format(np.real(e_exc_mp2)),\
                        level=2)
        print_logging_info("MP2 correlation energy = {:.12f}".format(np.real(e_total_mp2)), level=2)
        print_logging_info("{:.3f} seconds spent on MP2".format((end_time-start_time)), level=2)
        
        return [e_total_mp2, t_T_abij]

    def get_orb_inds(self, global_ind, dims):
        """
        Args:
            global_ind: int, the global index of an entry on a np array
            dims: list of ints, the dimensions of the np array

        Returns:
            inds: list of ints, the corresponding indices of the entry on the tensor
        """

        inds = []
        for i in range(len(dims)):
            ind = global_ind // np.prod(dims[i+1:])
            global_ind -= ind * np.prod(dims[i+1:])
            inds.append(int(ind))
        return inds
    
    def _solve_on_the_fly(self, eri, level_shift=0., **kwargs):
        """
        on-the-fly mp2 algorithm
        Note that eri should be able to calculate the necessary ERI blocks on-the-fly, e.g. oovv and vvoo for MP2.
        -------------
        Parameters:
            eri: ERI class object. 
                The ERI object containing the necessary tensors for MP2 calculation, and able to calculate them on-the-fly.
            level_shift: float.
                The level shift to be added to the energy denominator in MP2 amplitude calculation, to avoid divergence.
        """

        algo_name = "mp2.solve_on_the_fly"
        start_time = time.time()
        print_logging_info(algo_name,level=1)
        e_dir_mp2 = 0.
        e_exc_mp2 = 0.
        t_T_abij = None


        t_epsilon_i = eri.eps_occ
        t_epsilon_a = eri.eps_virt

        no = t_epsilon_i.size
        nv = t_epsilon_a.size

        # Calculate block size for memory management
        element_size = t_epsilon_i.dtype.itemsize  # Size of one element in bytes
        block_size = tensors.calculate_block_size(0, tuple((nv, nv, no, no)), element_size,
                                                  memory_fraction=0.75, is_shared_memory=False)
        if block_size != nv and block_size > 1:
            block_size = int(block_size/2)

        print_logging_info("Memory usage at start: {:.2f} GB".format(get_memory_usage()), level=3)
        print_logging_info("Using block size of {} for contration.".format(block_size), level=3)
        print_logging_info("Memory per block: {:.2f} GB".format(
            block_size * nv * no * no * element_size / (1024 ** 3)), level=3)
        
        for block_start in range(0, nv, block_size):
            start_time_block = time.time()
            block_end = min(block_start + block_size, nv)
            print_logging_info(" Calculating blocks [vvoo] [oovv]: {} to {}.".format(block_start, block_end), level=3)
            start_time_calc_block = time.time()
            indx = tuple((block_start, block_end, 0, nv, 0, no, 0, no))
            t_T_xbij = eri.get_pqrs('vvoo', idx=indx)
            indx= tuple((0, no, 0, no, block_start, block_end, 0, nv))
            t_V_ijxb = eri.get_pqrs('oovv', idx=indx)
            end_time_calc_block = time.time()
            print_logging_info(" Elapsed [vvoo] [oovv] integral time: {:.3f} seconds.".format(end_time_calc_block - start_time_calc_block), level=3)
            print_logging_info(" Memory after loading [vvoo] [oovv] blocks: {:.2f} GB".format(get_memory_usage()), level=3)
            start_time_contr_block = time.time()
            t_T_xbij /= (t_epsilon_i[None, None, :, None] + t_epsilon_i[None, None, None, :] - t_epsilon_a[block_start:block_end, None, None, None] - t_epsilon_a[None, :, None, None] + level_shift)
            e_dir_mp2 += 2.0*einsum('abij, ijab->',t_T_xbij, t_V_ijxb)
            e_exc_mp2 += -1.0*einsum('abij, jiab->',t_T_xbij, t_V_ijxb)
            end_time_contr_block = time.time()
            print_logging_info(" Elapsed [vvoo] [oovv] contraction time: {:.3f} seconds.".format(end_time_contr_block - start_time_contr_block), level=3)
            del t_V_ijxb, t_T_xbij
            end_time_block = time.time()
            print_logging_info(" Elapsed [vvoo] [oovv] block time: {:.3f} seconds.".format(end_time_block - start_time_block), level=3)
            print_logging_info(" Memory after block cleanup: {:.2f} GB".format(get_memory_usage()), level=3)
            gc.collect()
            sys.stdout.flush()

        print_logging_info("Memory usage after all blocks: {:.2f} GB".format(get_memory_usage()), level=3)   

        e_total_mp2 = e_dir_mp2 + e_exc_mp2
        end_time = time.time()
        print_logging_info("Direct contribution = {:.12f}".format(np.real(e_dir_mp2)),\
                        level=2)
        print_logging_info("Exchange contribution = {:.12f}".format(np.real(e_exc_mp2)),\
                        level=2)
        print_logging_info("MP2 correlation energy = {:.12f}".format(np.real(e_total_mp2)), level=2)
        print_logging_info("{:.3f} seconds spent on MP2".format((end_time-start_time)), level=2)
        
        return [e_total_mp2, t_T_abij]