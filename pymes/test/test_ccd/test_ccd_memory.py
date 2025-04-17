import numpy as np
import psutil
from pymes.solver.ccd import CCD
from pymes.util.tensors_util import calculate_block_size

def test_calculate_block_size():
    """
    Test the calculate_block_size function to ensure it calculates block sizes correctly.
    """
    # Mock parameters
    nv = 20  # Total elements along one dimension
    element_size = 8  # Assuming double precision (8 bytes per element)
    memory_fraction = 0.5
    total_elements_dimension = nv  # Total elements along one dimension

    available_memory = psutil.virtual_memory().available
    usable_memory = available_memory * memory_fraction
    max_elements_per_block = usable_memory // (element_size * nv**3)
    expected_block_size = min(total_elements_dimension, max_elements_per_block)

    # Call the function directly from tensors_util
    calculated_block_size = calculate_block_size(total_elements_dimension, element_size, memory_fraction)

    # Print internal variables for debugging
    print(f"Available memory: {available_memory} bytes")
    print(f"Memory fraction: {memory_fraction}")
    print(f"Usable memory: {usable_memory} bytes")
    print(f"Total elements along one dimension: {total_elements_dimension}")
    print(f"Calculated block size: {calculated_block_size}")
    print(f"Expected block size: {expected_block_size}")
    print(f"Number of blocks: {int(np.ceil(total_elements_dimension / calculated_block_size))}")

    assert calculated_block_size == expected_block_size, \
        f"Expected {expected_block_size}, got {calculated_block_size}"

def test_tensor_block_processing():
    """
    Test tensor slicing and block processing to ensure correctness.
    """
    # Mock tensor dimensions
    nv, no = 20, 10
    t_T_abij = np.random.rand(nv, nv, no, no)
    t_V_abcd = np.random.rand(nv, nv, nv, nv)

    # Mock eri object with a get_vvvv method
    class MockERI:
        def get_vvvv(self, indx):
            start, end, *_ = indx
            return t_V_abcd[start:end]

    eri = MockERI()

    # Instantiate CCD
    ccd = CCD(no=no)

    # Process tensor in blocks
    t_R_abij = np.zeros_like(t_T_abij)
    element_size = t_T_abij.dtype.itemsize
    total_elements_dimension = nv
    block_size = calculate_block_size(total_elements_dimension, element_size)

    # Print internal variables for debugging
    print(f"Tensor element size: {element_size} bytes")
    print(f"Total elements along one dimension: {total_elements_dimension}")
    print(f"Calculated block size: {block_size}")
    print(f"Number of blocks: {int(np.ceil(total_elements_dimension / block_size))}")

    for block_start in range(0, nv, block_size):
        block_end = min(block_start + block_size, nv)
        t_V_xbcd = eri.get_vvvv((block_start, block_end, 0, nv, 0, nv, 0, nv))
        t_R_xbij = np.einsum("xbcd, cdij -> xbij", t_V_xbcd, t_T_abij)
        t_R_abij[block_start:block_end, :, :, :] += t_R_xbij

    # Validate results
    expected_t_R_abij = np.einsum("abcd, cdij -> abij", t_V_abcd, t_T_abij)
    assert np.allclose(t_R_abij, expected_t_R_abij), "Block processing failed!"

if __name__ == "__main__":
    print("Testing block size calculation...")
    test_calculate_block_size()
    print("Block size calculation test passed!")
    print("Testing tensor block processing...")
    test_tensor_block_processing()
    print("Tensor block processing test passed!")
    # If all tests pass, print a success message.
    print("All tests passed!")
