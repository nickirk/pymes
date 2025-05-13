import os
from math import ceil

def get_thread_index_block( idx ):
    """
    Function to partition a one-dimensional array of indices idx[0]
    to idx[1] into blocks for each thread.

    Parameters
    ----------
    idx : tuple of two elements
        List of indices to be partitioned.
    Returns
    -------
    num_threads : int
        The number of threads used for partitioning.
    idx_blocks : list of tuples
        List of tuples, where each tuple contains the start and end indices for each block.
    """
    
    if len(idx) != 2:
        raise ValueError("The input idx must be a tuple of two elements.")
    if idx[0] >= idx[1]:
        raise ValueError("The first element of idx must be less than the second element.")
    if idx[0] < 0 or idx[1] < 0:
        raise ValueError("The elements of idx must be non-negative integers.")
    if idx[0] == idx[1]:
        return [(idx[0], idx[1])]
    
    import os

    # Get the value of the NUM_THREADS environment variable.
    cpu_threads = os.getenv('NUM_THREADS')
    
    # Check if the variable is set, and if not,
    # set it to the number of available CPU cores + 4
    # as default concurrent.futures 3.13
    if cpu_threads is None:
        cpu_threads = (os.cpu_count() or 1) - 4
        if cpu_threads < 1:
            cpu_threads = 1
    
    idx_range      = idx[1] - idx[0]
    num_threads    = min(cpu_threads, idx_range)
    print(f"Number of cpu threads: {cpu_threads}")
    num_threads    = 10
    idx_block_size = ceil( idx_range / num_threads )
    print(f"Index block size: {idx_block_size}")
    idx_blocks     = [(start, min(start + idx_block_size, idx[1])) \
                        for start in range(idx[0], idx[1], idx_block_size)]
    return num_threads, idx_blocks