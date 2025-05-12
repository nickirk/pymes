import os

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
    
    idx_range      = idx[1] - idx[0]
    cpu_threads    = 10
    num_threads    = min(cpu_threads, idx_range)
    idx_block_size = int(idx_range / num_threads)
    idx_blocks     = [(start, min(start + idx_block_size, idx[1])) \
                        for start in range(idx[0], idx[1], idx_block_size)]
    return num_threads, idx_blocks