import os
import sys
import psutil
import multiprocessing as mp

from math import ceil
from pymes.log import print_logging_info


def get_process_index_block( idx ):
    """
    Function to partition a one-dimensional array of indices idx[0]
    to idx[1] into blocks for each thread.

    Parameters
    ----------
    idx : tuple of two elements
        List of indices to be partitioned.
    Returns
    -------
    num_process : int
        The number of process used for partitioning.
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
    num_process      = min(cpu_threads, idx_range)
    #print(f"Number of cpu threads: {cpu_threads}")
    idx_block_size = ceil( idx_range / num_process )
    #print(f"Index block size: {idx_block_size}")
    idx_blocks     = [(start, min(start + idx_block_size, idx[1])) \
                        for start in range(idx[0], idx[1], idx_block_size)]
    return num_process, idx_blocks

def get_obj_tot_size(obj, seen=None):

    """
    Function to calculate the total memory size of an object, including its attributes.
    This function is recursive and handles various data types, including dictionaries,
    lists, and custom objects.
    Used to check the size of the object in memory when using the multiprocessing module.

    Parameters
    ----------
    obj : object
        The object whose size is to be calculated.
    seen : set, optional
        A set to keep track of already seen objects to avoid infinite recursion.
    Returns
    -------
    size : int
        The total memory size of the object in bytes.
    """

    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(get_obj_tot_size(v, seen) for v in obj.values())
        size += sum(get_obj_tot_size(k, seen) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += get_obj_tot_size(vars(obj), seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_obj_tot_size(i, seen) for i in obj)
    
    return size

def process_info():
    """
    Function to print the processor information from the multiprocessing module.
    """
    
    print_logging_info(f"Module  name : {mp.__name__}", level=3)
    print_logging_info(f"Process name : {mp.current_process().name}", level=3)
    print_logging_info(f"Parent proc. ID : {os.getppid()}", level=3)
    print_logging_info(f"Curr.  proc. ID : {mp.current_process().pid}", level=3)
    print_logging_info(f"Available memory : {psutil.virtual_memory().available / (1024**3):.2f} GB", level=3)