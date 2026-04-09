import psutil
from math import ceil

def calculate_block_size(idx, dim, element_size, memory_fraction=0.5, is_shared_memory=False):
        """
        Calculate the block size for tensor slicing one dimension based on available memory:
        A tensor [n0, n1, n2, n3] has a total of n0*n1*n2*n3 elements and each element has a size of
        element_size bytes. The function calculates the maximum number of elements that can be
        processed in one block without exceeding the available memory, considering a fraction of
        the available memory.
        That is [init:final,nd,nd,nd] for block sizes of |final-init|=block_size.

        Note:
            This function assumes that the tensor is a 4-index tensor with indices with the same 
            total number of elements.

        Args:
            idx (int):
                The dimension along which to calculate the block size.
            dim (tuple): 
                The shape of the tensor as a tuple (n0, n1, n2, n3).
            element_size (int): 
                Size of one element in bytes.
            memory_fraction (float): 
                Fraction of available memory to use (default: 50%).
            is_shared_memory (bool):
                Whether the memory is shared (default: False). If True, it reduces the usable memory
                to account for other processes using the shared memory.

        Returns:
            int: Block size for slicing.
        """
        other_dims_product = dim[0] * dim[1] * dim[2] * dim[3] // dim[idx]
        available_memory = psutil.virtual_memory().available
        usable_memory = available_memory * memory_fraction
        if is_shared_memory:
            usable_memory *= 0.45
        max_elements = ceil( usable_memory / ( element_size * other_dims_product))
        return min(dim[idx], max_elements)

def get_block_index( block_string, n_p, n_occ):
    """
    Calculates the corresponding index tuple for the corresponding block of the tensor:

    'full' = [:, :, :, :]
    'oooo' = [:no, :no, :no, :no]
    'ovvo' = [:no, no:, no:, :no]
    'voov' = [no:, :no, :no, no:]
    'oovv' = [:no, :no, no:, no:]
    'vvoo' = [no:, no:, :no, :no]
    'vovo' = [no:, :no, no:, :no]
    'ovov' = [:no, no:, :no, no:]
    'vvvv' = [no:, no:, no:, no:]

    Args:
        block_string (character): string for the 'XXXX' block.
        n_p (int): total number of spin orbitals.
        n_occ (int): number of occupied spin oritals.
    
    Returns:
        idx tuple containing the indices of each block as:
            {idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7]}
    """
    
    nP = int(n_p)
    no = int(n_occ)
    
    block = block_string.upper()
    
    if block == 'FULL':
          idx = tuple((0,nP,0,nP,0,nP,0,nP))
    elif block == 'OOOO':
          idx = tuple((0,no,0,no,0,no,0,no))
    elif block == 'OVVO':
          idx = tuple((0,no,no,nP,no,nP,0,no))
    elif block == 'VOOV':
          idx = tuple((no,nP,0,no,0,no,no,nP))
    elif block == 'OOVV':
          idx = tuple((0,no,0,no,no,nP,no,nP))
    elif block == 'VVOO':
          idx = tuple((no,nP,no,nP,0,no,0,no))
    elif block == 'VOVO':
          idx = tuple((no,nP,0,no,no,nP,0,no))
    elif block == 'OVOV':
          idx = tuple((0,no,no,nP,0,no,no,nP))
    elif block == 'VVVV':
          idx = tuple((no,nP,no,nP,no,nP,no,nP))
    else:
          raise ValueError("Tensor block not valid!")
      
    return idx

def write_one_index_tensor(tensor, filename):
      """
      Write a one-index tensor to a file in a human-readable format.
      """
      with open(filename, 'w') as f:
          if tensor is None:
              f.write("None\n")
          else:
              for i in range(tensor.shape[0]):
                  f.write(f"{i} {tensor[i]}\n")

def write_two_index_tensor(tensor, filename):
    """
    Write a two-index tensor to a file in a human-readable format.
    """
    with open(filename, 'w') as f:
        if tensor is None:
            f.write("None\n")
        else:
            for i in range(tensor.shape[0]):
                  for j in range(tensor.shape[1]):
                        f.write(f"{i} {j} {tensor[i, j]}\n")

def write_four_index_tensor(tensor, filename):
    """
    Write a four-index tensor to a file in a human-readable format.
    """
    with open(filename, 'w') as f:
        if tensor is None:
            f.write("None\n")
        else:
            for i in range(tensor.shape[0]):
                  for j in range(tensor.shape[1]):
                        for k in range(tensor.shape[2]):
                              for l in range(tensor.shape[3]):
                                    f.write(f"{i} {j} {k} {l} {tensor[i, j, k, l]}\n")