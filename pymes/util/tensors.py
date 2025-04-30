import psutil

def calculate_block_size(total_elements_dimension, element_size, memory_fraction=0.5):
        """
        Calculate the block size for tensor slicing one dimension based on available memory:
        A tensor [nd, nd, nd, nd] has a total of nd^4 elements and each element has a size of
        element_size bytes. The function calculates the maximum number of elements that can be
        processed in one block without exceeding the available memory, considering a fraction of
        the available memory.
        That is [init:final,nd,nd,nd] for block sizes of |final-init|=block_size.

        Note:
            This function assumes that the tensor is a 4-index tensor with indices with the same 
            total number of elements.

        Args:
            total_elements (int): Total number of elements in the tensor along the specified dimension.
            element_size (int): Size of one element in bytes.
            memory_fraction (float): Fraction of available memory to use (default: 50%).

        Returns:
            int: Block size for slicing.
        """
        total_elements = int(total_elements_dimension**4)
        available_memory = psutil.virtual_memory().available
        usable_memory = available_memory * memory_fraction
        max_elements = usable_memory // ( element_size * total_elements_dimension**3)
        return min(total_elements_dimension, max_elements)

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