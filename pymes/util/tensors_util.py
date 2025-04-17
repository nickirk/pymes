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