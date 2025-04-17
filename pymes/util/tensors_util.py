import psutil

def calculate_block_size(total_elements, element_size, memory_fraction=0.5):
        """
        Calculate the block size for tensor slicing based on available memory.

        Args:
            total_elements (int): Total number of elements in the tensor.
            element_size (int): Size of one element in bytes.
            memory_fraction (float): Fraction of available memory to use (default: 50%).

        Returns:
            int: Block size for slicing.
        """
        available_memory = psutil.virtual_memory().available
        usable_memory = available_memory * memory_fraction
        max_elements = usable_memory // element_size
        return min(total_elements, max_elements)