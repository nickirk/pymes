from pymes.basis_set.planewave import is_closed_shell, count_spatial_states

def main():
    # Test cases
    test_cases = [2, 14, 38, 54, 66, 114, 162, 186, 246]
    results = {N: is_closed_shell(N) for N in test_cases}
    
    for N, result in results.items():
        print(f"N = {N}: {'Closed shell' if result else 'Not closed shell'}")

if __name__ == "__main__":
    main()
# This code defines a function to count the number of spatial states for a given sum of squares,
# and checks if a given number of electrons corresponds to a closed-shell configuration.
# The main function tests this functionality with a set of test cases.
# The results are printed to the console.
