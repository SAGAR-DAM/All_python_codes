import subprocess
import numpy as np

def compute_eigenvalues(matrix):
    # Prepare input for C++ program
    input_data = f"{len(matrix)} {len(matrix[0])}\n"  # Rows and columns
    input_data += "\n".join([" ".join(map(str, row)) for row in matrix])

    # Run the C++ executable and capture output
    result = subprocess.run(["D:\\C++ codes\\c++ for python\\a.exe"], input=input_data.encode(), stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip()

    # Split output into coefficients
    coefficients = list(map(float, output.split()))

    print(f"coefficients: {coefficients}\n")
    # Use numpy to compute eigenvalues
    eigenvalues = np.roots(coefficients)

    return eigenvalues

if __name__ == "__main__":
    # Example usage
    input_matrix = np.random.uniform(low=-10, high=10, size=(2,2))
    input_matrix = np.floor(input_matrix)
    # input_matrix = [
    #     [4, 7, 2, 5, 4, 6],
    #     [3, 6, 1, 6, -8, -6],
    #     [2, 5, 1, -5, 7, 1],
    #     [7, -7, 5, 9, -2, -1],
    #     [2, 5, 4, 1, -5, 7],
    #     [5, 1, -5, 6, -8, -6]
    # ]
    
    
    print("Matrix:")
    print("=========")
    print(input_matrix)
    eigenvalues = compute_eigenvalues(input_matrix)
    print("""Eigenvalues:
====================""")
    for eig in eigenvalues:
        print(f"{eig}")
