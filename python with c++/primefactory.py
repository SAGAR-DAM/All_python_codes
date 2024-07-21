import subprocess
import matplotlib.pyplot as plt
import matplotlib as mpl

# Setting matplotlib parameters
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.dpi'] = 200  # High-res display

def get_primes(n):
    try:
        result = subprocess.run(
            ["D:\\C++ codes\\c++ for python\\primefactory.exe"],  # Replace with the path to your compiled C++ executable
            input=f"{n}\n".encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.stderr:
            raise RuntimeError(result.stderr.decode('utf-8'))
        
        output = result.stdout.decode('utf-8').strip()
        primes = list(map(int, output.split()))
        return primes

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def main():
    n = 10000  # Number of primes you want to find

    primes = get_primes(n)
    indices = list(range(1, n + 1))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(indices, primes, marker='o', linestyle='-', color='b')
    plt.title(f"Index vs. Prime Number (up to the {n}th prime)")
    plt.xlabel("Index (i)")
    plt.ylabel("Prime Number")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
