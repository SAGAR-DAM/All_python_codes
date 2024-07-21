import math
import matplotlib.pyplot as plt


import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi'] = 200  # highres display

# Function to check if a number is prime
def is_prime(num):
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False

    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6

    return True

# Function to find the nth prime
def nth_prime(n):
    if n < 1:
        print("n must be a positive integer.")
        return -1

    count = 0
    num = 1

    while count < n:
        num += 1
        if is_prime(num):
            count += 1

    return num

def main():
    # n = int(input("Enter the value of n: "))
    # print("\n\n")

    n = 6000
    
    primes = []
    indices = []

    for j in range(1, n + 1):
        result = nth_prime(j)
        if result != -1:
            # print(f"The {j}th prime number is {result}")
            indices.append(j)
            primes.append(result)

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
