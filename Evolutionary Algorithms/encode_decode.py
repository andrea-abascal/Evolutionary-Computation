import numpy as np

# Bits size
def bits_size(bounds,precision):
    b = bounds[0]
    num_bits = int(np.log2((b[1]-b[0])*(10**precision))+0.99)
    return num_bits

# Binary to real-value decoding
def decoding(bounds, bits, chromosome):
    real_chromosome = list()
    for i in range(len(bounds)):
        start = i * bits
        end = (i+1) * bits
        substring = chromosome[start:end]
        # Convert binary to decimal
        decimal_value = int(''.join(map(str, substring)), 2)
        # Scale decimal to real range
        min_bound, max_bound = bounds[i]
        real_value = min_bound + (decimal_value / (2**bits - 1)) * (max_bound - min_bound)
        real_chromosome.append(real_value)
    return real_chromosome
a = bits_size([[-2.5,2.5]],3)

print(a)
