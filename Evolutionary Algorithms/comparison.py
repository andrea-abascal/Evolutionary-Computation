
import matplotlib.pyplot as plt
import json

# Load the first JSON file
with open('/home/andrea/MCC/Repos/Evolutionary Computation/GA_results/binary_encoding.json', 'r') as f1:
    binary = json.load(f1)

# Load the second JSON file
with open('/home/andrea/MCC/Repos/Evolutionary Computation/GA_results/real_encoding.json', 'r') as f2:
    real = json.load(f2)

# Extract the lists from both files
b_1 = binary['test_case1']
b_2 = binary['test_case2']
b_3 = binary['test_case3']

r_1 = real['test_case1']
r_2 = real['test_case2']
r_3 = real['test_case3']

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Plot the first lists from both files in the first subplot
axes[0].plot(b_1, label='binary', color="blue")
axes[0].plot(r_1, label='real', color="red")
axes[0].set_title('Rosenbrock Function')
axes[0].legend()

# Plot the second lists from both files in the second subplot
axes[1].plot(b_2, label='binary', color="blue")
axes[1].plot(r_2, label='real', color="red")
axes[1].set_title('Rastringin Function n = 2')
axes[1].legend()

# Plot the third lists from both files in the third subplot
axes[2].plot(b_3, label='binary', color="blue")
axes[2].plot(r_3, label='real', color="red")
axes[2].set_title('Rastringin Function n = 5')
axes[2].legend()

# Display the plots
plt.tight_layout()
plt.savefig('genetic_algorithm_results.png')
plt.show()
