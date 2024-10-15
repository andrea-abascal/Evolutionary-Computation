
import matplotlib.pyplot as plt
import json
import numpy as np

# Load the first JSON file
with open('/home/andrea/MCC/Repos/Evolutionary Computation/MO_Problems/10/practice_10.json', 'r') as f1:
    de = json.load(f1)

with open('/home/andrea/MCC/Repos/Evolutionary Computation/MO_Problems/10/real_encoding.json', 'r') as f2:
    ga = json.load(f2)

folder = '/home/andrea/MCC/Repos/Evolutionary Computation/MO_Problems/10'

# Extract the lists from both files
f_1_de = de['test_case1_fitness']
f_2_de = de['test_case2_fitness']
p_1_de = de['test_case1_path']
p_2_de = de['test_case2_path']
t_1_de = de['test_case1_time']
t_2_de = de['test_case2_time']
s_1_de = de['test_case1_sol']
s_2_de = de['test_case2_sol']

f_1_ga = ga['test_case1_fitness']
f_2_ga = ga['test_case2_fitness']
p_1_ga = ga['test_case1_path']
p_2_ga = ga['test_case2_path']
t_1_ga = ga['test_case1_time']
t_2_ga = ga['test_case2_time']
s_1_ga = ga['test_case1_sol']
s_2_ga = ga['test_case2_sol']

# Plot the fitness values for DE and GA
plt.plot(f_1_de, label='DE - Rosenbrock Function Fitness', color='blue')
plt.plot(f_1_ga, label='GA - Rosenbrock Function Fitness', color='green')

# Add titles and labels
plt.title('Fitness Comparison: Rosenbrock Function(DE vs GA)')
plt.suptitle(f'DE time: {t_1_de} vs GA time: {t_1_ga}')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.legend()
# Display the plot
plt.savefig(f'{folder}/rosenbrock_fitness.png')
plt.show()


# Plot the fitness values for DE and GA
plt.plot(f_2_de, label='DE - Rastrigin Function Fitness', color='blue')
plt.plot(f_2_ga, label='GA - Rastrigin Function Fitness', color='green')

# Add titles and labels
plt.title('Fitness Comparison: Rastrigin Function 2 (DE vs GA)')
plt.suptitle(f'DE time: {t_2_de} vs GA time: {t_2_ga}')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.legend()
# Display the plot
plt.savefig(f'{folder}/rastrigin_fitness.png')
plt.show()

plt.close()


# Test cases
tests = [{
        "objective_function": lambda x: 100 * (x[0]**2 - x[1])**2 + (1 -x[0])**2,
        "bounds": [(-2.048, 2.048), (-2.048, 2.048)],
    },
    {
        "objective_function": lambda x: 20 + sum([(x[i]**2 - 10 * np.cos(2 * np.pi * x[i])) for i in range(2)]),
        "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
    }
]

def plot_contour(f, solution_de, solution_ga, bounds, path_de, path_ga, filename,name):
    """Plot contour of the function with paths for DE and GA."""
    x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Compute Z values for contour plot
    Z = np.array([[f([x1, x2]) for x1 in X1[i]] for i, x2 in enumerate(x2)])
    
    # Create contour plot
    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, Z, levels=np.linspace(np.min(Z), np.max(Z), 30), cmap='viridis')
    plt.colorbar(label='Objective function value')
    
    # Plot the paths of iterations for DE and GA
    path_de = np.array(path_de)
    path_ga = np.array(path_ga)
    
    # GA Path (in green)
    plt.plot(path_ga[:, 0], path_ga[:, 1], 'g-o', label='GA Path', markersize=4)
    plt.scatter(solution_ga[0], solution_ga[1], color='purple', s=100, zorder=5, label=f'GA Solution at {solution_ga}')

    # DE Path (in red)
    plt.plot(path_de[:, 0], path_de[:, 1], 'r-o', label='DE Path', markersize=4)
    plt.scatter(solution_de[0], solution_de[1], color='blue', s=100, zorder=5, label=f'DE Solution at {solution_de}')
    
    # Add labels, title, and legend
    plt.title(f'Contour Plot Comparison (DE vs GA) for {name} function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.savefig(filename)
    plt.show()
    # Save the plot as a PNG file
    
    plt.close()
    

# Loop through test cases and plot
for i, test in enumerate(tests):
    objective_function = test["objective_function"]
    bounds = np.array(test["bounds"])

    # Define paths and solutions for DE and GA for each test case
    if i == 0:  # Test case 1
        path_de = p_1_de
        path_ga = p_1_ga
        solution_de = s_1_de
        solution_ga = s_1_ga
        name = 'Rosenbrock'
    else:  # Test case 2
        path_de = p_2_de
        path_ga = p_2_ga
        solution_de = s_2_de
        solution_ga = s_2_ga
        name = 'Rastrigin'

    # Call the plot_contour function
    filename = f'{folder}/test_case_{i+1}_comparison.png'
    plot_contour(objective_function, solution_de, solution_ga, bounds, path_de, path_ga, filename, name)



