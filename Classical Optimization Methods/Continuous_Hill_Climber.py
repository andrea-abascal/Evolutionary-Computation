import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def hill_climber(objective_function, starting_point, step_size=1.0, max_iterations=1000):
    """Hill climber algorithm for continuous optimization."""
    current_point = np.array(starting_point)
    current_value = objective_function(current_point)
    path = [current_point.copy()]  # Store the path of iterations

    for i in range(max_iterations):
        neighbors = [current_point + step_size * direction 
                     for direction in [np.array([1, 0]), np.array([-1, 0]), 
                                       np.array([0, 1]), np.array([0, -1])]]
        
        neighbor_values = [objective_function(neighbor) for neighbor in neighbors]
        min_value_index = np.argmin(neighbor_values)
        min_value = neighbor_values[min_value_index]

        if min_value >= current_value:
            break

        current_point = neighbors[min_value_index]
        current_value = min_value
        path.append(current_point.copy())  # Store the new point

    return current_point, current_value, i, path

def error(optimal_val, obtained_val):
    return np.linalg.norm(optimal_val-obtained_val)

def plot_contour(f, solution, bounds, path, filename, step_size):
    """Plot contour of the function and the solution path."""
    x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.array([[f([x1, x2]) for x1 in X1[i]] for i, x2 in enumerate(x2)])

    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Objective function value')

    # Plot the path of iterations
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'r-o', label='Hill Climber Path')

    plt.title(f'Contour Plot with Iteration Path with a Step size of {step_size}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    # Save the plot as a PNG file
    plt.savefig(filename)
    plt.show()

# Test cases
"""
tests = [
    {
        "objective_function": lambda x: -1 * (-2 * x[0]**2 + 3 * x[0] * x[1] - 1.5 * x[1]**2 - 1.3),
        "starting_point": [-4, 4],
        "bounds": [(-6, 6), (-6, 6)],
        "step_size": 0.01,
        "filename": "hill_climber_A.png"
    },
    {
        "objective_function": lambda x: (4 - 2.1 * x[0]**2 + (x[0]**4)/3) * x[0]**2 + x[0] * x[1] + (-4 + 4 * x[1]**2) * x[1]**2,
        "starting_point": [0.5, 1],
        "bounds": [(-3, 3), (-2, 2)],
        "step_size": 0.01,
        "filename": "hill_climber_B.png"
    },
    {
        "objective_function": lambda x: 20 + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])),
        "starting_point": [-2, 2],
        "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
        "step_size": 0.99,
        "filename": "hill_climber_C.png"
    }
]
"""
tests = [
    {   "objective_function": lambda x: 100 * (x[0]**2 - x[1])**2 + (1 -x[0])**2,
        "bounds": [(-2.048, 2.048), (-2.048, 2.048)],
        "starting_point": [0.5, 1],
        "step_size": 0.1,
        "filename": "/home/andrea/MCC/Repos/Evolutionary Computation/GA_results/hill_climber_rosenbrock.png"},
    {
        "objective_function": lambda x: 20 + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])),
        "starting_point": [-2, 2],
        "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
        "step_size": 0.99,
        "filename": "/home/andrea/MCC/Repos/Evolutionary Computation/GA_results/hill_climber_C.png"
    }
]



# Iterate over each test case
for i, test in enumerate(tests):
    print(f"--- Test Case {i+1} ---")

    objective_function = test["objective_function"]
    starting_point = test["starting_point"]
    bounds = test["bounds"]
    step_size = test["step_size"]
    filename = test["filename"]

    # Run the hill climber algorithm
    solution_hill_climber, solution_value_hill_climber, operations, path = hill_climber(objective_function, starting_point, step_size=step_size)
    # Validate with scipy
    result = minimize(objective_function, starting_point, bounds=bounds)

    # Display the results
    print(f"Hill Climber Optimal Point: {solution_hill_climber}")
    print(f"Hill Climber Optimal Value: {solution_value_hill_climber}")
    print(f"Hill Climber Total Operations: {operations}\n")
    print("Validation using scipy.optimize:")
    print(f"Scipy Optimal Point: {result.x}")
    print(f"Scipy Optimal Value: {result.fun}\n")
    print(f"Two norm error: {error(result.fun, solution_value_hill_climber )}\n")

    plot_contour(objective_function, solution_hill_climber, bounds, path,filename,step_size)