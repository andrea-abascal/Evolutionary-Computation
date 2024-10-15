import autograd.numpy as np
from autograd import grad, hessian
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def newton_method(f, x_init, tol=1e-5, max_iter=30):
    """Newton's method for optimization"""
    x = np.array(x_init)
    grad_f = grad(f)
    hess_f = hessian(f)
    iteration = 0
    path = [x.copy()]  # Store the path of iterations
    for _ in range(max_iter):
    #while np.linalg.norm(grad_f(x)) > tol and iteration < max_iter:        
        grad_val = -grad_f(x)
        hess_val = hess_f(x)
        
        # Update step: x = x - H^(-1) * grad_f
        step = np.linalg.solve(hess_val, grad_val)
        x = x + step        
        iteration += 1
        path.append(x.copy())  # Store the new point
        if np.linalg.norm(step) < tol:
            break
    
    value = f(x)
    return x, value, iteration, path

def plot_contour(f, solution, bounds, path, filename):
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
    plt.plot(path[:, 0], path[:, 1], 'r-o', label='Newton Method Path')

    plt.title('Contour Plot with Iteration Path')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    # Save the plot as a PNG file
    plt.savefig(filename)
    plt.show()

def error(optimal_val, obtained_val):
    return np.linalg.norm(optimal_val-obtained_val)
"""
# Test cases
tests = [
    {
        "objective_function": lambda x: -1 * (-2 * x[0]**2 + 3 * x[0] * x[1] - 1.5 * x[1]**2 - 1.3),
        "starting_point": [-4, 4],
        "bounds": [(-6, 6), (-6, 6)],
        "filename": "Classical Optimization Methods/results/newton_method_A.png" 
    },
    {
        "objective_function": lambda x: (4 - 2.1 * x[0]**2 + (x[0]**4)/3) * x[0]**2 + x[0] * x[1] + (-4 + 4 * x[1]**2) * x[1]**2,
        "starting_point": [0.5, 1],
        "bounds": [(-3, 3), (-2, 2)],
        "filename": "Classical Optimization Methods/results/newton_method_B.png"
    },
    {
        "objective_function": lambda x: 20 + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])),
        "starting_point": [-2, 2],
        "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
        "filename": "Classical Optimization Methods/results/newton_method_C.png"
    }
]
"""
tests = [
    {   "objective_function": lambda x: 100 * (x[0]**2 - x[1])**2 + (1 -x[0])**2,
        "bounds": [(-2.048, 2.048), (-2.048, 2.048)],
        "starting_point": [0.5, 1],
        "filename": "/home/andrea/MCC/Repos/Evolutionary Computation/GA_results/newton_method_rosenbrock.png"},
    {
        "objective_function": lambda x: 20 + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])),
        "starting_point": [-2, 2],
        "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
        "filename": "/home/andrea/MCC/Repos/Evolutionary Computation/GA_results/newton_method_C.png"
    }
]
# Iterate over each test case
for i, test in enumerate(tests):
    print(f"--- Test Case {i + 1} ---")

    objective_function = test["objective_function"]
    starting_point = [float(i) for i in test["starting_point"]]
    bounds = test["bounds"]
    filename = test["filename"]

    # Run the Newton method algorithm
    solution_newton_method, solution_value_newton_method, iterations, path = newton_method(objective_function, starting_point)

    # Validate with scipy
    result = minimize(objective_function, starting_point, bounds=bounds)

    # Display the results
    print(f"Newton's Method Optimal Point: {solution_newton_method}")
    print(f"Newton's Method Optimal Value: {solution_value_newton_method}")
    print(f"Newton's Method Total Iterations: {iterations}\n")
    print("Validation using scipy.optimize:")
    print(f"Scipy Optimal Point: {result.x}")
    print(f"Scipy Optimal Value: {result.fun}\n")
    print(f"Two norm error: {error(result.fun, solution_value_newton_method )}")

    plot_contour(objective_function, solution_newton_method, bounds, path, filename)
