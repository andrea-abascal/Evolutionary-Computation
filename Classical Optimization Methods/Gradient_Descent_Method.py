import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def wolfe_conditions(f, grad_f, x, p, alpha):
    """ Check Wolfe conditions for a given step size alpha """
    # Wolfe condition parameters
    c1 = 1e-4
    c2 = 0.9

    f_alpha = f(x + alpha * p)
    f_0 = f(x)
    f_prime_alpha = np.dot(np.array(grad_f(x + alpha * p)), p)
    f_prime_0 = np.dot(np.array(grad_f(x)), p)
    
    # Wolfe conditions
    armijo = f_alpha <= f_0 + c1 * alpha * f_prime_0 # Armijo or sufficient decrease condition
    curvature = f_prime_alpha >= c2 * f_prime_0 # Curvature condition
    
    return armijo , curvature

def line_search(f, grad_f, x, d, alpha_init=1.0):
    """ Line search to find a step size that satisfies Wolfe conditions """
    alpha = alpha_init
    
    # Simple backtracking for Wolfe condition
    flag = False
    while not flag:
        armijo, curvature = wolfe_conditions(f, grad_f, x, d, alpha)
        if not armijo:
            alpha *= 0.5
            if not curvature:
                alpha *= 2
        else:
            flag = True
    
    return alpha

def gradient_descent(f, x_init, tol=1e-5, max_iter=1000, alpha_init=10):
    """ Gradient descent with Wolfe condition line search """
    x = x_init
    grad_f = grad(f)
    i = 0

    path = [x.copy()]  # Store the path of iterations
    
    while np.linalg.norm(np.array(grad_f(x))) > tol and i < max_iter:
        d = -np.array(grad_f(x))  # Direction: steepest descent
        
        # Wolfe condition line search
        alpha = line_search(f, grad_f, x, d, alpha_init)
        x = x + alpha * d
        path.append(x.copy())  # Store the new point
        i += 1
        #print(f"Iteration {iteration}: x1 = {x[0]}, x2 = {x[1]}, f(x) = {f(x)}, alpha = {alpha}")
    
    value = f(x)
    return x, value, i, path

def error(optimal_val, obtained_val):
    return np.linalg.norm(optimal_val-obtained_val)

def plot_contour(f, solution, bounds, path,filename):
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
    plt.plot(path[:, 0], path[:, 1], 'r-o', label='Gradient Descent Path')

    plt.title('Contour Plot with Iteration Path')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    # Save the plot as a PNG file
    plt.savefig(filename)
    plt.show()
"""
# Test cases
tests = [
    {
        "objective_function": lambda x: -1 * (-2 * x[0]**2 + 3 * x[0] * x[1] - 1.5 * x[1]**2 - 1.3),
        "starting_point": [-4, 4],
        "bounds": [(-6, 6), (-6, 6)],
        "step_size": 0.1,
        "filename": "gradient_descent_A.png"
    },
    {
        "objective_function": lambda x: (4 - 2.1 * x[0]**2 + (x[0]**4)/3) * x[0]**2 + x[0] * x[1] + (-4 + 4 * x[1]**2) * x[1]**2,
        "starting_point": [0.5, 1],
        "bounds": [(-3, 3), (-2, 2)],
        "step_size": 0.09,
        "filename": "gradient_descent_B.png"
    },
    {
        "objective_function": lambda x: 20 + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])),
        "starting_point": [-2, 2],
        "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
        "step_size": 1.,
        "filename": "gradient_descent_C.png"
    }
]
"""
tests = [
    {   "objective_function": lambda x: 100 * (x[0]**2 - x[1])**2 + (1 -x[0])**2,
        "bounds": [(-2.048, 2.048), (-2.048, 2.048)],
        "starting_point": [0.5, 1],
        "step_size": 0.9,
        "filename": "/home/andrea/MCC/Repos/Evolutionary Computation/GA_results/gradient_descent_rosenbrock.png"},
    {
        "objective_function": lambda x: 20 + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])),
        "starting_point": [-2, 2],
        "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
        "step_size": 1.,
        "filename": "/home/andrea/MCC/Repos/Evolutionary Computation/GA_results/gradient_descent_C.png"
    }
]

# Iterate over each test case
for i, test in enumerate(tests):
    print(f"--- Test Case {i+1} ---")

    objective_function = test["objective_function"]
    starting_point = [float(i) for i in test["starting_point"]]
    bounds = test["bounds"]
    step_size = test["step_size"]
    filename = test["filename"]
    # Run the hill climber algorithm
    solution_gradient_descend, solution_value_gradient_descend, operations, path = gradient_descent(objective_function, starting_point, alpha_init=step_size)

    # Validate with scipy
    result = minimize(objective_function, starting_point, bounds=bounds)

    # Display the results
    print(f"Gradient Descent Optimal Point: {solution_gradient_descend}")
    print(f"Gradient Descent Optimal Value: {solution_value_gradient_descend}")
    print(f"Gradient Descent Total Operations: {operations}\n")
    print("Validation using scipy.optimize:")
    print(f"Scipy Optimal Point: {result.x}")
    print(f"Scipy Optimal Value: {result.fun}\n")
    print(f"Two norm error: {error(result.fun, solution_value_gradient_descend )}\n")

    plot_contour(objective_function, solution_gradient_descend, bounds, path,filename)