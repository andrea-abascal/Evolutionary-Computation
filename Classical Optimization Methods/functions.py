import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Test cases
tests = [
    {
        "objective_function": lambda x:  -1* ((-2 * x[0]**2 )+ (3 * x[0] * x[1]) - (1.5 * x[1]**2) - 1.3),
        "starting_point": [-4, 4],
        "bounds": [(-6, 6), (-6, 6)],
        "step_size": 0.01,
        "filename": "objective_function_A.png"
    },
    {
        "objective_function": lambda x: (4 - 2.1 * x[0]**2 + (x[0]**4)/3) * x[0]**2 + x[0] * x[1] + (-4 + 4 * x[1]**2) * x[1]**2,
        "starting_point": [0.5, 1],
        "bounds": [(-3, 3), (-2, 2)],
        "step_size": 0.01,
        "filename": "objective_function_B.png"
    },
    {
        "objective_function": lambda x: 20 + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])),
        "starting_point": [-2, 2],
        "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
        "step_size": 0.99,
        "filename": "objective_function_C.png"
    }
]

# Create 3D plots for each objective function
for test in tests:
    objective_function = test["objective_function"]
    filename = test["filename"]
    
    # Create a meshgrid for plotting
    x = np.linspace(-2, 2, 800)
    y = np.linspace(-2, 2, 800)
    X, Y = np.meshgrid(x, y)
    Z = objective_function([X, Y])
    
    # Plot the objective function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    
    # Set labels
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    
    # Save the plot
    plt.savefig(filename)
    plt.show()
    plt.close()


print("Plots generated and saved.")
