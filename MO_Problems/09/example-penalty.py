import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This is required for 3D projection

def g(x, y, A, B):
    return (x**2 + y**2 + 
            A * np.maximum(0, x - y)**2 + 
            B * np.abs(x**2 + y**2 - 0.5)**2)

def f(x, y):
    return x**2 + y**2

# Define constants A and B
A = 2
B = 10

# Create a grid of points
x = np.linspace(-1, 1, 400)  # x in [-1, 1]
y = np.linspace(-1, 1, 400)  # y in [-1, 1]
X, Y = np.meshgrid(x, y)

# Compute g(x, y) and f(x, y)
Z_g = g(X, Y, A, B)
Z_f = f(X, Y)

# Create the plots
fig = plt.figure(figsize=(12, 8))

# 3D Surface Plot for g(x, y)
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z_g, cmap='viridis', alpha=0.7)
ax1.set_title('3D Surface Plot of g(x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('g(x, y)')
ax1.view_init(45, 45)  # Set the view angle for better visualization

# 3D Surface Plot for f(x, y)
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_f, cmap='plasma', alpha=0.7)
ax2.set_title('3D Surface Plot of f(x, y)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x, y)')
ax2.view_init(45, 45)  # Set the view angle for better visualization

# Show plots
plt.tight_layout()
plt.show()

