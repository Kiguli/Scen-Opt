import os
import numpy as np

def rk4_step(f, x, u, dt):
    """
    Compute the next state using RK4 for xdot = f(x, u).
    f: function f(x, u) returning xdot
    x: current state (numpy array)
    u: input/control (numpy array)
    dt: time step (float)
    """
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next

def f(x, u):
    alpha = np.arctan(np.tan(u[1]) / 2.0)
    dx = np.zeros(3)
    dx[0] = u[0] * np.cos(alpha + x[2]) / np.cos(alpha)
    dx[1] = u[0] * np.sin(alpha + x[2]) / np.cos(alpha)
    dx[2] = u[0] * np.tan(u[1])
    return dx

# Parameters
num_points = 3127
center = np.array([0, 1.2, 0])
width = 1.6
dt = 0.03
u = np.array([0.3, 0.3])  # fixed control input

# Sample points uniformly in the hypercube
low = center - width / 2
high = center + width / 2
samples = np.random.uniform(low, high, size=(num_points, 3))

# Compute next state for each sample
matrix = np.zeros((num_points, 6))
for i in range(num_points):
    x = samples[i]
    x_next = rk4_step(f, x, u, dt)
    matrix[i, :3] = x
    matrix[i, 3:] = x_next

# Calculate x_next for the center point
x_center = np.array([0, 1.2, 0])
x_next_center = rk4_step(f, x_center, u, dt)

# Compute absolute differences
diffs = np.zeros((num_points, 6))
for i in range(num_points):
    diffs[i, :3] = np.abs(matrix[i, :3] - x_center)
    diffs[i, 3:] = np.abs(matrix[i, 3:] - x_next_center)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)
np.savetxt(os.path.join(data_dir, 'growth_bound.csv'), diffs, delimiter=',')
