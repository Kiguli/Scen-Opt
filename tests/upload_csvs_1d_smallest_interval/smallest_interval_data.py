import numpy as np
import csv

# Generate random points inside the interval [0, 1] using a uniform distribution
num_points = 100  # Number of points to generate
points = np.random.uniform(0, 1, num_points)

# Save points to a CSV file
csv_filename = 'smallest_interval_1d.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Point'])  # Header
    for point in points:
        writer.writerow([point])

print(f"Random points saved to {csv_filename}")

# CONSTRAINTS FOR TOOL: LP ROBUST
# c = [0,1]^T
#A(d) = [[-1 -1],[1,-1]]
#B(d) = [delta[0],-delta[0]]^T
# A = 0
# B = 0


import numpy as np
import matplotlib.pyplot as plt

# Load points from CSV
points = np.loadtxt('smallest_interval_1d.csv', delimiter=',', skiprows=1)

# Plot black line at y=0
plt.plot([0, 1], [0, 0], color='black', linewidth=0.5)

# Plot red points on the black line
plt.scatter(points, np.zeros_like(points), color='red', zorder=3)

# Blue marker line above, spanning min to max of points
y_marker = 0.1
min_pt, max_pt = np.min(points), np.max(points)
plt.plot([min_pt, max_pt], [y_marker, y_marker], color='blue', linewidth=3)

# Green marker at the center of the blue line
center_pt = (min_pt + max_pt) / 2
plt.scatter([center_pt], [y_marker], color='green', s=100, zorder=4)

plt.axis('off')
plt.show()