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
