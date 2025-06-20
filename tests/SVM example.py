import numpy as np
import csv

# Parameters
num_points = 100  # Number of points to generate
domain_min, domain_max = -2, 2  # Domain bounds

# Generate random points in the domain [-2, 2]^2
points = np.random.uniform(domain_min, domain_max, (num_points, 2))

# Classify points based on x1
classifications = np.where(points[:, 0] < 0, -1, 1)

# Save points and classifications to a CSV file
csv_filename = 'SVM_example.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x1', 'x2', 'classification'])  # Header
    for point, classification in zip(points, classifications):
        writer.writerow([point[0], point[1], classification])

print(f"Data saved to {csv_filename}")