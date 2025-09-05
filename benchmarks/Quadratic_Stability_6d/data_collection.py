import numpy as np

# Generate 100 unique pairs (x, y) with x, y in [-1, 1]
num_samples = 1162
samples = set()

while len(samples) < num_samples:
    new_samples = np.random.uniform(-1, 1, (num_samples, 2))
    for pair in new_samples:
        samples.add(tuple(pair))
    samples = set(list(samples)[:num_samples])  # Ensure only 100 pairs

samples = np.array(list(samples))
samples = samples[:num_samples]

# Save to CSV
np.savetxt('quadratic_stability_data.csv', samples, delimiter=',')