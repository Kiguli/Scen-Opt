import numpy as np

# Sample 100 unique real values in the interval [-0.22, 1]
samples = np.random.uniform(-0.22, 1, 100)
samples = np.unique(samples)
while len(samples) < 100:
    # Add more samples if duplicates were removed
    new_samples = np.random.uniform(-0.22, 1, 100 - len(samples))
    samples = np.unique(np.concatenate([samples, new_samples]))

samples = np.sort(samples)

# Save to CSV file
np.savetxt('scheduling_variable_samples.csv', samples, delimiter=',')