#!/usr/bin/env python3
"""Generate scenario data for the LPV Stability 3D benchmark."""

import os
import numpy as np

benchmark_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(benchmark_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# Sample 100 unique real values in the interval [-0.22, 1]
samples = np.random.uniform(-0.22, 1, 100)
samples = np.unique(samples)
while len(samples) < 100:
    new_samples = np.random.uniform(-0.22, 1, 100 - len(samples))
    samples = np.unique(np.concatenate([samples, new_samples]))

samples = np.sort(samples)

out_path = os.path.join(data_dir, 'scenarios.csv')
np.savetxt(out_path, samples, delimiter=',')
print(f'Saved: {out_path}')