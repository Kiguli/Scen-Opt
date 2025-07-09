import json
import numpy as np

# Save
matrices = {'A': [[1, 2], [2, 3]], 'B': [[0, 1], [1, 0]]}
with open('matrices.json', 'w') as f:
    json.dump(matrices, f)

# Load
with open('matrices.json', 'r') as f:
    matrices = json.load(f)
A = np.array(matrices['A'])
B = np.array(matrices['B'])

