import matplotlib.pyplot as plt
import numpy as np
from src.Risk import quantify_risk

# Parameters
beta = 1e-6  # Confidence level
N_values = [2000, 4000, 8000]  # N1, N2, N3
N1 = N_values[1]
k = np.linspace(0,N1)

# Calculate risk bounds for each k value
epsL_values = []
epsU_values = []

for k_value in k:
    print("k = ", k_value)
    epsL, epsU = quantify_risk(k_value, N1, beta)
    epsL_values.append(epsL)
    epsU_values.append(epsU)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(k, epsL_values, label="Lower Bound (epsL)", marker="o")
plt.plot(k, epsU_values, label="Upper Bound (epsU)", marker="o")
plt.xlabel("k")
plt.ylabel("Risk Bounds")
plt.title("Risk Bounds vs k")
plt.legend()
plt.grid(True)
plt.show()