import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV
data = np.loadtxt('iris_minimal.csv', delimiter=',')
X = data[:, :2]
y = data[:, 2]

# Plot points by class
plt.figure(figsize=(6, 6))
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', label='Iris-setosa')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='orange',label='Other Iris')

# Plot decision boundary: 1.294061908745149*x1 + 0.8236170401941096*x2 - 3.78816446941025 = 0
x_vals = np.linspace(0, 7, 100)
# optimal solution found by cvxpy
y_vals = (3.78816446941025 - 1.294061908745149 * x_vals) / 0.8236170401941096
plt.plot(x_vals, y_vals, color='green', label='Decision Boundary') #TODO: double check/do manually.
plt.xlim(0,7)
plt.ylim(0,3)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Classification of Iris Species')
plt.legend(loc = 'lower right')
#plt.grid(True)
plt.show()