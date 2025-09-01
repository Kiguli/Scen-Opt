import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset
df = pd.read_csv('iris.csv')

# Create a pairplot colored by class
sns.pairplot(df, hue='class', diag_kind='hist', palette='Set2')

plt.suptitle('Iris Dataset Feature Pairplot', y=1.02)
plt.show()
