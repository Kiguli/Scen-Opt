import pandas as pd

# Define column names for the iris dataset
columns = [
    "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
]

# Read the iris.data file
df = pd.read_csv('iris.data', header=None, names=columns)

# Drop any empty rows (if present)
df = df.dropna()

# Save to CSV
df.to_csv('iris.csv', index=False)