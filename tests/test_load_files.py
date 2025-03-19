from src.Miscellaneous import load_file
import numpy as np
import pandas as pd

def save_numpy_to_file(array, file_path):
    """
    Saves a NumPy array to a file in CSV, TXT, XLSX, or JSON format.

    Parameters:
        array (np.ndarray): NumPy array to save.
        file_path (str): Path to the output file.
    """
    try:
        if file_path.endswith('.csv'):
            pd.DataFrame(array).to_csv(file_path, index=False, header=False)
        elif file_path.endswith('.txt'):
            np.savetxt(file_path, array, fmt='%s')
        elif file_path.endswith('.xlsx'):
            pd.DataFrame(array).to_excel(file_path, index=False, header=False)
        elif file_path.endswith('.json'):
            pd.DataFrame(array).to_json(file_path)
        else:
            raise ValueError("Unsupported file format")

        print(f"File saved successfully: {file_path}")
    except Exception as e:
        print(f"Error: {e}")


# Example usage:
deltas = np.array([[2, 1, -100], [3, 2, -120], [-1, 0, 0], [0, -1, 0]])
save_numpy_to_file(deltas, "data.csv")
save_numpy_to_file(deltas, "data.txt")
save_numpy_to_file(deltas, "data.xlsx")
save_numpy_to_file(deltas, "data.json")

#TEST CSV
array = load_file("data.csv")
print(type(array))
print(array)

#TEST TXT
array = load_file("data.txt")
print(type(array))
print(array)

#TEST EXCEL
array = load_file("data.xlsx")
print(type(array))
print(array)

#TEST JSON
array = load_file("data.json")
print(type(array))
print(array)