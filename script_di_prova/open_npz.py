import numpy as np

# Load the .npz file
file_path = r"C:\Users\Lorenzo\Desktop\PD020_N3conn_specific_channels_149.npz"
data = np.load(file_path)

# Print the keys of the arrays stored in the file
print("Keys in the .npz file:", data.files)
print("Number of connectivity channels:", len(data['connectivity_channels']))


# Access individual arrays by their keys
for key in data.files:
    print(f"Array for key '{key}':\n{data[key]}")

'''

# Itera su ogni array contenuto e mostra il numero di righe
for key in data.files:
    array_shape = data[key].shape
    print(f"Array '{key}' ha {array_shape[0]} righe." if len(array_shape) > 0 else f"Array '{key}' è vuoto.")
'''