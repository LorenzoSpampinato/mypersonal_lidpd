import numpy as np

# Specifica il percorso del file .npy
npy_file_path = r"D:\TESI\sleep_stages_09_12\clinical_annotations\PD045EEG_stages.npy"

# Carica i dati
data = np.load(npy_file_path)

# Stampa i dati
print("Dati caricati:")
print(data)

# Controlla la forma dei dati
print(f"Forma dei dati: {data.shape}")

import numpy as np

# Specifica il percorso del file .npy
npy_file_path = r"D:\TESI\sleep_stages_09_12\clinical_annotations\PD045EEG_stages.npy"
#npy_file_path = r"C:\Users\Lorenzo\Desktop\sleep_stages_09_12\clinical_annotations\stagesPD020.npy"

# Carica i dati
data = np.load(npy_file_path)

# Trova gli indici delle epoche con valore 3 (N3)
n3_epochs = np.where(data == 3)[0]

# Stampa le epoche corrispondenti a N3
print(f"Epoche N3 trovate: {len(n3_epochs)}")
print("Indici delle epoche N3:", n3_epochs)


# Conta il totale dei numeri
total_numbers = data.size
print(f"Totale numeri nel file: {total_numbers}")

# Conta la frequenza di ciascun valore unico
unique_values, counts = np.unique(data, return_counts=True)
print("Valori unici e frequenze:")
for value, count in zip(unique_values, counts):
    print(f"Valore {value}: {count} occorrenze")

