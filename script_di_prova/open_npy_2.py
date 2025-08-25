import numpy as np

# Percorso del file
file_path = r"C:\Users\Lorenzo\Desktop\sleep_stages_09_12\new\PD007EEG_stages.npy"

# Carica il file
try:
    stages = np.load(file_path)
    print("Contenuto di PD007EEG_stages.npy:")
    print(stages[0:60])
except FileNotFoundError:
    print(f"Errore: Il file '{file_path}' non esiste.")
except Exception as e:
    print(f"Errore durante il caricamento del file: {e}")
