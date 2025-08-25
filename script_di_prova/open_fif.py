import mne
import matplotlib.pyplot as plt

# Specifica il percorso del file .fif
file_path = r"C:\Users\Lorenzo\Desktop\PD002.fif"

# Carica il file raw usando MNE
raw = mne.io.read_raw_fif(file_path, preload=False)

# Stampa informazioni del file
print("=== Informazioni del file ===")
print(raw.info)

# Estrai i dati e i tempi dai primi tre canali
data, times = raw[:3]  # Prende i primi tre canali (indice 0, 1, 2)
data *= 1e6  # Converte i dati in µV (opzionale, per leggibilità)

# Crea subplots per ogni canale
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)

# Plot per ogni canale
for i, ax in enumerate(axes):
    ax.plot(times, data[i], label=f"Channel: {raw.ch_names[i]}")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend(loc="upper right")
    ax.grid(True)

# Etichetta comune sull'asse X
axes[-1].set_xlabel("Time (s)")

# Titolo generale
plt.suptitle("EEG Signals from First 3 Channels", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
plt.show()
