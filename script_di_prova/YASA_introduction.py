import yasa
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne import channel_type

# Path of the file with the hypnogram from hypnogram_definition
hypnogram_file_path = "D:/TESI/lid-data-samples/lid-data-samples/Labels/CTL\PD009.npy"

# Carica i dati dell'ipnogramma
hypnogram_data = np.load(hypnogram_file_path)

# Ottieni la forma dei dati dell'ipnogramma
hypnogram_shape = hypnogram_data.shape
print("Dimensioni dell'ipnogramma:", hypnogram_shape)

# Definisci il percorso del file
percorso_file = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/signal1.fif"

# Carica il file con preload=True
raw = mne.io.read_raw_fif(percorso_file, preload=True, verbose=False)

# Load the hypnogram produced by hypnogram_definition
hypno = np.load(hypnogram_file_path)

# Check the unique values to ensure they are valid sleep stages (0-4)
print("Unique values in hypnogram:", np.unique(hypno))

# Plot the hypnogram
yasa.plot_hypnogram(hypno)
plt.title("Hypnogram")
plt.xlabel("Hours")
plt.ylabel("Sleep Stages")
plt.show()  # Ensure you call this to display the plot

# Calculate sleep statistics
stats = yasa.sleep_statistics(hypno, sf_hyp=1/30)
print(stats)

hypno_up = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=raw)
print(len(hypno_up))
yasa.plot_spectrogram(data[channel_type("")], sf, hypno_up);

