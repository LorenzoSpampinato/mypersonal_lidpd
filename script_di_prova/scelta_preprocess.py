import mne
import matplotlib.pyplot as plt

# Parametri del filtro
fs = 250  # Frequenza di campionamento in Hz
l_freq = 0.50  # Frequenza di taglio inferiore
h_freq = 35 # Frequenza di taglio superiore
fir_window = 'hamming'  # Tipo di finestra
filter_length='auto'
l_trans_bandwidth=0.5
h_trans_bandwidth='auto'
fir_design='firwin'
method='fir'
phase= 'zero'

# Crea il filtro
filt_pars = mne.filter.create_filter(data=None, sfreq=fs, l_freq=l_freq, h_freq=h_freq, filter_length=filter_length,
                                     l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth,
                                     method=method, phase= phase,
                                     fir_window=fir_window, fir_design=fir_design, verbose=None)

# Calcola la larghezza della banda di transizione
#l_trans_bandwidth = min(max(l_freq * 0.25, 2), l_freq)
h_trans_bandwidth = min(max(h_freq * 0.25, 2), fs/2 - h_freq)
#l_trans_bandwidth=0.3
#h_trans_bandwidth=8

# Risposta del filtro in secondi (larghezza del filtro in campioni)
ordine_filtro = len(filt_pars)
# Converti la lunghezza del filtro in secondi
filtro_length_seconds = ordine_filtro / fs

# Calcola la banda di roll-off
rolloff_bandwidth = l_trans_bandwidth + h_trans_bandwidth

# Visualizza la risposta in frequenza del filtro
mne.viz.plot_filter(filt_pars, sfreq=fs, freq=None, gain='both')
plt.show()

# Stampa dei risultati e delle formule
print("Parametri del filtro FIR calcolati:")
print(f" - Frequenza di taglio inferiore: {l_freq} Hz")
print(f" - Frequenza di taglio superiore: {h_freq} Hz")
print(f" - Larghezza della banda di transizione inferiore: {l_trans_bandwidth:.2f} Hz")
print(f" - Larghezza della banda di transizione superiore: {h_trans_bandwidth:.2f} Hz")
print(f" - Ordine del filtro (numero di campioni): {ordine_filtro}")
print(f" - Tipo di finestra: {fir_window}")
print(f" - Risposta del filtro in secondi: {filtro_length_seconds:.4f} s")
print(f" - Banda di roll-off: {rolloff_bandwidth:.2f} Hz")

#####################################################################################################
'''
# Path to the FIF file (replace with the actual path)
file_path = 'D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/PD012_cropped.fif'

# Load the FIF file
raw = mne.io.read_raw_fif(file_path, preload=True)

# Apply FIR filter to the signal
raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, filter_length=filter_length,
                                 l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth,
                                 method=method, phase=phase,
                                 fir_window=fir_window, fir_design=fir_design, verbose=None)

# Select a channel to visualize the signal over time (e.g., the first channel)
channel = 0  # You can choose a specific channel
#start, stop = raw.time_as_index([60, 360])  # Time interval (in seconds) to visualize
start, stop = raw.time_as_index([503.6, 508])  # Time interval (in seconds) to visualize

# Extract data for the original and filtered signals
original_data, times = raw[channel, start:stop]
filtered_data, _ = raw_filtered[channel, start:stop]

# Converti i dati in microvolt
original_data_microvolt = original_data * 1e6  # Converti in μV
filtered_data_microvolt = filtered_data * 1e6  # Converti in μV

# Plot signals in the time domain
plt.figure(figsize=(12, 6))

# Original signal
plt.subplot(2, 1, 1)
plt.plot(times, original_data_microvolt[0], label="Original Signal", color="black")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.title("Original Signal")
plt.legend()

# Filtered signal
plt.subplot(2, 1, 2)
plt.plot(times, filtered_data_microvolt[0], label="Filtered Signal", color="black")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.title("Filtered Signal")
plt.legend()

plt.tight_layout()
plt.show()

'''