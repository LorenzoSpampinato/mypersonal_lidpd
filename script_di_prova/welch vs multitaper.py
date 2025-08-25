import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from mne.time_frequency import psd_array_multitaper

# Generare un segnale EEG simulato (oscillazioni alfa + rumore)
fs = 128  # Frequenza di campionamento (1000 Hz)
T = 10     # Durata in secondi
t = np.arange(0, T, 1/fs)  # Asse temporale

# Simulare un segnale EEG (oscillazione a 10 Hz con rumore)
f_alpha = 10  # Frequenza oscillatoria (tipica per onde alfa EEG)
eeg_signal = np.sin(2 * np.pi * f_alpha * t) + 0.5 * np.random.randn(len(t))  # Segnale EEG con rumore

# Calcolare la PSD usando il metodo Welch
f_welch, Pxx_welch = welch(eeg_signal, fs=fs, nperseg=1024, noverlap=512)

# Calcolare la PSD usando il metodo Multitaper
Pxx_multitaper, f_multitaper = psd_array_multitaper(eeg_signal, sfreq=fs, fmin=0, fmax=fs/2, adaptive=True,
                    normalization='full', verbose=0)

# Plot dei risultati
plt.figure(figsize=(10, 6))
plt.semilogy(f_welch, Pxx_welch, label='Welch', color='blue')
plt.semilogy(f_multitaper, Pxx_multitaper, label='Multitaper', color='red')
plt.title('Confronto tra PSD (Welch vs Multitaper) - Segnale EEG Simulato')
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Densità spettrale di potenza (dB/Hz)')
plt.legend()
plt.grid(True)
plt.show()
