import mne
import numpy as np
import matplotlib.pyplot as plt

# Percorso del file EEG
percorso_file = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/signal/"
file_eeg = percorso_file + "PD012_parziale.fif"  # Assicurati di rinominare il file!

# Caricamento del file EEG
raw = mne.io.read_raw_fif(file_eeg, preload=True, verbose=False)

#plt.rcParams.update({"font.size": 6.8})  # Imposta la dimensione del font a 8 (più piccolo)
raw.plot_sensors(ch_type="eeg", show_names=True)

# Controlla se il file ha una montatura
if raw.get_montage() is None:
    print("Nessuna montatura trovata. Imposto la montatura standard 10-20.")
    raw.set_montage("standard_1020")

# Estrai i dati EEG da un istante temporale
data, times = raw[:, 1]  # Usa il campione 100 come esempio
data = data.squeeze()  # **RISOLVE IL PROBLEMA DI DIMENSIONE (257, 1) → (257,)**

# Ottieni la posizione dei canali
montage = raw.get_montage()
posizioni = montage.get_positions()['ch_pos']

# Filtra solo i canali con posizione definita
canali_validi = [ch for ch in raw.info['ch_names'] if ch in posizioni]
if not canali_validi:
    raise ValueError("Nessuna posizione valida trovata per i canali EEG.")

# Ottieni coordinate X e Y per la topomap
x, y = np.array([posizioni[ch][:2] for ch in canali_validi]).T
pos = np.column_stack((x, y))  # Combina X e Y in un array 2D

# **Plot della Topomap con nomi dei canali**
fig, ax = plt.subplots()
mne.viz.plot_topomap(
    data[:len(pos)], pos, axes=ax, show=True, cmap='grey',
    contours=0, names=canali_validi, sensors=True  # <-- Mostra i nomi dei canali!
)

plt.show()


import mne
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib

# Abilita il backend interattivo
matplotlib.use('qt5agg')

# Classe per suddividere gli elettrodi in regioni cerebrali
class EEGRegionsDivider:
    def __init__(self):
        self.fp = np.sort(np.array([27, 33, 34, 38, 39, 47, 48, 26, 20, 19, 12, 11, 3, 2, 222]))
        self.f = np.sort(np.array([16, 22, 23, 24, 28, 29, 30, 35, 36, 40, 41, 42, 49, 50, 21, 15, 7, 14, 6,
                                   207, 13, 5, 215, 4, 224, 223, 214, 206, 213, 205]))
        self.c = np.sort(np.array([9, 17, 43, 44, 45, 51, 52, 53, 57, 58, 59, 60, 64, 65, 66, 71, 72, 8,
                                    81, 186, 198, 197, 185, 132, 196, 184, 144, 204, 195, 183, 155,
                                   194, 182, 164, 181, 173]))
        self.t = np.sort(np.array([55, 56, 62, 63, 69, 70, 74, 75, 84, 85, 96, 221, 212, 211, 203,
                                   202, 193, 192, 180, 179, 171, 170]))
        self.p = np.sort(np.array([76, 77, 78, 79, 80, 86, 87, 88, 89, 97, 98, 99, 100, 110, 90,
                                   101, 119, 172, 163, 154, 143, 131, 162, 153, 142, 130, 161,
                                   152, 141, 129, 128]))
        self.o = np.sort(np.array([107, 108, 109, 116, 117, 118, 125, 126, 160, 151, 140, 150,
                                   139, 127, 138]))

    def get_region_colors(self):
        colors = {}
        colors.update({ch: 'red' for ch in self.fp})
        colors.update({ch: 'blue' for ch in self.f})
        colors.update({ch: 'green' for ch in self.c})
        colors.update({ch: 'purple' for ch in self.t})
        colors.update({ch: 'orange' for ch in self.p})
        colors.update({ch: 'brown' for ch in self.o})
        return colors

# Definizione del percorso del file
percorso_file = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/signal/"

# Caricamento del file
raw = mne.io.read_raw_fif(percorso_file + 'PD012_parziale.fif', preload=True, verbose=False)

# Ottieni informazioni sulle posizioni
channel_names = raw.info['ch_names']
channel_positions = np.array([ch['loc'][:3] for ch in raw.info['chs'] if np.any(ch['loc'])])

# Istanzia la classe e ottieni i colori
divider = EEGRegionsDivider()
region_colors = divider.get_region_colors()

# Colora i canali in base alla loro regione
channel_colors = []
for idx, ch_name in enumerate(channel_names):
    ch_number = idx + 1
    channel_colors.append(region_colors.get(ch_number, 'gray'))

# Creazione del plot 3D
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot dei canali
for pos, color, ch_name in zip(channel_positions, channel_colors, channel_names):
    ax.scatter(pos[0], pos[1], pos[2], c=color, s=100, label=ch_name)
    ax.text(pos[0], pos[1], pos[2], ch_name, fontsize=8, ha='center', va='center', color='black')

# Aggiungi legenda
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=region)
    for region, color in zip(['Fp', 'F', 'C', 'T', 'P', 'O'],
                             ['red', 'blue', 'green', 'purple', 'orange', 'brown'])
]
ax.legend(handles=handles, loc='upper right', title="Regioni Cerebrali")

# Personalizza il grafico
ax.set_title("Disposizione degli elettrodi (Plot 3D)")
ax.set_xlabel("Asse X")
ax.set_ylabel("Asse Y")
ax.set_zlabel("Asse Z")

plt.show()
