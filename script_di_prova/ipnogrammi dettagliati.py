import numpy as np
import matplotlib.pyplot as plt
import os

# Directory principale
main_directory = r"D:\TESI\sleep_stages_09_12\clinical_scorings"

# Definizione delle fasi di sonno
stages_mapping = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    5: 'REM'
}

# Funzione per creare e salvare un ipnogramma
def create_hypnogram(directory, stages_file, arousals_file=None, bad_epochs_file=None, bad_muscle_flat_file=None):
    # Carica i file
    stages = np.load(stages_file) if stages_file else None
    arousals = np.load(arousals_file) if arousals_file else np.array([])
    bad_epochs = np.load(bad_epochs_file) if bad_epochs_file else np.array([])
    bad_muscle_flat = np.load(bad_muscle_flat_file) if bad_muscle_flat_file else np.array([])

    # Trova gli indici degli arousals (valore 1)
    arousal_indices = np.where(arousals == 1)[0] if len(arousals) > 0 else np.array([])

    # Crea l'ipnogramma
    epochs = np.arange(len(stages))
    plt.figure(figsize=(12, 7))

    # Disegna le fasi del sonno
    plt.plot(epochs, stages, drawstyle='steps-post', label='Sleep Stages', color='blue')

    # Marca le epoche escluse con colori diversi e cerchi più piccoli
    if len(arousal_indices) > 0:
        plt.scatter(arousal_indices, stages[arousal_indices], color='red', label='Arousals', s=20, zorder=3)
    if len(bad_epochs) > 0:
        plt.scatter(bad_epochs, stages[bad_epochs], color='green', label='Bad Epochs', s=20, zorder=3)
    if len(bad_muscle_flat) > 0:
        plt.scatter(bad_muscle_flat, stages[bad_muscle_flat], color='purple', label='Bad Muscle Flat', s=20, zorder=3)

    # Configurazione del grafico
    plt.gca().invert_yaxis()  # Inverte l'asse Y
    plt.yticks(list(stages_mapping.keys()), list(stages_mapping.values()))
    plt.xlabel('Epochs')
    plt.ylabel('Sleep Stages')
    plt.title('Hypnogram (Inverted) with Excluded Epochs')
    plt.legend()
    plt.grid(True)

    # Salva il grafico come immagine PNG
    output_file = os.path.join(directory, os.path.basename(stages_file).replace("_stages.npy", "_hypnogram.png"))
    plt.savefig(output_file)
    plt.close()
    print(f"Hypnogram saved: {output_file}")

# Cerca ricorsivamente tutti i file _stages.npy nella directory principale
for root, _, files in os.walk(main_directory):
    for file in files:
        if file.endswith("_stages.npy"):
            stages_file = os.path.join(root, file)

            # Identifica i file associati nella stessa directory
            arousals_file = os.path.join(root, file.replace("_stages.npy", "_arousals.npy"))
            bad_epochs_file = os.path.join(root, file.replace("_stages.npy", "_bad_epochs.npy"))
            bad_muscle_flat_file = os.path.join(root, file.replace("_stages.npy", "_bad_muscle_flat.npy"))

            # Verifica se i file associati esistono
            arousals_file = arousals_file if os.path.exists(arousals_file) else None
            bad_epochs_file = bad_epochs_file if os.path.exists(bad_epochs_file) else None
            bad_muscle_flat_file = bad_muscle_flat_file if os.path.exists(bad_muscle_flat_file) else None

            # Crea e salva l'ipnogramma
            create_hypnogram(root, stages_file, arousals_file, bad_epochs_file, bad_muscle_flat_file)
