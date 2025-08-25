import os
import mne

def convert_fif_to_set(input_folder, export_dir, overwrite=True):
    """Converts all .fif files in the input folder to .set files."""
    os.makedirs(export_dir, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".fif"):
            file_path = os.path.join(input_folder, file_name)
            file_name_base = os.path.splitext(file_name)[0]  # Rimuove l'estensione .fif

            print(f"Processing file: {file_path}")

            try:
                # Carica il file .fif
                raw_from_fif = mne.io.read_raw_fif(file_path, preload=True)

                # Definisce il percorso per salvare il file .set
                set_file_path = os.path.join(export_dir, f"{file_name_base}.set")

                # Esporta in formato .set usando MNE
                mne.export.export_raw(set_file_path, raw_from_fif, fmt="eeglab", overwrite=overwrite)
                print(f"File salvato come .set: {set_file_path}")

            except Exception as e:
                print(f"Errore durante la conversione di {file_name}: {e}")

# Configura i percorsi
input_folder = r"C:\Users\Lorenzo\Desktop\PD017"
export_dir = r"C:\Users\Lorenzo\Desktop\PD017"

convert_fif_to_set(input_folder, export_dir)
