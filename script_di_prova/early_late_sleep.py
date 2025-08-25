import numpy as np
import pandas as pd
import os

# Percorso della cartella contenente i file .npy
input_folder = r"C:\Users\Lorenzo\Desktop\sleep_stages_09_12\clinical_annotations"
output_folder = r"C:\Users\Lorenzo\Desktop\sleep_stages_09_12\clinical_annotations"
#input_folder = r"C:\Users\Lorenzo\Desktop\sleep_stages_usleep"
#output_folder = r"C:\Users\Lorenzo\Desktop\sleep_stages_usleep"


##################clinical_annotations############################################
# Crea la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Ottieni tutti i file .npy nella cartella di input
npy_files = [f for f in os.listdir(input_folder) if f.endswith('_stages.npy')]

# Funzione per elaborare ogni file .npy
def process_sleep_data(input_npy, output_csv):
    # Carica i dati dal file .npy
    sleep_stages = np.load(input_npy)
    print(f"Elaborando {input_npy}")

    # Genera indici delle epoche
    epochs = np.arange(len(sleep_stages))

    # Converte in DataFrame
    df = pd.DataFrame({'Epochs': epochs, 'Stage': sleep_stages})

    # Filtra solo la parte NREM (N1, N2, N3)
    nrem_df = df[df['Stage'].isin([1, 2, 3])]

    # Dimensione delle epoche NREM e calcolo dei segmenti
    total_nrem_epochs = len(nrem_df)
    segment_size = total_nrem_epochs // 10

    # Dividi in 10 segmenti
    segments = [nrem_df.iloc[i * segment_size:(i + 1) * segment_size] for i in range(10)]

    # Estrai early (segmenti 2, 3, 4) e late (segmenti 7, 8, 9)
    early_sleep = pd.concat(segments[0:3])
    late_sleep = pd.concat(segments[7:10])

    # Aggiungi una colonna "Phase" per distinguere early e late
    early_sleep['Phase'] = 'Early'
    late_sleep['Phase'] = 'Late'

    # Combina i dati early e late
    combined_sleep = pd.concat([early_sleep, late_sleep])

    # Trova le epoche che non sono early né late e chiamale "Not considered"
    not_considered = nrem_df[~nrem_df['Epochs'].isin(combined_sleep['Epochs'])]
    not_considered['Phase'] = 'Not considered'

    # Combina tutti i dati (early, late, not considered)
    final_sleep = pd.concat([combined_sleep, not_considered])

    # Ordina i dati in base al numero di epoca
    final_sleep = final_sleep.sort_values(by="Epochs")

    # Filtra solo le epoche con stadio N3
    n3_sleep = final_sleep[final_sleep['Stage'] == 3]

    # Salva il risultato in un file CSV
    n3_sleep.to_csv(output_csv, index=False)

    # Output per verifica
    print(f"File salvato: {output_csv}")
    print(n3_sleep.head())

# Elabora ogni file .npy nella cartella
for npy_file in npy_files:
    # Crea il percorso completo del file di input
    input_npy_path = os.path.join(input_folder, npy_file)

    # Estrai il nome del paziente (senza estensione)
    patient_name = os.path.splitext(npy_file)[0]

    # Crea il percorso di output per il CSV
    output_csv_path = os.path.join(output_folder, f"{patient_name}_e_l_03710.csv")

    # Elabora e salva il risultato
    process_sleep_data(input_npy_path, output_csv_path)
    ##################################################################################
'''
##############usleep##########################################################
# Crea la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Ottieni tutti i file .npy nella cartella di input
npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

# Funzione per elaborare ogni file .npy
def process_sleep_data(input_npy, output_csv):
    # Carica i dati dal file .npy e assicurati che sia un array 1D
    sleep_stages = np.load(input_npy).astype(int).ravel()  # Appiattisci l'array se non è 1D
    print(f"Elaborando {input_npy}")

    # Genera indici delle epoche
    epochs = np.arange(len(sleep_stages))

    # Converte in DataFrame
    df = pd.DataFrame({'Epochs': epochs, 'Stage': sleep_stages})

    # Filtra solo la parte NREM (N1, N2, N3)
    nrem_df = df[df['Stage'].isin([1, 2, 3])]

    # Dimensione delle epoche NREM e calcolo dei segmenti
    total_nrem_epochs = len(nrem_df)
    segment_size = total_nrem_epochs // 10

    # Dividi in 10 segmenti
    segments = [nrem_df.iloc[i * segment_size:(i + 1) * segment_size] for i in range(10)]

    # Estrai early (segmenti 2, 3, 4) e late (segmenti 7, 8, 9)
    early_sleep = pd.concat(segments[1:4])
    late_sleep = pd.concat(segments[6:9])

    # Aggiungi una colonna "Phase" per distinguere early e late
    early_sleep['Phase'] = 'Early'
    late_sleep['Phase'] = 'Late'

    # Combina i dati early e late
    combined_sleep = pd.concat([early_sleep, late_sleep])

    # Trova le epoche che non sono early né late e chiamale "Not considered"
    not_considered = nrem_df[~nrem_df['Epochs'].isin(combined_sleep['Epochs'])]
    not_considered['Phase'] = 'Not considered'

    # Combina tutti i dati (early, late, not considered)
    final_sleep = pd.concat([combined_sleep, not_considered])

    # Ordina i dati in base al numero di epoca
    final_sleep = final_sleep.sort_values(by="Epochs")

    # Filtra solo le epoche con stadio N3
    n3_sleep = final_sleep[final_sleep['Stage'] == 3]

    # Salva il risultato in un file CSV
    n3_sleep.to_csv(output_csv, index=False)

    # Output per verifica
    print(f"File salvato: {output_csv}")
    print(n3_sleep.head())
    

# Elabora ogni file .npy nella cartella
for npy_file in npy_files:
    # Crea il percorso completo del file di input
    input_npy_path = os.path.join(input_folder, npy_file)

    # Estrai il nome del paziente (senza estensione)
    patient_name = os.path.splitext(npy_file)[0]

    # Crea il percorso di output per il CSV
    output_csv_path = os.path.join(output_folder, f"{patient_name}_early_and_late_n3.csv")

    # Elabora e salva il risultato
    process_sleep_data(input_npy_path, output_csv_path)
    ########################################################################################
    '''

