import numpy as np
import pandas as pd
import os



import numpy as np
import pandas as pd

def process_early_late(input_npy, output_csv):
    """
    Processa i dati del sonno da un file .npy e salva i risultati filtrati in un file CSV.
    Calcola le percentuali di N3 per ogni segmento e le stampa.
    """
    sleep_stages = np.load(input_npy)
    print(f"Elaborando {input_npy}")

    # Genera indici delle epoche
    epochs = np.arange(len(sleep_stages))

    # Converte in DataFrame
    df = pd.DataFrame({'Epoch': epochs, 'Stage': sleep_stages})

    # Filtra solo la parte NREM (N1, N2, N3)
    nrem_df = df[df['Stage'].isin([1, 2, 3])]

    # Dimensione delle epoche NREM e calcolo dei segmenti
    total_nrem_epochs = len(nrem_df)
    segment_size = total_nrem_epochs // 10

    # Dividi in 10 segmenti
    segments = [nrem_df.iloc[i * segment_size:(i + 1) * segment_size] for i in range(10)]

    # Calcola e stampa le percentuali di N3 per ogni segmento
    for i, segment in enumerate(segments, start=1):
        n3_percentage = (segment['Stage'] == 3).sum() / len(segment) * 100 if len(segment) > 0 else 0
        print(f"Segmento {i}: {n3_percentage:.2f}%")

    # Estrai early (segmenti 2, 3, 4) e late (segmenti 7, 8, 9)
    early_sleep = pd.concat(segments[1:4])
    late_sleep = pd.concat(segments[6:9])

    # Aggiungi una colonna "Phase" per distinguere early e late
    early_sleep['Phase'] = 'Early'
    late_sleep['Phase'] = 'Late'

    # Combina i dati early e late
    combined_sleep = pd.concat([early_sleep, late_sleep])

    # Trova le epoche che non sono early né late e chiamale "Not considered"
    not_considered = nrem_df[~nrem_df['Epoch'].isin(combined_sleep['Epoch'])]
    not_considered['Phase'] = 'Not considered'

    # Combina tutti i dati (early, late, not considered)
    final_sleep = pd.concat([combined_sleep, not_considered])

    # Ordina i dati in base al numero di epoca
    final_sleep = final_sleep.sort_values(by="Epoch")

    # Filtra solo le epoche con stadio N2 o N3
    n2_n3_sleep = final_sleep[final_sleep['Stage'].isin([2, 3])]

    # Salva il risultato in un file CSV
    n2_n3_sleep.to_csv(output_csv, index=False)
    print(f"File salvato: {output_csv}")



def process_all_sleep_files(input_folder, output_folder):
    """
    Elabora tutti i file .npy nella cartella di input e salva i risultati nella cartella di output.
    """
    os.makedirs(output_folder, exist_ok=True)
    npy_files = [f for f in os.listdir(input_folder) if f.endswith('_stages.npy')]

    for npy_file in npy_files:
        input_npy_path = os.path.join(input_folder, npy_file)
        patient_name = os.path.splitext(npy_file)[0]
        output_csv_path = os.path.join(output_folder, f"{patient_name}_e_l_1469.csv")
        process_early_late(input_npy_path, output_csv_path)


######################################################################################################################

def assign_phases_to_patient(features_path, phases_path, output_path):
    """
    Assegna le fasi (early, late, not considered) ai dati delle feature di un singolo paziente.
    """
    features_df = pd.read_csv(features_path)
    print(f"Caricando features: {features_path} -> {features_df.shape}")

    phases_df = pd.read_csv(phases_path)
    print(f"Caricando fasi: {phases_path} -> {phases_df.shape}")

    # Assegna le fasi ai dati delle feature
    phases_df['Phase_Assigned'] = phases_df['Phase'].apply(
        lambda phase: 'Early' if 'Early' in phase else ('Late' if 'Late' in phase else 'Not considered')
    )

    # Merge basato su Epochs
    merged_df = pd.merge(features_df, phases_df[['Epoch', 'Phase_Assigned']], on='Epoch', how='left')

    # Salva il file aggiornato per il paziente
    merged_df.to_csv(output_path, index=False)
    print(f"File con fasi salvato: {output_path}")


def process_all_patients(features_folder_path, phases_folder_path, output_folder):
    """
    Processa ogni paziente separatamente assegnando le fasi e salvando il file aggiornato.
    """
    os.makedirs(output_folder, exist_ok=True)

    feature_files = [f for f in os.listdir(features_folder_path) if
                     f.endswith("_N3conn100_specific_channels_149.csv")]
    print("File delle feature trovati:", feature_files)

    for feature_file in feature_files:
        patient_name = feature_file.replace("_N3conn100_specific_channels_149.csv", "")
        print("Elaborazione del paziente:", patient_name)
        features_path = os.path.join(features_folder_path, feature_file)
        print("Path delle feature:", features_path)
        phases_path = os.path.join(phases_folder_path, f"{patient_name}EEG_stages_e_l_1469.csv")
        print("Path delle fasi:", phases_path)
        output_path = os.path.join(output_folder, f"{patient_name}_features_with_phases.csv")
        print("Path di output:", output_path)

        if os.path.exists(phases_path):
            assign_phases_to_patient(features_path, phases_path, output_path)
        else:
            print(f"Fasi non trovate per {patient_name}, saltato.")


######################################################################################################################

def concatenate_features_with_phases(output_folder, final_output_path):
    """
    Concatena tutti i file dei pazienti con le fasi assegnate in un unico DataFrame.
    """
    dataframes = []
    for file_name in os.listdir(output_folder):
        if file_name.endswith("_features_with_phases.csv"):
            file_path = os.path.join(output_folder, file_name)
            print(f"Caricando: {file_path}")
            dataframes.append(pd.read_csv(file_path))

    aggregated_df = pd.concat(dataframes, ignore_index=True)
    aggregated_df.to_csv(final_output_path, index=False)
    print(f"File aggregato salvato in: {final_output_path}")


######################################################################################################################

def main():
    # Cartelle di input e output
    input_folder = r"C:\Users\Lorenzo\Desktop\sleep_stages_09_12\clinical_annotations"
    output_folder_phases = r"C:\Users\Lorenzo\Desktop\sleep_stages_09_12\clinical_annotations"
    features_folder_path = r"D:\TESI\prova statistica\N3CONN100_specific_channels_149"
    output_folder_features = r"D:\TESI\prova statistica\N3CONN100_specific_channels_149"
    final_output_path = os.path.join(features_folder_path,
                                     "_N3CONN100_specific_channels_149_aggregated_with_phases.csv")

    # Step 1: Elaborazione dei file .npy per estrarre early/late e salvarli per ogni paziente
    process_all_sleep_files(input_folder, output_folder_phases)

    # Step 2: Assegnazione delle fasi ai dati delle feature per ogni paziente
    process_all_patients(features_folder_path, output_folder_phases, output_folder_features)

    # Step 3: Concatenazione di tutti i file aggiornati in un unico file finale
    concatenate_features_with_phases(output_folder_features, final_output_path)

if __name__ == "__main__":
    main()
