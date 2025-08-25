import pandas as pd

# Percorsi ai file
csv_file = r"C:\Users\Lorenzo\Desktop\all_subjects_features - Copia.csv"  # File CSV originale
summary_file = r"D:\TESI\lid-data-samples\lid-data-samples\output_hypnograms\sleep_cycles_summary.txt"  # File con i cicli del sonno
output_csv = r"C:\Users\Lorenzo\Desktop\all_subjects_features_with_period.csv"  # File CSV di output

# Leggi il file CSV originale
df = pd.read_csv(csv_file)

# Leggi il file di riassunto dei cicli
with open(summary_file, "r") as f:
    summary_data = f.readlines()

# Prepara una colonna "Period" inizialmente vuota
df["Period"] = ""

# Funzione per analizzare il file di riassunto e assegnare i periodi
period_data = {}
current_patient = None

for line in summary_data:
    line = line.strip()

    # Identifica il paziente
    if line.startswith("Patient"):
        current_patient = line.split()[1].replace("_preprocessed", "")
        period_data[current_patient] = []
        continue

    # Identifica i cicli e i relativi intervalli
    if "NREMP" in line or "REMP" in line:
        parts = line.split(":")
        cycle_type = parts[0].strip()  # Es: "NREMP 1", "REMP 1"
        indices = parts[1].split(",")  # Es: "Start = 51, End = 232"
        start = int(indices[0].split("=")[1].strip())
        end = int(indices[1].split("=")[1].strip())
        period_data[current_patient].append((cycle_type, start, end))

# Itera sui pazienti nel CSV
unique_patients = df["Subject"].unique()
current_offset = 0  # Tiene traccia della posizione globale nel CSV

for patient in unique_patients:
    if patient not in period_data:
        print(f"Attenzione: {patient} non trovato nel file TXT.")
        continue

    # Ottieni tutte le righe del paziente
    patient_rows = df[df["Subject"] == patient]
    num_rows = len(patient_rows)
    num_epochs = num_rows // 6  # Calcola il numero di epoche (6 regioni per epoca)

    # Crea un set per gli indici coperti (NREMP e REMP)
    covered_indices = set()

    # Processa ogni periodo per il paziente
    for cycle_type, start, end in period_data[patient]:
        # Itera su ciascuna regione cerebrale
        for region_idx in range(6):  # 6 regioni
            region_offset = region_idx * num_epochs  # Sposta di un blocco per ogni regione
            adjusted_start = current_offset + region_offset + start
            adjusted_end = current_offset + region_offset + end
            df.loc[adjusted_start:adjusted_end, "Period"] = cycle_type
            covered_indices.update(range(adjusted_start, adjusted_end + 1))  # Aggiungi gli indici coperti

    # Identifica gli intervalli non coperti da NREMP o REMP
    not_sleep_intervals = []
    current_start = None

    for i in range(current_offset, current_offset + num_rows):
        if i not in covered_indices:
            if current_start is None:
                current_start = i
        elif current_start is not None:
            not_sleep_intervals.append((current_start, i - 1))
            current_start = None

    # Gestisci l'intervallo finale
    if current_start is not None:
        not_sleep_intervals.append((current_start, current_offset + num_rows - 1))

    # Assegna il periodo "Not_Sleep" agli intervalli non coperti
    for start, end in not_sleep_intervals:
        df.loc[start:end, "Period"] = "Not_Sleep"

    # Aggiorna l'offset globale per il prossimo paziente
    current_offset += len(patient_rows)

# Reorganizza le colonne spostando "Period" tra "Brain region" e "Stage"
columns = list(df.columns)
period_index = columns.index("Brain region") + 1
columns.insert(period_index, columns.pop(columns.index("Period")))
df = df[columns]

# Salva il file CSV aggiornato
df.to_csv(output_csv, index=False)

print(f"File aggiornato salvato in: {output_csv}")
