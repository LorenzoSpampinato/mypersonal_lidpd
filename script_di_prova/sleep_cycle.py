import os
import numpy as np
import matplotlib.pyplot as plt

# Threshold parameters for customization
MAX_NREMP_DURATION_MIN = 120  # Maximum NREMP duration in minutes
MAX_REM_EPISODES_IN_NREMP = 6  # Maximum allowed REM episodes within an NREMP
MIN_REMP_DURATION = 2  # Minimum duration for a valid REMP in minutes
TIME_RESOLUTION_SEC = 30  # Time resolution of the data in seconds (default 30s)
#MIN_SPLIT_GAP_MIN = 12  # Minimum gap duration without N3 to trigger a split (in minutes)

# Directory containing the .npy files
input_folder = "D:/TESI/lid-data-samples/lid-data-samples/predicted_scorings"
output_folder = "D:/TESI/lid-data-samples/lid-data-samples/output_hypnograms"  # Directory for saving the hypnogram images

# Make sure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def identify_sleep_cycles(sleep_stages):
    """
    Identify sleep cycles (NREMP and REMP) based on sleep stage data.
    """
    cycles = []
    n = len(sleep_stages)
    i = 0  # Current index
    first_episode = True  # Flag for identifying the first NREMP
    first_remp = True  # Flag for identifying the first REMP

    while i < n:
        # Skip initial wake periods
        if sleep_stages[i] == 0:
            i += 1
            continue

        if (first_episode and sleep_stages[i] in [1, 2]) or (not first_episode and sleep_stages[i] in [1, 2, 3]):
            start_nremp = i
            duration_nremp = 0
            rem_included = 0  # Count REM episodes within the NREMP
            in_rem_episode = False  # Track whether we are in a continuous REM episode

            while i < n and sleep_stages[i] in [1, 2, 3, 4, 0]:
                if sleep_stages[i] in [1, 2, 3]:  # Valid NREM stages
                    duration_nremp += 1
                    rem_included = 0  # Reset REM episode counter
                    in_rem_episode = False  # Reset REM episode flag
                elif sleep_stages[i] == 4:  # REM stage
                    if not in_rem_episode:  # If starting a new REM episode
                        in_rem_episode = True  # Set flag indicating we are in a REM episode
                    rem_included += 1  # Increment REM episode counter
                elif sleep_stages[i] == 0:  # Wake state
                    rem_included = 0  # Reset REM episode counter
                    in_rem_episode = False  # Reset REM episode flag
                else:
                    rem_included = 0  # Reset REM counter for unexpected states
                    in_rem_episode = False

                if rem_included > MAX_REM_EPISODES_IN_NREMP:  # Check if we exceed the REM limit
                    break

                i += 1

            end_nremp = i - MAX_REM_EPISODES_IN_NREMP  # Exclude the last REM episode

            # Check if splitting is needed
            max_duration_indices = int(MAX_NREMP_DURATION_MIN * 60 / TIME_RESOLUTION_SEC)
            if end_nremp - start_nremp + 1 > max_duration_indices:
                # Look for a split point based on the last N3 before the end of the cycle
                last_n3_index = -1  # Initialize variable to store the index of the last N3

                for idx in range(end_nremp - 1, start_nremp - 1, -1):
                    if sleep_stages[idx] == 3:  # N3 found
                        last_n3_index = idx
                        break  # Stop as soon as the last N3 is found

                if last_n3_index != -1:  # If a valid N3 is found, we can split
                    # Set the split point at the end of the last N3
                    split_point = last_n3_index + 1

                    # Add the first part of the NREMP as a cycle
                    cycles.append({
                        "type": "NREMP",
                        "start": start_nremp,
                        "end": split_point - 1,
                        "duration_min": (split_point - start_nremp) * (TIME_RESOLUTION_SEC / 60),
                    })

                    # Update the start of the next NREMP
                    start_nremp = split_point

            # Add the remaining or complete NREMP as a cycle
            cycles.append({
                "type": "NREMP",
                "start": start_nremp,
                "end": end_nremp,
                "duration_min": (end_nremp - start_nremp + 1) * (TIME_RESOLUTION_SEC / 60),
            })
            first_episode = False

        # Identify a REMP
        if i < n and sleep_stages[i] == 4:  # Start of a REMP
            start_remp = i - MAX_REM_EPISODES_IN_NREMP + 1
            duration_remp = 0
            allowed_transitions = 2  # Allow up to n exceptions for N1, N2, or N3

            while i < n:
                if sleep_stages[i] == 4:  # REMP stages
                    duration_remp += 1
                # Allow up to 3 transitions to N1, N2, or N3
                elif sleep_stages[i] in [1, 2, 3] and allowed_transitions > 0:
                    duration_remp += 1  # Count these indices as part of the REMP duration
                    allowed_transitions -= 1  # Use one allowed transition
                elif sleep_stages[i] == 0:  # Ignore wake without stopping the analysis
                    pass
                else:
                    break  # Stop if another non-REM stage is encountered after using all exceptions
                i += 1

            # Add the identified REMP cycle if valid
            if first_remp or (duration_remp * (TIME_RESOLUTION_SEC / 60)) >= MIN_REMP_DURATION:
                cycles.append({
                    "type": "REMP",
                    "start": start_remp,
                    "end": i - 1,
                    "duration_min": duration_remp * (TIME_RESOLUTION_SEC / 60),  # Convert to minutes
                })
                first_remp = False

    # Group NREMP and REMP into cycles
    sleep_cycles = []
    current_cycle = {}

    for cycle in cycles:
        if cycle["type"] == "NREMP":
            # Append any incomplete cycle before starting a new NREMP
            if current_cycle and "NREMP" in current_cycle:
                sleep_cycles.append(current_cycle)
            current_cycle = {"NREMP": cycle, "REMP": None}
        elif cycle["type"] == "REMP" and current_cycle:
            current_cycle["REMP"] = cycle
            sleep_cycles.append(current_cycle)
            current_cycle = {}

    # Include the last NREMP if it is not paired with a REMP
    if current_cycle:
        sleep_cycles.append(current_cycle)

    return sleep_cycles



def plot_hypnogram(sleep_stages, sleep_cycles, patient_id, output_folder):
    """
    Plot the hypnogram with sleep cycles annotated and save it as a PNG file.
    """
    time_axis = np.arange(len(sleep_stages)) * (TIME_RESOLUTION_SEC / 60)  # Convert indices to minutes
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, sleep_stages, drawstyle='steps-post', label="Hypnogram")

    cycle_colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2']  # Colors for sleep cycles

    # Add horizontal lines for sleep cycles
    for idx, cycle in enumerate(sleep_cycles):
        if "NREMP" in cycle:
            nremp = cycle["NREMP"]
            plt.hlines(y=5, xmin=nremp["start"] * (TIME_RESOLUTION_SEC / 60),
                       xmax=nremp["end"] * (TIME_RESOLUTION_SEC / 60),
                       color=cycle_colors[idx % len(cycle_colors)], linestyle='-', linewidth=4,
                       label=f'NREMP {idx + 1}')
        if "REMP" in cycle and cycle["REMP"]:
            remp = cycle["REMP"]
            plt.hlines(y=6, xmin=remp["start"] * (TIME_RESOLUTION_SEC / 60),
                       xmax=remp["end"] * (TIME_RESOLUTION_SEC / 60),
                       color=cycle_colors[idx % len(cycle_colors)], linestyle='-', linewidth=4, label=f'REMP {idx + 1}')

    plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["Wake", "N1", "N2", "N3", "REM", "NREMP", "REMP"])
    plt.gca().invert_yaxis()
    plt.title(f"Hypnogram for Patient {patient_id}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Sleep Stage")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Salva l'ipnogramma come file PNG nella cartella dei dati del paziente
    plt.savefig(os.path.join(output_folder, f"hypnogram_{patient_id}.png"))


def write_cycles_to_file(sleep_cycles, patient_id, output_file, total_length):
    """
    Write sleep cycle details to a text file, marking gaps as Not_Sleep.
    """
    with open(output_file, "a") as f:
        f.write(f"Patient {patient_id}\n")

        # Traccia gli indici coperti da NREMP o REMP
        covered_indices = set()
        for idx, cycle in enumerate(sleep_cycles):
            nremp = cycle["NREMP"]
            remp = cycle["REMP"]

            # Scrivi i dettagli di NREMP
            f.write(
                f"NREMP {idx + 1}: Start = {nremp['start']}, End = {nremp['end']}, Duration = {nremp['duration_min']} min\n")
            covered_indices.update(range(nremp["start"], nremp["end"] + 1))

            # Scrivi i dettagli di REMP se esiste
            if remp:
                f.write(
                    f"  REMP {idx + 1}: Start = {remp['start']}, End = {remp['end']}, Duration = {remp['duration_min']} min\n")
                covered_indices.update(range(remp["start"], remp["end"] + 1))

        # Identifica i periodi non coperti e segnalali come Not_Sleep
        not_sleep_intervals = []
        current_start = None

        for i in range(total_length):
            if i not in covered_indices:
                if current_start is None:
                    current_start = i
            elif current_start is not None:
                not_sleep_intervals.append((current_start, i - 1))
                current_start = None

        # Gestisci l'intervallo finale
        if current_start is not None:
            not_sleep_intervals.append((current_start, total_length - 1))

        # Scrivi i dettagli dei periodi Not_Sleep
        for start, end in not_sleep_intervals:
            duration_min = (end - start + 1) * (TIME_RESOLUTION_SEC / 60)
            f.write(f"Not_Sleep: Start = {start}, End = {end}, Duration = {duration_min:.2f} min\n")
        f.write("\n")


# File di output per i dettagli dei cicli
output_file = os.path.join(output_folder, "sleep_cycles_summary.txt")

# Assicuriamoci che il file sia vuoto all'inizio
if os.path.exists(output_file):
    os.remove(output_file)

# Processa i file dei dati dei pazienti
for file_name in os.listdir(input_folder):
    if file_name.endswith(".npy"):
        patient_id = file_name.split(".")[0]  # Assume the file name is the patient ID
        file_path = os.path.join(input_folder, file_name)

        # Load the sleep stages data (a numpy array)
        sleep_stages = np.load(file_path)

        # Identify the sleep cycles
        sleep_cycles = identify_sleep_cycles(sleep_stages)

        # Plot and save the hypnogram for the patient
        plot_hypnogram(sleep_stages, sleep_cycles, patient_id, output_folder)

        # Write the cycle details to the summary file
        write_cycles_to_file(sleep_cycles, patient_id, output_file, total_length=len(sleep_stages))
