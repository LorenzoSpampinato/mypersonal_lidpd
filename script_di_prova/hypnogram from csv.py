import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SleepDataProcessor:
    def __init__(self, csv_file_path, npy_save_path, plot_save_path=None, sampling_rate=250, epoch_duration=30, process_type="stages"):
        self.csv_file_path = csv_file_path
        self.npy_save_path = npy_save_path
        self.plot_save_path = plot_save_path  # Solo per gli stages
        self.sampling_rate = sampling_rate
        self.epoch_duration = epoch_duration
        self.process_type = process_type  # "stages" o "arousals"

    def process_and_plot(self):
        if self.process_type == "stages":
            epoch_scores = self.process_data()
            self.plot_hypnogram(epoch_scores)
        elif self.process_type == "arousals":
            self.process_arousals()

    def process_data(self):
        """Elabora i dati degli stages"""
        samples_per_epoch = self.sampling_rate * self.epoch_duration

        data = pd.read_csv(self.csv_file_path, header=None)
        scores = data.iloc[:, 0]

        epochs = []
        for epoch_index, i in enumerate(range(0, len(scores), samples_per_epoch)):
            epoch_data = scores[i:i + samples_per_epoch]
            if not epoch_data.empty:
                unique_values = epoch_data.unique()
                if len(unique_values) > 1:
                    print(f"Epoch {epoch_index}: Different values found {unique_values}")
                epochs.append(epoch_data.iloc[0])

        epoch_scores = np.array(epochs)
        np.save(self.npy_save_path, epoch_scores)
        print(f"Stages saved: {self.npy_save_path}")

        return epoch_scores

    def process_arousals(self):
        """Elabora i dati degli arousals"""
        samples_per_epoch = self.sampling_rate * self.epoch_duration

        data = pd.read_csv(self.csv_file_path, header=None)
        arousals = data.iloc[:, 0]

        arousal_epochs = []
        for epoch_index, i in enumerate(range(0, len(arousals), samples_per_epoch)):
            epoch_data = arousals[i:i + samples_per_epoch]
            if not epoch_data.empty:
                arousal_epochs.append(int(epoch_data.max()))  # 1 se c'è un arousal, altrimenti 0

        arousal_epochs = np.array(arousal_epochs)
        np.save(self.npy_save_path, arousal_epochs)
        print(f"Arousals saved: {self.npy_save_path}")

    def plot_hypnogram(self, epoch_scores):
        """Genera il grafico dell'ipnogramma solo per gli stages"""
        if self.plot_save_path is None:
            return

        sleep_stage_labels = {
            0: "Awake",
            1: "NREM 1",
            2: "NREM 2",
            3: "NREM 3",
            5: "REM"
        }

        time_in_hours = np.arange(len(epoch_scores)) * (30 / 60 / 60)

        plt.figure(figsize=(12, 6))
        plt.step(time_in_hours, epoch_scores, where='mid', label='Sleep State', color='blue')

        plt.yticks(list(sleep_stage_labels.keys()), list(sleep_stage_labels.values()))
        plt.xlabel("Time (hours)")
        plt.ylabel("Sleep State")
        plt.title("Hypnogram")

        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig(self.plot_save_path)
        print(f"Plot saved: {self.plot_save_path}")
        plt.close()


# Percorso della cartella con i file CSV
input_folder = r"C:\Users\Lorenzo\Desktop\sleep_stages_09_12\new"

# Processa i file _stages.csv
for file_name in os.listdir(input_folder):
    if file_name.endswith("_stages.csv"):
        csv_file_path = os.path.join(input_folder, file_name)
        print(f"Processing STAGES file: {csv_file_path}")

        base_name = os.path.splitext(file_name)[0]
        npy_save_path = os.path.join(input_folder, f"{base_name}.npy")
        plot_save_path = os.path.join(input_folder, f"{base_name}_hypnogram.png")

        processor = SleepDataProcessor(csv_file_path, npy_save_path, plot_save_path, process_type="stages")
        processor.process_and_plot()

# Processa i file _arousals.csv
for file_name in os.listdir(input_folder):
    if file_name.endswith("_arousals.csv"):
        csv_file_path = os.path.join(input_folder, file_name)
        print(f"Processing AROUSALS file: {csv_file_path}")

        base_name = os.path.splitext(file_name)[0]
        npy_save_path = os.path.join(input_folder, f"{base_name}.npy")

        processor = SleepDataProcessor(csv_file_path, npy_save_path, process_type="arousals")
        processor.process_and_plot()
