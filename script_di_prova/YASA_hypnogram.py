import mne
import yasa
import numpy as np
from scipy.signal import detrend
import matplotlib.pyplot as plt
from utilities.hypnogram import hypnogram_definition
from lidpd_main import get_args

file_path = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/PD012_cropped.fif"
raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
raw_e8 = raw.copy().pick_channels(['E8'])  # Select the E8 channel

# Apply detrend and remove the mean value
data, times = raw_e8[:]
data_detrended = detrend(data, axis=1)

# Create a new Raw object with the detrended data
info = raw_e8.info
raw_e8_detrended = mne.io.RawArray(data_detrended, info)

# Apply band-pass filter (0.5-40 Hz)
raw_e8_detrended.filter(l_freq=0.35, h_freq=40, method='fir', fir_window='hamming', fir_design='firwin', verbose=False)

# Create 30-second epochs
epoch_duration = 30  # Duration of the epochs in seconds
events = mne.make_fixed_length_events(raw_e8_detrended, duration=epoch_duration)
epochs = mne.Epochs(raw_e8_detrended, events, event_id=None, tmin=0, tmax=epoch_duration, baseline=None, preload=True)
##############################################################################################
# Use YASA for automatic sleep stage classification
sls = yasa.SleepStaging(raw_e8_detrended, eeg_name='E8')

# Perform the classification
hypno = sls.predict()  # The output should be similar to the one you provided

# Map labels to integer values
stage_labels = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}
hypno_int = np.array([stage_labels[label] for label in hypno])

# Plot the hypnogram
fig, ax = plt.subplots(figsize=(10, 4))
yasa.plot_hypnogram(hypno_int, ax=ax)
plt.title("Sleep Stages Hypnogram")
plt.xlabel("Epochs (30s)")
plt.ylabel("Sleep Stage")
plt.show()

# Print the hypnogram (string values)
print("Hypnogram (string values):", hypno)

# Count how many times each sleep stage appears
unique_stages, counts = np.unique(hypno, return_counts=True)
stage_counts = dict(zip(unique_stages, counts))

print("Sleep stage counts:")
for stage, count in stage_counts.items():
    print(f"{stage}: {count}")

args = get_args()
hypnogram_definition(args.label_path, args.run_hypnogram)

#############################################################################################
#####################   Comparison between YASA and Labels   ################################
#############################################################################################
#############################################################################################
# Path of the file with the hypnogram from hypnogram_definition
hypnogram_file_path = "D:/TESI/lid-data-samples/lid-data-samples/Labels/DYS/PD012.npy"
# Load the hypnogram produced by hypnogram_definition
hypnogram_from_file = np.load(hypnogram_file_path)

# Check if hypno_int is defined
if 'hypno_int' not in locals():
    raise ValueError("The YASA hypnogram (hypno_int) is not defined.")

total_samples = len(hypnogram_from_file)
print(total_samples)

# Minimum and maximum time chosen by the user in seconds
time_min = 3 * 3600     # Insert minimum time in seconds
time_max = 3.5 * 3600   # Insert maximum time in seconds (e.g., 3600 seconds = 1 hour)

# Calculate the indices corresponding to the minimum and maximum time
index_min = int(time_min / 30) -1  # 30 seconds per epoch
index_max = int(time_max / 30) -1

hypnogram_from_file_truncated = hypnogram_from_file[index_min:index_max]

#######################################################################################################
# Comparison
comparison = hypnogram_from_file_truncated == hypno_int

# Calculate the percentage of similarity
similarity_percentage = np.mean(comparison) * 100

# Visualize the results
plt.figure(figsize=(12, 6))

# Plot the hypnogram from hypnogram_definition
plt.subplot(2, 1, 1)
plt.plot(hypnogram_from_file_truncated, label='Hypnogram from hypnogram_definition', color='blue')
plt.title('Hypnogram from hypnogram_definition')
plt.xlabel('Epochs (30s)')
plt.ylabel('Sleep Stage')
plt.xticks(ticks=np.arange(len(hypnogram_from_file_truncated)), labels=np.arange(len(hypnogram_from_file_truncated)))
plt.yticks(ticks=[0, 1, 2, 3, 4], labels=['Awake', 'N1', 'N2', 'N3', 'REM'])
plt.legend()

# Plot the hypnogram from YASA
plt.subplot(2, 1, 2)
plt.plot(hypno_int, label='Hypnogram from YASA', color='orange')
plt.title('Hypnogram from YASA')
plt.xlabel('Epochs (30s)')
plt.ylabel('Sleep Stage')
plt.xticks(ticks=np.arange(len(hypno_int)), labels=np.arange(len(hypno_int)))
plt.yticks(ticks=[0, 1, 2, 3, 4], labels=['Awake', 'N1', 'N2', 'N3', 'REM'])
plt.legend()

plt.tight_layout()
plt.show()

# Print the similarity percentage
print(f"Percentage of concordance between the hypnograms: {similarity_percentage:.2f}%")


