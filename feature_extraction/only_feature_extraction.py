import os
import numpy as np
import mne
from feature_extraction import ChannelConnectivityFeatureExtractor, NetworkFeatureExtractor, SingleChannelFeatureExtractor
import time
from utilities import EEGRegionsDivider
import matplotlib.pyplot as plt


class EEGFeatureExtractor:
    """
    A class for processing EEG data, segmenting it into epochs, identifying and excluding bad epochs,
    extracting features, and saving the results.

    Parameters:
        data_path (str): Path to the directory containing EEG data.
        label_path (str): Path to the directory containing labels or annotations.
        save_path (str): Path to the directory for saving processed data and extracted features.
        only_stage (str, optional): Stage of interest for filtering epochs.
        only_class (str, optional): Class of interest for filtering.
        only_patient (str, optional): Specific patient identifier for filtering.
    """

    def __init__(self, data_path, label_path, save_path, only_stage=None, only_class=None, only_patient=None):
        self.data_path = data_path
        self.label_path = label_path
        self.save_path = save_path
        self.only_stage = only_stage
        self.only_class = only_class
        self.only_patient = only_patient

        # Initialize the EEG region divider
        self.divider = EEGRegionsDivider()
        self.regions = self.divider.get_all_regions()
        self.idx_chs = self.divider.get_index_channels()

        # Channel names based on region indices
        self.names = ['E' + str(idx_ch) for idx_ch in self.idx_chs]

    def process_and_save_features(self, sub_fold, preprocessed_raw):
        """
        Process EEG data by segmenting into epochs, identifying bad epochs, filtering,
        extracting features, and saving the results.

        Parameters:
            sub_fold (str): Path to the specific folder of the dataset being processed.
            preprocessed_raw (mne.io.Raw): Preprocessed raw EEG data object.
        """
        # Validate the input raw object
        if isinstance(preprocessed_raw, mne.io.Raw):
            print("The object 'raw' is a valid MNE Raw object.")

        # Step 1: Apply annotations (e.g., bad segments and muscle artifacts)
        #self.bad_epochs_muscle_flat(preprocessed_raw)

        # Step 2: Segment the data into fixed-length epochs (ignoring annotations)
        epochs = self.segment_epochs(preprocessed_raw, epoch_duration=30.0)
        # Ottieni i canali da utilizzare, ad esempio quelli selezionati da EEGRegionsDivider
        idx = self.idx_chs  # oppure un sottoinsieme se desiderato
        picks = mne.pick_channels(epochs.info['ch_names'],
                                  include=[epochs.ch_names[i] for i in idx if i < len(epochs.ch_names)])

        # Estrai le coordinate 3D dei canali selezionati
        sensor_loc = []
        for i in picks:
            loc = epochs.info['chs'][i]['loc'][:3]  # xyz coordinate
            if not np.allclose(loc, 0):  # Evita locazioni nulle
                sensor_loc.append(loc)
        sensor_loc = np.array(sensor_loc)


        # Add bad epochs based on annotations, error folder, or arousals
        self.add_bad_epochs(preprocessed_raw, bad_epochs_muscle_flat=True, use_error_folder=True, arousals=True,
                       bad_epochs_amplitude=True)

        # Filter epochs based on clinical scoring and exclude bad epochs
        filtered_epoched_data, selected_indices, clinical_scores = self.filter_epochs(epochs, preprocessed_raw)


        # Save the filtered epochs
        #self.save_filtered_epochs(filtered_epoched_data, selected_indices, clinical_scores)

        # Extract features from the filtered epochs

        #avg_features, feature_names, raw_features, channel_names, all_conmats, selected_channels, average_channels, specific_channels = self.extract_features(
        avg_features, feature_names, raw_features, channel_names, average_channels, specific_channels = self.extract_features(
            filtered_epoched_data, fs=preprocessed_raw.info['sfreq'], average_channels=False,
            specific_channels=[37, 26, 18, 54, 36, 15, 224, 1, 57, 43, 197, 204, 92, 74, 66, 90, 164, 192,
                               209, 85, 100, 129, 171, 105, 108, 126, 151, 177, 137, 135, 157, 46, 34,
                               12, 10, 48, 23, 6, 222, 51, 8, 196, 71, 79, 143, 181, 86, 162, 106, 117,
                               139, 169, 115, 116, 150, 159, 62, 211, 83, 191, 113, 176, 147])

        # Save the extracted features
        self.save_features(avg_features, feature_names, raw_features, channel_names, clinical_scores, selected_indices, sub_fold,average_channels, specific_channels)
        #
        '''# Poi:
        self.save_features(
            avg_features, feature_names, raw_features, channel_names,
            clinical_scores, selected_indices, sub_fold,
            average_channels, specific_channels,
            all_conmats=all_conmats,
            selected_channels=selected_channels
        )
        '''
        #

    def segment_epochs(self, preprocessed_raw, epoch_duration=30.0):
        """
        Segment EEG data into fixed-length epochs.

        Parameters:
            preprocessed_raw (mne.io.Raw): Preprocessed raw EEG data.
            epoch_duration (float): Duration of each epoch in seconds.
            reject_by_annotation (bool): Whether to reject epochs based on annotations.

        Returns:
            mne.Epochs: The segmented EEG data as epochs.
        """
        events = mne.make_fixed_length_events(raw=preprocessed_raw, duration=epoch_duration)
        epochs = mne.Epochs(
            raw=preprocessed_raw, events=events, tmin=0.0, tmax=epoch_duration,
            baseline=None, preload=True, verbose=False
        )
        print(f"Number of epochs created: {len(epochs)}")
        return epochs

    def bad_epochs_muscle_flat(self, raw=None, flat=1e-10, min_duration=0.8, threshold_muscle=4):
        """
        Identify bad epochs based on amplitude thresholds (flat) and muscle activity,
        log onset and duration of each epoch, and save combined indices in one file.

        Parameters:
        - raw: The MNE Raw object to annotate. If None, use self.raw.
        - flat: Minimum allowable peak-to-peak amplitude (in Volts).
        - min_duration: Minimum duration for bad segments (in seconds).
        - threshold_muscle: Z-score threshold for detecting muscle artifacts.

        Returns:
        - bad_epoch_indices: NumPy array with combined indices of bad epochs.
        """
        if raw is None:
            raw = self.raw

        print("Annotating based on flat and muscle activity")

        # --- Annotate based on amplitude (flat) ---
        annotations_flat, _ = mne.preprocessing.annotate_amplitude(
            raw,
            flat=flat,
            min_duration=min_duration,
            picks="eeg"
        )

        print("Flat-based annotations:")
        print(annotations_flat)

        # Filter flat annotations with duration > 0.5 seconds
        bad_flat_epochs = []  # To log onset and duration
        bad_flat_epoch_indices = []  # To save only indices

        for onset, duration, description in zip(annotations_flat.onset, annotations_flat.duration,
                                                annotations_flat.description):
            if duration > 0.5:
                epoch_index = int(onset // 30)  # Assuming epoch duration = 30 seconds
                bad_flat_epochs.append([onset, duration, description])
                bad_flat_epoch_indices.append(epoch_index)

        # --- Annotate based on muscle activity ---
        annot_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
            raw,
            ch_type="eeg",
            threshold=threshold_muscle,
            min_length_good=0.1,
            filter_freq=[50, 63],
        )

        print("Muscle activity-based annotations:")
        print(annot_muscle)

        # Filter muscle annotations with duration > 0.5 seconds
        bad_muscle_epochs = []  # To log onset and duration
        bad_muscle_epoch_indices = []  # To save only indices

        for onset, duration, description in zip(annot_muscle.onset, annot_muscle.duration, annot_muscle.description):
            if duration > 0.5:
                epoch_index = int(onset // 30)  # Assuming epoch duration = 30 seconds
                bad_muscle_epochs.append([onset, duration, epoch_index, description])  # Include epoch index
                bad_muscle_epoch_indices.append(epoch_index)

        # --- Combine and Deduplicate Epoch Indices ---
        combined_epoch_indices = list(set(bad_flat_epoch_indices + bad_muscle_epoch_indices))
        combined_epoch_indices.sort()  # Ensure indices are in ascending order
        combined_epoch_indices = np.array(combined_epoch_indices, dtype=int)

        # --- Log Detailed Information ---
        print("Filtered bad flat epochs with duration > 0.5 seconds:")
        for epoch in bad_flat_epochs:
            print(f"Onset: {epoch[0]:.2f}s, Duration: {epoch[1]:.2f}s, Epoch Index: {int(epoch[0] // 30)}, Description: {epoch[2]}")

        print("Filtered bad muscle epochs with duration > 0.5 seconds:")
        for epoch in bad_muscle_epochs:
            print(f"Onset: {epoch[0]:.2f}s, Duration: {epoch[1]:.2f}s, Epoch Index: {epoch[2]}, Description: {epoch[3]}")

        # --- Save Combined Epoch Indices ---
        export_dir = os.path.join(
            self.label_path,
            "clinical_scorings",
            self.only_class,
            self.only_patient
        )
        os.makedirs(export_dir, exist_ok=True)

        combined_file = os.path.join(export_dir, f"{self.only_patient}EEG_bad_muscle_flat.npy")
        np.save(combined_file, combined_epoch_indices)

        print(f"Combined bad epoch indices saved to {combined_file}")

        # Return the combined epoch indices
        return combined_epoch_indices

    def add_bad_epochs(self, preprocessed_raw, bad_epochs_muscle_flat=True, use_error_folder=True, arousals=True,
                       bad_epochs_amplitude=True):
        """
        Identify bad epochs based on muscle/flat annotations, error folders, arousals file, or amplitude-based bad epochs
        and mark them.

        Parameters:
            preprocessed_raw (mne.io.Raw): Preprocessed raw EEG data.
            bad_epochs_muscle_flat (bool): Whether to use muscle/flat-based bad epochs (loaded from EEG_bad_muscle_flat.npy).
            use_error_folder (bool): Whether to use error folder for identifying bad epochs.
            arousals (bool): Whether to use arousals file for identifying bad epochs.
            bad_epochs_amplitude (bool): Whether to use amplitude-based bad epochs (loaded from EEG_bad_epochs.npy).
        """
        bad_epochs = []

        # Add bad epochs from muscle/flat annotations if requested
        if bad_epochs_muscle_flat:
            print("Using muscle/flat-based annotations to determine bad epochs...")
            bad_epochs_file = os.path.join(
                self.label_path, 'clinical_scorings', self.only_class, self.only_patient,
                f'{self.only_patient}EEG_bad_muscle_flat.npy'
            )
            if os.path.exists(bad_epochs_file):
                try:
                    bad_epochs_muscle_flat_data = np.load(bad_epochs_file)
                    bad_epochs.extend(bad_epochs_muscle_flat_data.tolist())
                    print(
                        f"Bad epochs from muscle/flat annotations: {bad_epochs_muscle_flat_data}" if bad_epochs_muscle_flat_data.size > 0 else "No bad epochs found in muscle/flat annotations.")
                except Exception as e:
                    print(f"Error while loading muscle/flat annotations file: {e}")

        # Add bad epochs from the error folder if requested
        if use_error_folder:
            print("Using error folder subdirectories to determine bad epochs...")
            errors_dir = os.path.join(self.save_path, "PLOT", self.only_class, self.only_patient,
                                      "ICA0.8_individual_epochs", "errors")
            if os.path.exists(errors_dir):
                for subdir in os.listdir(errors_dir):
                    subdir_path = os.path.join(errors_dir, subdir)
                    if os.path.isdir(subdir_path) and subdir.startswith("Epoch_"):
                        try:
                            epoch_number = int(subdir.split("_")[1])
                            if epoch_number not in bad_epochs:
                                bad_epochs.append(epoch_number)
                        except ValueError:
                            print(f"Invalid subdirectory name: {subdir}. Could not extract the epoch number.")
                print(
                    f"Bad epochs from error folder: {bad_epochs}" if bad_epochs else "No bad epochs found in error folder.")

        # Add bad epochs from arousals file if requested
        if arousals:
            print("Using arousals file to determine bad epochs...")
            clinical_scoring_path = os.path.join(
                self.label_path, 'clinical_scorings', self.only_class, self.only_patient,
                f'{self.only_patient}EEG_arousals.npy'
            )
            if os.path.exists(clinical_scoring_path):
                try:
                    epoch_classifications = np.load(clinical_scoring_path)
                    bad_epochs_arousals = np.where(epoch_classifications == 1)[0]
                    bad_epochs.extend(bad_epochs_arousals.tolist())
                    print(
                        f"Bad epochs from arousals file: {bad_epochs_arousals}" if bad_epochs_arousals.size > 0 else "No bad epochs found in arousals file.")
                except Exception as e:
                    print(f"Error while loading arousals file: {e}")

        # Add bad epochs from amplitude-based file if requested
        if bad_epochs_amplitude:
            print("Using amplitude-based bad epochs from EEG_bad_epochs.npy...")
            bad_epochs_file = os.path.join(
                self.label_path, 'clinical_scorings', self.only_class, self.only_patient,
                f'{self.only_patient}EEG_bad_epochs.npy'
            )
            if os.path.exists(bad_epochs_file):
                try:
                    bad_epochs_amplitude_data = np.load(bad_epochs_file)
                    bad_epochs.extend(bad_epochs_amplitude_data.tolist())
                    print(
                        f"Bad epochs from amplitude file: {bad_epochs_amplitude_data}" if bad_epochs_amplitude_data.size > 0 else "No bad epochs found in amplitude file.")
                except Exception as e:
                    print(f"Error while loading amplitude bad epochs file: {e}")

        # Update the raw object with the list of bad epochs
        if bad_epochs:
            preprocessed_raw._bad_epochs = sorted(set(bad_epochs))
            print(f"Bad epochs added to raw._bad_epochs: {preprocessed_raw._bad_epochs}")
        else:
            print("No bad epochs found.")

    def filter_epochs(self, epochs, preprocessed_raw):
        """
        Filter epochs based on clinical scoring and exclude bad epochs.

        Parameters:
            epochs (mne.Epochs): The segmented EEG data.
            preprocessed_raw (mne.io.Raw): The raw EEG data.

        Returns:
            tuple: Filtered epochs, indices of the selected epochs, and clinical scores.
        """
        clinical_scoring_path = os.path.join(self.label_path, 'clinical_scorings', self.only_class, self.only_patient,
                                             f'{self.only_patient}EEG_stages.npy')
        print('clinical_scoring_path:', clinical_scoring_path)

        # Check if the scoring file exists
        if not os.path.exists(clinical_scoring_path):
            raise FileNotFoundError(f"Clinical scoring file not found: {clinical_scoring_path}")

        # Load clinical scores
        clinical_scores = np.load(clinical_scoring_path)

        '''
        # Validate matching lengths
        if len(clinical_scores) - 1 != len(epochs):
            raise ValueError(
                f"Mismatch between number of epochs ({len(epochs)}) and clinical scoring entries ({len(clinical_scores)})")

        '''

        # Select indices of epochs with clinical scores '2' or '3' (N2 or N3)
        selected_indices = np.where(np.isin(clinical_scores, [2, 3]))[0]
        #selected_indices = np.where(np.isin(clinical_scores, [3]))[0]
        print("Selected indices (clinical scores '3'):", selected_indices)

        # Exclude bad epochs if they are marked in the raw object
        if hasattr(preprocessed_raw, '_bad_epochs'):
            bad_epochs = set(preprocessed_raw._bad_epochs)
            selected_indices = [idx for idx in selected_indices if idx not in bad_epochs]
            print("Selected indices after excluding bad epochs:", selected_indices)

        # Filter epochs based on the selected indices
        filtered_epoched_data = epochs[selected_indices]
        print(f"Selected {len(selected_indices)} epochs with clinical scores '3', excluding bad epochs.")

        return filtered_epoched_data, selected_indices, clinical_scores

    def save_filtered_epochs(self, filtered_epoched_data, selected_indices, clinical_scores):
        """
        Save filtered EEG epochs and their corresponding visualizations in separate folders for N2 and N3.

        Parameters:
            filtered_epoched_data (mne.Epochs): The EEG epochs that passed the filtering criteria.
            selected_indices (list): The indices of the selected epochs (corresponding to the original epochs).
            clinical_scores (np.ndarray): The clinical scores for the epochs.

        """
        # Define the channels to visualize
        channels_to_plot = ['E26', 'E15', 'E81', 'E101','E137']

        # Define the base path for saving both plots and epochs
        save_base_path = os.path.join(self.save_path, "PLOT", self.only_class, self.only_patient)

        # Create separate folders for N2 and N3
        n3_save_path = os.path.join(save_base_path, "N3_0.5Hz_EPOCHS_SELECTED")
        n2_save_path = os.path.join(save_base_path, "N2_0.5Hz_EPOCHS_SELECTED")

        os.makedirs(n3_save_path, exist_ok=True)
        os.makedirs(n2_save_path, exist_ok=True)

        def plot_and_save(epoch_data, title, save_path, channels_to_plot):
            """
            Helper function to create plots for the given epoch and save them to disk.

            Parameters:
                epoch_data (mne.Epochs): Data for a single epoch to plot.
                title (str): Title of the plot.
                save_path (str): Path where the plot will be saved.
                channels_to_plot (list): List of EEG channels to include in the plot.
            """
            time = epoch_data.times  # Time axis in seconds

            # Create subplots for each channel
            fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(12, 8), sharex=True)
            for ch_idx, ax in enumerate(axes):
                ch_name = channels_to_plot[ch_idx]
                if ch_name in epoch_data.info['ch_names']:
                    ch_idx_in_data = epoch_data.info['ch_names'].index(ch_name)
                    ch_data = epoch_data.get_data()[:, ch_idx_in_data, :]  # Extract channel data
                    ax.plot(time, ch_data[0].T)
                    ax.set_title(f"Channel {ch_name}")
                    ax.set_ylabel("Amplitude (µV)")

            # Add labels and save the figure
            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(title, fontsize=16)
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved plot to: {save_path}")

            # Save the epoch data as .npy
            npy_save_path = save_path.replace('.png', '.npy')
            np.save(npy_save_path, epoch_data.get_data())
            print(f"Saved epoch data as .npy to: {npy_save_path}")

        # Save each epoch
        for idx, epoch_idx in enumerate(selected_indices):
            epoch_data = filtered_epoched_data.get_data()[idx]
            clinical_score = clinical_scores[epoch_idx]  # Get the clinical score for this epoch

            # Generate and save the plot for this epoch
            if clinical_score == 3:
                # Save to N3 folder
                png_save_path = os.path.join(n3_save_path, f"epoch_{epoch_idx}.png")
                title = f"Epoch {epoch_idx} (N3)"
                plot_and_save(filtered_epoched_data[idx], title, png_save_path, channels_to_plot)
            elif clinical_score == 2:
                # Save to N2 folder
                png_save_path = os.path.join(n2_save_path, f"epoch_{epoch_idx}.png")
                title = f"Epoch {epoch_idx} (N2)"
                plot_and_save(filtered_epoched_data[idx], title, png_save_path, channels_to_plot)

    def extract_features(self, filtered_epoched_data, fs, average_channels=False,
                         specific_channels=None):
        """
        Extracts EEG features from filtered epochs.

        Parameters:
        - filtered_epoched_data (mne.Epochs): The filtered EEG epochs.
        - fs (int): Sampling frequency.
        - average_channels (bool): If True, calculates features averaged across channels.
        - specific_channels (list): List of specific channels to analyze (if None, analyzes all channels).

        Returns:
        - avg_features (np.ndarray): Matrix of averaged features (if requested).
        - feature_names (list): Names of the extracted features.
        - raw_features (np.ndarray): Matrix of raw features for each channel.
        - channel_names (list): Names of the channels analyzed.
        - average_channels (bool): Parameter used for feature extraction.
        - specific_channels (list): List of channels used for feature extraction.
        """
        print("Extracting features...")
        processing_times = {}
        start_time = time.time()

        # Create the feature extractor
        sc_extractor = SingleChannelFeatureExtractor(
            epochs=filtered_epoched_data,
            fs=fs,
            ch_reg=sorted(self.regions),
            save_path=self.save_path,
            only_stage=self.only_stage,
            only_patient=self.only_patient
        )

        # Extract features
        avg_features, feature_names, raw_features, channel_names = sc_extractor.extract_features(
            average_channels=average_channels,
            specific_channels=specific_channels
        )
        '''
        # --- Channel Connectivity Feature Extraction ---
        start_time = time.time()
        cc_extractor = ChannelConnectivityFeatureExtractor(
            epochs=filtered_epoched_data,
            fs=fs,
            ch_reg=sorted(self.regions),
            save_path=self.save_path,
            only_stage=self.only_stage,
            only_patient=self.only_patient,
            only_class = self.only_class
        )

        avg_features, feature_names, raw_features, channel_names, average_channels, specific_channels, all_conmats, selected_channels = cc_extractor.extract_features_c(
            average_channels=average_channels,
            specific_channels=specific_channels
        )
        '''
        # Calculate processing time
        processing_times['Single-Channel Feature Extraction'] = time.time() - start_time
        print(f"Feature extraction completed in {processing_times['Single-Channel Feature Extraction']} seconds.")

        return avg_features, feature_names, raw_features, channel_names, average_channels, specific_channels
        #return avg_features, feature_names, raw_features, channel_names, all_conmats, selected_channels, average_channels, specific_channels



    def save_features(self, avg_features, feature_names, raw_features, channel_names, clinical_scores, selected_indices, sub_fold,
                      average_channels, specific_channels):
        """
        Saves extracted features to `.npz` files, adapting filenames based on provided parameters.

        Parameters:
        - avg_features (np.ndarray): Matrix of averaged features.
        - feature_names (list): Names of the extracted features.
        - raw_features (np.ndarray): Matrix of raw features (per channel).
        - channel_names (list): Names of the channels used.
        - selected_indices (list): Indices of the selected epochs.
        - sub_fold (str): Sub-folder for saving the files.
        - average_channels (bool): Indicates whether features are averaged.
        - specific_channels (list): List of specific channels used.
        """
        # Identify brain regions from the given regions
        brain_regions = [reg.split('_')[1] for reg in sorted(self.regions)]

        # Create the save directory
        res_sub_fold = sub_fold.replace(self.data_path, os.path.join(self.save_path, 'Features'))
        os.makedirs(res_sub_fold, exist_ok=True)

        # Build the filename suffix based on parameters
        avg_suffix = "_mean" if average_channels else "_no_mean"
        specific_suffix = f"_specific_channels_{len(specific_channels)}" if specific_channels else "_all_channels"

        # Paths for saving files
        avg_save_path = os.path.join(
            res_sub_fold,
            os.path.basename(res_sub_fold) + f'{avg_suffix}_N2_N3_SWA_ENTROPY{specific_suffix}.npz'
        )
        electrode_save_path = os.path.join(
            res_sub_fold,
            os.path.basename(res_sub_fold) + f'_N3_SWA_ENTROPY{specific_suffix}.npz'
        )

        # Determine whether to save brain regions or specific channels
        avg_file_metadata = {"data": avg_features, "feats": feature_names}

        if not average_channels and specific_channels:
            avg_file_metadata["channels"] = specific_channels
        else:
            avg_file_metadata["regions"] = brain_regions

        avg_file_metadata["selected_indices"] = selected_indices

        # Aggiungi le etichette delle classi cliniche
        epoch_labels = clinical_scores[selected_indices]
        avg_file_metadata["epoch_labels"]=epoch_labels

        # Save averaged features
        np.savez(
            avg_save_path,
            **avg_file_metadata
        )
        print(f"Mean features saved for {sub_fold} in {avg_save_path}")

        # Save channel-specific features
        np.savez(
            electrode_save_path,
            data=raw_features,
            feats=feature_names,
            regions=brain_regions,
            channels=channel_names,
            selected_indices=selected_indices,
            epoch_labels=epoch_labels
        )
        print(f"Electrode features saved for {sub_fold} in {electrode_save_path}")

'''
    def save_features(self, avg_features, feature_names, raw_features, channel_names,
                      clinical_scores, selected_indices, sub_fold,
                      average_channels, specific_channels,
                      all_conmats=None, selected_channels=None):
        """
        Save extracted features and optionally connectivity matrices.
        """

        brain_regions = [reg.split('_')[1] for reg in sorted(self.regions)]
        res_sub_fold = sub_fold.replace(self.data_path, os.path.join(self.save_path, 'Features'))
        os.makedirs(res_sub_fold, exist_ok=True)

        avg_suffix = "_mean" if average_channels else "_no_mean"
        specific_suffix = f"_specific_channels_{len(specific_channels)}" if specific_channels else "_all_channels"

        avg_save_path = os.path.join(
            res_sub_fold,
            os.path.basename(res_sub_fold) + f'{avg_suffix}_N3conn100{specific_suffix}.npz'
        )
        electrode_save_path = os.path.join(
            res_sub_fold,
            os.path.basename(res_sub_fold) + f'_N3conn100{specific_suffix}.npz'
        )

        # Metadata per avg features
        avg_file_metadata = {
            "data": avg_features,
            "feats": feature_names,
            "selected_indices": selected_indices,
            "epoch_labels": clinical_scores[selected_indices]
        }

        if not average_channels and specific_channels:
            avg_file_metadata["channels"] = specific_channels
        else:
            avg_file_metadata["regions"] = brain_regions

        # Salvataggio features aggregate
        np.savez(avg_save_path, **avg_file_metadata)
        print(f"Mean features saved for {sub_fold} in {avg_save_path}")

        # Metadata per raw features
        raw_file_metadata = {
            "data": raw_features,
            "feats": feature_names,
            "regions": brain_regions,
            "channels": channel_names,
            "selected_indices": selected_indices,
            "epoch_labels": clinical_scores[selected_indices]
        }

        # Se hai matrici di connettività e canali, puoi salvarle insieme
        if all_conmats is not None:
            raw_file_metadata["connectivity_matrices"] = np.array(all_conmats)
        if selected_channels is not None:
            raw_file_metadata["connectivity_channels"] = selected_channels

        np.savez(electrode_save_path, **raw_file_metadata)
        print(f"Electrode features saved for {sub_fold} in {electrode_save_path}")

'''
