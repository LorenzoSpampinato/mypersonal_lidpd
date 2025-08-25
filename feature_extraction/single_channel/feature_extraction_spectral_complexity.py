import numpy as np
from scipy.signal import welch
import antropy as ant
from utilities import EEGRegionsDivider
import EntropyHub as EH
from pyentrp import entropy as ent
import nolds
import time
import numpy as np
from mne.time_frequency import psd_array_multitaper
import mne
import antropy as ant
import time


class SingleChannelFeatureExtractor:
    def __init__(self, epochs, fs, ch_reg, save_path=None, only_stage=None, only_patient=None):
        self.epochs = epochs
        self.fs = fs
        self.ch_reg = ch_reg

        self.feats = [
            # PSD
            "Delta Power (Mean)",
            "Delta Power (Relative)",
            "Total Power (Mean)",

            # Entropy
            "Correlation Dimension",
            "Katz Fractal Dimension",

            # Multiscale Entropy - SampEn
            "SampEn Scale 1",
            "SampEn Scale 2",
            "SampEn Scale 3",
            "SampEn CI",

            # Multiscale Entropy - SpecEn
            "SpecEn Scale 1",
            "SpecEn Scale 2",
            "SpecEn Scale 3",
            "SpecEn CI"
        ]

        self.features_matrix = np.zeros([len(self.ch_reg), len(self.feats), len(self.epochs)])
        self.save_path = save_path
        self.only_stage = only_stage
        self.only_patient = only_patient
        self.divider = EEGRegionsDivider()
        self.regions = self.divider.get_all_regions()
        self.idx_chs = self.divider.get_index_channels()
        self.names = ['E' + str(idx_ch) for idx_ch in self.idx_chs]
        self.all_channels_features_matrix = []

    def extract_features(self, average_channels=False, specific_channels=None):
        def _compute_psd_features(psds, freqs):
            bands = { "SWA": (0.5, 4.0)
            }

            mean_psd_powers = [
                np.sum(psds[:, (freqs >= low) & (freqs <= high)], axis=-1) / np.sum((freqs >= low) & (freqs <= high))
                for _, (low, high) in bands.items()
            ]
            total_power = np.sum(psds[:, (freqs >= 0.5) & (freqs <= 40.0)], axis=-1)
            rel_powers = [
                np.sum(psds[:, (freqs >= low) & (freqs <= high)], axis=-1) / total_power
                for _, (low, high) in bands.items()
            ]
            mean_power = np.mean(psds[:, (freqs >= 0.5) & (freqs <= 40.0)], axis=-1)

            return np.array(mean_psd_powers + rel_powers + [mean_power])

        def _compute_complexity_features(x_sig_30):
            n_epochs = x_sig_30.shape[0]
            feats = np.zeros((2, n_epochs))

            for i, epoch in enumerate(x_sig_30):
                feats[0, i] = ant.katz_fd(epoch)
                feats[1, i] = nolds.corr_dim(epoch, emb_dim=3, lag=2)

            return feats  # shape (2, n_epochs)

        def _compute_multiscale_entropy(x_sig_30):
            ms_entropy_methods = [
                ('SampEn', {'m': 3, 'tau': 1}),
                ('SpecEn', {'Freqs': (0.5 / 64, 35 / 64)})
            ]

            entropy_results = []
            for entropy_name, params in ms_entropy_methods:
                Mobj = EH.MSobject(EnType=entropy_name, **params)
                entropy_values, CI_values = [], []

                for epoch in x_sig_30:
                    entropy_per_scale, CI = EH.MSEn(epoch, Mobj, Scales=3, Methodx='coarse')
                    entropy_values.append(entropy_per_scale[:3])  # Scala 1,2,3
                    CI_values.append(CI)

                entropy_values = np.array(entropy_values).T  # (3, n_epochs)
                CI_values = np.array(CI_values).reshape(1, -1)  # Ensure (1, n_epochs)
                entropy_results.append(entropy_values)
                entropy_results.append(CI_values)

            return np.concatenate(entropy_results, axis=0)  # Tutti (feature, epoche)

        def _prepare_channels():
            if average_channels:
                selected_channels = set()
                for region in self.regions:
                    region_channels = region.split('=')[1].strip().split(', ')
                    valid_channels = [ch for ch in region_channels if ch in self.names]
                    selected_channels.update(valid_channels)
                return list(selected_channels)
            else:
                return specific_channels if specific_channels else self.names

        selected_channels = _prepare_channels()
        epochs_chs = self.epochs.copy().pick(picks=selected_channels, verbose=False)
        features_matrix = np.zeros([len(selected_channels), len(self.feats), len(epochs_chs)])
        #features_matrix = np.zeros([len(selected_channels), len(self.feats), 2])
        channel_names = selected_channels
        all_channels_features_matrix = []

        for nch, ch in enumerate(selected_channels):
            x_sig_30 = epochs_chs.get_data()[:, nch, :]

            psds = np.array([
                psd_array_multitaper(epoch, sfreq=self.fs, fmin=0.5, fmax=35, adaptive=True,
                                     normalization='full', verbose=False)[0]
                for epoch in x_sig_30
            ])
            freqs = psd_array_multitaper(x_sig_30[0], sfreq=self.fs, fmin=0.5, fmax=35,
                                         adaptive=True, normalization='full', verbose=False)[1]

            psd_features = _compute_psd_features(psds, freqs)
            features_matrix[nch, :3, :] = psd_features

            features_matrix[nch, 3:5, :] = _compute_complexity_features(x_sig_30)

            start_time = time.time()
            features_matrix[nch, 5:, :] = _compute_multiscale_entropy(x_sig_30)
            print(f"[{ch}] Tempo _compute_multiscale_entropy: {time.time() - start_time:.2f}s")

            all_channels_features_matrix.append(features_matrix[nch, :, :])

        all_channels_features_matrix = np.stack(all_channels_features_matrix, axis=0)

        if average_channels:
            region_matrices = []
            for region in self.regions:
                region_channels = region.split('=')[1].strip().split(', ')
                valid_channels = [ch for ch in region_channels if ch in channel_names]
                if valid_channels:
                    idxs = [channel_names.index(ch) for ch in valid_channels]
                    region_matrix = np.mean(features_matrix[idxs, :, :], axis=0)
                    region_matrices.append(region_matrix)
            if region_matrices:
                features_matrix = np.stack(region_matrices, axis=0)

        return features_matrix, self.feats, all_channels_features_matrix, channel_names
