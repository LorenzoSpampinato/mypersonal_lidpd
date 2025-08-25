import os
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs
import networkx as nx
import matplotlib.pyplot as plt
from scipy import linalg
from utilities import EEGRegionsDivider


class ChannelConnectivityFeatureExtractor:
    def __init__(self, epochs, fs, ch_reg, save_path=None, only_stage=None, only_patient=None, only_class=None):
        self.epochs = epochs
        self.fs = fs
        self.ch_reg = ch_reg
        self.save_path = save_path
        self.only_stage = only_stage
        self.only_patient = only_patient
        self.only_class = only_class
        self.divider = EEGRegionsDivider()
        self.regions = self.divider.get_all_regions()
        self.idx_chs = self.divider.get_index_channels()
        self.names = ['E' + str(idx_ch) for idx_ch in self.idx_chs]
        self.feats = ['imaginary coherence']
        self.bands = {
            "delta": [0.5, 4.0],
            #"theta": [4.0, 8.0],
            #"alpha": [8.0, 13.0],
            #"sigma": [13.0, 15.0],
            #"beta": [15.0, 30.0],
            #"gamma": [30.0, 49.0]
        }

    def extract_features_c(self, average_channels=False, specific_channels=None, min_dist=0):
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
        info = epochs_chs.info

        epochs = len(epochs_chs)
        fmin = tuple([v[0] for v in self.bands.values()])
        fmax = tuple([v[1] for v in self.bands.values()])
        method = 'imcoh'

        data = epochs_chs.get_data()
        ch_names = epochs_chs.ch_names

        # === Prepara struttura per feature: nodali e globali ===
        self.feats = ["imcoh", "degree", "strength", "betweenness", "clustering", "glob_eff", "avg_path"]
        features_matrix = np.zeros((len(selected_channels), len(self.feats), epochs))
        all_conmats = []

        for i in range(epochs):
            epoch_data = data[i]  # shape (n_channels, n_times)
            single_epoch = mne.EpochsArray(epoch_data[np.newaxis, ...], info, verbose=False)

            con = spectral_connectivity_epochs(
                single_epoch, method=method, mode="multitaper", sfreq=self.fs,
                fmin=fmin, fmax=fmax, faverage=True, verbose=0
            )
            conmat = np.abs(con.get_data(output='dense'))  # (n_channels, n_channels, n_bands)
            all_conmats.append(conmat)

            for band_idx, band_name in enumerate(self.bands.keys()):
                mat = conmat[:, :, band_idx]

                # === PERMUTATION TESTING ===
                n_perm = 100
                null_distributions = []
                for _ in range(n_perm):
                    shuffled = epoch_data.copy()
                    for ch in range(shuffled.shape[0]):
                        np.random.shuffle(shuffled[ch])
                    surrogate_epoch = mne.EpochsArray(shuffled[np.newaxis, ...], info, verbose=False)
                    con_surrogate = spectral_connectivity_epochs(
                        surrogate_epoch, method=method, mode="multitaper", sfreq=self.fs,
                        fmin=fmin, fmax=fmax, faverage=True, verbose=0
                    )
                    surrogate_data = np.abs(con_surrogate.get_data(output='dense'))[:, :, band_idx]
                    null_distributions.append(surrogate_data)

                null_distributions = np.stack(null_distributions, axis=0)
                threshold_matrix = np.percentile(null_distributions, 95, axis=0)
                above_threshold_mask = mat > threshold_matrix
                mat = mat * above_threshold_mask

                sens_loc = np.array([info['chs'][info['ch_names'].index(ch)]['loc'][:3] for ch in ch_names])
                '''
                # === TOP 20% tra le connessioni significative ===
                nonzero_values = mat[np.tril_indices_from(mat, k=-1)]
                nonzero_values = nonzero_values[nonzero_values > 0]
                if len(nonzero_values) > 0:
                    perc_thresh = np.percentile(nonzero_values, 80)
                    mat[mat < perc_thresh] = 0
                '''
                # === Media finale imcoh ===
                final_values = mat[np.tril_indices_from(mat, k=-1)]
                final_values = final_values[final_values > 0]
                final_mean = np.mean(final_values) if len(final_values) > 0 else 0.0

                sens_loc = np.array([info['chs'][info['ch_names'].index(ch)]['loc'][:3] for ch in ch_names])
                ii, jj = np.tril_indices(len(ch_names), k=-1)
                con_nodes = []
                con_val = []
                for idx1, idx2 in zip(ii, jj):
                    p1 = sens_loc[idx1]
                    p2 = sens_loc[idx2]
                    dist = np.linalg.norm(p1 - p2)
                    con_nodes.append((idx1, idx2))
                    con_val.append(mat[idx1, idx2])
                con_val = np.abs(np.array(con_val))

                # === CREA GRAFO ===
                G = nx.Graph()
                for x in range(len(ch_names)):
                    G.add_node(x, pos=(sens_loc[x, 0], sens_loc[x, 1]))

                pos = nx.get_node_attributes(G, 'pos')
                filtered_edges = [(pair, val) for pair, val in zip(con_nodes, con_val) if val > 0]
                for (idx1, idx2), weight in filtered_edges:
                    G.add_edge(idx1, idx2, weight=weight)

                filtered_edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

                fig, ax = plt.subplots(figsize=(6, 6))
                feature_name = self.feats[0]
                title = f"{feature_name} - Epoch {i + 1} - {band_name}"
                ax.set_title(title)

                labels = {x: ch_names[x] for x in range(len(ch_names))}
                label_pos = {k: (v[0], v[1]) for k, v in pos.items()}
                weights = np.array(weights)

                # Disegno sullo specifico Axes
                nx.draw(G, pos, ax=ax, node_size=24, node_color='black',
                        edge_color=weights, edge_cmap=plt.cm.viridis,
                        edge_vmin=min(weights), edge_vmax=max(weights))
                nx.draw_networkx_labels(G, label_pos, labels, font_size=7, font_color='grey', ax=ax)

                # Colorbar collegata all'Axes corretto
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                           norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax)
                cbar.set_label('Connectivity strength')

                patient_plot_dir = os.path.join(self.save_path, "PLOT", self.only_class, self.only_patient)
                os.makedirs(patient_plot_dir, exist_ok=True)
                epoch_plot_base_dir = os.path.join(patient_plot_dir, "connectivity_100")
                os.makedirs(epoch_plot_base_dir, exist_ok=True)

                safe_feat_name = feature_name.replace(" ", "_").lower()
                fname = f"{safe_feat_name}_epoch_{i + 1:02d}_{band_name}.png"
                fpath = os.path.join(epoch_plot_base_dir, fname)
                fig.savefig(fpath, dpi=300)
                plt.close(fig)

                #--------------------------------------------------------------
                print(f"\n--- Epoch {i + 1} ---")
                deg = dict(G.degree(weight=None))
                strength = dict(G.degree(weight='weight'))
                print("strength:", deg)
                print("\n--- Strength per nodo ---")
                for node_idx in sorted(deg):
                    label = ch_names[node_idx] if node_idx < len(ch_names) else f"Idx{node_idx}"
                    print(f"Node {node_idx} ({label}): deg = {deg[node_idx]:.4f}")

                btw = nx.betweenness_centrality(G, weight='weight', normalized=True)
                clust = nx.clustering(G, weight='weight')

                # === Network metrics ===
                if not nx.is_connected(G):
                    largest_cc = max(nx.connected_components(G), key=len)
                    G_sub = G.subgraph(largest_cc).copy()
                glob_eff = nx.global_efficiency(G_sub)
                avg_path_len = nx.average_shortest_path_length(G_sub, weight='weight')

                # === Salvataggio delle metriche per ogni canale ===
                print("Graph global metrics:")
                print(f"Global efficiency: {glob_eff}")
                print(f"Average path length: {avg_path_len}")

                print("ch_names:", ch_names)
                print("selected_channels:", selected_channels)

                for sc_idx, ch_idx in enumerate(selected_channels):
                    print("Processing channel:", ch_idx)
                    print("sc_idx:", sc_idx)
                    try:
                        local_idx = ch_names.index(f"E{ch_idx+1}")  # Esempio: 27 → 'E27'
                    except ValueError:
                        print(f"⚠️ Canale 'E{ch_idx+1}' (index {sc_idx}) non presente in ch_names – salto.")
                        continue

                    ch_name = ch_names[local_idx]
                    print(
                        f"Processing channel '{ch_name}' (index in features: {sc_idx}, index in ch_names: {local_idx})")

                    # Ora puoi usare local_idx per accedere ai dati nel grafo
                    final_mean_val = final_mean
                    deg_val = deg.get(local_idx, 0)
                    strength_val = strength.get(local_idx, 0)
                    btw_val = btw.get(local_idx, 0)
                    clust_val = clust.get(local_idx, 0)

                    # Salvataggio delle feature nel vettore
                    features_matrix[sc_idx, 0, i] = final_mean_val
                    features_matrix[sc_idx, 1, i] = deg_val
                    features_matrix[sc_idx, 2, i] = strength_val
                    features_matrix[sc_idx, 3, i] = btw_val
                    features_matrix[sc_idx, 4, i] = clust_val
                    features_matrix[sc_idx, 5, i] = glob_eff
                    features_matrix[sc_idx, 6, i] = avg_path_len

                    # 🔍 Log dettagliato
                    print(
                        f"\n🎯 Feature del canale '{ch_name}' (index nel grafo: {local_idx}, index in features: {sc_idx})")
                    print(f"  → Mean ImCoh         : {final_mean_val:.4f}")
                    print(f"  → Degree             : {deg_val:.2f}")
                    print(f"  → Strength           : {strength_val:.2f}")
                    print(f"  → Betweenness        : {btw_val:.4f}")
                    print(f"  → Clustering Coeff.  : {clust_val:.4f}")
                    print(f"  → Global Efficiency  : {glob_eff:.4f}")
                    print(f"  → Avg Path Length    : {avg_path_len:.4f}")

        return (
            features_matrix,
            self.feats,
            features_matrix.copy(),  # all_channels_features_matrix
            selected_channels,  # channel_names
            average_channels,
            specific_channels,
            all_conmats,
            selected_channels
        )





