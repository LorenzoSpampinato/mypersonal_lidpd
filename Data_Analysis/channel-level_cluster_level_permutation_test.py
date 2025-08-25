import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import mne
import os
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_rel, wilcoxon, ttest_ind, mannwhitneyu
from mne.stats import permutation_cluster_test
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from torch.ao.nn.quantized.functional import threshold
import itertools
from matplotlib.colors import TwoSlopeNorm



def read_egi_electrode_coordinates(xml_file_path):
    """
    Legge le coordinate degli elettrodi dal file XML di EGI GSN.
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        namespace = ''
        if '}' in root.tag:
            namespace = root.tag.split('}')[0] + '}'
        coords = {}
        sensor_path = f".//{namespace}sensor" if namespace else ".//sensor"
        number_path = f"{namespace}number" if namespace else "number"
        x_path = f"{namespace}x" if namespace else "x"
        y_path = f"{namespace}y" if namespace else "y"
        z_path = f"{namespace}z" if namespace else "z"
        for sensor in root.findall(sensor_path):
            num_elem = sensor.find(number_path)
            if num_elem is not None and num_elem.text:
                electrode = f"E{int(num_elem.text)}"
                x = float(sensor.find(x_path).text)
                y = float(sensor.find(y_path).text)
                z = float(sensor.find(z_path).text)
                coords[electrode] = (x, y, z)
        print(f"Lette coordinate per {len(coords)} elettrodi")
        return coords
    except Exception as e:
        print(f"Errore lettura XML: {e}")
        return {}


def divide_to_regions():
    """
    Definisce i 150 elettrodi dello scalpo sul sistema EEG 10-20.
    """
    # liste di numeri di canali per regione
    fp = np.sort(np.array([27, 33, 34, 38, 39, 47, 48, 26, 20, 19, 12, 11, 3, 2, 222]))
    f = np.sort(np.array([16, 22, 23, 24, 28, 29, 30, 35, 36, 40, 41, 42, 49, 50, 21, 15, 7, 14, 6, 207, 13, 5, 215,
                          4, 224, 223, 214, 206, 213, 205]))
    c = np.sort(np.array([9, 17, 43, 44, 45, 51, 52, 53, 57, 58, 59, 60, 64, 65, 66, 71, 72, 8, 257, 81, 186, 198,
                          197, 185, 132, 196, 184, 144, 204, 195, 183, 155, 194, 182, 164, 181, 173]))
    t = np.sort(np.array([55, 56, 62, 63, 69, 70, 74, 75, 84, 85, 96, 221, 212, 211, 203, 202, 193, 192, 180, 179,
                          171, 170]))
    p = np.sort(np.array([76, 77, 78, 79, 80, 86, 87, 88, 89, 97, 98, 99, 100, 110, 90, 101, 119, 172, 163, 154,
                          143, 131, 162, 153, 142, 130, 161, 152, 141, 129, 128]))
    o = np.sort(np.array([107, 108, 109, 116, 117, 118, 125, 126, 160, 151, 140, 150, 139, 127, 138]))
    regions = {
        'Prefrontal': [f'E{ch}' for ch in fp],
        'Frontal':    [f'E{ch}' for ch in f],
        'Central':    [f'E{ch}' for ch in c],
        'Temporal':   [f'E{ch}' for ch in t],
        'Parietal':   [f'E{ch}' for ch in p],
        'Occipital':  [f'E{ch}' for ch in o]
    }
    all_electrodes = [el for regs in regions.values() for el in regs]
    return regions, all_electrodes


def create_mne_info_from_coordinates(coordinates_3d, sfreq=128):
    _, all_electrodes = divide_to_regions()
    valid = set(all_electrodes)
    ch_names = sorted(ch for ch in coordinates_3d if ch in valid)
    ch_pos = {ch: coordinates_3d[ch] for ch in ch_names}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)
    print(f"Usati {len(ch_names)} elettrodi validi su {len(coordinates_3d)} trovati")
    return info



def run_cluster_permutation_test_early_vs_late(swa_early, swa_late, mne_info, alpha=0.05, apply_log=False):
    if apply_log:
        eps = np.finfo(float).eps
        swa_early = np.log10(swa_early + eps)
        swa_late = np.log10(swa_late + eps)

    ch_names = mne_info.ch_names
    adjacency, _ = mne.channels.find_ch_adjacency(mne_info, ch_type='eeg')

    def custom_t(X, Y):
        #t_vals, _ = ttest_rel(X, Y, axis=0, alternative='greater')
        t_vals, _ = ttest_rel(X, Y, axis=0, alternative='greater')
        #t_vals, _ = ttest_rel(X, Y, axis=0, alternative='less')
        return t_vals

    def custom_wilcoxon(X, Y):
        # Calcola la statistica di Wilcoxon per ogni canale
        w_vals, _ = wilcoxon(X, Y, alternative='greater', zero_method='zsplit')
        return w_vals

    n_subjects = swa_early.shape[0]
    t_threshold = t.ppf(1 - alpha, df=n_subjects - 1) -0.72
    print(f"t-threshold (critical t-value) at alpha={alpha}: {t_threshold:.3f}")


    swa_early3d = swa_early[:, np.newaxis, :]
    print(f"Shape early: {swa_early3d.shape}")

    swa_late3d = swa_late[:, np.newaxis, :]
    X = [swa_early3d, swa_late3d]

    T_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
        X, n_permutations=5000, tail=-1,
        stat_fun=custom_t, threshold=-t_threshold,
        adjacency=adjacency, out_type='mask', n_jobs=1
    )
    print(f"P-value per ciascun cluster: {cluster_p_values}")
    for i, cluster_mask in enumerate(clusters):
        # cluster_mask è un array booleano, stesso shape di T_obs
        # Somma dei T_obs nei punti dove cluster_mask è True
        cluster_stat = np.sum(T_obs[cluster_mask])
        num_points = np.sum(cluster_mask)
        indices = np.where(cluster_mask)[0]

        print(f"Cluster {i + 1}:")
        print(f"  P-value = {cluster_p_values[i]}")
        print(f"  Numero di punti = {num_points}")
        print(f"  Indici dei punti = {list(indices)}")
        print(f"  Somma statistiche (T_obs) = {cluster_stat:.4f}")
        print("-" * 40)

    sig_inds = np.where(cluster_p_values < 0.05)[0]
    mask = np.zeros(swa_early.shape[1], dtype=bool)
    for idx in sig_inds:
        mask |= clusters[idx][0]

    mean_early = np.mean(swa_early, axis=0)
    mean_late = np.mean(swa_late, axis=0)
    mean_diff = mean_early - mean_late

    return mean_early, mean_late, mean_diff, mask

def plot_group_topomaps_early_vs_late(
        all_early, all_late, all_diff, all_masks,
        group_order, mne_info, feature_name='Feature',
        figsize=(20, 14), cmap='turbo', contours=6,
        save_path="group_comparison_topomaps.png"):
    """
    Disegna i topomaps per ogni gruppo: Early, Late e Diff (Early - Late).
    all_early, all_late, all_diff, all_masks: liste di array shape (n_chan,)
    group_order: liste di nomi dei gruppi
    mne_info: oggetto info di MNE con montaggio
    """
    # Calcola limiti per colorbar
    all_vals = {
        'common': np.hstack(all_early + all_late),
        'diff': np.hstack(all_diff)
    }
    vmin_common, vmax_common = np.min(all_vals['common']), np.max(all_vals['common'])
    diff_vmin, diff_vmax = np.min(all_vals['diff']), np.max(all_vals['diff'])

    # Setup figure e layout a griglia
    n_groups = len(group_order)
    gs = GridSpec(nrows=n_groups, ncols=5,
                  width_ratios=[1, 1, 0.05, 1, 0.05],
                  wspace=0.1, hspace=0.1)
    topo_cols = [0, 1, 3]
    #a = -0.15
    b = 0.05
    c = 0.7

    fig = plt.figure(figsize=figsize)
    # Ciclo sui gruppi e sulle 3 mappe
    for i in range(n_groups):
        for j, (data_, title) in enumerate(zip(
                [all_early[i], all_late[i], all_diff[i]],
                ['Early', 'Late', 'Diff\n(Early - Late)']
        )):
            ax = fig.add_subplot(gs[i, topo_cols[j]])
            mask = all_masks[i] if title.startswith('Diff') else None


            vmin, vmax = (diff_vmin, diff_vmax) if title.startswith('Diff') else (vmin_common, vmax_common)

            im, _ = mne.viz.plot_topomap(
                data_, mne_info,
                mask=mask,
                mask_params=dict(marker='o', markersize=5),
                cmap=cmap, contours=contours,
                show=False, extrapolate='head', outlines='head',
                sensors=True, axes=ax, vlim=(vmin, vmax)
            )
            im, _ = mne.viz.plot_topomap(
                data_, mne_info,
                mask=mask,
                mask_params=dict(marker='o', markersize=5),
                cmap=cmap, contours=contours,
                show=False, extrapolate='head', outlines='head',
                sensors=True, axes=ax, vlim=(vmin, vmax)
            )

            # === Naso (triangolo più aperto e più vicino al topoplot) ===
            nose_x = [0.44, 0.5, 0.56]  # più aperto
            nose_y = [1.00, 1.04, 1.00]  # più vicino al cerchio (prima era 1.08)
            ax.plot(nose_x, nose_y, color='k', lw=1.5, transform=ax.transAxes, clip_on=False)

            # === Orecchie (elissoidi verticali, stretti e lunghi) ===
            theta = np.linspace(0, 2 * np.pi, 100)
            # Parametri per le ellissi
            a = 0.02  # semi-asse orizzontale (stretto)
            b = 0.08  # semi-asse verticale (lungo)

            # Orecchio sinistro
            ear_left_x = 0 + a * np.cos(theta)
            ear_left_y = 0.5 + b * np.sin(theta)
            ax.plot(ear_left_x, ear_left_y, color='k', lw=1, transform=ax.transAxes, clip_on=False)

            # Orecchio destro
            ear_right_x = 1 + a * np.cos(theta)
            ear_right_y = 0.5 + b * np.sin(theta)
            ax.plot(ear_right_x, ear_right_y, color='k', lw=1, transform=ax.transAxes, clip_on=False)

            if i == 0:
                # Titoli più in alto
                ax.set_title(title, fontsize=25, pad=22)

            if j == 0:
                # Etichette gruppi più a sinistra usando ax.text (in coordinate assi)
                ax.text(-0.3, 0.5, group_order[i], va='center', ha='right',
                        fontsize=25, transform=ax.transAxes)

    # Colorbars

    cbar_ax_common = fig.add_subplot(gs[:, 2])

    sm_common = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin_common, vmax=vmax_common))
    cbar_common = plt.colorbar(sm_common, cax=cbar_ax_common)
    #cbar_common.set_label(f"Katz Fractal Dimension", fontsize=18)
    #cbar_common.set_label(r"$\log_{10}(\mathrm{Slow\ Wave\ Activity})$", fontsize=18)
    cbar_common.set_label(f"Sample Entropy", fontsize=18)
    #cbar_common.set_label(feature_name, fontsize=14)

    cbar_ax_diff = fig.add_subplot(gs[:, 4])
    sm_diff = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=diff_vmin, vmax=diff_vmax))
    cbar_diff = plt.colorbar(sm_diff, cax=cbar_ax_diff)
    #cbar_diff.set_label(r'Sample Entropy difference', fontsize=12)
    #cbar_diff.set_label(f"Katz Fractal Dimension difference", fontsize=18)
    #cbar_diff.set_label(r"$\log_{10}(\mathrm{Slow\ Wave\ Activity})$ difference", fontsize=18)
    cbar_diff.set_label(f"Sample Entropy difference", fontsize=18)

    fig.suptitle(f"Topographic plot of Sample Entropy", fontsize=32)

    #fig.suptitle("Sample Entropy: Topographic plot", fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Salva e mostra
    fig.savefig(save_path, dpi=300)
    plt.show()

def run_cluster_permutation_test_between_groups(data1, data2, mne_info, alpha=0.05, apply_log=False, tail=0):
    if apply_log:
        eps = np.finfo(float).eps
        data1 = np.log10(data1 + eps)
        data2 = np.log10(data2 + eps)

    adjacency, _ = mne.channels.find_ch_adjacency(mne_info, ch_type='eeg')
    n1, n2 = data1.shape[0], data2.shape[0]
    df = n1 + n2 - 2
    thresh = t.ppf(1 - alpha/2 if tail==0 else 1-alpha, df)
    print(f"t-threshold (critical t-value) at alpha={alpha}: {thresh:.3f}")
    print(f"Shape data1: {data1.shape}")
    print(f"Shape data2: {data2.shape}")
    X = [data1[:, None, :], data2[:, None, :]]
    print(f"Shape X: {X[0].shape}, {X[1].shape}")

    def stat_ind(X, Y):
        t_vals, _ = ttest_ind(X, Y, axis=0, equal_var=True,
                              alternative='two-sided' if tail==0 else
                                           'greater' if tail==1 else 'less')
        return t_vals

    T_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
        X, stat_fun=stat_ind, threshold=thresh,
        adjacency=adjacency, tail=tail,
        n_permutations=5000, out_type='mask', n_jobs=1
    )
    for i, cluster_mask in enumerate(clusters):
        # cluster_mask è un array booleano, stesso shape di T_obs
        # Somma dei T_obs nei punti dove cluster_mask è True
        cluster_stat = np.sum(T_obs[cluster_mask])
        num_points = np.sum(cluster_mask)
        indices = np.where(cluster_mask)[0]

        print(f"Cluster {i + 1}:")
        print(f"  P-value = {cluster_p_values[i]}")
        print(f"  Numero di punti = {num_points}")
        print(f"  Indici dei punti = {list(indices)}")
        print(f"  Somma statistiche (T_obs) = {cluster_stat:.4f}")
        print("-" * 40)
    print(f"P-value per ciascun cluster: {cluster_p_values}")
    mask = np.zeros(data1.shape[1], bool)
    for i, p in enumerate(cluster_p_values):
        if p < alpha:
            mask |= clusters[i][0]
    diff = np.mean(data1, axis=0) - np.mean(data2, axis=0)
    return diff, mask



def main():
    apply_log = False
    #file_path = r"D:\TESI\prova statistica\N3MULTISCALEENTROPY_specific_channels_149\_N3MULTISCALEENTROPY_specific_channels_149_aggregated_with_phases"
    file_path = r"D:\TESI\prova statistica\N2N3ALLENTROPY_specific_channels_149\_N2N3ALLENTROPY_specific_channels_149_aggregated_with_phases.csv"
    #file_path = r"D:\TESI\prova statistica\N3MULTISCALEENTROPY_specific_channels_149\_N3MULTISCALEENTROPY_specific_channels_149_aggregated_with_phases.csv"
    #file_path = r"D:\TESI\prova statistica\N3CORRDIM_specific_channels_149\_N3CORRDIM_specific_channels_149_aggregated_with_phases.csv"
    #file_path = r"D:\TESI\prova statistica\N3CONN100_specific_channels_149\_N3CONN100_specific_channels_149_aggregated_with_phases.csv"

    xml_path = r"C:\\Users\\Lorenzo\\Desktop\\coordinates.xml"
    features = ['SampEn2']
    #features = [f'SampEn{i}' for i in range(1, 21)]  # SampEn1 to SampEn20

    group_order = ['CTL', 'DNV', 'ADV', 'DYS']

    # Caricamento dati
    data = pd.read_csv(file_path)
    data = data[data['Stage'] == 3]

    # Lettura coordinate e info MNE
    coords = read_egi_electrode_coordinates(xml_path)
    mne_info = create_mne_info_from_coordinates(coords)
    ordered_nums = [int(ch[1:]) for ch in mne_info.ch_names]

    for feat in features:
        print(f"Processing {feat}...")
        early_data, late_data = {}, {}

        for group in group_order:
            grp = data[data['Group'] == group]

            e_piv = grp[grp['Phase_Assigned'] == 'Early'] \
                .pivot_table(index='Subject', columns='Channel', values=feat).fillna(0)
            l_piv = grp[grp['Phase_Assigned'] == 'Late'] \
                .pivot_table(index='Subject', columns='Channel', values=feat).fillna(0)
            '''
            e_piv = grp[grp['Phase_Assigned'] == 'Early'] \
                .pivot_table(index='Subject', columns='Channel', values=feat,
                             aggfunc='median').fillna(0)
            l_piv = grp[grp['Phase_Assigned'] == 'Late'] \
                .pivot_table(index='Subject', columns='Channel', values=feat,
                             aggfunc='median').fillna(0)
            '''
            e_arr = e_piv[ordered_nums].loc[e_piv.index.intersection(l_piv.index)].values
            l_arr = l_piv[ordered_nums].loc[e_piv.index.intersection(l_piv.index)].values

            if apply_log:
                eps = np.finfo(float).eps
                e_arr = np.log10(e_arr + eps)
                l_arr = np.log10(l_arr + eps)

            early_data[group] = e_arr
            late_data[group] = l_arr

        # Cluster test per ogni scala
        all_early, all_late, all_diff, all_masks = [], [], [], []
        for group in group_order:
            e_arr = early_data[group]
            l_arr = late_data[group]
            mean_e, mean_l, mean_diff, mask = run_cluster_permutation_test_early_vs_late(
                e_arr, l_arr, mne_info, alpha=0.05, apply_log=apply_log
            )
            all_early.append(mean_e)
            all_late.append(mean_l)
            all_diff.append(mean_diff)
            all_masks.append(mask)

        # Salvataggio con nome della feature
        save_name = f"group_topomap_{feat}.png"
        plot_group_topomaps_early_vs_late(
            all_early, all_late, all_diff, all_masks,
            group_order, mne_info,
            feature_name=feat,
            save_path=save_name
        )

    '''
    
    # Costruisci figure combinate con colorbar esterna
    for phase, dataset in [('Early', early_data), ('Late', late_data)]:
        pairs = list(itertools.combinations(group_order, 2))
        n = len(pairs)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        im = None
        for ax, (g1, g2) in zip(axes, pairs):
            diff, mask = run_cluster_permutation_test_between_groups(
                dataset[g1], dataset[g2], mne_info,
                alpha=0.05, apply_log=apply_log)
            im, _ = mne.viz.plot_topomap(
                diff, mne_info, mask=mask,
                mask_params=dict(marker='o', markersize=5),
                show=False, axes=ax, vlim=(None, None), cmap='turbo')
            ax.set_title(f"{g1} - {g2}", fontsize=12)
        # rimuovi assi extra
        for ax in axes[n:]:
            ax.axis('off')

        # colorbar esterna a destra
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r'Differenza Media (log$_{10}$)', fontsize=12)

        # Regola layout dei topomaps leggermente più piccoli
        plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.05, wspace=0.3, hspace=0.3)
        fig.suptitle(f"Topomaps {phase} comparisons", fontsize=16)
        plt.show()
    '''

if __name__ == '__main__':
    main()
