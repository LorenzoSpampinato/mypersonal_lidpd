import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.spatial.distance import euclidean

# === CONFIG ===
spectral_file = r"D:\TESI\prova statistica\N2N3multitaperMeanPSD_specific_channels_149\_N2N3multitaperMeanPSD_specific_channels_149_aggregated_with_phases.csv"
#entropy_file = r"D:\TESI\prova statistica\N2N3justentropy_specific_channels_26\_no_mean_N2N3justentropy_specific_channels_26_aggregated_with_phases.csv"
entropy_file = r"D:\TESI\prova statistica\N2N3ALLENTROPY_specific_channels_149\_N2N3ALLENTROPY_specific_channels_149_aggregated_with_phases.csv"
output_dir = r"D:\TESI\grafici"
selected_channels = [27, 33, 34, 38, 39, 47, 48, 26, 20, 19, 12, 11, 3, 2, 222,16, 22, 23, 24, 28, 29, 30, 35, 36, 40, 41, 42, 49, 50, 21, 15, 7, 14, 6,
                            207, 13, 5, 215, 4, 224, 223, 214, 206, 213, 205]
os.makedirs(output_dir, exist_ok=True)

# === NOMI FEATURE ===
#x_feature = 'logarithm of the ratio of high-frequency to low-frequency power'
x_feature = 'Mean PSD Total Delta'
y_feature = 'SampEn2'

# === CARICAMENTO DATI ===
spectral_data = pd.read_csv(spectral_file)
entropy_data = pd.read_csv(entropy_file)

# === FILTRAGGI ===
spectral_data = spectral_data[
    (spectral_data['Stage'].isin([3])) & (spectral_data['Phase_Assigned'].isin(['Early', 'Late']))]
entropy_data = entropy_data[
    (entropy_data['Stage'].isin([3])) & (entropy_data['Phase_Assigned'].isin(['Early', 'Late']))]

spectral_data = spectral_data[spectral_data['Channel'].isin(selected_channels)]
entropy_data = entropy_data[entropy_data['Channel'].isin(selected_channels)]

# === MERGE DATI ===
merged = pd.merge(
    entropy_data[['Group', 'Subject', 'Epoch', 'Channel', 'Phase_Assigned', y_feature]],
    spectral_data[['Group', 'Subject', 'Epoch', 'Channel', 'Phase_Assigned',
                   'Mean PSD Alpha', 'Mean PSD Sigma','Mean PSD Scalp Slow-Wave','Mean PSD Total Beta',
                   'Mean PSD Total Delta', 'Mean PSD Total Theta']],
    on=['Group', 'Subject', 'Epoch', 'Channel', 'Phase_Assigned'],
    how='inner'
)


# === CALCOLO NUOVA FEATURE ===
def compute_new_feature(row):
    num = row['Mean PSD Alpha'] + row['Mean PSD Sigma'] + row['Mean PSD Total Beta']
    denom = row['Mean PSD Total Delta'] + row['Mean PSD Total Theta']
    return np.log(num / denom) if denom > 0 else np.nan


merged['logarithm of the ratio of high-frequency to low-frequency power'] = merged.apply(compute_new_feature, axis=1)

# === MEDIA TRA EPOCHE ===
epoch_avg = merged.groupby(['Group', 'Subject', 'Phase_Assigned', 'Epoch']).agg({
    y_feature: 'mean',
    x_feature: 'mean'
}).reset_index()

# === MEDIA TRA CANALI ===
agg_data_avg = epoch_avg.groupby(['Group', 'Subject', 'Phase_Assigned']).agg({
    y_feature: 'mean',
    x_feature: 'mean'
}).reset_index()

# === CALCOLO BARICENTRI ===
centroids = agg_data_avg.groupby(['Group', 'Phase_Assigned'])[[x_feature, y_feature]].mean().reset_index()


# === DISTANZA DAL BARICENTRO ===
def compute_distance_from_centroid(row, centroids_df):
    group = row['Group']
    phase = row['Phase_Assigned']
    centroid = centroids_df[(centroids_df['Group'] == group) & (centroids_df['Phase_Assigned'] == phase)]
    if centroid.empty:
        return np.nan
    centroid_x = centroid[x_feature].values[0]
    centroid_y = centroid[y_feature].values[0]
    return euclidean((row[x_feature], row[y_feature]), (centroid_x, centroid_y))


agg_data_avg['Dist_from_Centroid'] = agg_data_avg.apply(lambda row: compute_distance_from_centroid(row, centroids),
                                                        axis=1)

# === DEFINIZIONE PALETTE DI COLORI ===
unique_groups = agg_data_avg['Group'].unique()
palette = sns.color_palette("Set2", n_colors=len(unique_groups))
group_colors = dict(zip(unique_groups, palette))
'''
# === GRAFICI CON BARICENTRI ===
for phase in ['Early', 'Late']:
    phase_data = agg_data_avg[agg_data_avg['Phase_Assigned'] == phase]
    centroids_phase = centroids[centroids['Phase_Assigned'] == phase]

    plt.figure(figsize=(10, 6))

    # Traccia i punti dei dati
    for group in unique_groups:
        group_data = phase_data[phase_data['Group'] == group]
        plt.scatter(
            group_data[x_feature],
            group_data[y_feature],
            label=group,
            s=100,
            color=group_colors[group]
        )

    # Traccia i centroidi
    for _, row in centroids_phase.iterrows():
        group = row['Group']
        plt.scatter(
            row[x_feature],
            row[y_feature],
            color=group_colors[group],
            s=300,
            marker='X',
            edgecolor='black',
            linewidth=1.5,
            label=f"Centroid {group}"
        )

    plt.title(f'Mean {y_feature} vs {x_feature} - {phase}')
    plt.xlabel(f'Mean {x_feature} (avg epoche → avg canali)')
    plt.ylabel(f'Mean {y_feature} (avg epoche → avg canali)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plot_path = os.path.join(output_dir, f"{y_feature.replace(' ', '')}_vs_{x_feature.replace(' ', '')}_{phase}.png")
    plt.savefig(plot_path)
    plt.show()
'''
plt.figure(figsize=(12, 8))

# Traccia le frecce soggetto per soggetto
for group in unique_groups:
    group_data = agg_data_avg[agg_data_avg['Group'] == group]
    for subject in group_data['Subject'].unique():
        subj_data = group_data[group_data['Subject'] == subject]
        if len(subj_data) == 2:
            early = subj_data[subj_data['Phase_Assigned'] == 'Early'].iloc[0]
            late = subj_data[subj_data['Phase_Assigned'] == 'Late'].iloc[0]
            plt.plot(
                [early[x_feature], late[x_feature]],
                [early[y_feature], late[y_feature]],
                linestyle='--',
                color=group_colors[group],
                alpha=0.5
            )

# Punti Early (cerchi)
early_data = agg_data_avg[agg_data_avg['Phase_Assigned'] == 'Early']
for group in unique_groups:
    group_data = early_data[early_data['Group'] == group]
    plt.scatter(
        group_data[x_feature],
        group_data[y_feature],
        label=f'{group} Early',
        color=group_colors[group],
        marker='o',
        s=100
    )

# Punti Late (triangoli)
late_data = agg_data_avg[agg_data_avg['Phase_Assigned'] == 'Late']
for group in unique_groups:
    group_data = late_data[late_data['Group'] == group]
    plt.scatter(
        group_data[x_feature],
        group_data[y_feature],
        label=f'{group} Late',
        color=group_colors[group],
        marker='^',
        s=100
    )

# Frecce e centroidi
for group in unique_groups:
    early_centroid = centroids[(centroids['Group'] == group) & (centroids['Phase_Assigned'] == 'Early')]
    late_centroid = centroids[(centroids['Group'] == group) & (centroids['Phase_Assigned'] == 'Late')]

    if not early_centroid.empty and not late_centroid.empty:
        # Freccia tra centroidi
        plt.plot(
            [early_centroid[x_feature].values[0], late_centroid[x_feature].values[0]],
            [early_centroid[y_feature].values[0], late_centroid[y_feature].values[0]],
            linestyle='--',
            linewidth=2,
            color=group_colors[group]
        )

        # Centroidi Early (X)
        plt.scatter(
            early_centroid[x_feature],
            early_centroid[y_feature],
            color=group_colors[group],
            s=120,
            marker='X',
            edgecolor='black',
            linewidth=1.5,
            label=f'{group} Centroid Early'
        )

        # Centroidi Late (rombo)
        plt.scatter(
            late_centroid[x_feature],
            late_centroid[y_feature],
            color=group_colors[group],
            s=120,
            marker='D',
            edgecolor='black',
            linewidth=1.5,
            label=f'{group} Centroid Late'
        )

plt.title(f'{y_feature} vs {x_feature} - Early and Late with transitions')
plt.xlabel(f'{x_feature} (avg epoche → avg canali)')
plt.ylabel(f'{y_feature} (avg epoche → avg canali)')
plt.grid(True)
plt.tight_layout()
plt.legend()
plot_path = os.path.join(output_dir, f"{y_feature.replace(' ', '')}_vs_{x_feature.replace(' ', '')}_combined_with_centroids.png")
plt.savefig(plot_path)
plt.show()

