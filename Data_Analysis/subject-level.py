import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kruskal, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import normaltest, kruskal, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway, wilcoxon, kruskal, normaltest, levene, bartlett, fligner, ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from scipy.stats import shapiro
from statsmodels.stats.power import FTestAnovaPower

def analyze_feature(df, feature, channel_to_analyze):
    results = {}

    print(f"\n //////////////////Processing Channel: {channel_to_analyze}//////////////////")
    channel_data = df[df['Channel'] == channel_to_analyze]

    for phase in ['Early', 'Late']:

        phase_data = channel_data[channel_data['Phase_Assigned'] == phase]


        groups = phase_data['Group'].unique()
        print(f"Groups: {groups}")
        data = [phase_data[phase_data['Group'] == group].groupby('Subject')[feature].mean().dropna() for group in groups]

        # Statistiche descrittive
        descriptive_stats = phase_data.groupby('Group')[feature].describe()
        print(f"Descriptive Stats for {feature}:\n", descriptive_stats)

        # Test di normalità
        normality = {group: shapiro(group_data)[1] if len(group_data) >= 3 else None for group, group_data in zip(groups, data)}
        print(f"Normality Test Results: {normality}")

        if all(p > 0.05 for p in normality.values() if p is not None):
            levene_p = levene(*data)[1]
            bartlett_p = bartlett(*data)[1]
            var_test = "Levene" if levene_p < 0.05 else "Bartlett"
            var_p = levene_p if levene_p < 0.05 else bartlett_p
        else:
            fligner_p = fligner(*data)[1]
            var_test = "Fligner-Killeen"
            var_p = fligner_p

        print(f"Test di varianza: {var_test} (p={var_p:.2e})")

        if all(p > 0.05 for p in normality.values() if p is not None):
            if var_p > 0.05:
                stat, p_val = f_oneway(*data)
                test_type = "ANOVA"
            else:
                stat, p_val = f_oneway(*data)
                test_type = "Welch's ANOVA"
        else:
            stat, p_val = kruskal(*data)
            test_type = "Kruskal-Wallis"

        print(f"Test statistico: {test_type} (stat={stat:.3f}, p={p_val:.2e})")

        post_hoc = None
        if p_val < 0.05:
            combined_data = phase_data[[feature, 'Group']].dropna()
            if test_type == "ANOVA":
                post_hoc = pairwise_tukeyhsd(combined_data[feature], combined_data['Group'])
                print(f"Tukey PostHoc Results:\n", post_hoc)
            else:
                post_hoc = sp.posthoc_dunn(combined_data, val_col=feature, group_col='Group', p_adjust='bonferroni')
                print(f"Dunn PostHoc Results:\n", post_hoc)

        results[(channel_to_analyze, phase)] = {
            'Descriptive': descriptive_stats,
            'Normality': normality,
            'Test': (test_type, stat, p_val),
            'PostHoc': post_hoc
        }

    # Confronto Early vs Late per ciascun gruppo
    for group in ['CTL', 'DNV', 'ADV', 'DYS']:
        print(f"Comparing Early vs Late for {group}")
        early = channel_data[(channel_data['Group'] == group) & (channel_data['Phase_Assigned'] == 'Early')]
        late = channel_data[(channel_data['Group'] == group) & (channel_data['Phase_Assigned'] == 'Late')]

        # Unione per soggetti presenti in entrambi
        early_group = early.groupby('Subject')[feature].mean()
        late_group = late.groupby('Subject')[feature].mean()
        common_subjects = early_group.index.intersection(late_group.index)

        early_vals = early_group.loc[common_subjects]
        late_vals = late_group.loc[common_subjects]
        diffs = early_vals - late_vals

        if len(diffs) >= 3:
            p_normal = shapiro(diffs)[1]
            print(f"Shapiro on differences p={p_normal:.3f}")

            if p_normal > 0.05:
                stat, p_val = ttest_rel(early_vals, late_vals)
                test_type = "Paired t-test"
            else:
                stat, p_val = wilcoxon(early_vals, late_vals)
                test_type = "Wilcoxon signed-rank"

            print(f"{test_type} Early vs Late: stat={stat:.3f}, p={p_val:.3e}")

            results[(channel_to_analyze, f'{group} Early vs Late')] = {
                'Normality on diff': p_normal,
                'Test': (test_type, stat, p_val),
                'N_subjects': len(diffs)
            }
        else:
            print("Not enough paired data for this group.")


    return results




def print_results(results):
    """
    Stampa i risultati dell'analisi in modo organizzato.
    """
    for key, value in results.items():
        channel, phase = key

        if phase in ['Early', 'Late']:
            print(f"\n===== Analisi di {channel} - {phase} =====")
            print("Statistiche descrittive:\n", value['Descriptive'])
            print("\nTest di normalità:", value['Normality'])
            print("\nTest statistico:", value['Test'])

            # Verifica il tipo di test post-hoc e stampa i risultati correttamente
            post_hoc = value['PostHoc']
            if post_hoc is not None:
                if hasattr(post_hoc, 'summary'):  # Se è un risultato Tukey HSD
                    print("\nTest post-hoc (Tukey HSD):\n", post_hoc.summary())
                else:
                    print("\nTest post-hoc (Dunn con correzione Bonferroni):\n", post_hoc)

        else:  # Confronto Early vs. Late per ogni gruppo
            print(f"\n===== Confronto {phase} per {channel} =====")
            print("\nTest di normalità:", value['Normality'])
            print("\nTest statistico:", value['Test'])


# Creare una mappa di colori globale per i soggetti
import seaborn as sns


def create_subject_color_map(data):
    unique_groups = data['Group'].unique()  # Prendi i gruppi (es. 'CTL', 'DNV', etc.)
    subject_color_map = {}

    for group in unique_groups:
        subjects_in_group = sorted(data[data['Group'] == group]['Subject'].unique())
        num_subjects = len(subjects_in_group)

        # Genera colori distinti per ogni gruppo
        group_colors = sns.color_palette("tab20", n_colors=num_subjects)

        # Mappa soggetti a colori
        for subject, color in zip(subjects_in_group, group_colors):
            subject_color_map[subject] = color

    return subject_color_map


def plot_feature_per_patient_violin_and_sd_subplot(data, feature_name, channel_name, group_order=None):
    """
    Crea un grafico con subplot organizzati per gruppo.
    Ogni sottotrama mostra:
    - Violin plot della distribuzione per Early e Late, separati.
    - Media e deviazione standard.
    I pazienti dello stesso gruppo sono disposti sulla stessa riga.
    Il titolo di ogni sottotrama include il paziente e il gruppo di appartenenza.
    Include una legenda per distinguere Early e Late.

    Parameters:
        data (pd.DataFrame): Contiene le colonne 'Group', 'Subject', 'Channel', 'Phase_Assigned', e la caratteristica specificata.
        feature_name (str): Il nome della caratteristica da plottare.
        channel_name (str): Nome del canale da analizzare.
        group_order (list): Ordine esplicito dei gruppi (opzionale).
    """
    if group_order:
        data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)

    # Filtra i dati per il canale specificato
    channel_data = data[data['Channel'] == channel_name]

    # Ottieni i gruppi unici e i pazienti in ogni gruppo
    groups = channel_data['Group'].unique()
    group_to_patients = {group: channel_data[channel_data['Group'] == group]['Subject'].unique() for group in groups}

    # Numero massimo di pazienti in un gruppo
    max_patients_per_group = max(len(patients) for patients in group_to_patients.values())

    # Crea una griglia: righe = gruppi, colonne = massimo numero di pazienti in un gruppo
    n_rows = len(groups)
    n_cols = max_patients_per_group

    # Imposta la figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows), sharey=True)

    # Se c'è solo un gruppo o un paziente, axes potrebbe non essere una matrice
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    # Colori per le fasi
    phase_colors = {'Early': 'skyblue', 'Late': 'salmon'}

    # Loop sui gruppi e pazienti
    for row, group in enumerate(groups):
        patients = group_to_patients[group]

        for col in range(n_cols):
            ax = axes[row][col]

            # Se il numero di pazienti è minore del massimo, lascia i subplot vuoti
            if col >= len(patients):
                ax.axis('off')
                continue

            patient = patients[col]
            patient_data = channel_data[channel_data['Subject'] == patient]

            # Separare i dati per fase
            early_data = patient_data[patient_data['Phase_Assigned'] == 'Early']
            late_data = patient_data[patient_data['Phase_Assigned'] == 'Late']

            # Plot dei violin plot separati
            if not early_data.empty:
                sns.violinplot(
                    x=[0] * len(early_data),  # Posizione sfalsata per Early
                    y=early_data[feature_name],
                    inner='box',
                    color=phase_colors['Early'],
                    width=0.6,
                    ax=ax,
                    alpha=0.6
                )
            if not late_data.empty:
                sns.violinplot(
                    x=[1] * len(late_data),  # Posizione sfalsata per Late
                    y=late_data[feature_name],
                    inner='box',
                    color=phase_colors['Late'],
                    width=0.6,
                    ax=ax,
                    alpha=0.6
                )

            # Aggiungi media e deviazione standard per Early e Late
            for phase, phase_data, offset in zip(
                ['Early', 'Late'], [early_data, late_data], [0, 1]
            ):
                if not phase_data.empty:
                    mean_val = phase_data[feature_name].mean()
                    std_val = phase_data[feature_name].std()
                    ax.errorbar(
                        x=offset,
                        y=mean_val,
                        yerr=std_val,
                        fmt='o',
                        color='black',
                        alpha=0.8
                    )

            # Impostazioni asse
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Early', 'Late'])
            ax.set_title(f"Patient {patient} ({group})", fontsize=10)

        # Etichetta per i gruppi
        axes[row][0].set_ylabel(group, fontsize=14, labelpad=20)

    # Legenda per i colori delle fasi
    handles = [
        plt.Line2D([0], [0], color=phase_colors['Early'], lw=4, label='Early'),
        plt.Line2D([0], [0], color=phase_colors['Late'], lw=4, label='Late')
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=12, title="Phase")

    # Etichette comuni
    fig.supxlabel("Phase", fontsize=14)
    fig.supylabel(feature_name, fontsize=14)
    fig.suptitle(f"Distribuzione, Media e SD per paziente nel canale: {channel_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Lascia spazio per il titolo principale
    plt.show()



def plot_early_and_late_barplots_minmaxscaler(data, feature_name, channel_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=None):
    """
    Creates and displays barplots for the "Early" and "Late" phases, grouping patients by group.
    The barplots represent the group means, and colored dots represent the individual patient means.

    Parameters:
    - data: DataFrame with the data.
    - feature_name: Name of the feature to plot (y column).
    - channel_name: Channel to plot.
    - group_order: Optional list to order the groups on the x-axis.
    """
    # **Apply MinMaxScaler to the entire dataset**
    scaler = MinMaxScaler()
    data[feature_name] = scaler.fit_transform(data[[feature_name]])

    # Ensure that the 'Group' column is categorical with the desired order
    data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)

    # Filter data by the specified channel
    channel_data = data[data['Channel'] == channel_name]
    #channel_data = data[data['Brain region'] == channel_name]

    # Filter data by "Early" and "Late" phases
    combined_data = channel_data[channel_data['Phase_Assigned'].isin(["Early", "Late"])]

    # Plot creation
    plt.figure(figsize=(12, 6))
    x_offset = 0.3  # Distance between Early and Late bars
    tick_positions = []
    tick_labels = []

    color_map = subject_color_map

    # Variable to handle patient legends (to avoid duplicates)
    patient_legend_handles = []

    # Plot data for each group
    for group_idx, group in enumerate(combined_data['Group'].cat.categories):
        group_data = combined_data[combined_data['Group'] == group]

        early_data = group_data[group_data['Phase_Assigned'] == "Early"]
        late_data = group_data[group_data['Phase_Assigned'] == "Late"]

        early_means = early_data.groupby('Subject')[feature_name].mean()
        late_means = late_data.groupby('Subject')[feature_name].mean()

        early_group_mean = early_means.mean()
        late_group_mean = late_means.mean()

        # **Barplot for group means**
        plt.bar(
            group_idx * 2 - x_offset, early_group_mean, color='red', width=0.5, label="Early" if group_idx == 0 else "",
            zorder=3
        )
        plt.bar(
            group_idx * 2 + x_offset, late_group_mean, color='blue', width=0.5, label="Late" if group_idx == 0 else "",
            zorder=3
        )

        # Add dots for each patient with a unique color
        for idx, subject in enumerate(early_means.index):
            plt.scatter(
                [group_idx * 2 - x_offset], [early_means.loc[subject]], color=color_map[subject], zorder=4, alpha=1,
                edgecolor='black', linewidth=1.5, s=70
            )

        for idx, subject in enumerate(late_means.index):
            plt.scatter(
                [group_idx * 2 + x_offset], [late_means.loc[subject]], color=color_map[subject], zorder=4, alpha=1,
                edgecolor='black', linewidth=1.5, s=70
            )

        tick_positions.append(group_idx * 2)
        tick_labels.append(group)

    # Customizing the X-axis
    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=12)
    plt.xlabel("Groups", fontsize=14)
    plt.ylabel(feature_name, fontsize=14)
    plt.title(f"Barplot Early and Late ({feature_name}) - Channel: {channel_name} - Scaler: MinMaxScaler", fontsize=16)

    # Create a unique legend for the patients (to avoid duplicates)
    for subject, color in color_map.items():
        patient_legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=subject))

    # Creiamo un dizionario per raggruppare i pazienti per gruppo
    grouped_legends = {}
    for subject, color in subject_color_map.items():
        patient_group = data[data["Subject"] == subject]["Group"].iloc[0]  # Trova il gruppo del paziente
        if patient_group not in grouped_legends:
            grouped_legends[patient_group] = []
        grouped_legends[patient_group].append((subject, color))

    # Creiamo la legenda ordinata per gruppi
    legend_handles = []
    for group in group_order:  # Seguiamo l'ordine definito nei gruppi
        if group in grouped_legends:
            legend_handles.append(
                plt.Line2D([0], [0], color='black', lw=0, label=f"Patients {group}:"))  # Titolo del gruppo
            for subject, color in grouped_legends[group]:
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=subject))

    # Aggiungiamo la legenda Early/Late
    legend_handles.append(plt.Line2D([0], [0], color='red', lw=4, label='Early'))
    legend_handles.append(plt.Line2D([0], [0], color='blue', lw=4, label='Late'))

    # Impostiamo la legenda finale
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, title="Patients")

    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_early_and_late_barplots_standardscaler(data, feature_name, channel_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=None):
    """
    Creates and displays barplots for the "Early" and "Late" phases, grouping patients by group.
    The barplots represent the group means, and colored dots represent the individual patient means.

    Parameters:
    - data: DataFrame with the data.
    - feature_name: Name of the feature to plot (y column).
    - channel_name: Channel to plot.
    - group_order: Optional list to order the groups on the x-axis.
    """
    # **Apply StandardScaler to the entire dataset**
    scaler = StandardScaler()
    data[feature_name] = scaler.fit_transform(data[[feature_name]])

    # Ensure that the 'Group' column is categorical with the desired order
    data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)

    # Filter data by the specified channel
    channel_data = data[data['Channel'] == channel_name]
    #channel_data = data[data['Brain region'] == channel_name]

    # Filter data by "Early" and "Late" phases
    combined_data = channel_data[channel_data['Phase_Assigned'].isin(["Early", "Late"])]

    # Plot creation
    plt.figure(figsize=(12, 6))
    x_offset = 0.3  # Distance between Early and Late bars
    tick_positions = []
    tick_labels = []

    color_map = subject_color_map

    # Variable to handle patient legends (to avoid duplicates)
    patient_legend_handles = []

    # Plot data for each group
    for group_idx, group in enumerate(combined_data['Group'].cat.categories):
        group_data = combined_data[combined_data['Group'] == group]

        early_data = group_data[group_data['Phase_Assigned'] == "Early"]
        late_data = group_data[group_data['Phase_Assigned'] == "Late"]

        early_means = early_data.groupby('Subject')[feature_name].mean()
        late_means = late_data.groupby('Subject')[feature_name].mean()

        early_group_mean = early_means.mean()
        late_group_mean = late_means.mean()

        # **Barplot for group means**
        plt.bar(
            group_idx * 2 - x_offset, early_group_mean, color='red', width=0.5, label="Early" if group_idx == 0 else "",
            zorder=3
        )
        plt.bar(
            group_idx * 2 + x_offset, late_group_mean, color='blue', width=0.5, label="Late" if group_idx == 0 else "",
            zorder=3
        )

        # Add dots for each patient with a unique color
        for idx, subject in enumerate(early_means.index):
            plt.scatter(
                [group_idx * 2 - x_offset], [early_means.loc[subject]], color=color_map[subject], zorder=4, alpha=1,
                edgecolor='black', linewidth=1.5, s=70
            )

        for idx, subject in enumerate(late_means.index):
            plt.scatter(
                [group_idx * 2 + x_offset], [late_means.loc[subject]], color=color_map[subject], zorder=4, alpha=1,
                edgecolor='black', linewidth=1.5, s=70
            )

        tick_positions.append(group_idx * 2)
        tick_labels.append(group)

    # Customizing the X-axis
    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=12)
    plt.xlabel("Groups", fontsize=14)
    plt.ylabel(feature_name, fontsize=14)
    plt.title(f"Barplot Early and Late ({feature_name}) - Channel: {channel_name} - Scaler: StandardScaler", fontsize=16)

    # Create a unique legend for the patients (to avoid duplicates)
    for subject, color in color_map.items():
        patient_legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=subject))

    # Creiamo un dizionario per raggruppare i pazienti per gruppo
    grouped_legends = {}
    for subject, color in subject_color_map.items():
        patient_group = data[data["Subject"] == subject]["Group"].iloc[0]  # Trova il gruppo del paziente
        if patient_group not in grouped_legends:
            grouped_legends[patient_group] = []
        grouped_legends[patient_group].append((subject, color))

    # Creiamo la legenda ordinata per gruppi
    legend_handles = []
    for group in group_order:  # Seguiamo l'ordine definito nei gruppi
        if group in grouped_legends:
            legend_handles.append(
                plt.Line2D([0], [0], color='black', lw=0, label=f"Patients {group}:"))  # Titolo del gruppo
            for subject, color in grouped_legends[group]:
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=subject))

    # Aggiungiamo la legenda Early/Late
    legend_handles.append(plt.Line2D([0], [0], color='red', lw=4, label='Early'))
    legend_handles.append(plt.Line2D([0], [0], color='blue', lw=4, label='Late'))

    # Impostiamo la legenda finale
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, title="Patients")

    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_early_and_late_barplots_robustscaler(data, feature_name, channel_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=None):
    """
    Creates and displays barplots for the "Early" and "Late" phases, grouping patients by group.
    The barplots represent the group means, and colored dots represent the individual patient means.

    Parameters:
    - data: DataFrame with the data.
    - feature_name: Name of the feature to plot (y column).
    - channel_name: Channel to plot.
    - group_order: Optional list to order the groups on the x-axis.
    """
    # **Apply RobustScaler to the entire dataset**
    scaler = RobustScaler()
    data[feature_name] = scaler.fit_transform(data[[feature_name]])

    # Ensure that the 'Group' column is categorical with the desired order
    data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)

    # Filter data by the specified channel
    channel_data = data[data['Channel'] == channel_name]
    #channel_data = data[data['Brain region'] == channel_name]

    # Filter data by "Early" and "Late" phases
    combined_data = channel_data[channel_data['Phase_Assigned'].isin(["Early", "Late"])]

    # Plot creation
    plt.figure(figsize=(12, 6))
    x_offset = 0.3  # Distance between Early and Late bars
    tick_positions = []
    tick_labels = []

    color_map = subject_color_map

    # Variable to handle patient legends (to avoid duplicates)
    patient_legend_handles = []

    # Plot data for each group
    for group_idx, group in enumerate(combined_data['Group'].cat.categories):
        group_data = combined_data[combined_data['Group'] == group]

        early_data = group_data[group_data['Phase_Assigned'] == "Early"]
        late_data = group_data[group_data['Phase_Assigned'] == "Late"]

        early_means = early_data.groupby('Subject')[feature_name].mean()
        late_means = late_data.groupby('Subject')[feature_name].mean()

        early_group_mean = early_means.mean()
        late_group_mean = late_means.mean()

        # **Barplot for group means**
        plt.bar(
            group_idx * 2 - x_offset, early_group_mean, color='red', width=0.5, label="Early" if group_idx == 0 else "",
            zorder=3
        )
        plt.bar(
            group_idx * 2 + x_offset, late_group_mean, color='blue', width=0.5, label="Late" if group_idx == 0 else "",
            zorder=3
        )

        # Add dots for each patient with a unique color
        for idx, subject in enumerate(early_means.index):
            plt.scatter(
                [group_idx * 2 - x_offset], [early_means.loc[subject]], color=color_map[subject], zorder=4, alpha=1,
                edgecolor='black', linewidth=1.5, s=70
            )

        for idx, subject in enumerate(late_means.index):
            plt.scatter(
                [group_idx * 2 + x_offset], [late_means.loc[subject]], color=color_map[subject], zorder=4, alpha=1,
                edgecolor='black', linewidth=1.5, s=70
            )

        tick_positions.append(group_idx * 2)
        tick_labels.append(group)

    # Customizing the X-axis
    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=12)
    plt.xlabel("Groups", fontsize=14)
    plt.ylabel(feature_name, fontsize=14)
    plt.title(f"Barplot Early and Late ({feature_name}) - Channel: {channel_name} - Scaler: RobustScaler", fontsize=16)

    # Create a unique legend for the patients (to avoid duplicates)
    for subject, color in color_map.items():
        patient_legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=subject))

    # Creiamo un dizionario per raggruppare i pazienti per gruppo
    grouped_legends = {}
    for subject, color in subject_color_map.items():
        patient_group = data[data["Subject"] == subject]["Group"].iloc[0]  # Trova il gruppo del paziente
        if patient_group not in grouped_legends:
            grouped_legends[patient_group] = []
        grouped_legends[patient_group].append((subject, color))

    # Creiamo la legenda ordinata per gruppi
    legend_handles = []
    for group in group_order:  # Seguiamo l'ordine definito nei gruppi
        if group in grouped_legends:
            legend_handles.append(
                plt.Line2D([0], [0], color='black', lw=0, label=f"Patients {group}:"))  # Titolo del gruppo
            for subject, color in grouped_legends[group]:
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=subject))

    # Aggiungiamo la legenda Early/Late
    legend_handles.append(plt.Line2D([0], [0], color='red', lw=4, label='Early'))
    legend_handles.append(plt.Line2D([0], [0], color='blue', lw=4, label='Late'))

    # Impostiamo la legenda finale
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, title="Patients")

    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def plot_early_and_late_phases_grouped_ordered(data, feature_name, channel_name,group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=None):
    """
    Crea e mostra i boxplot per le fasi "Early" e "Late", raggruppando i pazienti per gruppo, per un singolo canale.
    Ogni grafico rappresenta un canale specifico, con boxplot per le fasi "Early" e "Late", e barplot per la media dei gruppi
    con intervallo SEM. Ora include anche la media di ciascun paziente.

    Parametri:
    - data: DataFrame con i dati.
    - feature_name: Nome della feature da plottare (colonna y).
    - channel_name: Canale da plottare.
    - group_order: Lista opzionale per ordinare i gruppi sull'asse x.
    """

    # Assicurati che la colonna 'Group' sia categorica con l'ordine desiderato
    data['Group'] = pd.Categorical(data['Group'], categories=group_order)

    # Filtra i dati per il canale specifico
    channel_data = data[data['Channel'] == channel_name]

    #channel_data = data[data['Brain region'] == channel_name]

    # Filtra i dati per le fasi "Early" e "Late"
    combined_data = channel_data[channel_data['Phase_Assigned'].isin(["Early", "Late"])]

    # Conta il numero di epoche Early e Late per ogni soggetto
    epoch_counts = combined_data.groupby(['Subject', 'Phase_Assigned']).size().unstack(fill_value=0)

    from scipy.stats import shapiro, ttest_rel, wilcoxon, f_oneway, kruskal, fligner, bartlett, levene
    from statsmodels.stats.power import TTestPower, FTestAnovaPower
    '''
    # Analisi statistica tra gruppi (Early / Late separatamente)
    for phase in ["Early", "Late"]:
        print(f"\n===== {channel_name} | Phase: {phase} =====")
        phase_data = channel_data[channel_data['Phase_Assigned'] == phase]
        subject_means = phase_data.groupby(['Subject', 'Group'])[feature_name].mean().reset_index()

        grouped_data = [subject_means[subject_means['Group'] == g][feature_name].dropna() for g in group_order]
        normality = {g: shapiro(d)[1] if len(d) >= 3 else None for g, d in zip(group_order, grouped_data)}
        print(f"Normalità: {normality}")

        all_normal = all(p is None or p > 0.05 for p in normality.values())

        if all_normal:
            print("Distribution: Normal")
            levene_p = levene(*grouped_data)[1]
            bartlett_p = bartlett(*grouped_data)[1]
            var_test = "Levene" if levene_p < 0.05 else "Bartlett"
            var_p = levene_p if levene_p < 0.05 else bartlett_p
        else:
            print("Distribution: Not Normal")
            fligner_p = fligner(*grouped_data)[1]
            var_test = "Fligner-Killeen"
            var_p = fligner_p


        #print(f"Test di varianza ({var_test}): p = {var_p:.3e}")
        if var_p < 0.05:
            print(f"Variance Test: Heterogeneous (p = {var_p:.3e})")
        else:
            print(f"Variance Test: Homogeneous (p = {var_p:.3e})")


        if all_normal:
            if var_p > 0.05:
                stat, pval = f_oneway(*grouped_data)
                test_type = "One-Way ANOVA"
            else:
                stat, pval = f_oneway(*grouped_data)
                test_type = "Welch's ANOVA"

        else:
            stat, pval = kruskal(*grouped_data)
            test_type = "Kruskal-Wallis"

        print(f"{test_type}: stat = {stat:.3f}, p = {pval:.3e}")

        if test_type == "One-Way ANOVA":
            all_data = subject_means[[feature_name, 'Group']].dropna()
            grand_mean = all_data[feature_name].mean()
            ss_total = ((all_data[feature_name] - grand_mean) ** 2).sum()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in grouped_data)
            eta_squared = ss_between / ss_total
            print(f"Effect size (η²): {eta_squared:.3f}")
            power_analysis = FTestAnovaPower()
            n_per_group = power_analysis.solve_power(effect_size=np.sqrt(eta_squared / (1 - eta_squared)),
                                                     alpha=0.05, power=0.80, k_groups=len(group_order))
            print(f"Soggetti per gruppo per potenza 0.80: {np.ceil(n_per_group)}")
        elif test_type == "Kruskal-Wallis":
            n_total = sum(len(g) for g in grouped_data)
            epsilon_squared = (stat - len(group_order) + 1) / (n_total - len(group_order))
            eta_squared_kw = (stat * (n_total + 1)) / (n_total ** 2 - 1)
            print(f"Effect size (ε²): {epsilon_squared:.3f}")
            print(f"Stima eta² da Kruskal-Wallis: {eta_squared_kw:.3f}")
            power_analysis = FTestAnovaPower()
            try:
                effect_size_kw = np.sqrt(eta_squared_kw / (1 - eta_squared_kw))
                n_per_group_kw = power_analysis.solve_power(effect_size=effect_size_kw,
                                                            alpha=0.05, power=0.80, k_groups=len(group_order))
                print(f"Soggetti per gruppo per potenza 0.80 (da eta² approssimato): {np.ceil(n_per_group_kw)}")
            except:
                print("⚠ Errore nella stima del power da eta² (valore troppo estremo?)")

        # Stampa della mean e SEM per ciascun gruppo per fase (Early / Late)
        for group_idx, group in enumerate(group_order):
            group_phase_data = phase_data[phase_data['Group'] == group]
            mean_value = group_phase_data.groupby('Subject')[feature_name].mean().mean()
            sem_value = group_phase_data.groupby('Subject')[feature_name].mean().sem()
            print(f"Group: {group}, Phase: {phase} - Mean: {mean_value:.3f}, SEM: {sem_value:.3f}")

    # Confronto tra i gruppi per ogni fase

    # Confronto Early vs Late per ogni gruppo
    for group in group_order:
        print(f"\n===== {channel_name} | Confronto Early vs Late per {group} =====")
        early_group = channel_data[(channel_data['Group'] == group) & (channel_data['Phase_Assigned'] == 'Early')]
        late_group = channel_data[(channel_data['Group'] == group) & (channel_data['Phase_Assigned'] == 'Late')]

        early_means_raw = early_group.groupby('Subject')[feature_name].mean()
        late_means_raw = late_group.groupby('Subject')[feature_name].mean()

        common_subjects = early_means_raw.index.intersection(late_means_raw.index)
        early_means = early_means_raw.loc[common_subjects]
        late_means = late_means_raw.loc[common_subjects]
        differences = early_means - late_means
        mean_early = early_means.mean()
        mean_late = late_means.mean()
        diff = mean_late - mean_early
        diff_percent = (diff / mean_early) * 100
        print(f"Difference: {diff:.2f} ({diff_percent:+.2f}%)")

        shapiro_p = shapiro(differences)[1]
        print(f"Test di normalità sulle differenze (Early - Late) per {group}: p = {shapiro_p:.3e}")

        if shapiro_p > 0.05:
            stat, pval = ttest_rel(early_means, late_means, alternative="greater" )
            test_type = "T-test paired"
            print(f"{test_type}: stat = {stat:.3f}, p = {pval:.3e}")
            pooled_std = np.sqrt(((early_means.std() ** 2) + (late_means.std() ** 2)) / 2)
            cohen_d = np.abs(np.mean(differences)) / pooled_std
            print(f"Effect size (Cohen's d): {cohen_d:.3f}")
            power_analysis = TTestPower()
            n_required = power_analysis.solve_power(effect_size=cohen_d, alpha=0.05, power=0.80)
            print(f"Soggetti per gruppo per potenza 0.80: {np.ceil(n_required)}")
        else:
            stat, pval = wilcoxon(early_means, late_means)
            test_type = "Wilcoxon signed-rank"
            print(f"{test_type}: stat = {stat:.3f}, p = {pval:.3e}")

            # 1. Rimuove le differenze nulle
            diffs_nonzero = differences[differences != 0]
            n = len(diffs_nonzero)
            from scipy.stats import rankdata
            # 2. Calcola i ranghi dei valori assoluti
            ranks = rankdata(np.abs(diffs_nonzero))

            # 3. Somma dei ranghi dei valori positivi → T+
            T_pos = np.sum(ranks[diffs_nonzero > 0])

            # 4. Statistiche attese
            expected_T = n * (n + 1) / 4
            std_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

            # 5. Z-score con correzione di continuità
            z = (T_pos - expected_T - 0.5) / std_T

            # 6. Effect size r e d_W
            r = z / np.sqrt(n)
            print(f"Effect size (r): {r:.3f}")

            d_w = (2 * r) / np.sqrt(1 - r ** 2)
            print(f"Approximated Cohen's $d_W$: {d_w:.3f}")

            # 7. Stima potenza
            power_analysis = TTestPower()
            n_required = power_analysis.solve_power(effect_size=d_w, alpha=0.05, power=0.80)
            print(f"Soggetti per gruppo per potenza 0.80: {np.ceil(n_required)}")
    '''
    # Stampa il numero di epoche per ogni soggetto
    print("\nNumero di epoche per ciascun soggetto:")
    print(epoch_counts)



    color_map = subject_color_map

    # Creazione del grafico
    plt.figure(figsize=(16, 8))
    x_offset = 0.4
    tick_positions = []
    tick_labels = []

    # Traccia i dati per ogni gruppo
    for group_idx, group in enumerate(combined_data['Group'].cat.categories):
        group_data = combined_data[combined_data['Group'] == group]
        group_data = group_data.sort_values(by=["Subject"])

        early_data = group_data[group_data['Phase_Assigned'] == "Early"]
        late_data = group_data[group_data['Phase_Assigned'] == "Late"]

        early_means = early_data.groupby('Subject')[feature_name].mean()
        late_means = late_data.groupby('Subject')[feature_name].mean()

        early_group_mean = early_means.mean()
        late_group_mean = late_means.mean()
        early_sem = early_means.sem()
        late_sem = late_means.sem()

        sns.boxplot(
            x=np.full(len(early_data), group_idx * 2 - x_offset),
            y=early_data[feature_name],
            hue=early_data['Subject'],
            data=early_data,
            palette=color_map,
            dodge=True,
            width=0.6,
            boxprops=dict(alpha=0.8),
            showfliers=False
        )

        sns.boxplot(
            x=np.full(len(late_data), group_idx * 2 + x_offset),
            y=late_data[feature_name],
            hue=late_data['Subject'],
            data=late_data,
            palette=color_map,
            dodge=True,
            width=0.6,
            boxprops=dict(alpha=0.8),
            showfliers=False
        )

        plt.bar(
            group_idx * 2 - x_offset, early_group_mean, color='red', width=0.2, zorder=3
        )
        plt.bar(
            group_idx * 2 + x_offset, late_group_mean, color='blue', width=0.2, zorder=3
        )

        plt.errorbar(
            group_idx * 2 - x_offset, early_group_mean, yerr=early_sem, fmt='none', color='black', capsize=5, zorder=4
        )
        plt.errorbar(
            group_idx * 2 + x_offset, late_group_mean, yerr=late_sem, fmt='none', color='black', capsize=5, zorder=4
        )

        plt.scatter(
            np.full(len(early_means), group_idx * 2 - x_offset),
            early_means.values,
            color=[color_map[subj] for subj in early_means.index],
            edgecolor='black',
            linewidth=0.5,
            zorder=5,
            s=50
        )

        plt.scatter(
            np.full(len(late_means), group_idx * 2 + x_offset),
            late_means.values,
            color=[color_map[subj] for subj in late_means.index],
            edgecolor='black',
            linewidth=0.5,
            zorder=5,
            s=50
        )

        tick_positions.append(group_idx * 2)
        tick_labels.append(group)

    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=12)
    plt.xlabel("Groups", fontsize=14)
    plt.ylabel(f'{feature_name}', fontsize=14)
    plt.title(f"{feature_name}", fontsize=16)

    # Creiamo un dizionario per raggruppare i pazienti per gruppo
    grouped_legends = {}
    for subject, color in subject_color_map.items():
        patient_group = data[data["Subject"] == subject]["Group"].iloc[0]  # Trova il gruppo del paziente
        if patient_group not in grouped_legends:
            grouped_legends[patient_group] = []
        grouped_legends[patient_group].append((subject, color))

    # Creiamo la legenda ordinata per gruppi
    legend_handles = []

    # Aggiungiamo la legenda Early/Late
    legend_handles.append(plt.Line2D([0], [0], color='red', lw=4, label='Early'))
    legend_handles.append(plt.Line2D([0], [0], color='blue', lw=4, label='Late'))

    for group in group_order:  # Seguiamo l'ordine definito nei gruppi
        if group in grouped_legends:
            legend_handles.append(
                plt.Line2D([0], [0], color='black', lw=0, label=f"{group}:"))  # Titolo del gruppo
            for subject, color in grouped_legends[group]:
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=subject))


    # Impostiamo la legenda finale
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title='Legend')

    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # Imposta i limiti dell'asse y per iniziare da un valore maggiore di zero
    all_values = combined_data[feature_name].values
    ymin = np.min(all_values)
    ymax = np.max(all_values)
    padding = 0.1 * (ymax - ymin)
    plt.ylim(bottom=ymin - padding * 0.2, top=ymax + padding)
    plt.ylabel(f'SWA (µV²/Hz)', fontsize=14)
    plt.title(f"SWA at Subject-Level: Single Channel Fz (E15)", fontsize=16)
    plt.show()


def plot_early_and_late_bars_with_points(data, feature_name, channel_name,
                                         group_order=['CTL', 'DNV', 'ADV', 'DYS'],
                                         subject_color_map=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import shapiro, ttest_rel, wilcoxon, f_oneway, kruskal, fligner, bartlett, levene
    from statsmodels.stats.power import TTestPower, FTestAnovaPower

    data['Group'] = pd.Categorical(data['Group'], categories=group_order)
    channel_data = data[data['Channel'] == channel_name]
    combined_data = channel_data[channel_data['Phase_Assigned'].isin(["Early", "Late"])]

    plt.figure(figsize=(12, 6))
    x_offset = 0.3
    tick_positions = []
    tick_labels = []

    ymin, ymax = float('inf'), float('-inf')

    # Analisi statistica gruppi (Early / Late separatamente)
    for phase in ["Early", "Late"]:
        print(f"\n===== {channel_name} | Phase: {phase} =====")
        phase_data = channel_data[channel_data['Phase_Assigned'] == phase]
        subject_means = phase_data.groupby(['Subject', 'Group'])[feature_name].mean().reset_index()

        grouped_data = [subject_means[subject_means['Group'] == g][feature_name].dropna() for g in group_order]
        normality = {g: shapiro(d)[1] if len(d) >= 3 else None for g, d in zip(group_order, grouped_data)}
        print(f"Normalità: {normality}")

        all_normal = all(p is None or p > 0.05 for p in normality.values())

        if all_normal:
            print("Distribution: Normal")
            levene_p = levene(*grouped_data)[1]
            bartlett_p = bartlett(*grouped_data)[1]
            var_test = "Levene" if levene_p < 0.05 else "Bartlett"
            var_p = levene_p if levene_p < 0.05 else bartlett_p
        else:
            print("Distribution: Not Normal")
            fligner_p = fligner(*grouped_data)[1]
            var_test = "Fligner-Killeen"
            var_p = fligner_p

        if var_p < 0.05:
            print(f"Variance test: Heterogeneous (p = {var_p:.3e})")
        else:
            print(f"Variance test: Homogeneous (p = {var_p:.3e})")

        if all_normal:
            if var_p > 0.05:
                stat, pval = f_oneway(*grouped_data)
                test_type = "One-Way ANOVA"
            else:
                stat, pval = f_oneway(*grouped_data)
                test_type = "Welch's ANOVA"
        else:
            stat, pval = kruskal(*grouped_data)
            test_type = "Kruskal-Wallis"

        print(f"{test_type}: stat = {stat:.3f}, p = {pval:.3e}")

        if test_type == "One-Way ANOVA":
            all_data = subject_means[[feature_name, 'Group']].dropna()
            grand_mean = all_data[feature_name].mean()
            ss_total = ((all_data[feature_name] - grand_mean) ** 2).sum()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in grouped_data)
            eta_squared = ss_between / ss_total
            print(f"Effect size (η²): {eta_squared:.3f}")
            power_analysis = FTestAnovaPower()
            n_per_group = power_analysis.solve_power(effect_size=np.sqrt(eta_squared / (1 - eta_squared)),
                                                     alpha=0.05, power=0.80, k_groups=len(group_order))
            print(f"Soggetti per gruppo per potenza 0.80: {np.ceil(n_per_group)}")
        elif test_type == "Kruskal-Wallis":
            n_total = sum(len(g) for g in grouped_data)
            epsilon_squared = (stat - len(group_order) + 1) / (n_total - len(group_order))
            print(f"Effect size (ε²): {epsilon_squared:.3f}")
            eta_squared_kw = (stat * (n_total + 1)) / (n_total ** 2 - 1)
            print(f"Stima eta² da Kruskal-Wallis: {eta_squared_kw:.3f}")
            power_analysis = FTestAnovaPower()
            try:
                effect_size_kw = np.sqrt(eta_squared_kw / (1 - eta_squared_kw))
                n_per_group_kw = power_analysis.solve_power(effect_size=effect_size_kw,
                                                            alpha=0.05, power=0.80, k_groups=len(group_order))
                print(f"Soggetti per gruppo per potenza 0.80 (da eta² approssimato): {np.ceil(n_per_group_kw)}")
            except:
                print("⚠ Errore nella stima del power da eta² (valore troppo estremo?)")

        # Stampa della mean e SEM per ciascun gruppo per fase (Early / Late)
        for group_idx, group in enumerate(group_order):
            group_phase_data = phase_data[phase_data['Group'] == group]
            mean_value = group_phase_data.groupby('Subject')[feature_name].mean().mean()
            sem_value = group_phase_data.groupby('Subject')[feature_name].mean().sem()
            print(f"Group: {group}, Phase: {phase} - Mean: {mean_value:.3f}, SEM: {sem_value:.3f}")

    # Confronto Early vs Late per ogni gruppo
    for group in group_order:
        print(f"\n===== {channel_name} | Confronto Early vs Late per {group} =====")
        early_group = channel_data[(channel_data['Group'] == group) & (channel_data['Phase_Assigned'] == 'Early')]
        late_group = channel_data[(channel_data['Group'] == group) & (channel_data['Phase_Assigned'] == 'Late')]

        early_means_raw = early_group.groupby('Subject')[feature_name].mean()
        print(early_means_raw)
        late_means_raw = late_group.groupby('Subject')[feature_name].mean()

        common_subjects = early_means_raw.index.intersection(late_means_raw.index)
        early_means = early_means_raw.loc[common_subjects]
        late_means = late_means_raw.loc[common_subjects]
        differences = early_means - late_means
        mean_early = early_means.mean()
        mean_late = late_means.mean()
        diff = mean_late - mean_early
        diff_percent = (diff / mean_early) * 100
        print(f"Difference: {diff:.2f} ({diff_percent:+.2f}%)")

        shapiro_p = shapiro(differences)[1]
        print(f"Test di normalità sulle differenze (Early - Late) per {group}: p = {shapiro_p:.3e}")

        if shapiro_p > 0.05:
            stat, pval = ttest_rel(early_means, late_means)
            test_type = "T-test paired"
            print(f"{test_type}: stat = {stat:.3f}, p = {pval:.3e}")
            pooled_std = np.sqrt(((early_means.std() ** 2) + (late_means.std() ** 2)) / 2)
            cohen_d = np.abs(np.mean(differences)) / pooled_std
            print(f"Effect size (Cohen's d): {cohen_d:.3f}")
            power_analysis = TTestPower()
            n_required = power_analysis.solve_power(effect_size=cohen_d, alpha=0.05, power=0.80)
            print(f"Soggetti per gruppo per potenza 0.80: {np.ceil(n_required)}")
        else:
            stat, pval = wilcoxon(early_means, late_means)
            test_type = "Wilcoxon signed-rank"
            print(f"{test_type}: stat = {stat:.3f}, p = {pval:.3e}")

            # Calcolo z, r e d_W
            n = len(differences)
            diffs_nonzero = differences[differences != 0]
            T = np.sum(np.abs(diffs_nonzero.rank()))
            expected_T = n * (n + 1) / 4
            std_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            z = (T - expected_T - 0.5) / std_T
            r = z / np.sqrt(n)
            print(f"Effect size (r): {r:.3f}")

            # Conversione in Cohen's d_W
            d_w = (2 * r) / np.sqrt(1 - r ** 2)
            print(f"Approximated Cohen's $d_W$: {d_w:.3f}")

            # Stima della potenza con TTestPower usando d_W
            power_analysis = TTestPower()
            n_required = power_analysis.solve_power(effect_size=d_w, alpha=0.05, power=0.80)
            print(f"Soggetti per gruppo per potenza 0.80: {np.ceil(n_required)}")

        print(f"\n--- Dettaglio soggetti del gruppo {group} ---")
        for subj in common_subjects:
            early_val = early_means[subj]
            late_val = late_means[subj]
            diff = late_val - early_val  # differenza assoluta
            perc_diff = (diff / early_val) * 100 if early_val != 0 else float('inf')  # differenza percentuale

            if perc_diff > 15:
                trend = "↑ Late > Early"
            elif perc_diff < -15:
                trend = "↓ Late < Early"
            else:
                trend = "~ Stable"

            print(f"Subject: {subj} | Early: {early_val:.2f}, Late: {late_val:.2f}, "
                  f"Δ: {diff:+.2f}, Δ%: {perc_diff:+.2f}% → {trend}")

    # Plot
    shown_subjects = set()
    subject_handles = []

    for group_idx, group in enumerate(group_order):
        group_data = combined_data[combined_data['Group'] == group]
        early = group_data[group_data['Phase_Assigned'] == "Early"]
        late = group_data[group_data['Phase_Assigned'] == "Late"]

        early_means = early.groupby('Subject')[feature_name].mean()
        late_means = late.groupby('Subject')[feature_name].mean()

        early_mean = early_means.mean()
        late_mean = late_means.mean()
        early_sem = early_means.sem()
        late_sem = late_means.sem()

        plt.bar(group_idx * 2 - x_offset, early_mean, color='red', width=0.4, zorder=2,
                label='Early' if group_idx == 0 else "")
        plt.bar(group_idx * 2 + x_offset, late_mean, color='blue', width=0.4, zorder=2,
                label='Late' if group_idx == 0 else "")

        plt.errorbar(group_idx * 2 - x_offset, early_mean, yerr=early_sem, fmt='none', color='black', capsize=5,
                     zorder=3)
        plt.errorbar(group_idx * 2 + x_offset, late_mean, yerr=late_sem, fmt='none', color='black', capsize=5, zorder=3)

        for subj in early_means.index:
            color = subject_color_map.get(subj, 'gray')
            label = subj if subj not in shown_subjects else None
            plt.scatter(group_idx * 2 - x_offset, early_means[subj],
                        color=color, edgecolor='black', linewidth=0.5, zorder=4, s=50, label=label)
            shown_subjects.add(subj)

        for subj in late_means.index:
            color = subject_color_map.get(subj, 'gray')
            label = subj if subj not in shown_subjects else None
            plt.scatter(group_idx * 2 + x_offset, late_means[subj],
                        color=color, edgecolor='black', linewidth=0.5, zorder=4, s=50, label=label)
            shown_subjects.add(subj)

        tick_positions.append(group_idx * 2)
        tick_labels.append(group)
        ymin = min(ymin, early_means.min(), late_means.min())
        ymax = max(ymax, early_means.max(), late_means.max())

    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=12)
    plt.xlabel("Groups", fontsize=14)
    plt.ylabel(f'SWA (µV²/Hz)', fontsize=14)
    plt.title(f"SWA at Subject-Level: {channel_name} channels", fontsize=16)

    margin = (ymax - ymin) * 0.2 if ymax > ymin else 1
    plt.ylim(ymin - margin, ymax + margin)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # === Legenda soggetti come in plot_group_bars_no_early_late ===
    legend_handles = []
    grouped_legends = {}
    for subject, color in subject_color_map.items():
        subj_group_series = data[data['Subject'] == subject]['Group']
        subj_group = subj_group_series.iloc[0] if not subj_group_series.empty else 'Unknown'
        grouped_legends.setdefault(subj_group, []).append((subject, color))

    for group in group_order:
        if group in grouped_legends:
            legend_handles.append(plt.Line2D([0], [0], color='black', lw=0, label=f"{group}:"))
            for subject, color in grouped_legends[group]:
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                               markeredgecolor='black', markersize=7, label=subject)
                )

    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    # Legenda per Early e Late
    early_patch = mpatches.Patch(color='red', label='Early')
    late_patch = mpatches.Patch(color='blue', label='Late')

    # Legenda soggetti per gruppo
    subject_legend_handles = []
    grouped_legends = {}
    for subject, color in subject_color_map.items():
        subj_group_series = data[data['Subject'] == subject]['Group']
        subj_group = subj_group_series.iloc[0] if not subj_group_series.empty else 'Unknown'
        grouped_legends.setdefault(subj_group, []).append((subject, color))

    for group in group_order:
        if group in grouped_legends:
            subject_legend_handles.append(plt.Line2D([0], [0], color='black', lw=0, label=f"{group}:"))
            for subject, color in grouped_legends[group]:
                subject_legend_handles.append(
                    mlines.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, markeredgecolor='black',
                                  markersize=7, label=subject)
                )

    # Combinare tutti gli elementi
    legend_handles = [early_patch, late_patch] + subject_legend_handles

    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Legend", fontsize=6)
    # Imposta i limiti dell'asse y per iniziare da un valore maggiore di zero
    all_values = combined_data[feature_name].values
    ymin = np.min(all_values)
    ymax = np.max(all_values)
    padding = 0.1 * (ymax - ymin)
    plt.ylim(bottom=ymin - padding * 0.2, top=ymax + padding)
    plt.show()


def plot_group_bars_no_early_late(data, feature_name, channel_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'],
                                  subject_color_map=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import shapiro, levene, bartlett, fligner, f_oneway, kruskal
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import scikit_posthocs as sp
    from statsmodels.stats.power import FTestAnovaPower

    data['Group'] = pd.Categorical(data['Group'], categories=group_order)

    # Calcolo media per soggetto e regione
    channel_data = data[data['Channel'] == channel_name]
    subject_means = channel_data.groupby(['Subject', 'Group'])[feature_name].mean().reset_index()

    # === ANALISI STATISTICA ===
    grouped_data = [subject_means[subject_means['Group'] == g][feature_name].dropna() for g in group_order]
    normality = {g: shapiro(d)[1] if len(d) >= 3 else None for g, d in zip(group_order, grouped_data)}
    print(f"\nNormalità per {channel_name} - {feature_name}: {normality}")

    all_normal = all(p is None or p > 0.05 for p in normality.values())

    if all_normal:
        levene_p = levene(*grouped_data)[1]
        bartlett_p = bartlett(*grouped_data)[1]
        var_test = "Levene" if levene_p < 0.05 else "Bartlett"
        var_p = levene_p if levene_p < 0.05 else bartlett_p
    else:
        fligner_p = fligner(*grouped_data)[1]
        var_test = "Fligner-Killeen"
        var_p = fligner_p

    print(f"Test di varianza ({var_test}): p = {var_p:.3e}")

    # Test globale
    if all_normal:
        if var_p > 0.05:
            stat, p = f_oneway(*grouped_data)
            test_type = "ANOVA"
        else:
            stat, p = f_oneway(*grouped_data)  # Welch non disponibile in scipy f_oneway, va simulato
            test_type = "Welch's ANOVA (approssimato)"
    else:
        stat, p = kruskal(*grouped_data)
        test_type = "Kruskal-Wallis"

    print(f"Test globale: {test_type}, stat = {stat:.3f}, p = {p:.3e}")

    # Effect size
    if test_type.startswith("ANOVA"):
        all_data = subject_means[[feature_name, 'Group']].dropna()
        grand_mean = all_data[feature_name].mean()
        ss_total = ((all_data[feature_name] - grand_mean) ** 2).sum()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in grouped_data)
        eta_squared = ss_between / ss_total
        print(f"Effect Size (η²): {eta_squared:.3f}")
    elif test_type == "Kruskal-Wallis":
        # Approssimazione di epsilon squared
        n_total = sum(len(g) for g in grouped_data)
        epsilon_squared = (stat - len(group_order) + 1) / (n_total - len(group_order))
        print(f"Effect Size (ε²): {epsilon_squared:.3f}")

    # Post-hoc
    if p < 0.05:
        all_data = subject_means[[feature_name, 'Group']].dropna()
        if test_type == "ANOVA":
            tukey = pairwise_tukeyhsd(all_data[feature_name], all_data['Group'])
            print("Post-hoc Tukey:\n", tukey.summary())
        else:
            dunn = sp.posthoc_dunn(all_data, val_col=feature_name, group_col='Group', p_adjust='bonferroni')
            print("Post-hoc Dunn (Bonferroni):\n", dunn)

    # Stima potenza statistica
    if test_type.startswith("ANOVA") and eta_squared > 0:
        f_effect = np.sqrt(eta_squared / (1 - eta_squared))
        power_analysis = FTestAnovaPower()
        n_needed = power_analysis.solve_power(effect_size=f_effect, alpha=0.05, power=0.8, k_groups=len(group_order))
        print(f"Stima dimensione campionaria per gruppo (80% potenza): ~{int(np.ceil(n_needed))}")

    # === PLOTTING ===
    plt.figure(figsize=(12, 6))
    tick_positions = []
    tick_labels = []

    ymin, ymax = float('inf'), float('-inf')

    for group_idx, group in enumerate(group_order):
        group_subjects = subject_means[subject_means['Group'] == group]
        if group_subjects.empty:
            continue

        group_mean = group_subjects[feature_name].mean()
        group_sem = group_subjects[feature_name].sem()

        plt.bar(group_idx, group_mean, yerr=group_sem, color='gray', alpha=0.6, capsize=5, zorder=2)

        plt.scatter(
            np.full(len(group_subjects), group_idx),
            group_subjects[feature_name],
            color=[subject_color_map[subj] for subj in group_subjects['Subject']],
            edgecolor='black', linewidth=0.5, s=60, zorder=3
        )

        tick_positions.append(group_idx)
        tick_labels.append(group)
        ymin = min(ymin, group_subjects[feature_name].min())
        ymax = max(ymax, group_subjects[feature_name].max())

    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=12)
    plt.xlabel("Gruppi", fontsize=14)
    plt.ylabel(feature_name, fontsize=14)
    plt.title(f"{feature_name} per Gruppo - {channel_name}", fontsize=16)

    margin = (ymax - ymin) * 0.2 if ymax > ymin else 1
    plt.ylim(ymin - margin, ymax + margin)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    legend_handles = []
    grouped_legends = {}
    for subject, color in subject_color_map.items():
        subj_group = data[data['Subject'] == subject]['Group'].iloc[0]
        if subj_group not in grouped_legends:
            grouped_legends[subj_group] = []
        grouped_legends[subj_group].append((subject, color))

    for group in group_order:
        if group in grouped_legends:
            legend_handles.append(plt.Line2D([0], [0], color='black', lw=0, label=f"{group}:"))
            for subject, color in grouped_legends[group]:
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                               markeredgecolor='black', markersize=7, label=subject)
                )

    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Pazienti", fontsize=6)
    plt.tight_layout()
    plt.show()


# === Carica i dati ===
#file_path = r"D:\TESI\prova statistica\N2N3multitaperMeanPSD_specific_channels_149\_N2N3multitaperMeanPSD_specific_channels_149_aggregated_with_phases.csv"
file_path = r"D:\TESI\prova statistica\N2N3ALLENTROPY_specific_channels_149\_N2N3ALLENTROPY_specific_channels_149_aggregated_with_phases.csv"
#file_path = r"D:\TESI\prova statistica\N3MULTISCALEENTROPY_specific_channels_149\_N3MULTISCALEENTROPY_specific_channels_149_aggregated_with_phases"
#file_path=r"D:\TESI\prova statistica\N3CONN100_specific_channels_149\_N3CONN100_specific_channels_149_aggregated_with_phases.csv"

data = pd.read_csv(file_path)
data = data[data['Stage'] == 3]

# === Funzione (presunta già esistente) ===
# Assicurati che create_subject_color_map e plot_early_and_late_phases_grouped_ordered siano definite

# === Definisci le regioni cerebrali con i rispettivi canali ===
# === Definisci le regioni cerebrali con i canali aggiornati ===
regions = {
    "FP & F": np.sort(np.array([27, 33, 34, 38, 39, 47, 48, 26, 20, 19, 12, 11, 3, 2, 222,16, 22, 23, 24, 28, 29, 30, 35, 36, 40, 41, 42, 49, 50, 21, 15, 7, 14, 6,
                            207, 13, 5, 215, 4, 224, 223, 214, 206, 213, 205])),
    "ALL":  np.sort(np.array([27, 33, 34, 38, 39, 47, 48, 26, 20, 19, 12, 11, 3, 2, 222,16, 22, 23, 24, 28, 29, 30, 35, 36, 40, 41, 42, 49, 50, 21, 15, 7, 14, 6,
                            207, 13, 5, 215, 4, 224, 223, 214, 206, 213, 205,9, 17, 43, 44, 45, 51, 52, 53, 57, 58, 59, 60, 64, 65, 66, 71, 72, 8,
                            81, 186, 198, 197, 185, 132, 196, 184, 144, 204, 195, 183, 155,
                            194, 182, 164, 181, 173,55, 56, 62, 63, 69, 70, 74, 75, 84, 85, 96, 221, 212, 211, 203,
                            202, 193, 192, 180, 179, 171, 170,76, 77, 78, 79, 80, 86, 87, 88, 89, 97, 98, 99, 100, 110, 90,
                            101, 119, 172, 163, 154, 143, 131, 162, 153, 142, 130, 161,
                            152, 141, 129, 128,107, 108, 109, 116, 117, 118, 125, 126, 160, 151, 140, 150,
                            139, 127, 138])),
}
# === Specifica le feature da analizzare ===
feature_to_analyze = ['Mean PSD Total Delta']

# === Crea mappa dei colori per soggetto ===
subject_color_map = create_subject_color_map(data)

# === Loop su ogni regione e feature ===
for region_name, channel_list in regions.items():
    # Filtra i dati per i canali della regione
    region_data = data[data['Channel'].isin(channel_list)].copy()

    for feature_name in feature_to_analyze:
        # Media per soggetto, fase e gruppo
        aggregated_region_data = (
            region_data.groupby(['Subject', 'Phase_Assigned', 'Group'])[feature_name]
            .mean()
            .reset_index()
        )

        # Aggiungi colonna 'Channel' con il nome della regione per compatibilità con le funzioni
        aggregated_region_data['Channel'] = region_name

        print(f"\n### Grafico per caratteristica: {feature_name}, Regione: {region_name} ###")

        # === Plot multipli per la regione ===
        plot_early_and_late_bars_with_points(aggregated_region_data.copy(),feature_name=feature_name,channel_name=region_name,group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=subject_color_map)
        #plot_feature_per_patient_violin_and_sd_subplot(aggregated_region_data.copy(),feature_name=feature_name,channel_name=region_name,group_order=['CTL', 'DNV', 'ADV', 'DYS'])
        #plot_early_and_late_barplots_minmaxscaler(aggregated_region_data.copy(),feature_name=feature_name,channel_name=region_name,group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=subject_color_map)
        #plot_early_and_late_barplots_standardscaler(aggregated_region_data.copy(),feature_name=feature_name,channel_name=region_name,group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=subject_color_map)
        #plot_early_and_late_barplots_robustscaler(aggregated_region_data.copy(),feature_name=feature_name,channel_name=region_name,group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=subject_color_map)
        #plot_group_bars_no_early_late(aggregated_region_data.copy(),feature_name=feature_name,channel_name=region_name,group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=subject_color_map)

# === Analisi per un singolo canale specifico (fuori dal loop delle regioni) ===
single_channel = 15
channel_label = f"Fz"

for feature_name in feature_to_analyze:

    print(f"\n### Grafico per caratteristica: {feature_name}, Canale: {channel_label} ###")
    #plot_early_and_late_phases_grouped_ordered(data.copy(),feature_name=feature_name,channel_name=single_channel,group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=subject_color_map)

