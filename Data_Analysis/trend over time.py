import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib
from matplotlib.gridspec import GridSpec




def create_subject_color_map(data):
    """Genera una mappa di colori unica per ogni soggetto, mantenendo coerenza tra gruppi."""

    unique_groups = sorted(data['Group'].unique())  # Prendi i gruppi ordinati
    subject_color_map = {}

    for group in unique_groups:
        subjects_in_group = sorted(data[data['Group'] == group]['Subject'].unique())
        num_subjects = len(subjects_in_group)


        # Genera una palette di colori distinta per ogni gruppo
        group_colors = sns.color_palette("tab20", n_colors=num_subjects)

        # Mappa soggetti ai colori
        for subject, color in zip(subjects_in_group, group_colors):
            subject_color_map[subject] = color

    return subject_color_map

def plot_epochs_per_subject_by_group_with_linear_regression(data, feature_name, channel_name, subject_color_map):
    """ Plotta i dati per ogni soggetto diviso per gruppo con regressione lineare. """

    phase_column = 'Phase_Assigned' if 'Phase_Assigned' in data.columns else None
    channel_data = data[data['Channel'] == channel_name]
    #channel_data = data[data['Brain region'] == channel_name]
    groups = ['CTL', 'DNV', 'ADV', 'DYS']

    linear_regression_params = {'Group': [], 'Subject': [], 'Slope': [], 'Intercept': [], 'R_squared': []}
    num_groups = len(groups)

    fig, axes = plt.subplots(1, num_groups, figsize=(18, 8), sharey=True)
    if num_groups == 1:
        axes = [axes]

    vertical_offset_step = 0.01 * (data[feature_name].max() - data[feature_name].min())

    for i, group in enumerate(groups):
        group_data = channel_data[channel_data['Group'] == group]
        group_axes = axes[i]
        linear_model = LinearRegression()
        subject_offset = 0
        legend_elements = []

        group_subjects = sorted(group_data['Subject'].unique())

        for subject in group_subjects:
            subject_data = group_data[group_data['Subject'] == subject].copy().reset_index(drop=True)
            original_index = subject_data.index.values

            # Normalizza gli indici tra 0 e 100
            normalized_index = (
                (original_index - original_index.min()) / (original_index.max() - original_index.min()) * 100
                if len(original_index) > 1 else original_index
            )

            # Regressione lineare
            X = original_index.reshape(-1, 1)
            y = subject_data[feature_name].values
            linear_model.fit(X, y)
            y_pred = linear_model.predict(X)

            # Offset verticale per separare i soggetti
            adjusted_y_pred = y_pred + subject_offset
            adjusted_y = y + subject_offset

            # Determina il colore per soggetto
            subject_color = subject_color_map[subject]

            # Determina colori per fase
            if phase_column:
                phase_colors = subject_data[phase_column].map(
                    {'Early': 'red', 'Late': 'blue', 'Not considered': 'black'}
                ).fillna('black').tolist()
            else:
                phase_colors = ['black'] * len(subject_data)

            # Scatter plot con colori per soggetto
            group_axes.scatter(normalized_index, adjusted_y, s=5, color=subject_color, label=f"Subject {subject}")

            # Disegna la linea di regressione con colore per fase
            for j in range(len(original_index) - 1):
                group_axes.plot(
                    [normalized_index[j], normalized_index[j + 1]],
                    [adjusted_y_pred[j], adjusted_y_pred[j + 1]],
                    color=phase_colors[j], linewidth=2
                )

            # Salva parametri regressione
            linear_regression_params['Group'].append(group)
            linear_regression_params['Subject'].append(subject)
            linear_regression_params['Slope'].append(linear_model.coef_[0])
            linear_regression_params['Intercept'].append(linear_model.intercept_)
            linear_regression_params['R_squared'].append(linear_model.score(X, y))

            # Incrementa offset per il soggetto successivo
            subject_offset += vertical_offset_step

            # Aggiungi alla legenda
            legend_elements.append((f"Subject {subject}", subject_color))

        # Imposta il titolo e le etichette
        group_axes.set_title(f"Group {group}", fontsize=14)
        group_axes.set_xlabel("Normalized Epoch Index (0-100)", fontsize=12)
        if i == 0:
            group_axes.set_ylabel(feature_name, fontsize=12)
        else:
            group_axes.set_yticks([])

        group_axes.grid(True, linestyle="--", alpha=0.6)

        # Crea legenda con i colori dei soggetti
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=label)
            for label, color in legend_elements
        ]
        group_axes.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)

    plt.suptitle(f"Linear Regression - Channel: {channel_name}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, right=0.85)
    plt.show()

    # Creazione DataFrame parametri di regressione
    regression_comparison_df = pd.DataFrame(linear_regression_params)

    # Plotta il barplot per la pendenza (Slope)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Group', y='Slope', data=regression_comparison_df, ax=ax, color='blue', alpha=0.6)

    # Dizionario per separare i soggetti per gruppo
    grouped_handles = {}

    for i, row in regression_comparison_df.iterrows():
        subject = row['Subject']
        group = row['Group']
        scatter = ax.scatter(row['Group'], row['Slope'], color=subject_color_map[subject], s=50, zorder=5)

        # Salva l'handle per ogni gruppo
        if group not in grouped_handles:
            grouped_handles[group] = []
        grouped_handles[group].append((scatter, subject))

    ax.set_ylabel("Pendenza (Slope)", fontsize=12)
    ax.set_title("Confronto tra i gruppi - Pendenza della Regressione Lineare", fontsize=14)

    # Creazione della legenda divisa per gruppi
    legend_elements = []
    for group, items in grouped_handles.items():
        legend_elements.append((f"**{group}**", 'white'))  # Intestazione del gruppo
        for scatter, subject in items:
            legend_elements.append((f"Subject {subject}", subject_color_map[subject]))

    # Creazione degli handle per la legenda
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=label)
               for label, color in legend_elements]

    ax.legend(handles=handles, title="Subjects", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=7)

    plt.tight_layout()
    plt.show()

    # Plotta il barplot per l'intercetta (Intercept)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Group', y='Intercept', data=regression_comparison_df, ax=ax, color='orange', alpha=0.6)

    grouped_handles = {}

    for i, row in regression_comparison_df.iterrows():
        subject = row['Subject']
        group = row['Group']
        scatter = ax.scatter(row['Group'], row['Intercept'], color=subject_color_map[subject], s=50, zorder=5)

        if group not in grouped_handles:
            grouped_handles[group] = []
        grouped_handles[group].append((scatter, subject))

    ax.set_ylabel("Intercetta (Intercept)", fontsize=12)
    ax.set_title("Confronto tra i gruppi - Intercetta della Regressione Lineare", fontsize=14)

    legend_elements = []
    for group, items in grouped_handles.items():
        legend_elements.append((f"**{group}**", 'white'))
        for scatter, subject in items:
            legend_elements.append((f"Subject {subject}", subject_color_map[subject]))

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=label)
               for label, color in legend_elements]

    ax.legend(handles=handles, title="Subjects", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=7)

    plt.tight_layout()
    plt.show()

    # Plotta il barplot per R²
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Group', y='R_squared', data=regression_comparison_df, ax=ax, color='green', alpha=0.6)

    grouped_handles = {}

    for i, row in regression_comparison_df.iterrows():
        subject = row['Subject']
        group = row['Group']
        scatter = ax.scatter(row['Group'], row['R_squared'], color=subject_color_map[subject], s=50, zorder=5)

        if group not in grouped_handles:
            grouped_handles[group] = []
        grouped_handles[group].append((scatter, subject))

    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("Confronto tra i gruppi - R² della Regressione Lineare", fontsize=14)

    legend_elements = []
    for group, items in grouped_handles.items():
        legend_elements.append((f"**{group}**", 'white'))
        for scatter, subject in items:
            legend_elements.append((f"Subject {subject}", subject_color_map[subject]))

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=label)
               for label, color in legend_elements]

    ax.legend(handles=handles, title="Subjects", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_subjects_by_group_phases(data, feature_name, channel_name, subject_color_map):
    """ Crea piccoli subplots individuali per soggetto, con punti colorati per fase (senza regressione). """
    phase_column = 'Phase_Assigned' if 'Phase_Assigned' in data.columns else None
    channel_data = data[data['Channel'] == channel_name]
    groups = ['CTL', 'DNV', 'ADV', 'DYS']

    subject_list = sorted(channel_data['Subject'].unique())
    n_subjects = len(subject_list)

    n_cols = 6  # numero di colonne nei subplot
    n_rows = int(np.ceil(n_subjects / n_cols))

    fig = plt.figure(figsize=(n_cols * 3, n_rows * 2.5))
    gs = GridSpec(n_rows, n_cols, figure=fig)

    for idx, subject in enumerate(subject_list):
        subject_data = channel_data[channel_data['Subject'] == subject].copy().reset_index(drop=True)
        group = subject_data['Group'].iloc[0]
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])

        original_index = subject_data.index.values
        normalized_index = (
            (original_index - original_index.min()) / (original_index.max() - original_index.min()) * 100
            if len(original_index) > 1 else original_index
        )

        # Colori per fasi
        if phase_column:
            colors = subject_data[phase_column].map(
                {'Early': 'red', 'Late': 'blue'}
            ).fillna('black')
        else:
            colors = ['black'] * len(subject_data)

        ax.scatter(normalized_index, subject_data[feature_name], s=6, c=colors)

        ax.set_title(f"{group}-{subject}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle(f"Epoch Data per Soggetto - Channel: {channel_name}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


def plot_epochs_per_subject_by_group_with_polynomial_regression(data, feature_name, channel_name, poly_degree=2, subject_color_map=None):
    # Controlliamo se la colonna Phase_Assigned esiste
    phase_column = 'Phase_Assigned' if 'Phase_Assigned' in data.columns else None

    # Filtra i dati per il canale selezionato
    channel_data = data[data['Channel'] == channel_name]
    #channel_data = data[data['Brain region'] == channel_name]

    # Gruppi specifici
    groups = ['CTL', 'DNV', 'ADV', 'DYS']

    # Inizializza la struttura per salvare i parametri della regressione
    poly_regression_params = {
        'Group': [],
        'Subject': [],
        'Coefficients': [],
        'R_squared': [],
        'Global_Derivative': [],
        'Derivative_Area': [],
        'Derivative_Values': []
    }

    # Numero di gruppi
    num_groups = len(groups)

    # Creazione del plot con 1 riga e 4 colonne
    fig, axes = plt.subplots(1, num_groups, figsize=(14, 8), sharey=True)

    if num_groups == 1:
        axes = [axes]

    # Calcola l'offset verticale basato sull'intervallo dei dati
    vertical_offset_step = 0.15 * (data[feature_name].max() - data[feature_name].min())

    # Itera sui gruppi e plottane i dati
    for i, group in enumerate(groups):
        group_data = channel_data[channel_data['Group'] == group]
        num_patients = len(group_data['Subject'].unique())
        group_axes = axes[i]

        # Palette di colori per i soggetti
        color_map = subject_color_map

        poly_model = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression())

        subject_offset = 0  # Offset iniziale
        for subject in group_data['Subject'].unique():
            subject_data = group_data[group_data['Subject'] == subject].copy().reset_index(drop=True)

            # Indici originali
            original_index = subject_data.index.values.astype(float)

            # Normalizzazione dell'indice per scalare tra 0 e 100
            if original_index.max() > original_index.min():
                normalized_index = (original_index - original_index.min()) / (original_index.max() - original_index.min()) * 100
            else:
                normalized_index = np.zeros_like(original_index)  # Caso con un solo punto dati

            # Regressione polinomiale sugli indici originali
            X = original_index.reshape(-1, 1)
            y = subject_data[feature_name].values

            poly_model.fit(X, y)
            y_pred = poly_model.predict(X)

            # Applica l'offset verticale
            subject_offset += vertical_offset_step

            # Disegna la curva della regressione con il colore della fase
            for j in range(1, len(normalized_index)):
                start_idx = j - 1
                end_idx = j

                phase_start = subject_data[phase_column].iloc[start_idx] if phase_column else 'Not considered'
                phase_end = subject_data[phase_column].iloc[end_idx] if phase_column else 'Not considered'

                # Colore basato sulla fase
                if phase_start == 'Early' or phase_end == 'Early':
                    line_color = 'red'
                elif phase_start == 'Late' or phase_end == 'Late':
                    line_color = 'blue'
                else:
                    line_color = 'black'

                group_axes.plot(normalized_index[start_idx:end_idx + 1], y_pred[start_idx:end_idx + 1] + subject_offset,
                                color=line_color, linewidth=2)

            # Calcolo della derivata della regressione polinomiale
            r2_score = poly_model.score(X, y)
            coef = poly_model.named_steps['linearregression'].coef_

            derivative_function = np.poly1d(coef[::-1]).deriv()
            derivative_values = derivative_function(original_index)

            global_derivative = np.mean(derivative_values)
            area_under_derivative = np.trapz(derivative_values, original_index)

            # Salva i parametri
            poly_regression_params['Group'].append(group)
            poly_regression_params['Subject'].append(subject)
            poly_regression_params['Coefficients'].append(coef)
            poly_regression_params['R_squared'].append(r2_score)
            poly_regression_params['Global_Derivative'].append(global_derivative)
            poly_regression_params['Derivative_Area'].append(area_under_derivative)
            poly_regression_params['Derivative_Values'].append(derivative_values)

            # Scatter plot per il soggetto
            group_axes.scatter(normalized_index, subject_data[feature_name] + subject_offset,
                               color=subject_color_map[subject], alpha=1.0, s=10, label=f"Subject {subject}")

        group_axes.set_title(f"Group {group}", fontsize=14)
        group_axes.set_xlabel("Normalized Epoch Index (%)", fontsize=12)
        group_axes.set_ylabel(feature_name, fontsize=12)
        group_axes.grid(True, linestyle="--", alpha=0.6)

        # Rimuove voci duplicate nella legenda
        handles, labels = group_axes.get_legend_handles_labels()
        unique_handles_labels = {label: handle for handle, label in zip(handles, labels)}
        group_axes.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc='upper right', fontsize=7)

    plt.suptitle(f"Polynomial Regression ({poly_degree}° degree) - Channel: {channel_name}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    #####################
    # Analisi dei dati risultanti
    poly_regression_df = pd.DataFrame(poly_regression_params)

    # Verifica se ci sono valori NaN nella colonna 'Global_Derivative'
    print(poly_regression_df[poly_regression_df['Global_Derivative'].isna()])

    # Stampa il DataFrame per vedere se ci sono problemi con i dati
    print(poly_regression_df.head())

    # Mappa colori aggiornata per i soggetti
    color_map = subject_color_map

    #####################
    # *Plot della Derivata Globale per ogni gruppo*
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(x='Group', y='Global_Derivative', data=poly_regression_df, ax=ax, palette='Greens', alpha=0.6)

    grouped_handles = {}

    # Aggiungi i pallini con i valori individuali per ogni paziente
    for i, row in poly_regression_df.iterrows():
        subject = row['Subject']
        group = row['Group']
        scatter = ax.scatter(row['Group'], row['Global_Derivative'], color=color_map[subject], s=100, zorder=5)

        if group not in grouped_handles:
            grouped_handles[group] = []
        grouped_handles[group].append((scatter, subject))

    ax.set_ylabel("Global Derivative", fontsize=12)
    ax.set_title(f"Global Derivative (Degree {poly_degree})", fontsize=14)

    # Creazione della legenda divisa per gruppi
    legend_elements = []
    for group, items in grouped_handles.items():
        legend_elements.append((f"**{group}**", 'white'))
        for scatter, subject in items:
            legend_elements.append((f"Subject {subject}", color_map[subject]))

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=label)
               for label, color in legend_elements]

    ax.legend(handles=handles, title="Subjects", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=7)

    plt.tight_layout()
    plt.show()

    #####################
    # *Plot di R² per ogni gruppo*
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(x='Group', y='R_squared', data=poly_regression_df, ax=ax, palette='Blues', alpha=0.6)

    grouped_handles = {}

    # Aggiungi i pallini con i valori individuali per ogni paziente
    for i, row in poly_regression_df.iterrows():
        subject = row['Subject']
        group = row['Group']
        scatter = ax.scatter(row['Group'], row['R_squared'], color=color_map[subject], s=100, zorder=5)

        if group not in grouped_handles:
            grouped_handles[group] = []
        grouped_handles[group].append((scatter, subject))

    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title(f"R² Score (Degree {poly_degree})", fontsize=14)

    # Creazione della legenda divisa per gruppi
    legend_elements = []
    for group, items in grouped_handles.items():
        legend_elements.append((f"**{group}**", 'white'))
        for scatter, subject in items:
            legend_elements.append((f"Subject {subject}", color_map[subject]))

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=label)
               for label, color in legend_elements]

    ax.legend(handles=handles, title="Subjects", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=7)

    plt.tight_layout()
    plt.show()

    #####################
    # *Plot dell'Area sotto la curva della derivata per ogni gruppo*
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(x='Group', y='Derivative_Area', data=poly_regression_df, ax=ax, palette='Purples', alpha=0.6)

    grouped_handles = {}

    # Aggiungi i pallini con i valori individuali per ogni paziente
    for i, row in poly_regression_df.iterrows():
        subject = row['Subject']
        group = row['Group']
        scatter = ax.scatter(row['Group'], row['Derivative_Area'], color=color_map[subject], s=100, zorder=5)

        if group not in grouped_handles:
            grouped_handles[group] = []
        grouped_handles[group].append((scatter, subject))

    ax.set_ylabel("Area Under Derivative", fontsize=12)
    ax.set_title(f"Area Under Derivative (Degree {poly_degree})", fontsize=14)

    # Creazione della legenda divisa per gruppi
    legend_elements = []
    for group, items in grouped_handles.items():
        legend_elements.append((f"**{group}**", 'white'))
        for scatter, subject in items:
            legend_elements.append((f"Subject {subject}", color_map[subject]))

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=label)
               for label, color in legend_elements]

    ax.legend(handles=handles, title="Subjects", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=7)

    plt.tight_layout()
    plt.show()

    return

def plot_group_mean_and_std_over_normalized_sleep(data, feature_name, channel_name):
    channel_data = data[data['Channel'] == channel_name]
    groups = ['CTL', 'DNV', 'ADV', 'DYS']
    num_bins = 100  # punti percentuali

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, group in enumerate(groups):
        group_data = channel_data[channel_data['Group'] == group]
        grouped = []

        # Normalizza ogni soggetto a 100 punti
        for subject in group_data['Subject'].unique():
            subject_data = group_data[group_data['Subject'] == subject].copy()
            subject_data = subject_data.sort_index()

            # Crea indice normalizzato
            subject_data['Normalized_Index'] = np.linspace(0, 100, len(subject_data))

            # Bin per ottenere valori medi nei punti % comuni
            binned = subject_data.groupby(pd.cut(subject_data['Normalized_Index'], bins=num_bins)).agg({feature_name: 'mean'})
            binned.index = np.linspace(0, 100, num_bins)
            grouped.append(binned[feature_name].values)

        # Calcola media e std across soggetti
        group_array = np.array(grouped)
        mean_vals = np.nanmean(group_array, axis=0)
        std_vals = np.nanstd(group_array, axis=0)
        percent_axis = np.linspace(0, 100, num_bins)

        ax = axes[i]
        ax.plot(percent_axis, mean_vals, label='Mean', color='blue')
        ax.fill_between(percent_axis, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3, color='blue', label='±1 STD')

        ax.set_title(f"Group: {group}", fontsize=14)
        ax.set_xlabel("Normalized Sleep Progression (%)")
        ax.set_ylabel(feature_name)
        ax.grid(True)
        ax.legend()

    plt.suptitle(f"Mean ± STD of {feature_name} across subjects (Channel {channel_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


def find_best_poly_degree(data, feature_name, channel_name, max_degree=40):
    # Controlliamo se la colonna Phase_Assigned esiste
    phase_column = 'Phase_Assigned' if 'Phase_Assigned' in data.columns else None

    # Filtra i dati per il canale selezionato
    channel_data = data[data['Channel'] == channel_name]
    #channel_data = data[data['Brain region'] == channel_name]

    # Gruppi specifici
    groups = ['CTL', 'DNV', 'ADV', 'DYS']

    best_degrees = {}

    # Itera sui gruppi per determinare il miglior grado polinomiale per ciascuno
    for group in groups:
        group_data = channel_data[channel_data['Group'] == group]

        avg_r2_scores = []  # Salva gli R² medi per ogni grado
        for poly_degree in range(1, max_degree + 1):
            poly_model = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression())
            r2_scores = []

            for subject in group_data['Subject'].unique():
                subject_data = group_data[group_data['Subject'] == subject].copy().reset_index(drop=True)

                # Indici originali
                original_index = subject_data.index.values.astype(float)

                # Normalizzazione dell'indice per scalare tra 0 e 100
                if original_index.max() > original_index.min():
                    normalized_index = (original_index - original_index.min()) / (
                                original_index.max() - original_index.min()) * 100
                else:
                    normalized_index = np.zeros_like(original_index)  # Caso con un solo punto dati

                # Regressione polinomiale sugli indici originali
                X = original_index.reshape(-1, 1)
                y = subject_data[feature_name].values

                poly_model.fit(X, y)
                r2_scores.append(poly_model.score(X, y))  # Aggiungi il punteggio R²

            # Calcola la media dei punteggi R² per questo grado
            avg_r2_scores.append(np.mean(r2_scores))

        # Trova il miglior grado per questo gruppo
        best_degree = np.argmax(
            avg_r2_scores) + 1  # Indice di grado migliore (aggiungi 1 per allinearlo ai gradi reali)
        best_degrees[group] = best_degree
        print(
            f"Best polynomial degree for group {group}: {best_degree} (Average R²: {avg_r2_scores[best_degree - 1]:.3f})")

    return best_degrees

####################################################################################
from sklearn.preprocessing import MinMaxScaler


# Carica i dati
file_path = r"D:\TESI\prova statistica\N2N3multitaperMeanPSD_specific_channels_149\_N2N3multitaperMeanPSD_specific_channels_149_aggregated_with_phases.csv"
data = pd.read_csv(file_path)
data = data[data['Stage'] == 3]
print(data)

# Identifica le colonne numeriche (escludendo le categoriali)
numeric_columns = data.select_dtypes(include=[np.number]).columns

# Applica MinMaxScaler alle colonne numeriche
#scaler = MinMaxScaler()
#data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Imposta la feature e il canale da analizzare
feature_to_analyze = "Mean PSD Total Delta"
selected_channel = 15

# Trova il miglior grado polinomiale per ogni gruppo
best_degrees = find_best_poly_degree(data, feature_to_analyze, selected_channel, max_degree=40)
subject_color_map = create_subject_color_map(data)

# Stampiamo il miglior grado per ciascun gruppo
print("\nBest polynomial degrees for each group:")
for group, degree in best_degrees.items():
    print(f"Group {group}: Degree {degree}")

# Genera il grafico separato per ogni gruppo (CTL, ADV, DNV, DYS)
#plot_group_mean_and_std_over_normalized_sleep(data, feature_to_analyze, selected_channel)
#plot_epochs_per_subject_by_group_with_linear_regression(data, feature_to_analyze, selected_channel, subject_color_map=subject_color_map)
plot_subjects_by_group_phases(data, feature_to_analyze, selected_channel, subject_color_map=subject_color_map)

#plot_epochs_per_subject_by_group_with_polynomial_regression(data, feature_to_analyze, selected_channel, poly_degree=6, subject_color_map=subject_color_map)

