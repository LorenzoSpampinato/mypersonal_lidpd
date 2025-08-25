
import os
from scipy.stats import ttest_ind,ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import permutation_test


# 1) t-test indipendente (Student)
def independent_t_stat(x, y):
    """
    Restituisce il t-statistic per il t-test di Student a due campioni indipendenti.
    """
    t_stat, _ = ttest_ind(x, y, axis=0)
    return t_stat


# Mappa simboli per gruppi
SYMBOL_MAP = {'CTL': '+', 'DNV': '§', 'ADV': '$', 'DYS': '#'}


def analyze_and_plot(data, feature_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'],
                     selected_channels=None, output_excel="results.xlsx", n_resamples=5000):

    # Imposta l'ordine dei gruppi come categorica ordinata (utile per plotting e ordinamenti coerenti)
    data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)

    # Filtra i dati per le fasi di interesse, qui "Early" e "Late"
    filtered_data = data[data['Phase_Assigned'].isin(['Early', 'Late'])]

    # Se sono specificati canali selezionati, filtra ulteriormente i dati per quei canali, altrimenti preparati per analizzare tutti i canali
    if selected_channels is not None:
        filtered_data_selected = filtered_data[filtered_data['Channel'].isin(selected_channels)]
        filtered_data_all = filtered_data
    else:
        filtered_data_selected = pd.DataFrame()  # vuoto se nessun canale selezionato
        filtered_data_all = filtered_data        # tutti i dati filtrati

    # --- Calcolo delle medie ---
    # 1. Calcola la media per ogni soggetto e canale, per ogni gruppo e fase
    subject_means_all = filtered_data_all.groupby(
        ['Group', 'Channel', 'Subject', 'Phase_Assigned'], observed=True)[feature_name].mean().reset_index()

    # 2. Calcola la media di queste medie per canale e gruppo, per ogni fase
    channel_means_all = subject_means_all.groupby(
        ['Group', 'Channel', 'Phase_Assigned'], observed=True)[feature_name].mean().reset_index()

    # 3. Calcola la media globale per gruppo e fase (media di tutti i canali)
    overall_means_all = channel_means_all.groupby(
        ['Group', 'Phase_Assigned'], observed=True)[feature_name].mean().reset_index()

    # Se canali selezionati, ripeti i calcoli per questi canali (altrimenti dataframe vuoti)
    if selected_channels is not None:
        subject_means_selected = filtered_data_selected.groupby(
            ['Group', 'Channel', 'Subject', 'Phase_Assigned'], observed=True)[feature_name].mean().reset_index()
        channel_means_selected = subject_means_selected.groupby(
            ['Group', 'Channel', 'Phase_Assigned'], observed=True)[feature_name].mean().reset_index()
        overall_means_selected = channel_means_selected.groupby(
            ['Group', 'Phase_Assigned'], observed=True)[feature_name].mean().reset_index()
    else:
        overall_means_selected = pd.DataFrame()
        channel_means_selected = pd.DataFrame()

    # --- Preparazione per salvare i risultati statistici ---
    statistical_results = []
    compare_group_difference = []
    sig_marks = {'All Channels': {}, 'Frontal Channels': {}}  # struttura per annotare significatività (per due set di canali)

    # --- Funzione interna per test statistici tra gruppi per ogni fase ---
    def statistical_tests_on_early_vs_early_or_late_vs_late_across_groups(channel_means, label):
        # Per ciascuna fase (Early e Late)
        for phase in ['Early', 'Late']:
            # Seleziona i dati di interesse per la fase
            phase_data = channel_means[channel_means['Phase_Assigned'] == phase]

            # Trova i gruppi presenti in questa fase
            groups = list(phase_data['Group'].unique())

            # Estrai i dati del feature per ogni gruppo in array numpy (rimuovendo NaN)
            data_groups = [phase_data[phase_data['Group'] == g][feature_name].dropna().values for g in groups]

            # Calcola media e deviazione standard per ogni gruppo
            group_stats = {
                g: {
                    'mean': np.mean(data_groups[i]),
                    'std': np.std(data_groups[i], ddof=1)
                } for i, g in enumerate(groups)
            }

            # --- Test statistici pairwise tra gruppi ---
            pair_keys, raw_pvals, delta_diffs = [], [], {}  # chiavi confronto, p-value grezzi, differenze stat
            all_corrected_pvals = []
            all_comparisons_info = []  # lista di tuple (chiave confronto, stat, p_val corretto)

            # Loop su tutte le coppie di gruppi (senza ripetizioni)
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    g1, g2 = groups[i], groups[j]
                    data1, data2 = data_groups[i], data_groups[j]
                    mean_g1, mean_g2 = np.mean(data1), np.mean(data2)

                    # Test di permutazione indipendente, usando alternativa "less" (controlla se mean_g1 < mean_g2)
                    # Si ordina i dati in modo da fare il test "less" correttamente
                    # !!!! la funzione permutation_test se riceve in input più di due gruppi, non può dare p-value dei confronti specifici tra le coppie di gruppi, ma solo quello generale
                    # Quindi si fa il test per ogni coppia di gruppi: 4 gruppi -> 6 test separati.
                    if mean_g1 < mean_g2:
                        res = permutation_test([data1, data2], independent_t_stat,
                                               permutation_type='independent', n_resamples=n_resamples,
                                               alternative='less', vectorized=False)
                        key = f"{g1} vs {g2}"
                        delta_stat = res.statistic  # stat test (già g2 - g1)
                    else:
                        res = permutation_test([data2, data1], independent_t_stat,
                                               permutation_type='independent', n_resamples=n_resamples,
                                               alternative='less', vectorized=False)
                        key = f"{g2} vs {g1}"
                        delta_stat = res.statistic

                    pair_keys.append(key)
                    raw_pvals.append(res.pvalue)
                    delta_diffs[key] = delta_stat


            #Applico 2 volte Holm correction:
            # la prima volta perché sto facendo permutation test con 5000 p-value, quindi lo applico a questi
            # la seconda volta perché sto facendo six pairwise comparisons between groups, quindi voglio essere più restrittivo applicando holm ai 6 p-value ottenuti
            # Correzione multipla di Holm sui p-value raw
            if raw_pvals:
                arr = np.array(raw_pvals)
                m = len(arr)
                idxs = np.argsort(arr)
                sorted_p = arr[idxs]

                # Calcola p adjusted con Holm manuale (correzione step-down)
                adj = np.minimum(1, sorted_p * (m - np.arange(m)))
                for k in range(m - 1):
                    if adj[k + 1] < adj[k]:
                        adj[k + 1] = adj[k]

                p_holm = np.empty(m)
                p_holm[idxs] = adj

                # Costruisci dizionario con risultati post-hoc
                post_hoc = {}
                for i, k in enumerate(pair_keys):
                    stat = delta_diffs[k]
                    p_h = p_holm[i]
                    post_hoc[k] = f"stat = {stat:.2f}, p (Holm) = {p_h:.2e}"
                    all_corrected_pvals.append(p_h)
                    all_comparisons_info.append((k, stat, p_h))
            else:
                post_hoc = None

            # Inizializza struttura per annotazioni significatività per questo label e fase
            sig_marks[label].setdefault(phase, {g: [] for g in groups})

            # Correzione globale su tutti i p-values corretti di Holm (seconda correzione)
            if all_corrected_pvals:
                _, pvals_global_holm = multipletests(all_corrected_pvals, method='holm')[:2]

                # Aggiorna annotazioni significatività
                for i, (comp_key, stat, p_h) in enumerate(all_comparisons_info):
                    p_g = pvals_global_holm[i]

                    # Se significativo dopo correzione globale, aggiungi simbolo di significatività
                    if p_g < 0.05:
                        g1, _, g2 = comp_key.split()
                        sig_marks[label][phase][g1].append(SYMBOL_MAP[g2])

                    # Aggiungi p-value globale nel testo dei risultati post-hoc
                    if post_hoc and comp_key in post_hoc:
                        post_hoc[comp_key] = (
                            f"stat = {stat:.2f}, "
                            f"p (Holm) = {p_h:.2e}, "
                            f"p (Holm-global) = {p_g:.2e}"
                        )

            # Salva i risultati statistici per questa fase e label
            statistical_results.append({
                'Phase': phase,
                'Label': label,
                'Group Stats': group_stats,
                'Post hoc': post_hoc
            })

            # Stampa i risultati post-hoc a schermo
            print(f"\nPost hoc results for {label} - {phase}:\n")
            if post_hoc:
                for comp, txt in post_hoc.items():
                    print(f"{comp}: {txt}")
                    # Se significativo dopo correzione globale, stampa messaggio
                    if "Holm-global" in txt and float(txt.split("p (Holm-global) = ")[1]) < 0.05:
                        print("  ↳ SIGNIFICANT after global correction (p < 0.05)")
            else:
                print("No significant pairwise comparisons.")

    def statistical_tests_on_differences_late_early_across_groups(channel_means, feature_name, label):
        # Creo una tabella pivot con indice 'Group' e 'Channel',
        # colonne le fasi 'Early' e 'Late', valori la feature di interesse
        pivot = channel_means.pivot_table(index=['Group', 'Channel'],
                                          columns='Phase_Assigned',
                                          values=feature_name)

        # Elimino righe dove mancano dati in 'Early' o 'Late' (canali incompleti)
        pivot = pivot.dropna(subset=['Early', 'Late'])

        # Calcolo la differenza tra Late ed Early per ogni gruppo-canale
        pivot['Diff'] = pivot['Late'] - pivot['Early']  #############################################

        # Raggruppo le differenze calcolate per gruppo (lista di differenze per gruppo)
        group_diffs = pivot.groupby('Group')['Diff'].apply(list)

        # Calcolo media e deviazione standard delle differenze per ogni gruppo
        group_stats = {}
        for g in group_diffs.index:
            data = group_diffs[g]
            group_stats[g] = {
                'mean': np.mean(data),
                'std': np.std(data, ddof=1)
            }

        print(f"\n===== Group Comparison of Early-Late Differences ({label}) =====")

        # Definisco l'ordine di confronto dei gruppi (ordine personalizzato)
        group_order = ['DYS', 'CTL', 'DNV', 'ADV']
        # Seleziono solo i gruppi presenti nel dataset, mantenendo l'ordine definito
        group_names_sorted = [group for group in group_order if group in group_diffs.index]
        group_diffs_sorted = group_diffs[group_names_sorted]

        # Inizializzo liste per memorizzare risultati dei test post hoc
        pair_keys = []
        raw_pvals = []
        delta_diffs = {}

        # Ciclo su tutte le coppie di gruppi per confrontare le differenze Early-Late
        for i in range(len(group_names_sorted)):
            for j in range(i + 1, len(group_names_sorted)):
                g1, g2 = group_names_sorted[i], group_names_sorted[j]

                # Calcolo la media delle differenze per ciascun gruppo della coppia
                mean_g1 = np.mean(group_diffs_sorted[g1])
                mean_g2 = np.mean(group_diffs_sorted[g2])
                print(f"Mean {g1}: {mean_g1:.2f}")
                print(f"Mean {g2}: {mean_g2:.2f}")

                # Eseguo test di permutazione (permutazione t-test indipendente)
                # nel verso corretto, cioè confronto dal gruppo con media minore verso quello con media maggiore
                # !!!! la funzione permutation_test se riceve in input più di due gruppi, non può dare p-value dei confronti specifici tra le coppie di gruppi, ma solo quello generale
                # Quindi si fa il test per ogni coppia di gruppi: 4 gruppi -> 6 test separati.
                if mean_g1 < mean_g2:
                    res_pair = permutation_test([group_diffs_sorted[g1], group_diffs_sorted[g2]], independent_t_stat,
                                                permutation_type='independent',
                                                n_resamples=n_resamples,
                                                alternative='less',
                                                vectorized=False)
                    key = f"{g1} vs {g2}"  # chiave descrittiva del confronto
                    delta_stat = res_pair.statistic  # statistica del test (g2 - g1)
                else:
                    res_pair = permutation_test([group_diffs_sorted[g2], group_diffs_sorted[g1]], independent_t_stat,
                                                permutation_type='independent',
                                                n_resamples=n_resamples,
                                                alternative='less',
                                                vectorized=False)
                    key = f"{g2} vs {g1}"
                    delta_stat = res_pair.statistic

                # Memorizzo chiave confronto, statistica e p-value raw
                pair_keys.append(key)
                delta_diffs[key] = delta_stat
                raw_pvals.append(res_pair.pvalue)
                print(f"Post hoc {key}: Δdiff = {delta_stat:.2f}, raw p = {res_pair.pvalue:.2e}")

        # Applico 2 volte Holm correction:
        # la prima volta perché sto facendo permutation test con 5000 p-value, quindi lo applico a questi
        # la seconda volta perché sto facendo six pairwise comparisons between groups, quindi voglio essere più restrittivo applicando holm ai 6 p-value ottenuti
        # Se ci sono p-value da correggere, applico correzione di Holm locale
        post_hoc = None
        if len(raw_pvals) > 0:
            raw_pvals = np.array(raw_pvals)
            m = len(raw_pvals)
            sorted_indices = np.argsort(raw_pvals)
            adjusted = np.empty(m, dtype=float)

            # Calcolo p-value corretti con metodo Holm
            for i, idx in enumerate(sorted_indices):
                adjusted[idx] = min(raw_pvals[idx] * (m - i), 1.0)

            # Correggo eventuali discontinuità per mantenere ordine monotono dei p
            for i in range(m - 1):
                if adjusted[sorted_indices[i]] > adjusted[sorted_indices[i + 1]]:
                    adjusted[sorted_indices[i + 1]] = adjusted[sorted_indices[i]]

            post_hoc = {}
            all_corrected_pvals = []
            all_comparisons_info = []  # lista tuple (chiave confronto, stat, p corretti)

            # Creo dizionario di risultati post hoc con p-value corretti localmente
            for i, k in enumerate(pair_keys):
                delta_diff = delta_diffs[k]
                p_h = adjusted[i]
                post_hoc[k] = f"Δdiff = {delta_diff:.2f}, p (Holm) = {p_h:.2e}"
                all_corrected_pvals.append(p_h)
                all_comparisons_info.append((k, delta_diff, p_h))

            # Importo funzione multipletests per una seconda correzione globale Holm su tutti i test post hoc
            from statsmodels.stats.multitest import multipletests
            if all_corrected_pvals:
                _, pvals_global_holm = multipletests(all_corrected_pvals, method='holm')[:2]

                # Aggiungo p-value corretti globalmente ai risultati post hoc, eventualmente per visualizzazione/significatività globale
                for i, (comp_key, stat, p_h) in enumerate(all_comparisons_info):
                    p_g = pvals_global_holm[i]
                    post_hoc[comp_key] = (
                        f"Δdiff = {stat:.2f}, "
                        f"p (Holm) = {p_h:.2e}, "
                        f"p (Holm-global) = {p_g:.2e}"
                    )

            # Stampo risultati post hoc dopo doppia correzione Holm
            print("Post hoc (Holm corrected with global correction):")
            for comparison, result in post_hoc.items():
                print(f"{comparison}: {result}")
        else:
            post_hoc = None

        # Salvo i risultati descrittivi e post hoc in una lista globale per uso successivo (ad es. plotting)
        compare_group_difference.append({
            'Phase': 'Early-Late Δ',
            'Label': label,
            'Descriptive Stats': "; ".join(
                [f"{g}: mean Δ = {np.mean(d):.2f} ± {np.std(d, ddof=1):.2f}" for g, d in group_diffs_sorted.items()]),
            'Post hoc': post_hoc
        })

        # Ritorno i risultati post hoc (per ulteriori usi nel codice)
        return post_hoc


    ##############################

    # Eseguo il test di permutazione su tutti i canali (tutte le medie canale per gruppo e fase)
    print("Permutation test on all channels")
    statistical_tests_on_early_vs_early_or_late_vs_late_across_groups(channel_means_all, "All Channels")

    # Se sono stati selezionati canali specifici (ad esempio canali frontali),
    # eseguo lo stesso test solo su quei canali
    if selected_channels is not None:
        print("Permutation test on selected (frontal) channels")
        statistical_tests_on_early_vs_early_or_late_vs_late_across_groups(channel_means_selected, "Frontal Channels")

    # Ora confronto le differenze (delta) Early-Late tra i gruppi su tutti i canali
    print("Comparing Early-Late differences across groups - All Channels")
    post_hoc_all = statistical_tests_on_differences_late_early_across_groups(channel_means_all, feature_name,
                                                                             "All Channels")
    print("Post-hoc All Channels:", post_hoc_all)

    post_hoc_frontal = None
    # Se sono stati selezionati canali specifici, faccio lo stesso confronto sui canali selezionati
    if selected_channels is not None:
        print("Comparing Early-Late differences across groups - Frontal Channels")
        post_hoc_frontal = statistical_tests_on_differences_late_early_across_groups(channel_means_selected,
                                                                                     feature_name, "Frontal Channels")
        print("Post-hoc Frontal Channels:", post_hoc_frontal)

    # Salvo tutti i risultati statistici ottenuti in un file Excel
    # Creo DataFrame dai risultati salvati durante i test (statistical_results e compare_group_difference)
    df_statistical_results = pd.DataFrame(statistical_results)
    df_compare_group_difference = pd.DataFrame(compare_group_difference)

    # Scrivo i risultati in due fogli diversi del file Excel specificato da output_excel
    with pd.ExcelWriter(output_excel) as writer:
        df_statistical_results.to_excel(writer, sheet_name="Early vs early, late vs late", index=False)
        df_compare_group_difference.to_excel(writer, sheet_name="delta difference across groups", index=False)

    print("Excel file with results saved as:", output_excel)

    ##############################

    def plot_data(post_hoc_all=None, post_hoc_frontal=None):
        color_map = {'Early': 'red', 'Late': 'blue'}
        offset = {'Early': -0.2, 'Late': 0.2}
        plt.figure(figsize=(12, 7))

        all_vals = list(channel_means_all[feature_name])
        if not channel_means_selected.empty:
            all_vals += list(channel_means_selected[feature_name])
        y_min, y_max = (min(all_vals), max(all_vals)) if all_vals else (0, 1)
        y_range = y_max - y_min
        symbol_gap = 0.1 * y_range
        #plt.ylim(bottom=y_min - 1.5 * symbol_gap, top=y_max + 3 * symbol_gap)
        plt.ylim(bottom=y_min - 1.5 * symbol_gap, top=y_max + 1 * symbol_gap)

        # Plot All Channels (rossi e blu)
        for phase in ['Early', 'Late']:
            phase_data = channel_means_all[channel_means_all['Phase_Assigned'] == phase]
            x = [group_order.index(g) + offset[phase] for g in phase_data['Group']]
            y = phase_data[feature_name]
            plt.scatter(x, y,
                        color=color_map[phase],
                        label=f"{phase} - All Channels",
                        alpha=0.6,
                        edgecolors='black',
                        s=75)

            # Simboli $§# per All Channels
            for g, marks in sig_marks['All Channels'].get(phase, {}).items():
                if marks:
                    xi = group_order.index(g) + offset[phase]
                    #max_y = phase_data[phase_data['Group'] == g][feature_name].max()
                    #y0 = max_y + symbol_gap
                    min_y = phase_data[phase_data['Group'] == g][feature_name].min()
                    y0 = min_y - symbol_gap

                    txt = ''.join(marks)
                    plt.text(xi, y0, txt, color=color_map[phase], ha='center', va='bottom', fontsize=14)

            # Linee verticali che collegano le medie Early e Late
            for i, g in enumerate(group_order):
                # All Channels
                early_val = \
                channel_means_all[(channel_means_all['Phase_Assigned'] == 'Early') & (channel_means_all['Group'] == g)][
                    feature_name].mean()
                late_val = \
                channel_means_all[(channel_means_all['Phase_Assigned'] == 'Late') & (channel_means_all['Group'] == g)][
                    feature_name].mean()
                if not np.isnan(early_val) and not np.isnan(late_val):
                    plt.plot([i + 0.05, i + 0.05], [early_val, late_val], color='black', linewidth=2)

                # Frontal Channels
                if not channel_means_selected.empty:
                    early_val_f = channel_means_selected[
                        (channel_means_selected['Phase_Assigned'] == 'Early') & (channel_means_selected['Group'] == g)][
                        feature_name].mean()
                    late_val_f = channel_means_selected[
                        (channel_means_selected['Phase_Assigned'] == 'Late') & (channel_means_selected['Group'] == g)][
                        feature_name].mean()
                    if not np.isnan(early_val_f) and not np.isnan(late_val_f):
                        plt.plot([i - 0.05, i - 0.05], [early_val_f, late_val_f], color='green', linewidth=2)

        # Plot Frontal Channels (verdi)
        if not channel_means_selected.empty:
            for phase in ['Early', 'Late']:
                phase_data = channel_means_selected[channel_means_selected['Phase_Assigned'] == phase]
                x = [group_order.index(g) + offset[phase] for g in phase_data['Group']]
                y = phase_data[feature_name]
                plt.scatter(x, y,
                            color='green',
                            alpha=0.4,
                            edgecolors='red' if phase == 'Early' else 'blue',
                            s=75,
                            label='Frontal Channels' if phase == 'Early' else None)

                # Simboli $§# per Frontal Channels
                for g in group_order:
                    marks = sig_marks['Frontal Channels'].get(phase, {}).get(g, [])
                    if marks:
                        xi = group_order.index(g) + offset[phase]

                        # Trova il min_y dei dati frontal
                        min_y_f = phase_data[phase_data['Group'] == g][feature_name].min()
                        # Trova il min_y dei dati all (per stesso gruppo e fase)
                        min_y_all = channel_means_all[
                            (channel_means_all['Group'] == g) & (channel_means_all['Phase_Assigned'] == phase)
                            ][feature_name].min()

                        # y0 dei frontal = simbolo all - 1.0
                        y0_all = min_y_all - symbol_gap
                        #y0_all = min_y_all + symbol_gap
                        #y0 = y0_all -0.025 #per sample entropy
                        y0 = y0_all - 0.025  # per Katz FD
                        #y0 = y0_all - 0.015  #correlation dimension
                        #y0 = y0_all -7 # per slow wave activity
                        #y0 = y0_all - 0.005 # per spectral entropy

                        txt = ''.join(marks)
                        print(f"Plotting {g} {phase} at ({xi}, {y0}) with marks: {marks}")
                        plt.text(xi, y0, txt, color='green', ha='center', va='bottom', fontsize=14)

            # Linee medie
            if not overall_means_all.empty:
                for phase in ['Early', 'Late']:
                    mval = overall_means_all[overall_means_all['Phase_Assigned'] == phase][feature_name].values
                    plt.hlines(y=mval,
                               xmin=np.arange(len(group_order)) + offset[phase] - 0.1,
                               xmax=np.arange(len(group_order)) + offset[phase] + 0.1,
                               colors=color_map[phase],
                               linestyles='dashed',
                               linewidth=2,
                               label=f"Mean All Channels {phase}")
            if not overall_means_selected.empty:
                for phase in ['Early', 'Late']:
                    mval = overall_means_selected[overall_means_selected['Phase_Assigned'] == phase][
                        feature_name].values
                    plt.hlines(y=mval,
                               xmin=np.arange(len(group_order)) + offset[phase] - 0.1,
                               xmax=np.arange(len(group_order)) + offset[phase] + 0.1,
                               colors='green',
                               linestyles='dashed',
                               linewidth=2,
                               label="Mean Frontal Channels" if phase == 'Early' else None)

            def parse_posthoc_dict(ph_dict):
                parsed = {g: [] for g in group_order}
                if not isinstance(ph_dict, dict):
                    return parsed
                for comp, txt in ph_dict.items():
                    try:
                        g1, g2 = comp.split(" vs ")
                        pval = float(txt.split("p (Holm-global) = ")[1])
                    except:
                        continue
                    if pval < 0.05:
                        # metti in g1 il simbolo di g2
                        parsed[g1].append(SYMBOL_MAP[g2])
                return parsed

            sig_all = parse_posthoc_dict(post_hoc_all)
            sig_frontal = parse_posthoc_dict(post_hoc_frontal)

            # Ora aggiungi linee verticali e simboli:
            for i, g in enumerate(group_order):
                e_all = channel_means_all.query("Group==@g & Phase_Assigned=='Early'")[feature_name].mean()
                l_all = channel_means_all.query("Group==@g & Phase_Assigned=='Late'")[feature_name].mean()
                if g in sig_all and sig_all[g]:
                    y0 = max(e_all, l_all) + 0.02 * (y_max - y_min)
                    plt.text(i + 0.05, y0, ''.join(sig_all[g]),
                             ha='center', va='bottom', fontsize=16, color='black')

                if not channel_means_selected.empty:
                    e_f = channel_means_selected.query("Group==@g & Phase_Assigned=='Early'")[feature_name].mean()
                    l_f = channel_means_selected.query("Group==@g & Phase_Assigned=='Late'")[feature_name].mean()
                    if g in sig_frontal and sig_frontal[g]:
                        y0f = max(e_f, l_f) + 0.02 * (y_max - y_min)
                        plt.text(i - 0.05, y0f, ''.join(sig_frontal[g]),
                                 ha='center', va='bottom', fontsize=16, color='green')

            # Resto del plotting...
            plt.xlabel('Groups', fontsize=12)
            plt.ylabel(f'Katz Fractal Dimension', fontsize=12)
            plt.title(f"Distribution of channel-wise means", fontsize=14)
            #y_max_rounded = np.ceil(y_max / 20) * 20
            #plt.yticks(np.arange(0, y_max_rounded + 1, 20)) # ad esempio 10 valori equidistanti
            #plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(range(len(group_order)), group_order)
            # Gestione legenda
            handles, labels = plt.gca().get_legend_handles_labels()
            # Aggiungi descrizione simboli post-hoc
            symbol_legend = (
                "p < 0.05:\n"
                "+ vs. CTL\n"
                "§ vs. DNV\n"
                "$ vs. ADV\n"
                "# vs. DYS\n"
            )
            plt.legend(handles, labels, loc='upper right', fontsize=8, title="Legend")
            #plt.gcf().text(0.665, 0.25, symbol_legend, fontsize=10, verticalalignment='top')
            plt.gcf().text(0.665, 0.84, symbol_legend, fontsize=10, verticalalignment='top')


            plt.show()

    plot_data(post_hoc_all=post_hoc_all, post_hoc_frontal=post_hoc_frontal)


# Carica i dati da un file CSV specificato nel percorso file_path
file_path = r"D:\TESI\prova statistica\N2N3ALLENTROPY_specific_channels_149\_N2N3ALLENTROPY_specific_channels_149_aggregated_with_phases.csv"
data = pd.read_csv(file_path)

# Filtra i dati per selezionare solo la fase di sonno N3 (Stage == 3)
data = data[data['Stage'] == 3]

# Definisci la lista delle feature da analizzare, in questo caso solo "Katz FD"
feature_to_analyze = ["Katz FD"]

# Definisci i canali EEG selezionati specifici (frontopolari e frontali),
# da usare per analisi e grafici più mirati
# !!!!!!! senza specificarlo le analisi vengono fatte sempre anche su tutti i canali, quindi usa selected_channels se vuoi aggiungere anche regioni specifiche
selected_channels = [27, 33, 34, 38, 39, 47, 48, 26, 20, 19, 12, 11, 3, 2, 222, 16, 22, 23, 24, 28, 29, 30, 35, 36, 40,
                     41, 42, 49, 50, 21, 15, 7, 14, 6,
                     207, 13, 5, 215, 4, 224, 223, 214, 206, 213, 205]


# Definisci il percorso della cartella in cui salvare i file Excel di output
output_dir = r"D:\TESI\excel\topographic"

# Se la cartella non esiste, creala per evitare errori al salvataggio
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cicla sulle feature da analizzare (in questo caso solo "Katz FD")
for feature_name in feature_to_analyze:
    # Costruisci il percorso completo del file Excel dove salvare i risultati
    output_excel = os.path.join(output_dir, f"{feature_name.replace(' ', '_')}.xlsx")

    # Esegui la funzione di analisi e generazione del grafico,
    # passando i dati filtrati, la feature da analizzare,
    # l’ordine dei gruppi da visualizzare, i canali selezionati,
    # e il percorso per salvare il file Excel
    analyze_and_plot(data=data.copy(), feature_name=feature_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'],
                     selected_channels=selected_channels, output_excel=output_excel)

