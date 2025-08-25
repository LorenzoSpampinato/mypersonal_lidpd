from scipy.stats import f_oneway, wilcoxon, kruskal, normaltest, levene, bartlett, fligner, ttest_rel, mannwhitneyu, \
    shapiro
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations
import os
from scipy.stats import ttest_ind, mannwhitneyu, ttest_rel, wilcoxon
import numpy as np
from statsmodels.stats.multitest import multipletests

def welch_anova(data, dv, between):
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    # Backup nome originale
    original_dv = dv

    # Rinomina temporanea per compatibilità con Patsy (spazi → underscore)
    safe_dv = dv.replace(" ", "_")
    data = data.rename(columns={dv: safe_dv})

    # Costruzione formula e modello
    formula = f"{safe_dv} ~ C({between})"
    model = ols(formula, data).fit()

    # Calcolo ANOVA di Welch
    aov_table = anova_lm(model, typ=2, robust='hc3')  # 'hc3' è robusto a eteroschedasticità

    # Ripristina nome originale nella tabella risultante, se presente
    aov_table.index = [original_dv if idx.startswith(safe_dv) else idx for idx in aov_table.index]

    return aov_table


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import f_oneway, permutation_test


# Funzione statistica per il test globale tra gruppi (ANOVA)
def group_statistic_anova(*samples):
    return f_oneway(*samples)[0]


# Funzione statistica per il test globale (Kruskal-Wallis)
def group_statistic_kruskal(*samples):
    return kruskal(*samples)[0]


def diff_of_means(*samples):
    return np.mean(samples[0]) - np.mean(samples[1])  # Restituiamo la differenza direzionale


# Funzione per il test appaiato: differenza media tra Late ed Early
def paired_mean_diff(x, y):
    return np.mean(x - y)


# 1) t-test indipendente (Student)
def independent_t_stat(x, y):
    """
    Restituisce il t-statistic per il t-test di Student a due campioni indipendenti.
    """
    t_stat, _ = ttest_ind(x, y, axis=0)
    return t_stat


# 2) Mann–Whitney U (Wilcoxon rank-sum per campioni indipendenti)
def mannwhitney_stat(x, y):
    """
    Restituisce l'U-statistic per il test di Mann–Whitney.
    """
    # alternative='two-sided' per test bilaterale
    res = mannwhitneyu(x, y, alternative='two-sided', use_continuity=True)
    return res.statistic


# 3) t-test appaiato (Student paired)
def paired_t_stat(x, y):
    """
    Restituisce il t-statistic per il t-test di Student su campioni appaiati (relativo).
    """
    t_stat, _ = ttest_rel(x, y, axis=0, alternative='two-sided')
    return t_stat


# 4) Wilcoxon signed-rank test (campioni appaiati, non parametrico)
def wilcoxon_stat(x, y):
    """
    Restituisce il W-statistic per il test di Wilcoxon sui dati appaiati.
    """
    # zero_method='wilcox' esclude differenze nulle, alternative='two-sided'
    res = wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
    return res.statistic


# Mappa simboli per gruppi
SYMBOL_MAP = {'CTL': '+', 'DNV': '§', 'ADV': '$', 'DYS': '#'}


def analyze_and_plot(data, feature_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'],
                     selected_channels=None, output_excel="results.xlsx", n_resamples=2):
    # Imposta l'ordine dei gruppi
    data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)
    filtered_data = data[data['Phase_Assigned'].isin(['Early', 'Late'])]

    if selected_channels is not None:
        filtered_data_selected = filtered_data[filtered_data['Channel'].isin(selected_channels)]
        filtered_data_all = filtered_data
    else:
        filtered_data_selected = pd.DataFrame()
        filtered_data_all = filtered_data

    # Calcolo delle medie per soggetto e canale
    subject_means_all = filtered_data_all.groupby(
        ['Group', 'Channel', 'Subject', 'Phase_Assigned'], observed=True)[feature_name].mean().reset_index()
    channel_means_all = subject_means_all.groupby(
        ['Group', 'Channel', 'Phase_Assigned'], observed=True)[feature_name].mean().reset_index()
    overall_means_all = channel_means_all.groupby(
        ['Group', 'Phase_Assigned'], observed=True)[feature_name].mean().reset_index()

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

    # Liste per salvare i risultati
    statistical_results = []
    early_late_results = []
    compare_group_difference = []
    sig_marks = {'All Channels': {}, 'Frontal Channels': {}}

    def statistical_tests(channel_means, label):
        for phase in ['Early', 'Late']:
            phase_data = channel_means[channel_means['Phase_Assigned'] == phase]
            groups = list(phase_data['Group'].unique())
            data_groups = [phase_data[phase_data['Group'] == g][feature_name].dropna().values for g in groups]

            # Calcolo mean e std per ogni gruppo
            group_stats = {
                g: {
                    'mean': np.mean(data_groups[i]),
                    'std': np.std(data_groups[i], ddof=1)
                } for i, g in enumerate(groups)
            }

            # Test pairwise + Holm
            pair_keys, raw_pvals, delta_diffs = [], [], {}
            all_corrected_pvals = []
            all_comparisons_info = []  # (comp_key, stat, p_h)

            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    g1, g2 = groups[i], groups[j]
                    data1, data2 = data_groups[i], data_groups[j]
                    mean_g1, mean_g2 = np.mean(data1), np.mean(data2)

                    if mean_g1< mean_g2:
                        res = permutation_test([data1, data2], independent_t_stat,
                                               permutation_type='independent', n_resamples=n_resamples,
                                               alternative='less', vectorized=False)
                        key = f"{g1} vs {g2}"  # direzione coerente con g2 - g1
                        delta_stat = res.statistic  # già g2 - g1
                    else:
                        res = permutation_test([data2, data1], independent_t_stat,
                                               permutation_type='independent', n_resamples=n_resamples,
                                               alternative='less', vectorized=False)
                        key = f"{g2} vs {g1}"  # ancora direzione coerente con g2 - g1
                        delta_stat = res.statistic

                    pair_keys.append(key)
                    raw_pvals.append(res.pvalue)
                    delta_diffs[key] = delta_stat

            if raw_pvals:
                arr = np.array(raw_pvals)
                m = len(arr)
                idxs = np.argsort(arr)
                sorted_p = arr[idxs]
                adj = np.minimum(1, sorted_p * (m - np.arange(m)))
                for k in range(m - 1):
                    if adj[k + 1] < adj[k]:
                        adj[k + 1] = adj[k]
                p_holm = np.empty(m)
                p_holm[idxs] = adj
                post_hoc = {}
                for i, k in enumerate(pair_keys):
                    stat = delta_diffs[k]
                    p_h = p_holm[i]
                    post_hoc[k] = f"stat = {stat:.2f}, p (Holm) = {p_h:.2e}"
                    all_corrected_pvals.append(p_h)
                    all_comparisons_info.append((k, stat, p_h))
            else:
                post_hoc = None

            sig_marks[label].setdefault(phase, {g: [] for g in groups})

            # Correzione globale (seconda correzione)
            if all_corrected_pvals:
                _, pvals_global_holm = multipletests(all_corrected_pvals, method='holm')[:2]

                for i, (comp_key, stat, p_h) in enumerate(all_comparisons_info):
                    p_g = pvals_global_holm[i]

                    # Aggiungi simbolo se significativo globalmente
                    if p_g < 0.05:
                        g1, _, g2 = comp_key.split()
                        sig_marks[label][phase][g1].append(SYMBOL_MAP[g2])

                    # Aggiungi entrambi i valori di p nella descrizione
                    if post_hoc and comp_key in post_hoc:
                        post_hoc[comp_key] = (
                            f"stat = {stat:.2f}, "
                            f"p (Holm) = {p_h:.2e}, "
                            f"p (Holm-global) = {p_g:.2e}"
                        )

            # Salva i risultati solo dopo aver aggiornato post_hoc
            statistical_results.append({
                'Phase': phase,
                'Label': label,
                'Group Stats': group_stats,
                'Post hoc': post_hoc
            })

            # Stampa i risultati
            print(f"\nPost hoc results for {label} - {phase}:\n")
            if post_hoc:
                for comp, txt in post_hoc.items():
                    print(f"{comp}: {txt}")
                    if "Holm-global" in txt and float(txt.split("p (Holm-global) = ")[1]) < 0.05:
                        print("  ↳ SIGNIFICANT after global correction (p < 0.05)")
            else:
                print("No significant pairwise comparisons.")

        return post_hoc


    def compare_group_differences(channel_means, feature_name, label):
        # Pivot per ottenere la differenza Early-Late per ogni canale e gruppo
        pivot = channel_means.pivot_table(index=['Group', 'Channel'],
                                          columns='Phase_Assigned',
                                          values=feature_name)

        # Rimuove canali mancanti
        pivot = pivot.dropna(subset=['Early', 'Late'])

        # Calcola la differenza Early - Late
        #pivot['Diff'] = pivot['Late'] - pivot['Early'] #############################################
        pivot['Diff'] = pivot['Early'] - pivot['Late']

        # Raccolta delle differenze per gruppo
        group_diffs = pivot.groupby('Group')['Diff'].apply(list)

        group_stats = {}
        for g in group_diffs.index:
            data = group_diffs[g]
            group_stats[g] = {
                'mean': np.mean(data),
                'std': np.std(data, ddof=1)
            }

        print(f"\n===== Group Comparison of Early-Late Differences ({label}) =====")

        # Ordinare i gruppi nell'ordine desiderato: DYS, CTL, DNV, ADV
        group_order = ['DYS', 'CTL', 'DNV', 'ADV']
        group_names_sorted = [group for group in group_order if group in group_diffs.index]
        group_diffs_sorted = group_diffs[group_names_sorted]

        # Post hoc tra gruppi
        pair_keys = []
        raw_pvals = []
        delta_diffs = {}

        for i in range(len(group_names_sorted)):
            for j in range(i + 1, len(group_names_sorted)):
                g1, g2 = group_names_sorted[i], group_names_sorted[j]
                mean_g1 = np.mean(group_diffs_sorted[g1])
                mean_g2 = np.mean(group_diffs_sorted[g2])
                print(f"Mean {g1}: {mean_g1:.2f}")
                print(f"Mean {g2}: {mean_g2:.2f}")

                if mean_g1 < mean_g2:
                    # Test g1 vs g2 → confronto nella direzione giusta
                    res_pair = permutation_test([group_diffs_sorted[g1], group_diffs_sorted[g2]], independent_t_stat,
                                                permutation_type='independent',
                                                n_resamples=n_resamples,
                                                alternative='less',
                                                vectorized=False)
                    key = f"{g1} vs {g2}"  # direzione del confronto
                    delta_stat = res_pair.statistic  # già g2 - g1
                else:
                    # Test g2 vs g1 → ancora nella direzione g2 - g1
                    res_pair = permutation_test([group_diffs_sorted[g2], group_diffs_sorted[g1]], independent_t_stat,
                                                permutation_type='independent',
                                                n_resamples=n_resamples,
                                                alternative='less',
                                                vectorized=False)
                    key = f"{g2} vs {g1}"  # mantieni sempre questa forma
                    delta_stat = res_pair.statistic  # già g2 - g1

                pair_keys.append(key)
                delta_diffs[key] = delta_stat
                raw_pvals.append(res_pair.pvalue)
                print(f"Post hoc {key}: Δdiff = {delta_stat:.2f}, raw p = {res_pair.pvalue:.2e}")

        # Holm correction locale
        post_hoc = None
        if len(raw_pvals) > 0:
            raw_pvals = np.array(raw_pvals)
            m = len(raw_pvals)
            sorted_indices = np.argsort(raw_pvals)
            adjusted = np.empty(m, dtype=float)

            for i, idx in enumerate(sorted_indices):
                adjusted[idx] = min(raw_pvals[idx] * (m - i), 1.0)

            for i in range(m - 1):
                if adjusted[sorted_indices[i]] > adjusted[sorted_indices[i + 1]]:
                    adjusted[sorted_indices[i + 1]] = adjusted[sorted_indices[i]]

            post_hoc = {}
            all_corrected_pvals = []
            all_comparisons_info = []  # (comp_key, stat, p_h)

            for i, k in enumerate(pair_keys):
                delta_diff = delta_diffs[k]
                p_h = adjusted[i]
                post_hoc[k] = f"Δdiff = {delta_diff:.2f}, p (Holm) = {p_h:.2e}"
                all_corrected_pvals.append(p_h)
                all_comparisons_info.append((k, delta_diff, p_h))

            # Correzione globale Holm (seconda correzione)
            from statsmodels.stats.multitest import multipletests
            if all_corrected_pvals:
                _, pvals_global_holm = multipletests(all_corrected_pvals, method='holm')[:2]

                # Simboli di significatività globale - qui puoi adattare a seconda di come vuoi mostrarli
                for i, (comp_key, stat, p_h) in enumerate(all_comparisons_info):
                    p_g = pvals_global_holm[i]
                    # Se vuoi salvare simboli o altro, puoi fare qui (ad esempio usare sig_marks)
                    # Aggiungi entrambi i valori p nella descrizione testuale
                    post_hoc[comp_key] = (
                        f"Δdiff = {stat:.2f}, "
                        f"p (Holm) = {p_h:.2e}, "
                        f"p (Holm-global) = {p_g:.2e}"
                    )

            print("Post hoc (Holm corrected with global correction):")
            for comparison, result in post_hoc.items():
                print(f"{comparison}: {result}")
        else:
            post_hoc = None

        compare_group_difference.append({
            'Phase': 'Early-Late Δ',
            'Label': label,
            'Descriptive Stats': "; ".join(
                [f"{g}: mean Δ = {np.mean(d):.2f} ± {np.std(d, ddof=1):.2f}" for g, d in group_diffs_sorted.items()]),
            'Post hoc': post_hoc
        })

        # Ritorna i risultati post hoc per l'uso nel plot
        return post_hoc

    '''
    # Esegui tutti i test UNA SOLA VOLTA e salva i risultati
    print("Permutation test on all channels")
    statistical_tests(channel_means_all, "All Channels")

    if selected_channels is not None:
        print("Permutation test on selected (frontal) channels")
        statistical_tests(channel_means_selected, "Frontal Channels")
    '''
    # Differenze nei delta tra gruppi
    print("Comparing Early-Late differences across groups - All Channels")
    post_hoc_all = compare_group_differences(channel_means_all, feature_name, "All Channels")
    print("Post-hoc All Channels:", post_hoc_all)

    post_hoc_frontal = None
    if selected_channels is not None:
        print("Comparing Early-Late differences across groups - Frontal Channels")
        post_hoc_frontal = compare_group_differences(channel_means_selected, feature_name, "Frontal Channels")
        print("Post-hoc Frontal Channels:", post_hoc_frontal)

    # Salva i risultati in un file Excel
    df_statistical_results = pd.DataFrame(statistical_results)
    df_compare_group_difference = pd.DataFrame(compare_group_difference)
    #df_early_late_results = pd.DataFrame(early_late_results)
    with pd.ExcelWriter(output_excel) as writer:
        df_statistical_results.to_excel(writer, sheet_name="Early vs early, late vs late", index=False)
        #df_early_late_results.to_excel(writer, sheet_name="Early vs Late Results", index=False)
        df_compare_group_difference.to_excel(writer, sheet_name="delta difference across groups", index=False)
    print("Excel file with results saved as:", output_excel)

    #def plot_data(post_hoc_all=None, post_hoc_frontal=None, all_pvals=None, frontal_pvals=None):
    def plot_data(post_hoc_all=None, post_hoc_frontal=None):
        import seaborn as sns
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        color_map = {'Early': 'red', 'Late': 'blue'}
        offset = {'Early': -0.2, 'Late': 0.2}
        plt.figure(figsize=(12, 7))

        all_vals = list(channel_means_all[feature_name])
        if not channel_means_selected.empty:
            all_vals += list(channel_means_selected[feature_name])
        y_min, y_max = (min(all_vals), max(all_vals)) if all_vals else (0, 1)
        y_range = y_max - y_min
        symbol_gap = 0.1 * y_range
        plt.ylim(bottom=y_min - 1.5 * symbol_gap, top=y_max + 1 * symbol_gap)

        # --- Strip plot: All Channels ---
        df_all = channel_means_all.copy()
        sns.stripplot(data=df_all,
                      x='Group',
                      y=feature_name,
                      hue='Phase_Assigned',
                      dodge=True,
                      jitter=True,
                      alpha=0.7,
                      palette=color_map,
                      order=group_order,
                      hue_order=['Early', 'Late'],
                      edgecolor='black',
                      linewidth=1,
                      size=6)

        # Simboli $§# per All Channels
        for phase in ['Early', 'Late']:
            phase_data = df_all[df_all['Phase_Assigned'] == phase]
            for g, marks in sig_marks['All Channels'].get(phase, {}).items():
                if marks:
                    xi = group_order.index(g) + offset[phase]
                    min_y = phase_data[phase_data['Group'] == g][feature_name].min()
                    y0 = min_y - symbol_gap
                    txt = ''.join(marks)
                    plt.text(xi, y0, txt, color=color_map[phase], ha='center', va='bottom', fontsize=14)


        # --- Strip plot: Frontal Channels ---
        df_frontal = channel_means_selected.copy()
        if not df_frontal.empty:
            sns.stripplot(data=df_frontal,
                          x='Group',
                          y=feature_name,
                          hue='Phase_Assigned',
                          dodge=True,
                          jitter=True,
                          alpha=0.7,
                          palette={'Early': 'green', 'Late': 'green'},
                          order=group_order,
                          hue_order=['Early', 'Late'],
                          edgecolor='black',
                          linewidth=1,
                          size=6)

            # Simboli $§# per Frontal Channels
            for phase in ['Early', 'Late']:
                phase_data = df_frontal[df_frontal['Phase_Assigned'] == phase]
                for g in group_order:
                    marks = sig_marks['Frontal Channels'].get(phase, {}).get(g, [])
                    if marks:
                        xi = group_order.index(g) + offset[phase]
                        min_y_all = df_all[
                            (df_all['Group'] == g) & (df_all['Phase_Assigned'] == phase)
                            ][feature_name].min()
                        y0_all = min_y_all - symbol_gap
                        y0 = y0_all - 7      #per SWA
                        #y0 = y0_all - 0.025 # per sample entropy
                        #y0 = y0_all - 0.005  # per spectral entropy
                        #y0 = y0_all - 0.025  # per Katz FD
                        #y0 = y0_all - 0.015  #correlation dimension
                        txt = ''.join(marks)
                        print(f"Plotting {g} {phase} at ({xi}, {y0}) with marks: {marks}")
                        plt.text(xi, y0, txt, color='green', ha='center', va='bottom', fontsize=14)

        # --- Parsing Post-hoc e simboli sopra ---
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
                    parsed[g1].append(SYMBOL_MAP[g2])
            return parsed

        sig_all = parse_posthoc_dict(post_hoc_all)
        sig_frontal = parse_posthoc_dict(post_hoc_frontal)

        # Dashed lines: medie All Channels per gruppo
        for i, g in enumerate(group_order):
            for phase, color in color_map.items():
                mean_val = df_all.query("Group==@g & Phase_Assigned==@phase")[feature_name].mean()
                if not np.isnan(mean_val):
                    x = i + offset[phase]
                    plt.hlines(mean_val, x - 0.15, x + 0.15, colors=color, linestyles='--', linewidth=2)

        # Dashed lines: medie Frontal Channels per gruppo
        if not df_frontal.empty:
            for i, g in enumerate(group_order):
                for phase in ['Early', 'Late']:
                    mean_val = df_frontal.query("Group==@g & Phase_Assigned==@phase")[feature_name].mean()
                    if not np.isnan(mean_val):
                        x = i + offset[phase]
                        plt.hlines(mean_val, x - 0.15, x + 0.15, colors='green', linestyles='--', linewidth=2)

        # --- Titolo, assi, legenda ---
        plt.xlabel('Groups', fontsize=12)
        plt.ylabel('Sample Entropy', fontsize=12)
        plt.title(f"Distribution of channel-wise means", fontsize=14)
        plt.xticks(range(len(group_order)), group_order, fontsize= 12)

        # Legenda: punti e linee
        handles = [
            Patch(facecolor='red', edgecolor='black', label='Early - All channels'),
            Patch(facecolor='blue', edgecolor='black', label='Late - All channels'),
            #Patch(facecolor='green', edgecolor='black', label='Frontal channels'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Mean - Early'),
            Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='Mean - Late'),
            #Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Mean - Frontal'),
        ]

        # --- Grid verticale tra i gruppi ---
        for i in range(len(group_order) - 1):
            x_pos = i + 0.5
            plt.axvline(x=x_pos, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)

        plt.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.7)



        symbol_legend = (
            "p < 0.05:\n"
            "+ vs. CTL\n"
            "§ vs. DNV\n"
            "$ vs. ADV\n"
            "# vs. DYS\n"
        )
        plt.legend(handles=handles, loc='upper right', fontsize=9, title='Legend', title_fontsize=10)
        #plt.legend(handles=handles, loc='lower right', fontsize=9)
        plt.gcf().text(0.78, 0.90, symbol_legend, fontsize=10, verticalalignment='top')
        #plt.gcf().text(0.78, 0.22, symbol_legend, fontsize=10, verticalalignment='top')
        plt.ylim(0.55, None)

        plt.tight_layout()
        plt.show()

    def plot_group_diff_stripplot(post_hoc_all=None, post_hoc_frontal=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        SYMBOL_MAP = {
            'CTL': '+',
            'DNV': '§',
            'ADV': '$',
            'DYS': '#'
        }

        plt.figure(figsize=(12, 7))

        all_vals = list(channel_means_all[feature_name])
        if not channel_means_selected.empty:
            all_vals += list(channel_means_selected[feature_name])
        y_min, y_max = (min(all_vals), max(all_vals)) if all_vals else (0, 1)
        y_range = y_max - y_min
        symbol_gap = 0.1 * y_range

        # dati "All Channels"
        pivot = channel_means_all.pivot_table(index=['Group', 'Channel'],
                                              columns='Phase_Assigned',
                                              values=feature_name)
        pivot = pivot.dropna(subset=['Early', 'Late'])
        pivot['Diff'] = pivot['Late'] - pivot['Early']
        #pivot['Diff'] = pivot['Early'] - pivot['Late']

        pivot.reset_index(inplace=True)

        # dati "Frontal + Pre Frontal"
        if not channel_means_selected.empty:
            pivot_frontal = channel_means_selected.pivot_table(index=['Group', 'Channel'],
                                                               columns='Phase_Assigned',
                                                               values=feature_name)
            pivot_frontal = pivot_frontal.dropna(subset=['Early', 'Late'])
            pivot_frontal['Diff'] = pivot_frontal['Late'] - pivot_frontal['Early']
            #pivot_frontal['Diff'] = pivot_frontal['Early'] - pivot_frontal['Late']
            pivot_frontal.reset_index(inplace=True)
        else:
            pivot_frontal = None

        # Plot All Channels - viola, bordo nero visibile con linewidth=1.5
        sns.stripplot(data=pivot, x='Group', y='Diff', order=group_order,
                      jitter=True, size=7, edgecolor='black', linewidth=1.5, alpha=0.7,
                      color='pink', label='All Channels')

        # Plot Frontal + Pre Frontal - verde chiaro, bordo nero visibile

        if pivot_frontal is not None:
            sns.stripplot(data=pivot_frontal, x='Group', y='Diff', order=group_order,
                          jitter=True, size=7, edgecolor='black', linewidth=1.5, alpha=0.7,
                          color='green', label='Frontal channels')

        # Linee medie dashed
        means = pivot.groupby('Group')['Diff'].mean()
        for i, g in enumerate(group_order):
            if g in means.index:
                plt.hlines(y=means[g], xmin=i - 0.3, xmax=i + 0.3,
                           colors='purple', linestyles='dashed', lw=2)

        if pivot_frontal is not None:
            means_f = pivot_frontal.groupby('Group')['Diff'].mean()
            for i, g in enumerate(group_order):
                if g in means_f.index:
                    plt.hlines(y=means_f[g], xmin=i - 0.3, xmax=i + 0.3,
                               colors='green', linestyles='dashed', lw=2)

        def parse_posthoc_symbols(ph_dict):
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
                    parsed[g1].append(SYMBOL_MAP.get(g2, '?'))
            return parsed

        sig_symbols = parse_posthoc_symbols(post_hoc_all)
        sig_symbols_frontal = parse_posthoc_symbols(post_hoc_frontal)

        ymin, ymax = plt.ylim()
        y_pos_all = ymin - 0.05 * (ymax - ymin)
        y_pos_frontal = y_pos_all - 0.05 * (ymax - ymin)

        for i, g in enumerate(group_order):
            if sig_symbols.get(g):
                plt.text(i, y_pos_all, ''.join(sig_symbols[g]),
                         ha='center', va='top', fontsize=16, color='black')
            if sig_symbols_frontal.get(g):
                plt.text(i, y_pos_frontal, ''.join(sig_symbols_frontal[g]),
                         ha='center', va='top', fontsize=16, color='green')

        plt.title(f'Distribution of channel-wise mean differences (Late - Early)')
        plt.ylabel('Late - Early Difference of Sample Entropy')
        plt.xlabel('Group')
        plt.xticks(fontsize=12)
        plt.ylim(y_pos_frontal - 0.05 * (ymax - ymin), ymax)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # Legenda manuale con punti e dashed lines
        legend_handles = [
            Patch(facecolor='pink', edgecolor='black', label='All Channels'),
            #Patch(facecolor='green', edgecolor='black', label='Frontal Channels'),
            Line2D([0], [0], color='purple', lw=1.5, linestyle='dashed', label='Mean - All Channels'),
            #Line2D([0], [0], color='green', lw=1.5, linestyle='dashed', label='Mean - Frontal Channels')
        ]
        #plt.legend(handles=legend_handles, loc='lower left', fontsize=10, title="Legend")
        plt.legend(handles=legend_handles, loc='upper right', fontsize=10, title='Legend', title_fontsize=10)
        #plt.legend(handles=handles, loc='lower right', fontsize=9)
        # --- Grid verticale tra i gruppi ---
        for i in range(len(group_order) - 1):
            x_pos = i + 0.5
            plt.axvline(x=x_pos, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)

        symbol_legend = (
            "p < 0.05:\n"
            "+ vs. CTL\n"
            "§ vs. DNV\n"
            "$ vs. ADV\n"
            "# vs. DYS\n"
        )
        #plt.gcf().text(0.65, 0.84, symbol_legend, fontsize=10, verticalalignment='top')
        #plt.gcf().text(0.30, 0.24, symbol_legend, fontsize=10, verticalalignment='top')
        plt.gcf().text(0.67, 0.86, symbol_legend, fontsize=10, verticalalignment='top')
        plt.ylim(-0.12, None)  # imposta il limite inferiore a -2, il superiore resta automatico

        plt.show()

    #all_pvals = compare_early_late(channel_means_all, feature_name, label="All Channels")
    #frontal_pvals = compare_early_late(channel_means_selected, feature_name, label="Frontal Channels")
    #plot_data(post_hoc_all=post_hoc_all,post_hoc_frontal=post_hoc_frontal,all_pvals=all_pvals,frontal_pvals=frontal_pvals)
    plot_data(post_hoc_all=None, post_hoc_frontal=None)
    plot_group_diff_stripplot(post_hoc_all=post_hoc_all, post_hoc_frontal=post_hoc_frontal)

    #plot_data(post_hoc_all=None, post_hoc_frontal=None)


# Carica i dati
# file_path = r"D:\TESI\prova statistica\N2N3multitaperMeanPSD_specific_channels_149\_N2N3multitaperMeanPSD_specific_channels_149_aggregated_with_phases.csv"
file_path = r"D:\TESI\prova statistica\N2N3ALLENTROPY_specific_channels_149\_N2N3ALLENTROPY_specific_channels_149_aggregated_with_phases.csv"
#file_path = r"D:\TESI\prova statistica\N3MULTISCALEENTROPY_specific_channels_149\_N3MULTISCALEENTROPY_specific_channels_149_aggregated_with_phases.csv"
#file_path = r"D:\TESI\prova statistica\N3CORRDIM_specific_channels_149\_N3CORRDIM_specific_channels_149_aggregated_with_phases.csv"
#file_path=r"D:\TESI\prova statistica\N3CONN_specific_channels_149\_N3CONN_specific_channels_149_aggregated_with_phases.csv"
data = pd.read_csv(file_path)
data = data[data['Stage'] == 3]
feature_to_analyze=["SampEn2"]
'''
feature_to_analyze = [
    "Sample Entropy", "Spectral Entropy", "Permutation Entropy",
    "SampEn1", "SampEn2", "SampEn3", "SampEn4", "SampEn5",
    "SampEn6", "SampEn7", "SampEn8", "SampEn9", "SampEn10",
    "SampEn11", "SampEn12", "SampEn13", "SampEn14", "SampEn15",
    "SampEn16", "SampEn17", "SampEn18", "SampEn19", "SampEn20"
]
'''
# feature_to_analyze = ['2. Absolute High Delta Power']
'''
# Ciclo per analizzare ogni caratteristica
for feature in feature_to_analyze:
    print(f"\n### Analisi per caratteristica: {feature} ###")
    results = analyze_feature(data, feature)
'''
# Seleziona solo alcuni canali
'''selected_channels = [27, 33, 34, 38, 39, 47, 48, 26, 20, 19, 12, 11, 3, 2, 222, 16, 22, 23, 24, 28, 29, 30, 35, 36, 40,
                     41, 42, 49, 50, 21, 15, 7, 14, 6,
                     207, 13, 5, 215, 4, 224, 223, 214, 206, 213, 205] '''
                      # Aggiungi i canali di interesse
selected_channels=None
# selected_channels=None
# Percorso della cartella in cui salvare i file Excel
output_dir = r"D:\TESI\excel\topographic"

# Crea la cartella se non esiste
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Genera il grafico solo per i canali selezionati
for feature_name in feature_to_analyze:
    # Utilizza il nome della feature come nome del file Excel
    output_excel = os.path.join(output_dir, f"{feature_name.replace(' ', '_')}.xlsx")

    analyze_and_plot(data=data.copy(), feature_name=feature_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'],
                     selected_channels=selected_channels, output_excel=output_excel)
    # plot_selected_channels_mean_per_group(data.copy(), feature_name=feature_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'], selected_channels=selected_channels)
    # plot_early_and_late_per_channel(data.copy(), feature_name=feature_name, group_order=['CTL', 'DNV', 'ADV', 'DYS' ], selected_channels=selected_channels)