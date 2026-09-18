"""
plot_report_incertezza_componibile_unificato.py
─────────────────────────────────────────────────────────────────────────────
Sostituisce plot_report_incertezza_componibile_optuna.py (diretta) e
plot_report_incertezza_residual_componibile_optuna.py (residuo) con UN solo
script di grafici, costruito sopra build_report_incertezza_componibile_unificato.py
(le 6 famiglie: 3 metodi × diretta/residuo, tutte Optuna — vedi quel file
per la spiegazione completa).

Rispetto ai due script che sostituisce: i grafici "per orizzonte"
(MAE/coverage/calibrazione/ranking/evento-non evento) ora mostrano tutte e
6 le famiglie insieme (colore per metodo, tratteggio = diretta, pieno =
residuo — stessa convenzione visiva già usata nel vecchio grafico 13 di
confronto diretta/residuo, qui estesa a tutti i grafici). I grafici dedicati
al confronto diretta-vs-residuo (MAE/coverage a barre appaiate, epistemica/
aleatoria sovrapposta) restano come viste mirate a "partire dalla fisica
Penman riduce l'incertezza, o no?" — ora alimentati direttamente dallo
stesso dataframe unificato (non da un secondo caricamento separato).

Output in confronto/grafici_incertezza_optuna/:
  01_mae_per_orizzonte.png
  02_coverage_90_per_orizzonte.png
  03_curva_di_calibrazione.png
  04_epistemica_aleatoria_diretta_vs_residuo.png
  05_intervallo_temporale.png              — griglia 3 metodi x 2 (diretta/residuo)
  06_ranking_mae.png
  07_coverage_eventi_vs_non_eventi.png
  08_sharpness_eventi_vs_non_eventi.png
  09_mae_diretta_vs_residuo.png
  10_coverage_diretta_vs_residuo.png
  11_aloni_incertezza_mc-dropout_diretta.png
  12_aloni_incertezza_mc-dropout_residuo.png
  13_aloni_incertezza_deep-ensemble_diretta.png
  14_aloni_incertezza_deep-ensemble_residuo.png
"""

# ─────────────────────────────────────────────────────────────────────────────
# Versione per pipeline_corrette/: legge i risultati prodotti dalle pipeline di
# QUESTA cartella (pioggia ed ET0 previsti aggregati correttamente), non da
# ai_feature_componibile/. Nessun import fuori da pipeline_corrette/.
# Nota: penman_baseline/results_osservato/ e' un ORACOLO (pioggia realmente
# caduta) e non va confrontato alla pari con le pipeline previsionali.
# ─────────────────────────────────────────────────────────────────────────────


import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import build_report_incertezza_optuna as bri

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "grafici_incertezza_optuna")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

HORIZ_X = list(range(1, 8))
Z_90 = 1.645   # quantile normale per un intervallo simmetrico al 90%
EVENT_SHADE_COLOR = "#FDE68A"
DIRETTA_COLOR = "#9CA3AF"
RESIDUO_COLOR = "#1D4ED8"

# Colore per METODO (usato su tutti i grafici "per orizzonte"): tratteggio
# = diretta, pieno = residuo — stessa serie di colori, solo lo stile della
# linea distingue diretta/residuo.
METHOD_COLOR = {
    bri.METHOD_MC: "#1D4ED8",
    bri.METHOD_ENSEMBLE: "#059669",
    bri.METHOD_QUANTILE: "#B45309",
    bri.METHOD_VI: "#7C3AED",
    bri.METHOD_LAPLACE: "#4F46E5",
    bri.METHOD_MC_WARMUP: "#0E7490",
    bri.METHOD_ENSEMBLE_WARMUP: "#4D7C0F",
}
LINESTYLE = {bri.APPROCCIO_DIRETTA: "--", bri.APPROCCIO_RESIDUO: "-"}
ALPHA_LINE = {bri.APPROCCIO_DIRETTA: 0.6, bri.APPROCCIO_RESIDUO: 1.0}

# Colore per TIPO (usato dove diretta/residuo compaiono come categorie
# separate sullo stesso asse, es. barre): stessa tinta del metodo, più chiara
# per diretta, più scura per residuo — stessa logica di TIPO_FILL nel report
# Excel. I due metodi con warm-up riprendono la famiglia di colore di quello
# da cui derivano.
_TIPO_TINTE = {
    bri.METHOD_MC: ("#93C5FD", "#1D4ED8"),
    bri.METHOD_ENSEMBLE: ("#6EE7B7", "#059669"),
    bri.METHOD_QUANTILE: ("#FCD34D", "#B45309"),
    bri.METHOD_VI: ("#C4B5FD", "#7C3AED"),
    bri.METHOD_LAPLACE: ("#A5B4FC", "#4F46E5"),
    bri.METHOD_MC_WARMUP: ("#67E8F9", "#0E7490"),
    bri.METHOD_ENSEMBLE_WARMUP: ("#BEF264", "#4D7C0F"),
}
TIPO_COLOR = {
    bri.tipo_label(metodo, approccio): tinte[i]
    for metodo, tinte in _TIPO_TINTE.items()
    for i, approccio in enumerate(bri.APPROCCI)
}


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def horizon_series(combined, tipo, feature_set, column):
    subset = combined[(combined["tipo"] == tipo) & (combined["feature_set"] == feature_set)]
    subset = subset.set_index("horizon").reindex(bri.HORIZONS)
    return subset[column].values.astype(float)


def _pick_representative_sensor(preds):
    test_rows = preds[
        (preds["split"] == "test") & (preds["horizon"] == "t+1") &
        (preds["feature_set"] == preds["feature_set"].iloc[0])
    ]
    counts = test_rows.groupby("sensor").size().sort_values(ascending=False)
    return counts.index[0]


def metodi_con_dati(best_feature_set, solo_gaussiani=False):
    """I metodi che hanno almeno una variante con risultati. Con i run
    lanciati un po' alla volta, ciclare su bri.METHODS produrrebbe pannelli
    vuoti."""
    metodi = []
    for method in bri.METHODS:
        if solo_gaussiani and bri.METHOD_KIND[method] != "gaussian":
            continue
        if any(best_feature_set.get(bri.tipo_label(method, a)) for a in bri.APPROCCI):
            metodi.append(method)
    return metodi


def _griglia_pannelli(n_metodi, larghezza=4.6, altezza=3.6, sharey=True):
    """Griglia di pannelli, uno per metodo: con 14 famiglie un grafico unico
    sarebbe illeggibile, mentre due linee per pannello (diretta e residuo) si
    leggono, e l'asse condiviso tiene i metodi confrontabili fra loro."""
    ncols = min(4, n_metodi)
    nrows = math.ceil(n_metodi / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(larghezza * ncols, altezza * nrows),
                             squeeze=False, sharex=True, sharey=sharey)
    flat = axes.ravel()
    for ax in flat[n_metodi:]:
        ax.axis("off")

    # con sharex le etichette dell'asse x restano solo sull'ultima riga: nelle
    # colonne che finiscono prima (l'ultima riga è incompleta) il pannello più
    # in basso resterebbe senza t+1..t+7. Si riaccendono lì.
    for i in range(n_metodi):
        if i + ncols >= n_metodi:
            flat[i].tick_params(labelbottom=True)
    return fig, flat[:n_metodi]


def _assi_orizzonte(ax):
    ax.set_xticks(HORIZ_X)
    ax.set_xticklabels([f"t+{h}" for h in HORIZ_X])


def _shade_event_days(ax, dates, event_flags):
    """Sfondo colorato in corrispondenza dei giorni con evento acqua (pioggia
    o irrigazione) — un rettangolo largo ~1 giorno per ciascuna data marcata
    has_water_event=True, così l'occhio associa subito i picchi di incertezza
    ai giorni di evento senza dover incrociare due grafici separati."""
    half_day = pd.Timedelta(hours=12)
    for d, is_event in zip(dates, event_flags):
        if is_event:
            ax.axvspan(d - half_day, d + half_day, color=EVENT_SHADE_COLOR, alpha=0.6, zorder=0, linewidth=0)


# ═══════════════════════════════════════════════════════════════════════════
# 01 — MAE per orizzonte, tutte e 6 le famiglie
# ═══════════════════════════════════════════════════════════════════════════

def plot_mae_per_horizon(combined, best_feature_set):
    metodi = metodi_con_dati(best_feature_set)
    fig, axes = _griglia_pannelli(len(metodi))

    for ax, method in zip(axes, metodi):
        for approccio in bri.APPROCCI:
            tipo = bri.tipo_label(method, approccio)
            fset = best_feature_set.get(tipo)
            if fset is None:
                continue
            y = horizon_series(combined, tipo, fset, "MAE")
            ax.plot(HORIZ_X, y, marker="o", label=f"{approccio} ({fset})",
                    color=METHOD_COLOR[method], linestyle=LINESTYLE[approccio],
                    alpha=ALPHA_LINE[approccio], linewidth=2)
        ax.set_title(method, fontsize=11)
        ax.set_xlabel("Orizzonte")
        ax.set_ylabel("MAE (punti % umidità)")
        _assi_orizzonte(ax)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("MAE per orizzonte — migliore di ciascuna famiglia (Optuna, tratteggio = diretta, pieno = residuo)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_mae_per_orizzonte.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 02 — coverage al 90% per orizzonte
# ═══════════════════════════════════════════════════════════════════════════

def plot_coverage_90_per_horizon(combined, best_feature_set):
    metodi = metodi_con_dati(best_feature_set)
    fig, axes = _griglia_pannelli(len(metodi))

    for ax, method in zip(axes, metodi):
        for approccio in bri.APPROCCI:
            tipo = bri.tipo_label(method, approccio)
            fset = best_feature_set.get(tipo)
            if fset is None:
                continue
            y = horizon_series(combined, tipo, fset, "coverage_90")
            ax.plot(HORIZ_X, y, marker="o", label=f"{approccio} ({fset})",
                    color=METHOD_COLOR[method], linestyle=LINESTYLE[approccio],
                    alpha=ALPHA_LINE[approccio], linewidth=2)
        ax.axhline(0.90, color="black", linestyle=":", linewidth=1)
        ax.set_ylim(0.5, 1.02)
        ax.set_title(method, fontsize=11)
        ax.set_xlabel("Orizzonte")
        ax.set_ylabel("Coverage al 90%")
        _assi_orizzonte(ax)
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Coverage al 90% per orizzonte — sotto la riga punteggiata = intervalli troppo stretti "
                 "(tratteggio = diretta, pieno = residuo)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_coverage_90_per_orizzonte.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 03 — curva di calibrazione (nominale vs empirica)
# ═══════════════════════════════════════════════════════════════════════════

def plot_calibration_curve(combined, best_feature_set):
    gaussian_levels = [50, 68, 80, 90, 95, 99]
    quantile_levels = [50, 90]

    metodi = metodi_con_dati(best_feature_set)
    fig, axes = _griglia_pannelli(len(metodi), larghezza=4.2, altezza=4.2)

    all_rows = combined[combined["horizon"] == bri.ALL_KEY]
    for ax, method in zip(axes, metodi):
        # i livelli gaussiani non si applicano alla Quantile Regression, che
        # ha solo i percentili che ha stimato: la distinzione segue
        # METHOD_KIND, non il nome del metodo
        levels = gaussian_levels if bri.METHOD_KIND[method] == "gaussian" else quantile_levels
        for approccio in bri.APPROCCI:
            tipo = bri.tipo_label(method, approccio)
            fset = best_feature_set.get(tipo)
            if fset is None:
                continue
            row = all_rows[(all_rows["tipo"] == tipo) & (all_rows["feature_set"] == fset)]
            if row.empty:
                continue
            row = row.iloc[0]
            nominal = [lv / 100.0 for lv in levels]
            empirical = [float(row[f"coverage_{lv}"]) for lv in levels]
            ax.plot(nominal, empirical, marker="o", label=f"{approccio} ({fset})",
                    color=METHOD_COLOR[method], linestyle=LINESTYLE[approccio],
                    alpha=ALPHA_LINE[approccio], linewidth=2)

        ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1)
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0.45, 1.02)
        ax.set_title(method, fontsize=11)
        ax.set_xlabel("Livello nominale")
        ax.set_ylabel("Coverage empirica")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Curva di calibrazione (Complessivo 7gg, test) — la diagonale punteggiata è la "
                 "calibrazione perfetta", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_curva_di_calibrazione.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 04 — epistemica vs aleatoria per orizzonte, Diretta (tratteggiata) vs
# Residuo (piena) sovrapposte — un pannello per ogni metodo gaussiano con
# dati. La Quantile Regression resta fuori: stima percentili e non scompone
# l'incertezza nelle due componenti.
# ═══════════════════════════════════════════════════════════════════════════

def plot_epistemic_aleatoric_diretta_vs_residuo(combined, best_feature_set):
    metodi = metodi_con_dati(best_feature_set, solo_gaussiani=True)
    if not metodi:
        return
    fig, axes = _griglia_pannelli(len(metodi), larghezza=4.8, altezza=3.8)

    for ax, method in zip(axes, metodi):
        for approccio, larghezza, alpha in ((bri.APPROCCIO_DIRETTA, 1.5, 0.6),
                                            (bri.APPROCCIO_RESIDUO, 2.2, 1.0)):
            fset = best_feature_set.get(bri.tipo_label(method, approccio))
            if fset is None:
                continue
            tipo = bri.tipo_label(method, approccio)
            epistemica = horizon_series(combined, tipo, fset, "mean_epistemic_std")
            aleatoria = horizon_series(combined, tipo, fset, "mean_aleatoric_std")
            ax.plot(HORIZ_X, epistemica, marker="o", linestyle=LINESTYLE[approccio],
                    label=f"epistemica ({approccio}, {fset})", color="#7C3AED",
                    linewidth=larghezza, alpha=alpha)
            ax.plot(HORIZ_X, aleatoria, marker="s", linestyle=LINESTYLE[approccio],
                    label=f"aleatoria ({approccio}, {fset})", color="#DC2626",
                    linewidth=larghezza, alpha=alpha)

        ax.set_xlabel("Orizzonte")
        ax.set_ylabel("Deviazione standard (punti % umidità)")
        _assi_orizzonte(ax)
        ax.set_title(method, fontsize=11)
        ax.legend(loc="upper left", fontsize=7)

    fig.suptitle("Incertezza epistemica e aleatoria per orizzonte — diretta (tratteggiata) vs residuo (piena)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_epistemica_aleatoria_diretta_vs_residuo.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 05 — andamento nel tempo con banda di previsione al 90%: griglia 3 metodi
# x 2 (diretta/residuo)
# ═══════════════════════════════════════════════════════════════════════════

def plot_time_series_with_band(best_feature_set):
    preds_cache = {}

    def get_preds(method, approccio):
        key = (method, approccio)
        if key not in preds_cache:
            preds_cache[key] = pd.read_csv(bri.SOURCES[key]["preds"])
        return preds_cache[key]

    metodi = metodi_con_dati(best_feature_set)
    sensor = _pick_representative_sensor(get_preds(*bri.FAMIGLIE_DISPONIBILI[0]))

    fig, axes = plt.subplots(len(metodi), 2, figsize=(15, 4.3 * len(metodi)),
                             sharex=True, squeeze=False)
    for row_i, method in enumerate(metodi):
        for col_i, approccio in enumerate(bri.APPROCCI):
            ax = axes[row_i, col_i]
            tipo = bri.tipo_label(method, approccio)
            fset = best_feature_set.get(tipo)
            if fset is None:
                ax.set_visible(False)
                continue
            preds = get_preds(method, approccio)
            sub = preds[(preds["split"] == "test") & (preds["horizon"] == "t+1") &
                       (preds["feature_set"] == fset) & (preds["sensor"] == sensor)].sort_values("date")

            if bri.METHOD_KIND[method] == "quantile":
                ax.fill_between(sub["date"], sub["q05"], sub["q95"], color=METHOD_COLOR[method], alpha=0.25, label="intervallo 90% (Q05-Q95)")
                ax.plot(sub["date"], sub["q50"], color=METHOD_COLOR[method], linewidth=1.5, label="mediana prevista")
            else:
                lower = sub["mu_pred"] - Z_90 * sub["total_std"]
                upper = sub["mu_pred"] + Z_90 * sub["total_std"]
                ax.fill_between(sub["date"], lower, upper, color=METHOD_COLOR[method], alpha=0.25, label="intervallo 90%")
                ax.plot(sub["date"], sub["mu_pred"], color=METHOD_COLOR[method], linewidth=1.5, label="previsto")

            ax.scatter(sub["date"], sub["y_true"], color="black", s=12, zorder=5, label="reale")
            ax.set_title(f"{tipo} ({fset})", fontsize=10)
            ax.legend(loc="upper left", fontsize=7)
            if row_i == len(metodi) - 1:
                ax.tick_params(axis="x", rotation=45)
        axes[row_i, 0].set_ylabel("Umidità (%)")

    fig.suptitle(f"Andamento nel tempo con banda al 90% — {sensor}, orizzonte t+1")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_intervallo_temporale.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 06 — ranking completo per MAE, tutte e 6 le famiglie
# ═══════════════════════════════════════════════════════════════════════════

def plot_ranking_mae(combined):
    all_rows = combined[combined["horizon"] == bri.ALL_KEY].copy()
    all_rows = all_rows.sort_values("MAE").reset_index(drop=True)
    labels = [f"{r['tipo']} — {r['feature_set']}" for _, r in all_rows.iterrows()]
    colors = [TIPO_COLOR[r["tipo"]] for _, r in all_rows.iterrows()]

    fig, ax = plt.subplots(figsize=(9, max(6, 0.3 * len(all_rows))))
    y_pos = np.arange(len(all_rows))[::-1]
    ax.barh(y_pos, all_rows["MAE"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("MAE complessivo (7 orizzonti, test)")
    ax.set_title("Ranking completo per MAE — tutte le combinazioni (Optuna, diretta+residuo)")

    handles = [Patch(color=TIPO_COLOR[t], label=t) for t in bri.TIPO_ORDER]
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "06_ranking_mae.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 07/08 — eventi vs non-eventi (coverage_90 e sharpness_90, Complessivo 7gg)
# ═══════════════════════════════════════════════════════════════════════════

def _event_vs_nonevent_bars(event_comparison, value_col_evento, value_col_non_evento, ylabel, title, out_name, nominal_line=None):
    all_rows = event_comparison[event_comparison["_horizon_key"] == bri.ALL_KEY]

    tipi = [t for t in bri.TIPO_ORDER if t in all_rows["tipo"].values]
    x = np.arange(len(tipi))
    width = 0.35

    evento_values = [float(all_rows[all_rows["tipo"] == t][value_col_evento].iloc[0]) for t in tipi]
    non_evento_values = [float(all_rows[all_rows["tipo"] == t][value_col_non_evento].iloc[0]) for t in tipi]

    # la larghezza cresce con il numero di famiglie: a 14 categorie le
    # etichette in un grafico da 11 pollici si sovrappongono
    fig, ax = plt.subplots(figsize=(max(11, 1.15 * len(tipi) + 4), 6.5))
    ax.bar(x - width / 2, evento_values, width, label="giorni con evento (pioggia/irrigazione)", color="#1D4ED8")
    ax.bar(x + width / 2, non_evento_values, width, label="giorni senza evento", color="#9CA3AF")

    if nominal_line is not None:
        ax.axhline(nominal_line, color="black", linestyle="--", linewidth=1, label="livello nominale (90%)")

    ax.set_xticks(x)
    ax.set_xticklabels(tipi, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name))
    plt.close(fig)


def plot_event_vs_nonevent_coverage(event_comparison):
    _event_vs_nonevent_bars(
        event_comparison, "coverage_90_evento", "coverage_90_non_evento",
        ylabel="Coverage empirica (intervallo al 90%)",
        title="Coverage al 90%: giorni con evento vs senza (Complessivo 7gg, test, Optuna)",
        out_name="07_coverage_eventi_vs_non_eventi.png",
        nominal_line=0.90,
    )


def plot_event_vs_nonevent_sharpness(event_comparison):
    _event_vs_nonevent_bars(
        event_comparison, "sharpness_90_evento", "sharpness_90_non_evento",
        ylabel="Sharpness (ampiezza media intervallo al 90%)",
        title="Sharpness al 90%: giorni con evento vs senza (Complessivo 7gg, test, Optuna)",
        out_name="08_sharpness_eventi_vs_non_eventi.png",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 09/10 — Diretta vs Residuo: MAE e coverage_90 (Complessivo 7gg, per
# metodo, STESSO feature-set — quello scelto dal residuo come migliore)
# ═══════════════════════════════════════════════════════════════════════════

def _diretta_vs_residuo_bars(diretta_vs_residuo, metric_col, ylabel, title, out_name, nominal_line=None):
    metodi = [m for m in bri.METHODS if m in diretta_vs_residuo["metodo"].values]
    x = np.arange(len(metodi))
    width = 0.35

    diretta_values = [float(diretta_vs_residuo[diretta_vs_residuo["metodo"] == m][f"diretta_{metric_col}"].iloc[0]) for m in metodi]
    residuo_values = [float(diretta_vs_residuo[diretta_vs_residuo["metodo"] == m][f"residuo_{metric_col}"].iloc[0]) for m in metodi]

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(metodi) + 2), 5.8))
    ax.bar(x - width / 2, diretta_values, width, label="Diretta (target assoluto)", color=DIRETTA_COLOR)
    ax.bar(x + width / 2, residuo_values, width, label="Residuo (target − Penman)", color=RESIDUO_COLOR)

    if nominal_line is not None:
        ax.axhline(nominal_line, color="black", linestyle="--", linewidth=1, label="livello nominale (90%)")

    ax.set_xticks(x)
    ax.set_xticklabels(metodi, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name))
    plt.close(fig)


def plot_mae_diretta_vs_residuo(diretta_vs_residuo):
    _diretta_vs_residuo_bars(
        diretta_vs_residuo, "MAE",
        ylabel="MAE (punti % umidità)",
        title="MAE: Diretta vs Residuo, stesso feature-set (Complessivo 7gg, test, Optuna)",
        out_name="09_mae_diretta_vs_residuo.png",
    )


def plot_coverage_diretta_vs_residuo(diretta_vs_residuo):
    _diretta_vs_residuo_bars(
        diretta_vs_residuo, "coverage_90",
        ylabel="Coverage empirica (intervallo al 90%)",
        title="Coverage al 90%: Diretta vs Residuo, stesso feature-set (Complessivo 7gg, test, Optuna)",
        out_name="10_coverage_diretta_vs_residuo.png",
        nominal_line=0.90,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 11+ — "aloni" di incertezza nel tempo, con sfondo evento/non-evento — una
# immagine per ciascuna famiglia gaussiana disponibile (la Quantile
# Regression non scompone l'incertezza, quindi non ha aloni da mostrare).
# ═══════════════════════════════════════════════════════════════════════════

def plot_uncertainty_halo(preds_df, tipo, fset, sensor, horizon, out_name, color):
    sub = preds_df[(preds_df["split"] == "test") & (preds_df["horizon"] == horizon) &
                   (preds_df["feature_set"] == fset) & (preds_df["sensor"] == sensor)].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date")

    dates = sub["date"]
    mu = sub["mu_pred"].values
    epistemic = sub["epistemic_std"].values
    aleatoric = sub["aleatoric_std"].values
    total = sub["total_std"].values
    y_true = sub["y_true"].values
    event = sub["has_water_event"].values.astype(bool)

    aleatoric_lower = mu - Z_90 * aleatoric
    aleatoric_upper = mu + Z_90 * aleatoric
    total_lower = mu - Z_90 * total
    total_upper = mu + Z_90 * total

    fig, (ax_halo, ax_std) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True,
                                          gridspec_kw={"height_ratios": [2, 1]})

    _shade_event_days(ax_halo, dates, event)
    _shade_event_days(ax_std, dates, event)

    ax_halo.fill_between(dates, total_lower, total_upper, color=color, alpha=0.15,
                         label="alone esterno: incertezza totale (90%)", zorder=1)
    ax_halo.fill_between(dates, aleatoric_lower, aleatoric_upper, color=color, alpha=0.40,
                         label="alone interno: incertezza aleatoria (90%)", zorder=2)
    ax_halo.plot(dates, mu, color="black", linewidth=1.3, label="previsto (mu)", zorder=3)
    ax_halo.scatter(dates, y_true, color="black", s=12, zorder=4, label="reale")

    ax_halo.set_ylabel("Umidità (%)")
    ax_halo.set_title(f"{tipo} ({fset}) — {sensor}, {horizon}, test — aloni di incertezza\n"
                      f"(divario tra alone esterno e interno = incertezza epistemica; sfondo giallo = giorno con evento acqua)")
    event_patch = Patch(facecolor=EVENT_SHADE_COLOR, alpha=0.6, label="giorno con evento acqua")
    handles, labels = ax_halo.get_legend_handles_labels()
    ax_halo.legend(handles + [event_patch], labels + ["giorno con evento acqua"], loc="upper left", fontsize=8)

    ax_std.plot(dates, aleatoric, color="#DC2626", linewidth=1.5, label="aleatoric_std")
    ax_std.plot(dates, epistemic, color="#7C3AED", linewidth=1.5, label="epistemic_std")
    ax_std.set_ylabel("Deviazione standard")
    ax_std.set_xlabel("Data")
    ax_std.legend(loc="upper left", fontsize=8)
    ax_std.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name))
    plt.close(fig)


def plot_halo(method, approccio, best_feature_set, sensor, idx):
    tipo = bri.tipo_label(method, approccio)
    fset = best_feature_set.get(tipo)
    if fset is None:
        print(f"  salto alone {tipo}: nessun feature-set migliore disponibile")
        return
    preds = pd.read_csv(bri.SOURCES[(method, approccio)]["preds"])
    slug = method.lower().replace(" ", "-")
    out_name = f"{idx}_aloni_incertezza_{slug}_{approccio}.png"
    plot_uncertainty_halo(preds, tipo, fset, sensor, "t+1", out_name, TIPO_COLOR[tipo])


def main():
    combined = bri.load_combined_metrics()
    best_feature_set = bri.pick_best_feature_set_per_tipo(combined)
    print("Migliori feature-set per famiglia (Optuna, diretta+residuo):", best_feature_set)

    print("Calcolo confronto evento vs non-evento...")
    event_comparison = bri.compute_event_vs_nonevent(best_feature_set)

    print("Calcolo confronto Diretta vs Residuo (stesso feature-set)...")
    diretta_vs_residuo = bri.compute_diretta_vs_residuo(combined, best_feature_set)

    plot_mae_per_horizon(combined, best_feature_set)
    plot_coverage_90_per_horizon(combined, best_feature_set)
    plot_calibration_curve(combined, best_feature_set)
    plot_epistemic_aleatoric_diretta_vs_residuo(combined, best_feature_set)
    plot_time_series_with_band(best_feature_set)
    plot_ranking_mae(combined)
    plot_event_vs_nonevent_coverage(event_comparison)
    plot_event_vs_nonevent_sharpness(event_comparison)
    plot_mae_diretta_vs_residuo(diretta_vs_residuo)
    plot_coverage_diretta_vs_residuo(diretta_vs_residuo)

    # sensore rappresentativo dalla prima famiglia disponibile: MC-Dropout
    # diretta potrebbe non essere ancora stata prodotta
    prima_preds = pd.read_csv(bri.SOURCES[bri.FAMIGLIE_DISPONIBILI[0]]["preds"])
    sensor = _pick_representative_sensor(prima_preds)
    print(f"Sensore rappresentativo per gli aloni: {sensor}")

    # un alone per ogni famiglia gaussiana disponibile, numerati da 11 in poi
    idx = 11
    for method, approccio in bri.FAMIGLIE_DISPONIBILI:
        if bri.METHOD_KIND[method] != "gaussian":
            continue
        plot_halo(method, approccio, best_feature_set, sensor, str(idx))
        idx += 1

    print(f"\nGrafici salvati in: {OUT_DIR}")


if __name__ == "__main__":
    main()