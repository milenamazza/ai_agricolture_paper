"""
plot_report_incertezza_componibile.py
─────────────────────────────────────────────────────────────────────────────
Grafici di confronto tra i 3 metodi di incertezza COMPONIBILI (MC-Dropout,
Deep Ensemble, Quantile Regression), sul migliore feature-set di ciascuno
(stessa selezione di build_report_incertezza_componibile.py, riusata da qui
in sola lettura). Versione componibile di
confronto/plot_report_incertezza.py: stessi 6 grafici originali, più 2 nuovi
per il confronto giorni-con-evento vs giorni-senza-evento (stessi numeri del
foglio "Eventi vs non-eventi" del report Excel).

Output in confronto/grafici_incertezza/:
  01_mae_per_orizzonte.png       — MAE per t+1..t+7, i 3 migliori
  02_coverage_90_per_orizzonte.png — coverage al 90% per orizzonte, con riga
                                     di riferimento al livello nominale
  03_curva_di_calibrazione.png   — coverage nominale vs empirica (tutti i
                                     livelli disponibili per famiglia)
  04_epistemica_vs_aleatoria.png — le due componenti di incertezza per
                                     orizzonte (solo MC-Dropout/Ensemble,
                                     Quantile Regression non le scompone)
  05_intervallo_temporale.png    — andamento nel tempo (un sensore) con banda
                                     di previsione al 90%, un pannello per
                                     metodo
  06_ranking_mae.png             — ranking completo per MAE (tutte le
                                     combinazioni famiglia x feature-set)
  07_coverage_eventi_vs_non_eventi.png — coverage_90 nei giorni con evento
                                     acqua (pioggia/irrigazione) vs senza,
                                     una barra per famiglia, Complessivo 7gg
  08_sharpness_eventi_vs_non_eventi.png — stesso confronto per sharpness_90
"""

# ─────────────────────────────────────────────────────────────────────────────
# Versione per pipeline_corrette/: legge i risultati prodotti dalle pipeline di
# QUESTA cartella (pioggia ed ET0 previsti aggregati correttamente), non da
# ai_feature_componibile/. Nessun import fuori da pipeline_corrette/.
# Nota: penman_baseline/results_osservato/ e' un ORACOLO (pioggia realmente
# caduta) e non va confrontato alla pari con le pipeline previsionali.
# ─────────────────────────────────────────────────────────────────────────────


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import build_report_incertezza as bri

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "grafici_incertezza")
os.makedirs(OUT_DIR, exist_ok=True)

MC_DROPOUT_PREDS = bri.MC_DROPOUT_PREDS
ENSEMBLE_PREDS = bri.ENSEMBLE_PREDS
QUANTILE_PREDS = bri.QUANTILE_PREDS

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

# Sei famiglie (diretta e residuo per ciascun metodo): stesso colore base per
# metodo, tinta piu' scura per il residuo — come nella versione Optuna.
COLOR = {
    bri.TIPO_MC_DROPOUT: "#1D4ED8",
    bri.TIPO_MC_DROPOUT_RES: "#1E3A8A",
    bri.TIPO_ENSEMBLE: "#059669",
    bri.TIPO_ENSEMBLE_RES: "#065F46",
    bri.TIPO_QUANTILE: "#B45309",
    bri.TIPO_QUANTILE_RES: "#7C2D12",
}
# tratteggio per la variante residuo, cosi' si distinguono anche in stampa
LINESTYLE = {t: ("--" if t.endswith("(residuo)") else "-") for t in bri.TIPO_ORDER}
Z_90 = 1.645   # quantile normale per un intervallo simmetrico al 90%


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def horizon_series(combined, tipo, feature_set, column):
    subset = combined[(combined["tipo"] == tipo) & (combined["feature_set"] == feature_set)]
    subset = subset.set_index("horizon").reindex(bri.HORIZONS)
    return subset[column].values.astype(float)


# ═══════════════════════════════════════════════════════════════════════════
# 01 — MAE per orizzonte
# ═══════════════════════════════════════════════════════════════════════════

def plot_mae_per_horizon(combined, best_feature_set):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tipo in bri.TIPO_ORDER:
        fset = best_feature_set.get(tipo)
        if fset is None:
            continue
        y = horizon_series(combined, tipo, fset, "MAE")
        ax.plot(HORIZ_X, y, marker="o", label=f"{tipo} ({fset})", color=COLOR[tipo],
                linestyle=LINESTYLE[tipo], linewidth=2)
    ax.set_xlabel("Orizzonte di previsione")
    ax.set_ylabel("MAE (punti % umidità)")
    ax.set_xticks(HORIZ_X)
    ax.set_xticklabels([f"t+{h}" for h in HORIZ_X])
    ax.set_title("MAE per orizzonte — migliore di ciascun metodo (componibile)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_mae_per_orizzonte.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 02 — coverage al 90% per orizzonte
# ═══════════════════════════════════════════════════════════════════════════

def plot_coverage_90_per_horizon(combined, best_feature_set):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tipo in bri.TIPO_ORDER:
        fset = best_feature_set.get(tipo)
        if fset is None:
            continue
        y = horizon_series(combined, tipo, fset, "coverage_90")
        ax.plot(HORIZ_X, y, marker="o", label=f"{tipo} ({fset})", color=COLOR[tipo],
                linestyle=LINESTYLE[tipo], linewidth=2)
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1, label="livello nominale (90%)")
    ax.set_xlabel("Orizzonte di previsione")
    ax.set_ylabel("Coverage empirica (intervallo al 90%)")
    ax.set_xticks(HORIZ_X)
    ax.set_xticklabels([f"t+{h}" for h in HORIZ_X])
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Coverage al 90% per orizzonte — sotto la riga = overconfident")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_coverage_90_per_orizzonte.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 03 — curva di calibrazione (nominale vs empirica)
# ═══════════════════════════════════════════════════════════════════════════

def plot_calibration_curve(combined, best_feature_set):
    fig, ax = plt.subplots(figsize=(7, 7))

    gaussian_levels = [50, 68, 80, 90, 95, 99]
    quantile_levels = [50, 90]

    all_rows = combined[combined["horizon"] == bri.ALL_KEY]
    for tipo in bri.TIPO_ORDER:
        fset = best_feature_set.get(tipo)
        if fset is None:
            continue
        row = all_rows[(all_rows["tipo"] == tipo) & (all_rows["feature_set"] == fset)].iloc[0]
        # la Quantile Regression stima solo i quantili 05/25/50/75/95, quindi
        # ha coverage solo a 50 e 90: vale per la variante diretta E per quella
        # residuo, altrimenti la seconda chiederebbe coverage_68 e compagnia e
        # finirebbe con quattro punti su sei a NaN
        e_quantile = tipo in (bri.TIPO_QUANTILE, bri.TIPO_QUANTILE_RES)
        levels = quantile_levels if e_quantile else gaussian_levels
        nominal = [lv / 100.0 for lv in levels]
        empirical = [float(row[f"coverage_{lv}"]) for lv in levels]
        ax.plot(nominal, empirical, marker="o", label=f"{tipo} ({fset})", color=COLOR[tipo],
                linestyle=LINESTYLE[tipo], linewidth=2)

    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="calibrazione perfetta")
    ax.set_xlabel("Livello di confidenza nominale")
    ax.set_ylabel("Coverage empirica")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Curva di calibrazione (Complessivo 7gg, test)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_curva_di_calibrazione.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 04 — epistemica vs aleatoria per orizzonte (solo MC-Dropout/Ensemble)
# ═══════════════════════════════════════════════════════════════════════════

def plot_epistemic_vs_aleatoric(combined, best_feature_set):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)

    for ax, tipo in zip(axes, [bri.TIPO_MC_DROPOUT, bri.TIPO_ENSEMBLE]):
        fset = best_feature_set.get(tipo)
        epistemic = horizon_series(combined, tipo, fset, "mean_epistemic_std")
        aleatoric = horizon_series(combined, tipo, fset, "mean_aleatoric_std")
        ax.plot(HORIZ_X, epistemic, marker="o", label="epistemica", color="#7C3AED", linewidth=2)
        ax.plot(HORIZ_X, aleatoric, marker="o", label="aleatoria", color="#DC2626", linewidth=2)
        ax.set_xlabel("Orizzonte di previsione")
        ax.set_xticks(HORIZ_X)
        ax.set_xticklabels([f"t+{h}" for h in HORIZ_X])
        ax.set_title(f"{tipo} ({fset})")
        ax.legend(loc="upper left", fontsize=9)

    axes[0].set_ylabel("Deviazione standard (punti % umidità)")
    fig.suptitle("Incertezza epistemica vs aleatoria per orizzonte (componibile)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_epistemica_vs_aleatoria.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 05 — andamento nel tempo con banda di previsione al 90%
# ═══════════════════════════════════════════════════════════════════════════

def _pick_representative_sensor(mc_dropout_preds):
    test_rows = mc_dropout_preds[
        (mc_dropout_preds["split"] == "test") & (mc_dropout_preds["horizon"] == "t+1") &
        (mc_dropout_preds["feature_set"] == mc_dropout_preds["feature_set"].iloc[0])
    ]
    counts = test_rows.groupby("sensor").size().sort_values(ascending=False)
    return counts.index[0]


def plot_time_series_with_band(best_feature_set):
    mc_dropout_preds = pd.read_csv(MC_DROPOUT_PREDS)
    ensemble_preds = pd.read_csv(ENSEMBLE_PREDS)
    quantile_preds = pd.read_csv(QUANTILE_PREDS)

    sensor = _pick_representative_sensor(mc_dropout_preds)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # MC-Dropout
    fset = best_feature_set[bri.TIPO_MC_DROPOUT]
    sub = mc_dropout_preds[(mc_dropout_preds["split"] == "test") & (mc_dropout_preds["horizon"] == "t+1") &
                           (mc_dropout_preds["feature_set"] == fset) & (mc_dropout_preds["sensor"] == sensor)]
    sub = sub.sort_values("date")
    lower = sub["mu_pred"] - Z_90 * sub["total_std"]
    upper = sub["mu_pred"] + Z_90 * sub["total_std"]
    axes[0].fill_between(sub["date"], lower, upper, color=COLOR[bri.TIPO_MC_DROPOUT], alpha=0.25, label="intervallo 90%")
    axes[0].plot(sub["date"], sub["mu_pred"], color=COLOR[bri.TIPO_MC_DROPOUT], linewidth=1.5, label="previsto")
    axes[0].scatter(sub["date"], sub["y_true"], color="black", s=14, zorder=5, label="reale")
    axes[0].set_title(f"MC-Dropout ({fset}) — {sensor}, orizzonte t+1")
    axes[0].legend(loc="upper left", fontsize=8)

    # Deep Ensemble
    fset = best_feature_set[bri.TIPO_ENSEMBLE]
    sub = ensemble_preds[(ensemble_preds["split"] == "test") & (ensemble_preds["horizon"] == "t+1") &
                         (ensemble_preds["feature_set"] == fset) & (ensemble_preds["sensor"] == sensor)]
    sub = sub.sort_values("date")
    lower = sub["mu_pred"] - Z_90 * sub["total_std"]
    upper = sub["mu_pred"] + Z_90 * sub["total_std"]
    axes[1].fill_between(sub["date"], lower, upper, color=COLOR[bri.TIPO_ENSEMBLE], alpha=0.25, label="intervallo 90%")
    axes[1].plot(sub["date"], sub["mu_pred"], color=COLOR[bri.TIPO_ENSEMBLE], linewidth=1.5, label="previsto")
    axes[1].scatter(sub["date"], sub["y_true"], color="black", s=14, zorder=5, label="reale")
    axes[1].set_title(f"Deep Ensemble ({fset}) — {sensor}, orizzonte t+1")
    axes[1].legend(loc="upper left", fontsize=8)

    # Quantile Regression
    fset = best_feature_set[bri.TIPO_QUANTILE]
    sub = quantile_preds[(quantile_preds["split"] == "test") & (quantile_preds["horizon"] == "t+1") &
                         (quantile_preds["feature_set"] == fset) & (quantile_preds["sensor"] == sensor)]
    sub = sub.sort_values("date")
    axes[2].fill_between(sub["date"], sub["q05"], sub["q95"], color=COLOR[bri.TIPO_QUANTILE], alpha=0.25, label="intervallo 90% (Q05-Q95)")
    axes[2].plot(sub["date"], sub["q50"], color=COLOR[bri.TIPO_QUANTILE], linewidth=1.5, label="mediana prevista")
    axes[2].scatter(sub["date"], sub["y_true"], color="black", s=14, zorder=5, label="reale")
    axes[2].set_title(f"Quantile Regression ({fset}) — {sensor}, orizzonte t+1")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].tick_params(axis="x", rotation=45)

    for ax in axes:
        ax.set_ylabel("Umidità (%)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_intervallo_temporale.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 06 — ranking completo per MAE
# ═══════════════════════════════════════════════════════════════════════════

def plot_ranking_mae(combined):
    all_rows = combined[combined["horizon"] == bri.ALL_KEY].copy()
    all_rows = all_rows.sort_values("MAE").reset_index(drop=True)
    labels = [f"{r['tipo']} — {r['feature_set']}" for _, r in all_rows.iterrows()]
    colors = [COLOR[r["tipo"]] for _, r in all_rows.iterrows()]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(all_rows))))
    y_pos = np.arange(len(all_rows))[::-1]
    ax.barh(y_pos, all_rows["MAE"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("MAE complessivo (7 orizzonti, test)")
    ax.set_title("Ranking completo per MAE — tutte le combinazioni (componibile)")

    from matplotlib.patches import Patch
    handles = [Patch(color=COLOR[t], label=t) for t in bri.TIPO_ORDER]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
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

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - width / 2, evento_values, width, label="giorni con evento (pioggia/irrigazione)", color="#1D4ED8")
    ax.bar(x + width / 2, non_evento_values, width, label="giorni senza evento", color="#9CA3AF")

    if nominal_line is not None:
        ax.axhline(nominal_line, color="black", linestyle="--", linewidth=1, label="livello nominale (90%)")

    ax.set_xticks(x)
    ax.set_xticklabels(tipi)
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
        title="Coverage al 90%: giorni con evento vs senza (Complessivo 7gg, test)",
        out_name="07_coverage_eventi_vs_non_eventi.png",
        nominal_line=0.90,
    )


def plot_event_vs_nonevent_sharpness(event_comparison):
    _event_vs_nonevent_bars(
        event_comparison, "sharpness_90_evento", "sharpness_90_non_evento",
        ylabel="Sharpness (ampiezza media intervallo al 90%)",
        title="Sharpness al 90%: giorni con evento vs senza (Complessivo 7gg, test)",
        out_name="08_sharpness_eventi_vs_non_eventi.png",
    )


def main():
    combined = bri.load_combined_metrics()
    best_feature_set = bri.pick_best_feature_set_per_tipo(combined)
    print("Migliori feature-set per famiglia:", best_feature_set)

    print("Calcolo confronto eventi vs non-eventi...")
    event_comparison = bri.compute_event_vs_nonevent(best_feature_set)

    plot_mae_per_horizon(combined, best_feature_set)
    plot_coverage_90_per_horizon(combined, best_feature_set)
    plot_calibration_curve(combined, best_feature_set)
    plot_epistemic_vs_aleatoric(combined, best_feature_set)
    plot_time_series_with_band(best_feature_set)
    plot_ranking_mae(combined)
    plot_event_vs_nonevent_coverage(event_comparison)
    plot_event_vs_nonevent_sharpness(event_comparison)

    print(f"\nGrafici salvati in: {OUT_DIR}")


if __name__ == "__main__":
    main()