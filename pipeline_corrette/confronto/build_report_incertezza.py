"""
build_report_incertezza_componibile.py
─────────────────────────────────────────────────────────────────────────────
Confronto dei 3 metodi di incertezza adattati al sistema COMPONIBILE
(feature-set componibili + irrigazione ricostruita + split canonico
condiviso): MC-Dropout (uncertainty_componibile/), Deep Ensemble
(deep_ensemble_componibile/), Quantile Regression
(quantile_regression_componibile/). Versione componibile di
confronto/build_report_incertezza.py, stessa logica, riusata qui in sola
lettura per le parti invariate.

Tutti e 3 condividono lo STESSO split canonico (70/15/15 cronologico per
sensore, calcolato una sola volta in ai_feature_componibile/
dati_ricostruiti_util.py) — verificato via smoke test: n_train=342/val=72/
test=76 identico su tutte e 3. Nessun problema di allineamento da risolvere
qui.

Le 3 famiglie NON condividono tutte le metriche, per lo stesso motivo
dell'originale (MC-Dropout/Ensemble assumono una gaussiana e scompongono
l'incertezza in epistemica+aleatoria; Quantile Regression non assume nulla).
Le metriche comuni restano MAE/RMSE/R² e coverage/sharpness al 50%/90%.

Novità rispetto all'originale (richieste esplicitamente):
  - foglio "Guida coverage e sharpness": spiega COSA sono, COME si calcolano
    (formule riprese da uncertainty_ml/eval_uncertainty.py e
    quantile_regression_ml/eval_quantile.py, importate in sola lettura) e
    COME si interpretano, più consigli pratici su come intervenire con gli
    iperparametri.
  - foglio "Eventi vs non-eventi": le stesse metriche comuni, per orizzonte
    (t+1..t+7 + Complessivo), calcolate separatamente sui giorni con evento
    (pioggia o irrigazione, colonna has_water_event nei *_preds.csv) e sui
    giorni senza — per rispondere a "l'incertezza cambia nei periodi con
    eventi?". Richiede le colonne has_rain/has_irr/has_water_event nei file
    *_preds.csv (aggiunte ai 3 script run_*_componibile.py insieme a questo
    report).

Output: confronto/confronto_incertezza.xlsx
  - "Riepilogo"                  — nota metodologica + il migliore di
                                    ciascuna famiglia + confronto sulle
                                    metriche comuni
  - "Guida coverage e sharpness" — spiegazione testuale, cosa sono/come si
                                    calcolano/come si interpretano
  - "Per orizzonte"              — per i 3 migliori, dettaglio t+1..t+7 +
                                    Complessivo
  - "Eventi vs non-eventi"       — stesse metriche comuni, separate per
                                    giorni con/senza evento acqua
  - "Ranking completo"           — tutte le combinazioni famiglia x
                                    feature-set
  - "Dati completi"              — tabella lunga di riferimento
"""

# ─────────────────────────────────────────────────────────────────────────────
# Versione per pipeline_corrette/: legge i risultati prodotti dalle pipeline di
# QUESTA cartella (pioggia ed ET0 previsti aggregati correttamente), non da
# ai_feature_componibile/. Nessun import fuori da pipeline_corrette/.
# Nota: penman_baseline/results_osservato/ e' un ORACOLO (pioggia realmente
# caduta) e non va confrontato alla pari con le pipeline previsionali.
# ─────────────────────────────────────────────────────────────────────────────


import os
import sys

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter
from sklearn.metrics import mean_absolute_error

HERE = os.path.dirname(__file__)
PIPE = os.path.join(HERE, "..")            # pipeline_corrette/
CORE = os.path.join(PIPE, "core")

# feature_groups.py, eval_uncertainty.py ed eval_quantile.py stanno in
# pipeline_corrette/core/: CORE va aggiunto al path PRIMA di importarli.
sys.path.insert(0, CORE)

from feature_groups import THEMATIC_FEATURE_SETS  # noqa: E402
import eval_uncertainty as eu   # noqa: E402  (import in sola lettura, invariato)
import eval_quantile as eq      # noqa: E402  (import in sola lettura, invariato)

# Sorgenti a GRIGLIA (results/ e results_residual/), sia diretta che residuo.
# Le results_residual/ erano state lasciate fuori: i loro risultati esistevano
# ma non comparivano in nessun report. La versione Optuna le includeva gia',
# quindi qui si allinea la griglia alla stessa struttura a 6 famiglie.
MC_DROPOUT_METRICS = os.path.join(PIPE, "mc_dropout", "results", "uncertainty_metrics.csv")
ENSEMBLE_METRICS = os.path.join(PIPE, "deep_ensemble", "results", "ensemble_metrics.csv")
QUANTILE_METRICS = os.path.join(PIPE, "quantile_regression", "results", "quantile_metrics.csv")

MC_DROPOUT_PREDS = os.path.join(PIPE, "mc_dropout", "results", "uncertainty_preds.csv")
ENSEMBLE_PREDS = os.path.join(PIPE, "deep_ensemble", "results", "ensemble_preds.csv")
QUANTILE_PREDS = os.path.join(PIPE, "quantile_regression", "results", "quantile_preds.csv")

MC_DROPOUT_RES_METRICS = os.path.join(PIPE, "mc_dropout", "results_residual", "uncertainty_residual_metrics.csv")
ENSEMBLE_RES_METRICS = os.path.join(PIPE, "deep_ensemble", "results_residual", "ensemble_residual_metrics.csv")
QUANTILE_RES_METRICS = os.path.join(PIPE, "quantile_regression", "results_residual", "quantile_residual_metrics.csv")

MC_DROPOUT_RES_PREDS = os.path.join(PIPE, "mc_dropout", "results_residual", "uncertainty_residual_preds.csv")
ENSEMBLE_RES_PREDS = os.path.join(PIPE, "deep_ensemble", "results_residual", "ensemble_residual_preds.csv")
QUANTILE_RES_PREDS = os.path.join(PIPE, "quantile_regression", "results_residual", "quantile_residual_preds.csv")

OUT_PATH = os.path.join(HERE, "confronto_incertezza.xlsx")

FSET_ORDER = list(THEMATIC_FEATURE_SETS.keys())   # ACQUA, TERRA, METEO, TERRA_ACQUA, TERRA_METEO, ACQUA_METEO, TOTALE
HORIZONS = [f"t+{h}" for h in range(1, 8)]
ALL_KEY = "ALL"                        # valore usato nei CSV grezzi
ALL_LABEL = "Complessivo (7gg)"        # etichetta mostrata nel report

# Stessa convenzione di build_report_incertezza_optuna.py: 6 famiglie,
# raggruppate per metodo, ognuna nella variante diretta e residuo.
METODO_MC = "MC-Dropout"
METODO_ENSEMBLE = "Deep Ensemble"
METODO_QUANTILE = "Quantile Regression"
METODI = [METODO_MC, METODO_ENSEMBLE, METODO_QUANTILE]
APPROCCIO_DIRETTA = "diretta"
APPROCCIO_RESIDUO = "residuo"
APPROCCI = [APPROCCIO_DIRETTA, APPROCCIO_RESIDUO]


def tipo_label(metodo, approccio):
    return f"{metodo} ({approccio})"


TIPO_MC_DROPOUT = tipo_label(METODO_MC, APPROCCIO_DIRETTA)
TIPO_ENSEMBLE = tipo_label(METODO_ENSEMBLE, APPROCCIO_DIRETTA)
TIPO_QUANTILE = tipo_label(METODO_QUANTILE, APPROCCIO_DIRETTA)
TIPO_MC_DROPOUT_RES = tipo_label(METODO_MC, APPROCCIO_RESIDUO)
TIPO_ENSEMBLE_RES = tipo_label(METODO_ENSEMBLE, APPROCCIO_RESIDUO)
TIPO_QUANTILE_RES = tipo_label(METODO_QUANTILE, APPROCCIO_RESIDUO)
TIPO_ORDER = [tipo_label(m, a) for m in METODI for a in APPROCCI]

QUANTILE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
QUANTILE_COLS = ["q05", "q25", "q50", "q75", "q95"]

# Metriche direttamente confrontabili tra tutte e 3 le famiglie.
COMMON_METRIC_COLS = ["MAE", "RMSE", "R2", "coverage_50", "sharpness_50",
                      "coverage_90", "sharpness_90"]


# ═══════════════════════════════════════════════════════════════════════════
# Caricamento metriche aggregate (uguale all'originale, sorgenti componibili)
# ═══════════════════════════════════════════════════════════════════════════

def load_combined_metrics():
    """Le 6 famiglie a griglia. MC-Dropout e Deep Ensemble hanno la NLL come
    punteggio primario, la Quantile Regression la pinball loss: sono scale
    diverse, quindi il confronto fra famiglie si fa sulle metriche comuni,
    non sul punteggio primario."""
    sorgenti = [
        (MC_DROPOUT_METRICS, TIPO_MC_DROPOUT, "NLL"),
        (MC_DROPOUT_RES_METRICS, TIPO_MC_DROPOUT_RES, "NLL"),
        (ENSEMBLE_METRICS, TIPO_ENSEMBLE, "NLL"),
        (ENSEMBLE_RES_METRICS, TIPO_ENSEMBLE_RES, "NLL"),
        (QUANTILE_METRICS, TIPO_QUANTILE, "pinball_loss"),
        (QUANTILE_RES_METRICS, TIPO_QUANTILE_RES, "pinball_loss"),
    ]
    pezzi = []
    for percorso, tipo, punteggio in sorgenti:
        if not os.path.exists(percorso):
            print(f"  ATTENZIONE: manca {percorso} — famiglia '{tipo}' esclusa dal report")
            continue
        d = pd.read_csv(percorso)
        d = d[d["split"] == "test"].copy()
        d["tipo"] = tipo
        d["primary_score"] = d[punteggio] if punteggio in d.columns else np.nan
        d["primary_score_name"] = punteggio
        pezzi.append(d)
        print(f"  {tipo:34s} {len(d):4d} righe di test da {os.path.relpath(percorso, PIPE)}")
    if not pezzi:
        raise RuntimeError("Nessuna sorgente di metriche trovata.")
    return pd.concat(pezzi, ignore_index=True, sort=False)


def pick_best_feature_set_per_tipo(combined):
    """Per ciascuna famiglia, il feature-set con il punteggio primario di
    quella famiglia più basso sulla riga aggregata (tutti e 7 gli orizzonti
    insieme)."""
    best_feature_set = {}
    all_rows = combined[combined["horizon"] == ALL_KEY]
    for tipo, group in all_rows.groupby("tipo"):
        best_row = group.sort_values("primary_score").iloc[0]
        best_feature_set[tipo] = best_row["feature_set"]
    return best_feature_set


# ═══════════════════════════════════════════════════════════════════════════
# Confronto evento/non-evento: richiede le predizioni riga per riga (non le
# metriche aggregate già calcolate), filtrate per has_water_event, con le
# stesse funzioni di eval_uncertainty/eval_quantile usate in produzione.
# ═══════════════════════════════════════════════════════════════════════════

def _metrics_for_gaussian_subset(subset):
    """subset: righe di uncertainty_preds.csv/ensemble_preds.csv già filtrate
    (per split/feature_set/orizzonte/evento). Ritorna un dict con n, MAE,
    coverage_90, sharpness_90 — None per i valori non calcolabili (n<2)."""
    n = len(subset)
    if n == 0:
        return {"n": 0, "MAE": np.nan, "coverage_90": np.nan, "sharpness_90": np.nan}
    mae = mean_absolute_error(subset["y_true"], subset["mu_pred"])
    coverage_90 = eu.coverage_at_level(subset["y_true"], subset["mu_pred"], subset["total_std"], 0.90)
    sharpness_90 = eu.sharpness_at_level(subset["total_std"], 0.90)
    return {"n": n, "MAE": round(float(mae), 4),
            "coverage_90": round(float(coverage_90), 4),
            "sharpness_90": round(float(sharpness_90), 4)}


def _metrics_for_quantile_subset(subset):
    """Stesso ruolo di _metrics_for_gaussian_subset ma per quantile_preds.csv
    (colonne q05..q95 invece di mu_pred/total_std)."""
    n = len(subset)
    if n == 0:
        return {"n": 0, "MAE": np.nan, "coverage_90": np.nan, "sharpness_90": np.nan}
    quantile_predictions = subset[QUANTILE_COLS].values.astype(float)
    mae = mean_absolute_error(subset["y_true"], subset["q50"])
    coverage_90 = eq.coverage_from_quantiles(subset["y_true"].values, quantile_predictions, QUANTILE_LEVELS, 0.05, 0.95)
    sharpness_90 = eq.sharpness_from_quantiles(quantile_predictions, QUANTILE_LEVELS, 0.05, 0.95)
    return {"n": n, "MAE": round(float(mae), 4),
            "coverage_90": round(float(coverage_90), 4),
            "sharpness_90": round(float(sharpness_90), 4)}


def compute_event_vs_nonevent(best_feature_set):
    """Per ciascuna famiglia (sul suo feature-set migliore, split test), per
    ciascun orizzonte t+1..t+7 + Complessivo: le metriche comuni calcolate
    separatamente sui giorni con evento acqua (pioggia o irrigazione) e sui
    giorni senza. Ritorna una lista di righe piatte, una per
    (tipo, orizzonte)."""
    sources = {
        TIPO_MC_DROPOUT: (MC_DROPOUT_PREDS, _metrics_for_gaussian_subset),
        TIPO_MC_DROPOUT_RES: (MC_DROPOUT_RES_PREDS, _metrics_for_gaussian_subset),
        TIPO_ENSEMBLE: (ENSEMBLE_PREDS, _metrics_for_gaussian_subset),
        TIPO_ENSEMBLE_RES: (ENSEMBLE_RES_PREDS, _metrics_for_gaussian_subset),
        TIPO_QUANTILE: (QUANTILE_PREDS, _metrics_for_quantile_subset),
        TIPO_QUANTILE_RES: (QUANTILE_RES_PREDS, _metrics_for_quantile_subset),
    }

    rows = []
    for tipo in TIPO_ORDER:
        fset = best_feature_set.get(tipo)
        if fset is None:
            continue
        preds_path, metrics_fn = sources[tipo]
        preds = pd.read_csv(preds_path)
        preds = preds[(preds["split"] == "test") & (preds["feature_set"] == fset)]

        for horizon_key, horizon_display in list(zip(HORIZONS, HORIZONS)) + [(ALL_KEY, ALL_LABEL)]:
            horizon_subset = preds if horizon_key == ALL_KEY else preds[preds["horizon"] == horizon_key]

            event_metrics = metrics_fn(horizon_subset[horizon_subset["has_water_event"]])
            noevent_metrics = metrics_fn(horizon_subset[~horizon_subset["has_water_event"]])

            rows.append({
                "tipo": tipo, "feature_set": fset,
                "orizzonte": horizon_display, "_horizon_key": horizon_key,
                "n_evento": event_metrics["n"], "MAE_evento": event_metrics["MAE"],
                "coverage_90_evento": event_metrics["coverage_90"], "sharpness_90_evento": event_metrics["sharpness_90"],
                "n_non_evento": noevent_metrics["n"], "MAE_non_evento": noevent_metrics["MAE"],
                "coverage_90_non_evento": noevent_metrics["coverage_90"], "sharpness_90_non_evento": noevent_metrics["sharpness_90"],
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Stile (stessa palette dell'originale, per coerenza visiva col resto del
# progetto)
# ═══════════════════════════════════════════════════════════════════════════

HEADER_FILL = PatternFill("solid", fgColor="2F5233")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TITLE_FONT = Font(bold=True, size=13)
# Stessa palette della versione Optuna: tinta base per metodo, sfumatura piu'
# scura per la variante residuo.
TIPO_FILL = {
    TIPO_MC_DROPOUT: PatternFill("solid", fgColor="C9DAF8"),
    TIPO_MC_DROPOUT_RES: PatternFill("solid", fgColor="9FC5E8"),
    TIPO_ENSEMBLE: PatternFill("solid", fgColor="D9EAD3"),
    TIPO_ENSEMBLE_RES: PatternFill("solid", fgColor="B6D7A8"),
    TIPO_QUANTILE: PatternFill("solid", fgColor="FCE5CD"),
    TIPO_QUANTILE_RES: PatternFill("solid", fgColor="F9CB9C"),
}
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _style_header_row(ws, row_idx, n_cols):
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row_idx, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center")


# ═══════════════════════════════════════════════════════════════════════════
# Sheet "Riepilogo"
# ═══════════════════════════════════════════════════════════════════════════

def write_summary_sheet(wb, combined, best_feature_set):
    ws = wb.create_sheet("Riepilogo", 0)
    ws.append(["Confronto metodi di incertezza COMPONIBILI: MC-Dropout vs Deep Ensemble vs Quantile Regression"])
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([])

    ws.cell(row=ws.max_row + 1, column=1,
            value="Nota metodologica: le tre famiglie condividono lo STESSO split canonico "
                  "(calcolato una volta sola in ai_feature_componibile/dati_ricostruiti_util.py, "
                  "verificato identico su tutte e 3 via smoke test). Non condividono però lo "
                  "stesso modo di intendere l'incertezza: MC-Dropout e Deep Ensemble assumono "
                  "una gaussiana e la scompongono in epistemica+aleatoria (colonna primaria: "
                  "NLL); Quantile Regression non assume nulla e non scompone (colonna primaria: "
                  "pinball loss). Le due colonne NON sono confrontabili in valore assoluto tra "
                  "loro — sono mostrate separatamente, mai fuse in un unico punteggio.").font = Font(italic=True)
    ws.cell(row=ws.max_row + 1, column=1,
            value="Le metriche davvero confrontabili tra tutte e 3 sono MAE/RMSE/R² "
                  "(qualità della previsione puntuale) e coverage/sharpness al 50% e al 90% — "
                  "vedi il foglio \"Guida coverage e sharpness\" per il significato.").font = Font(italic=True)
    ws.append([])

    # ── Il migliore di ciascuna famiglia ────────────────────────────────────
    ws.cell(row=ws.max_row + 1, column=1, value="Il migliore di ciascuna famiglia (per punteggio primario, test set)").font = Font(bold=True, size=12)
    headers = ["Famiglia", "Feature-set migliore", "Punteggio primario", "Valore", "MAE", "coverage_90", "sharpness_90"]
    ws.append(headers)
    _style_header_row(ws, ws.max_row, len(headers))

    all_rows = combined[combined["horizon"] == ALL_KEY]
    for tipo in TIPO_ORDER:
        fset = best_feature_set.get(tipo)
        if fset is None:
            continue
        row = all_rows[(all_rows["tipo"] == tipo) & (all_rows["feature_set"] == fset)].iloc[0]
        ws.append([tipo, fset, row["primary_score_name"], round(float(row["primary_score"]), 4),
                  round(float(row["MAE"]), 4), round(float(row["coverage_90"]), 4),
                  round(float(row["sharpness_90"]), 4)])
        rr = ws.max_row
        for c in range(1, len(headers) + 1):
            cell = ws.cell(row=rr, column=c)
            cell.border = BORDER
            cell.fill = TIPO_FILL.get(tipo)

    ws.append([])
    ws.append([])

    # ── Tabella di confronto sulle metriche comuni ──────────────────────────
    ws.cell(row=ws.max_row + 1, column=1,
           value="Confronto sulle metriche comuni (i 3 migliori, Complessivo 7gg)").font = Font(bold=True, size=12)
    headers2 = ["Famiglia", "Feature-set"] + COMMON_METRIC_COLS
    ws.append(headers2)
    _style_header_row(ws, ws.max_row, len(headers2))

    for tipo in TIPO_ORDER:
        fset = best_feature_set.get(tipo)
        if fset is None:
            continue
        row = all_rows[(all_rows["tipo"] == tipo) & (all_rows["feature_set"] == fset)].iloc[0]
        values = [tipo, fset] + [round(float(row[c]), 4) if pd.notna(row[c]) else None for c in COMMON_METRIC_COLS]
        ws.append(values)
        rr = ws.max_row
        for c in range(1, len(headers2) + 1):
            cell = ws.cell(row=rr, column=c)
            cell.border = BORDER
            cell.fill = TIPO_FILL.get(tipo)

    widths = [20, 26, 16, 12, 10, 12, 12]
    for i, w in enumerate(widths):
        ws.column_dimensions[get_column_letter(i + 1)].width = w
    return ws


# ═══════════════════════════════════════════════════════════════════════════
# Sheet "Guida coverage e sharpness" — documentazione testuale
# ═══════════════════════════════════════════════════════════════════════════

def write_guida_sheet(wb):
    ws = wb.create_sheet("Guida coverage e sharpness")
    ws.column_dimensions["A"].width = 110

    def title(text, size=13):
        ws.cell(row=ws.max_row + 1, column=1, value=text).font = Font(bold=True, size=size)

    def body(text):
        cell = ws.cell(row=ws.max_row + 1, column=1, value=text)
        cell.alignment = Alignment(wrap_text=True, vertical="top")

    def blank():
        ws.append([])

    title("Cosa sono coverage e sharpness", 14)
    blank()
    body("COVERAGE: la percentuale di giorni in cui il valore vero è caduto DENTRO l'intervallo "
         "di previsione dichiarato dal modello. coverage_90 = frazione di volte in cui il vero è "
         "dentro l'intervallo al 90%. Se il modello è ben calibrato, coverage_90 dovrebbe essere "
         "vicino a 0.90 (né sistematicamente più basso né più alto).")
    blank()
    body("  • coverage_90 MOLTO più bassa di 0.90 (es. 0.75)  →  il modello è OVERCONFIDENT: "
         "dichiara intervalli troppo STRETTI rispetto a quanto in realtà sbaglia. È il caso più "
         "pericoloso: l'utente si fida di un margine di errore che nella pratica è più ampio.")
    body("  • coverage_90 MOLTO più alta di 0.90 (es. 0.99)  →  il modello è troppo PRUDENTE: "
         "dichiara intervalli più larghi del necessario. Meno pericoloso, ma meno utile (un "
         "intervallo enorme è sempre \"corretto\" ma non dice niente).")
    blank()
    body("SHARPNESS: l'ampiezza MEDIA dell'intervallo di previsione (per sharpness_90, l'ampiezza "
         "dell'intervallo al 90%). Più piccola è, più il modello è \"nitido\" — ma la sharpness va "
         "SEMPRE guardata insieme alla coverage: un modello che dichiara sempre un intervallo "
         "larghissimo ha ottima coverage ma è inutile in pratica (sharpness pessima); un modello "
         "con sharpness bassissima ma coverage scarsa sta solo mentendo su quanto è sicuro.")
    blank()
    blank()

    title("Come si calcolano")
    blank()
    body("MC-Dropout e Deep Ensemble (uncertainty_componibile/, deep_ensemble_componibile/) "
         "assumono che l'errore segua una gaussiana N(mu, sigma^2), dove mu è la previsione "
         "puntuale e sigma (\"total_std\") è la deviazione standard totale dichiarata "
         "(epistemica + aleatoria). L'intervallo al 90% è [mu - z·sigma, mu + z·sigma] con "
         "z≈1.645 (il quantile normale per cui il 90% della gaussiana sta in quel range). "
         "coverage_90 = quante volte il vero cade in quell'intervallo; sharpness_90 = "
         "2·z·sigma mediato su tutte le righe. Formule esatte in "
         "uncertainty_ml/eval_uncertainty.py: coverage_at_level() e sharpness_at_level().")
    blank()
    body("Quantile Regression (quantile_regression_componibile/) non assume nessuna forma: "
         "l'intervallo al 90% è preso DIRETTAMENTE dai percentili che il modello predice, "
         "[Q05, Q95] (Q05 e Q95 sono due delle 5 uscite del modello, non calcolate da una "
         "formula). coverage_90 = quante volte il vero cade tra Q05 e Q95; sharpness_90 = "
         "Q95 - Q05 mediato su tutte le righe. Formule esatte in "
         "quantile_regression_ml/eval_quantile.py: coverage_from_quantiles() e "
         "sharpness_from_quantiles().")
    blank()
    blank()

    title("Come si interpretano insieme (non una sola delle due)")
    blank()
    body("Il confronto onesto tra due modelli guarda SEMPRE la coppia coverage+sharpness "
         "insieme, mai la sharpness da sola: un modello A con sharpness più piccola di un "
         "modello B è \"migliore\" SOLO SE la sua coverage è altrettanto vicina (o più vicina) "
         "al livello nominale. Se A ha sharpness più piccola ma coverage molto sotto 0.90, A "
         "non è migliore: sta solo dichiarando meno incertezza di quanta ne ha davvero.")
    blank()
    blank()

    title("Come abbassare l'incertezza provando nuovi iperparametri")
    blank()
    body("Le leve concrete sono nei GRID_MLP_* dei 3 script run_*_componibile.py (hidden, lr, "
         "dropout) e nei parametri fissi NN_FIXED_* (epochs, patience, batch_size).")
    blank()
    body("  • Sharpness alta ma coverage corretta (intervalli larghi ma ben calibrati): provare "
         "a ridurre il dropout e/o aumentare la capacità (hidden più grande, es. da (64,32) a "
         "(128,64,32)) — se il modello non sta ancora sfruttando tutto il segnale disponibile "
         "nelle feature, un po' più di capacità riduce l'incertezza aleatoria residua.")
    body("  • Coverage troppo bassa (overconfident): NON si risolve abbassando l'incertezza "
         "dichiarata, ma aumentandola finché la calibrazione non torna vicina al nominale — es. "
         "aumentare il dropout per MC-Dropout (più rumore in predizione = più incertezza "
         "epistemica dichiarata) o aumentare il numero di membri per Deep Ensemble.")
    body("  • Aleatoria alta e persistente su tutte le combinazioni di iperparametri provate: "
         "spesso è rumore reale nei dati o un feature-set poco informativo per quel target, più "
         "che un problema risolvibile con gli iperparametri — confrontare lo stesso modello su "
         "feature-set diversi (es. ACQUA vs TERRA) per capire se il problema è nelle feature "
         "prima di continuare a cercare iperparametri migliori.")

    return ws


# ═══════════════════════════════════════════════════════════════════════════
# Sheet "Per orizzonte" — dettaglio t+1..t+7 + Complessivo, per i 3 migliori
# ═══════════════════════════════════════════════════════════════════════════

def write_by_horizon_sheet(wb, combined, best_feature_set):
    ws = wb.create_sheet("Per orizzonte")

    detail_cols = ["MAE", "coverage_50", "sharpness_50", "coverage_90", "sharpness_90",
                  "NLL", "pinball_loss", "mean_epistemic_std", "mean_aleatoric_std"]
    col_headers = ["Orizzonte"] + detail_cols
    n_cols = len(col_headers)

    row_cursor = 1
    for tipo in TIPO_ORDER:
        fset = best_feature_set.get(tipo)
        if fset is None:
            continue

        ws.cell(row=row_cursor, column=1,
               value=f"{tipo} — feature-set: {fset}").font = TITLE_FONT
        row_cursor += 1

        hdr_row = row_cursor
        for i, h in enumerate(col_headers, start=1):
            ws.cell(row=hdr_row, column=i, value=h)
        _style_header_row(ws, hdr_row, n_cols)
        row_cursor += 1

        subset = combined[(combined["tipo"] == tipo) & (combined["feature_set"] == fset)]
        data_start = row_cursor
        for horizon_key, horizon_display in list(zip(HORIZONS, HORIZONS)) + [(ALL_KEY, ALL_LABEL)]:
            row_data = subset[subset["horizon"] == horizon_key]
            if row_data.empty:
                continue
            row = row_data.iloc[0]
            values = [horizon_display] + [
                round(float(row[c]), 4) if c in row.index and pd.notna(row[c]) else None
                for c in detail_cols
            ]
            ws.append(values)
            rr = ws.max_row
            for c in range(1, n_cols + 1):
                cell = ws.cell(row=rr, column=c)
                cell.border = BORDER
                if c == 1:
                    cell.fill = TIPO_FILL.get(tipo)
            if horizon_key == ALL_KEY:
                for c in range(1, n_cols + 1):
                    ws.cell(row=rr, column=c).font = Font(bold=True)
            row_cursor += 1
        data_end = row_cursor - 1

        if data_end >= data_start:
            mae_col_letter = get_column_letter(2)
            rng = f"{mae_col_letter}{data_start}:{mae_col_letter}{data_end}"
            rule = ColorScaleRule(start_type="min", start_color="63BE7B",
                                  mid_type="percentile", mid_value=50, mid_color="FFEB84",
                                  end_type="max", end_color="F8696B")
            ws.conditional_formatting.add(rng, rule)

        row_cursor += 1

    ws.freeze_panes = "B2"
    ws.column_dimensions["A"].width = 14
    for i in range(len(detail_cols)):
        ws.column_dimensions[get_column_letter(2 + i)].width = 16
    return ws


# ═══════════════════════════════════════════════════════════════════════════
# Sheet "Eventi vs non-eventi"
# ═══════════════════════════════════════════════════════════════════════════

def write_event_vs_nonevent_sheet(wb, event_comparison):
    ws = wb.create_sheet("Eventi vs non-eventi")
    ws.cell(row=1, column=1,
           value="Le stesse metriche comuni, calcolate separatamente sui giorni con evento acqua "
                 "(pioggia o irrigazione) e sui giorni senza — test split, feature-set migliore di "
                 "ciascuna famiglia.").font = Font(italic=True)
    ws.append([])

    col_headers = ["Famiglia", "Feature-set", "Orizzonte",
                   "n (evento)", "MAE (evento)", "coverage_90 (evento)", "sharpness_90 (evento)",
                   "n (non evento)", "MAE (non evento)", "coverage_90 (non evento)", "sharpness_90 (non evento)"]
    hdr_row = ws.max_row + 1
    for i, h in enumerate(col_headers, start=1):
        ws.cell(row=hdr_row, column=i, value=h)
    _style_header_row(ws, hdr_row, len(col_headers))

    for tipo in TIPO_ORDER:
        subset = event_comparison[event_comparison["tipo"] == tipo]
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            values = [
                tipo, row["feature_set"], row["orizzonte"],
                row["n_evento"], row["MAE_evento"], row["coverage_90_evento"], row["sharpness_90_evento"],
                row["n_non_evento"], row["MAE_non_evento"], row["coverage_90_non_evento"], row["sharpness_90_non_evento"],
            ]
            ws.append(values)
            rr = ws.max_row
            for c in range(1, len(col_headers) + 1):
                cell = ws.cell(row=rr, column=c)
                cell.border = BORDER
                cell.fill = TIPO_FILL.get(tipo)
            if row["_horizon_key"] == ALL_KEY:
                for c in range(1, len(col_headers) + 1):
                    ws.cell(row=rr, column=c).font = Font(bold=True)

    widths = [18, 14, 14, 11, 13, 18, 18, 14, 15, 20, 20]
    for i, w in enumerate(widths):
        ws.column_dimensions[get_column_letter(i + 1)].width = w
    ws.freeze_panes = "A4"
    return ws


# ═══════════════════════════════════════════════════════════════════════════
# Sheet "Ranking completo" — tutte le combinazioni famiglia x feature-set
# ═══════════════════════════════════════════════════════════════════════════

def write_ranking_sheet(wb, combined):
    ws = wb.create_sheet("Ranking completo")
    headers = ["Famiglia", "Feature-set", "Punteggio primario", "Valore primario"] + COMMON_METRIC_COLS
    ws.append(headers)
    _style_header_row(ws, 1, len(headers))

    all_rows = combined[combined["horizon"] == ALL_KEY].copy()
    all_rows["_tipo_order"] = all_rows["tipo"].map({t: i for i, t in enumerate(TIPO_ORDER)})
    all_rows = all_rows.sort_values(["_tipo_order", "primary_score"])

    for _, row in all_rows.iterrows():
        values = [row["tipo"], row["feature_set"], row["primary_score_name"],
                 round(float(row["primary_score"]), 4)]
        values += [round(float(row[c]), 4) if pd.notna(row[c]) else None for c in COMMON_METRIC_COLS]
        ws.append(values)
        rr = ws.max_row
        for c in range(1, len(headers) + 1):
            cell = ws.cell(row=rr, column=c)
            cell.border = BORDER
            if c <= 2:
                cell.fill = TIPO_FILL.get(row["tipo"])

    widths = [20, 14, 16, 14, 10, 10, 8, 12, 12, 12, 12]
    for i, w in enumerate(widths):
        ws.column_dimensions[get_column_letter(i + 1)].width = w
    ws.freeze_panes = "A2"
    return ws


def write_full_data_sheet(wb, combined):
    ws = wb.create_sheet("Dati completi")
    cols = [c for c in combined.columns if c not in ("primary_score", "primary_score_name")]
    ws.append(cols)
    _style_header_row(ws, 1, len(cols))
    for _, row in combined.iterrows():
        ws.append([row[c] if pd.notna(row[c]) else None for c in cols])
    ws.freeze_panes = "A2"
    for i, c in enumerate(cols):
        ws.column_dimensions[get_column_letter(i + 1)].width = max(10, min(28, len(c) + 4))
    return ws


def main():
    combined = load_combined_metrics()
    best_feature_set = pick_best_feature_set_per_tipo(combined)
    print("Migliori feature-set per famiglia:", best_feature_set)

    print("Calcolo confronto eventi vs non-eventi (righe di *_preds.csv filtrate per has_water_event)...")
    event_comparison = compute_event_vs_nonevent(best_feature_set)

    wb = Workbook()
    wb.remove(wb.active)
    write_summary_sheet(wb, combined, best_feature_set)
    write_guida_sheet(wb)
    write_by_horizon_sheet(wb, combined, best_feature_set)
    write_event_vs_nonevent_sheet(wb, event_comparison)
    write_ranking_sheet(wb, combined)
    write_full_data_sheet(wb, combined)

    wb.save(OUT_PATH)
    print(f"Salvato: {OUT_PATH}")
    print(f"Righe totali combinate: {len(combined)}")


if __name__ == "__main__":
    main()