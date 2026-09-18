"""
build_report_incertezza_componibile_unificato.py
─────────────────────────────────────────────────────────────────────────────
Sostituisce build_report_incertezza_componibile_optuna.py (diretta) e
build_report_incertezza_residual_componibile_optuna.py (residuo, target −
penman_pred, poi ricostruito — vedi confronto/report_incertezza_dettagliato.md
§3-4 per la dimostrazione di comparabilità) con UN report unico. Copre SOLO
le pipeline Optuna (le versioni a griglia esaustiva restano in
build_report_incertezza_componibile.py, invariato, non toccato da questo
script).

I metodi di incertezza, ciascuno nella variante diretta E residuo, diventano
"famiglie" trattate in modo uniforme in tutto il report: un unico dataframe
combinato con un valore di "tipo" per famiglia. Il foglio "Diretta vs
Residuo" resta un confronto mirato allo STESSO feature-set (quello scelto
come migliore dalla variante residuo di ciascun metodo), per isolare la
domanda "partire dalla fisica Penman riduce davvero l'incertezza residua, o
no?" dal rumore di un feature-set diverso.

SETTE METODI, QUATTORDICI FAMIGLIE
Ai tre storici (MC-Dropout, Deep Ensemble, Quantile Regression) se ne sono
aggiunti quattro: Bayesian VI, Laplace, e le versioni con warm-up della media
di MC-Dropout e Deep Ensemble. I quattro nuovi sono gaussiani come i primi
due, quindi passano per le stesse funzioni di metrica senza casi speciali;
solo la Quantile Regression lavora su percentili.

FAMIGLIE MANCANTI
I run delle pipeline sono lunghi e vengono lanciati un po' alla volta. Le
famiglie che non hanno ancora i CSV vengono SALTATE invece di far fallire il
report, e sono elencate nel foglio "Copertura" insieme al numero di sensori
di quelle presenti (una famiglia con meno sensori delle altre è ancora uno
smoke test e non è confrontabile in scala). Rieseguendo lo script a run
finiti, le famiglie rientrano da sole senza modifiche al codice.

Riuso in sola lettura (nessuna modifica ai file sorgenti):
  - core/feature_groups.py: THEMATIC_FEATURE_SETS.
  - core/eval_uncertainty.py: coverage_at_level, sharpness_at_level.
  - core/eval_quantile.py: coverage_from_quantiles, sharpness_from_quantiles.

Output: confronto/confronto_incertezza_optuna.xlsx
  - "Riepilogo", "Copertura", "Guida coverage e sharpness",
    "Confronto metodi (evento vs non-evento)", "Per orizzonte",
    "Diretta vs Residuo", "Ranking completo", "Dati completi".
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

OUT_PATH = os.path.join(HERE, "confronto_incertezza_optuna.xlsx")

FSET_ORDER = list(THEMATIC_FEATURE_SETS.keys())   # ACQUA, TERRA, METEO, TERRA_ACQUA, TERRA_METEO, ACQUA_METEO, TOTALE
HORIZONS = [f"t+{h}" for h in range(1, 8)]
ALL_KEY = "ALL"                        # valore usato nei CSV grezzi
ALL_LABEL = "Complessivo (7gg)"        # etichetta mostrata nel report

METHOD_MC = "MC-Dropout"
METHOD_ENSEMBLE = "Deep Ensemble"
METHOD_QUANTILE = "Quantile Regression"
METHOD_VI = "Bayesian VI"
METHOD_LAPLACE = "Laplace"
METHOD_MC_WARMUP = "MC-Dropout warm-up"
METHOD_ENSEMBLE_WARMUP = "Deep Ensemble warm-up"

# Ordine: prima i tre metodi storici, poi i quattro aggiunti dopo.
METHODS = [METHOD_MC, METHOD_ENSEMBLE, METHOD_QUANTILE,
           METHOD_VI, METHOD_LAPLACE, METHOD_MC_WARMUP, METHOD_ENSEMBLE_WARMUP]

# I quattro metodi nuovi sono tutti gaussiani: producono mu_pred,
# epistemic_std, aleatoric_std, total_std e NLL con le stesse colonne di
# MC-Dropout, quindi riusano _metrics_for_gaussian_subset senza modifiche.
# Solo la Quantile Regression lavora su percentili e non assume una gaussiana.
METHOD_KIND = {METHOD_MC: "gaussian", METHOD_ENSEMBLE: "gaussian", METHOD_QUANTILE: "quantile",
               METHOD_VI: "gaussian", METHOD_LAPLACE: "gaussian",
               METHOD_MC_WARMUP: "gaussian", METHOD_ENSEMBLE_WARMUP: "gaussian"}

APPROCCIO_DIRETTA = "diretta"
APPROCCIO_RESIDUO = "residuo"
APPROCCI = [APPROCCIO_DIRETTA, APPROCCIO_RESIDUO]


def tipo_label(method, approccio):
    return f"{method} ({approccio})"


# Tutte le famiglie previste (7 metodi x 2 approcci), raggruppate per metodo.
# Più sotto, dopo il controllo sui file, TIPO_ORDER viene ristretto a quelle
# effettivamente disponibili: i run sono lunghi e possono essere parziali.
TIPO_ORDER_COMPLETO = [tipo_label(m, a) for m in METHODS for a in APPROCCI]
TIPO_ORDER = TIPO_ORDER_COMPLETO

QUANTILE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
QUANTILE_COLS = ["q05", "q25", "q50", "q75", "q95"]

# Metriche direttamente confrontabili fra tutte le famiglie.
COMMON_METRIC_COLS = ["MAE", "RMSE", "R2", "coverage_50", "sharpness_50",
                      "coverage_90", "sharpness_90"]

# ═══════════════════════════════════════════════════════════════════════════
# Sorgenti dati (solo Optuna: results_optuna/, sia diretta che residuo)
# ═══════════════════════════════════════════════════════════════════════════

SOURCES = {
    (METHOD_MC, APPROCCIO_DIRETTA): {
        "metrics": os.path.join(PIPE, "mc_dropout", "results_optuna", "uncertainty_metrics.csv"),
        "preds": os.path.join(PIPE, "mc_dropout", "results_optuna", "uncertainty_preds.csv"),
    },
    (METHOD_MC, APPROCCIO_RESIDUO): {
        "metrics": os.path.join(PIPE, "mc_dropout", "results_residual_optuna", "uncertainty_residual_metrics.csv"),
        "preds": os.path.join(PIPE, "mc_dropout", "results_residual_optuna", "uncertainty_residual_preds.csv"),
    },
    (METHOD_ENSEMBLE, APPROCCIO_DIRETTA): {
        "metrics": os.path.join(PIPE, "deep_ensemble", "results_optuna", "ensemble_metrics.csv"),
        "preds": os.path.join(PIPE, "deep_ensemble", "results_optuna", "ensemble_preds.csv"),
    },
    (METHOD_ENSEMBLE, APPROCCIO_RESIDUO): {
        "metrics": os.path.join(PIPE, "deep_ensemble", "results_residual_optuna", "ensemble_residual_metrics.csv"),
        "preds": os.path.join(PIPE, "deep_ensemble", "results_residual_optuna", "ensemble_residual_preds.csv"),
    },
    (METHOD_QUANTILE, APPROCCIO_DIRETTA): {
        "metrics": os.path.join(PIPE, "quantile_regression", "results_optuna", "quantile_metrics.csv"),
        "preds": os.path.join(PIPE, "quantile_regression", "results_optuna", "quantile_preds.csv"),
    },
    (METHOD_QUANTILE, APPROCCIO_RESIDUO): {
        "metrics": os.path.join(PIPE, "quantile_regression", "results_residual_optuna", "quantile_residual_metrics.csv"),
        "preds": os.path.join(PIPE, "quantile_regression", "results_residual_optuna", "quantile_residual_preds.csv"),
    },
    (METHOD_VI, APPROCCIO_DIRETTA): {
        "metrics": os.path.join(PIPE, "bayesian_vi", "results_optuna", "bayesian_vi_metrics.csv"),
        "preds": os.path.join(PIPE, "bayesian_vi", "results_optuna", "bayesian_vi_preds.csv"),
    },
    (METHOD_VI, APPROCCIO_RESIDUO): {
        "metrics": os.path.join(PIPE, "bayesian_vi", "results_residual_optuna", "bayesian_vi_residual_metrics.csv"),
        "preds": os.path.join(PIPE, "bayesian_vi", "results_residual_optuna", "bayesian_vi_residual_preds.csv"),
    },
    (METHOD_LAPLACE, APPROCCIO_DIRETTA): {
        "metrics": os.path.join(PIPE, "laplace", "results_optuna", "laplace_metrics.csv"),
        "preds": os.path.join(PIPE, "laplace", "results_optuna", "laplace_preds.csv"),
    },
    (METHOD_LAPLACE, APPROCCIO_RESIDUO): {
        "metrics": os.path.join(PIPE, "laplace", "results_residual_optuna", "laplace_residual_metrics.csv"),
        "preds": os.path.join(PIPE, "laplace", "results_residual_optuna", "laplace_residual_preds.csv"),
    },
    (METHOD_MC_WARMUP, APPROCCIO_DIRETTA): {
        "metrics": os.path.join(PIPE, "mc_dropout_warmup", "results_optuna", "uncertainty_warmup_metrics.csv"),
        "preds": os.path.join(PIPE, "mc_dropout_warmup", "results_optuna", "uncertainty_warmup_preds.csv"),
    },
    (METHOD_MC_WARMUP, APPROCCIO_RESIDUO): {
        "metrics": os.path.join(PIPE, "mc_dropout_warmup", "results_residual_optuna", "uncertainty_warmup_residual_metrics.csv"),
        "preds": os.path.join(PIPE, "mc_dropout_warmup", "results_residual_optuna", "uncertainty_warmup_residual_preds.csv"),
    },
    (METHOD_ENSEMBLE_WARMUP, APPROCCIO_DIRETTA): {
        "metrics": os.path.join(PIPE, "deep_ensemble_warmup", "results_optuna", "ensemble_warmup_metrics.csv"),
        "preds": os.path.join(PIPE, "deep_ensemble_warmup", "results_optuna", "ensemble_warmup_preds.csv"),
    },
    (METHOD_ENSEMBLE_WARMUP, APPROCCIO_RESIDUO): {
        "metrics": os.path.join(PIPE, "deep_ensemble_warmup", "results_residual_optuna", "ensemble_warmup_residual_metrics.csv"),
        "preds": os.path.join(PIPE, "deep_ensemble_warmup", "results_residual_optuna", "ensemble_warmup_residual_preds.csv"),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Famiglie disponibili: i run possono essere ancora in corso
# ═══════════════════════════════════════════════════════════════════════════

def _csv_utilizzabile(percorso):
    """Vero se il file esiste e contiene almeno una riga di dati oltre
    all'intestazione. Un CSV a zero righe capita se una pipeline è stata
    interrotta mentre scriveva."""
    if not os.path.exists(percorso) or os.path.getsize(percorso) == 0:
        return False
    try:
        return len(pd.read_csv(percorso, nrows=1)) > 0
    except Exception:
        return False


def famiglie_disponibili():
    """Le (metodo, approccio) che hanno entrambi i CSV pronti.

    Serve perché i run delle pipeline sono lunghi e vengono lanciati un po'
    alla volta: senza questo controllo la prima cartella ancora vuota farebbe
    fallire tutto il report con un FileNotFoundError. Le famiglie mancanti
    vengono elencate nel foglio "Copertura", così il report dice sempre su
    cosa è stato costruito.
    """
    disponibili, mancanti = [], []
    for method in METHODS:
        for approccio in APPROCCI:
            src = SOURCES[(method, approccio)]
            if _csv_utilizzabile(src["metrics"]) and _csv_utilizzabile(src["preds"]):
                disponibili.append((method, approccio))
            else:
                mancanti.append((method, approccio))
    return disponibili, mancanti


FAMIGLIE_DISPONIBILI, FAMIGLIE_MANCANTI = famiglie_disponibili()
METODI_DISPONIBILI = [m for m in METHODS if any(mm == m for mm, _ in FAMIGLIE_DISPONIBILI)]
TIPO_ORDER = [tipo_label(m, a) for m, a in FAMIGLIE_DISPONIBILI]


# ═══════════════════════════════════════════════════════════════════════════
# Caricamento metriche aggregate (tutte le famiglie in un unico dataframe)
# ═══════════════════════════════════════════════════════════════════════════

def load_combined_metrics():
    if not FAMIGLIE_DISPONIBILI:
        raise SystemExit(
            "Nessuna famiglia disponibile: nessuna pipeline di incertezza ha ancora "
            "prodotto i CSV in results_optuna/ o results_residual_optuna/.\n"
            "Lancia almeno una pipeline prima di costruire il report.")

    frames = []
    for method, approccio in FAMIGLIE_DISPONIBILI:
        df = pd.read_csv(SOURCES[(method, approccio)]["metrics"])
        df = df[df["split"] == "test"].copy()
        df["tipo"] = tipo_label(method, approccio)
        df["metodo"] = method
        df["approccio"] = approccio
        if METHOD_KIND[method] == "gaussian":
            df["primary_score"] = df["NLL"]
            df["primary_score_name"] = "NLL"
        else:
            df["primary_score"] = df["pinball_loss"]
            df["primary_score_name"] = "pinball_loss"
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def pick_best_feature_set_per_tipo(combined):
    """Per ciascuna famiglia, il feature-set con il punteggio
    primario di quella famiglia più basso sulla riga aggregata (tutti e 7
    gli orizzonti insieme). Diretta e residuo dello stesso metodo sono
    scelti INDIPENDENTEMENTE (possono avere feature-set migliori diversi) —
    il foglio "Diretta vs Residuo" più sotto usa invece lo stesso
    feature-set per un confronto mirato."""
    best_feature_set = {}
    all_rows = combined[combined["horizon"] == ALL_KEY]
    for tipo, group in all_rows.groupby("tipo"):
        best_row = group.sort_values("primary_score").iloc[0]
        best_feature_set[tipo] = best_row["feature_set"]
    return best_feature_set


# ═══════════════════════════════════════════════════════════════════════════
# Confronto evento/non-evento, con sigma (totale + epistemica/aleatoria)
# ═══════════════════════════════════════════════════════════════════════════

def _metrics_for_gaussian_subset(subset):
    n = len(subset)
    if n == 0:
        return {"n": 0, "MAE": np.nan, "coverage_90": np.nan, "sharpness_90": np.nan,
                "sigma": np.nan, "epistemic_std": np.nan, "aleatoric_std": np.nan}
    mae = mean_absolute_error(subset["y_true"], subset["mu_pred"])
    coverage_90 = eu.coverage_at_level(subset["y_true"], subset["mu_pred"], subset["total_std"], 0.90)
    sharpness_90 = eu.sharpness_at_level(subset["total_std"], 0.90)
    return {"n": n, "MAE": round(float(mae), 4),
            "coverage_90": round(float(coverage_90), 4),
            "sharpness_90": round(float(sharpness_90), 4),
            "sigma": round(float(subset["total_std"].mean()), 4),
            "epistemic_std": round(float(subset["epistemic_std"].mean()), 4),
            "aleatoric_std": round(float(subset["aleatoric_std"].mean()), 4)}


def _metrics_for_quantile_subset(subset):
    n = len(subset)
    if n == 0:
        return {"n": 0, "MAE": np.nan, "coverage_90": np.nan, "sharpness_90": np.nan,
                "sigma": np.nan, "epistemic_std": np.nan, "aleatoric_std": np.nan}
    quantile_predictions = subset[QUANTILE_COLS].values.astype(float)
    mae = mean_absolute_error(subset["y_true"], subset["q50"])
    coverage_90 = eq.coverage_from_quantiles(subset["y_true"].values, quantile_predictions, QUANTILE_LEVELS, 0.05, 0.95)
    sharpness_90 = eq.sharpness_from_quantiles(quantile_predictions, QUANTILE_LEVELS, 0.05, 0.95)
    # Quantile Regression non ha un sigma letterale (nessuna gaussiana assunta) — resta NaN, atteso.
    return {"n": n, "MAE": round(float(mae), 4),
            "coverage_90": round(float(coverage_90), 4),
            "sharpness_90": round(float(sharpness_90), 4),
            "sigma": np.nan, "epistemic_std": np.nan, "aleatoric_std": np.nan}


def compute_event_vs_nonevent(best_feature_set):
    rows = []
    for method, approccio in FAMIGLIE_DISPONIBILI:
        metrics_fn = _metrics_for_gaussian_subset if METHOD_KIND[method] == "gaussian" else _metrics_for_quantile_subset
        tipo = tipo_label(method, approccio)
        fset = best_feature_set.get(tipo)
        if fset is None:
            continue
        preds = pd.read_csv(SOURCES[(method, approccio)]["preds"])
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
                "sigma_evento": event_metrics["sigma"], "epistemica_evento": event_metrics["epistemic_std"],
                "aleatoria_evento": event_metrics["aleatoric_std"],
                "n_non_evento": noevent_metrics["n"], "MAE_non_evento": noevent_metrics["MAE"],
                "coverage_90_non_evento": noevent_metrics["coverage_90"], "sharpness_90_non_evento": noevent_metrics["sharpness_90"],
                "sigma_non_evento": noevent_metrics["sigma"], "epistemica_non_evento": noevent_metrics["epistemic_std"],
                "aleatoria_non_evento": noevent_metrics["aleatoric_std"],
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Diretta vs Residuo: STESSO feature-set (quello scelto dal residuo come
# migliore), confronto diretto dei punteggi ALL/test — isola la domanda
# "partire dalla fisica Penman riduce davvero l'incertezza residua, o no?"
# ═══════════════════════════════════════════════════════════════════════════

def compute_diretta_vs_residuo(combined, best_feature_set):
    compare_cols = ["MAE", "R2", "coverage_90", "sharpness_90",
                    "mean_epistemic_std", "mean_aleatoric_std"]
    all_rows = combined[combined["horizon"] == ALL_KEY]

    rows = []
    for method in METHODS:
        tipo_residuo = tipo_label(method, APPROCCIO_RESIDUO)
        tipo_diretta = tipo_label(method, APPROCCIO_DIRETTA)
        fset = best_feature_set.get(tipo_residuo)
        if fset is None:
            continue

        diretta_row = all_rows[(all_rows["tipo"] == tipo_diretta) & (all_rows["feature_set"] == fset)]
        residuo_row = all_rows[(all_rows["tipo"] == tipo_residuo) & (all_rows["feature_set"] == fset)]
        if diretta_row.empty or residuo_row.empty:
            continue
        diretta_row = diretta_row.iloc[0]
        residuo_row = residuo_row.iloc[0]

        primary_col = "NLL" if METHOD_KIND[method] == "gaussian" else "pinball_loss"
        row = {"metodo": method, "feature_set": fset, "primary_col": primary_col}
        for col in compare_cols + [primary_col]:
            row[f"diretta_{col}"] = float(diretta_row[col]) if col in diretta_row.index and pd.notna(diretta_row[col]) else np.nan
            row[f"residuo_{col}"] = float(residuo_row[col]) if col in residuo_row.index and pd.notna(residuo_row[col]) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Stile (stessa palette del resto del progetto) — TIPO_FILL ora a 6 voci:
# stessa tinta base per metodo, sfumatura più scura per "residuo".
# ═══════════════════════════════════════════════════════════════════════════

HEADER_FILL = PatternFill("solid", fgColor="2F5233")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TITLE_FONT = Font(bold=True, size=13)
# Una tinta per metodo: chiara per la diretta, più satura per il residuo.
# I due metodi con warm-up riprendono la tinta del metodo da cui derivano
# (azzurro per MC-Dropout, verde per Deep Ensemble) in una variante più
# fredda, così nel foglio si legge subito la parentela.
METHOD_TINTS = {
    METHOD_MC: ("C9DAF8", "9FC5E8"),
    METHOD_ENSEMBLE: ("D9EAD3", "B6D7A8"),
    METHOD_QUANTILE: ("FCE5CD", "F9CB9C"),
    METHOD_VI: ("E6D9F2", "C9A8E0"),
    METHOD_LAPLACE: ("D9D2E9", "B4A7D6"),
    METHOD_MC_WARMUP: ("D0E8EC", "A2C4C9"),
    METHOD_ENSEMBLE_WARMUP: ("E2EFD9", "C7E0B4"),
}
TIPO_FILL = {
    tipo_label(metodo, approccio): PatternFill("solid", fgColor=tinte[i])
    for metodo, tinte in METHOD_TINTS.items()
    for i, approccio in enumerate(APPROCCI)
}
METHOD_FILL = {   # per il foglio "Diretta vs Residuo", righe per metodo (non per tipo)
    metodo: PatternFill("solid", fgColor=tinte[0]) for metodo, tinte in METHOD_TINTS.items()
}
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
BETTER_FILL = PatternFill("solid", fgColor="D9EAD3")
WORSE_FILL = PatternFill("solid", fgColor="F4CCCC")


def _style_header_row(ws, row_idx, n_cols):
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row_idx, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center")


# ═══════════════════════════════════════════════════════════════════════════
# Sheet "Riepilogo"
# ═══════════════════════════════════════════════════════════════════════════

def _quantile_residuo_is_smoke_test():
    """Controlla la dimensione reale del CSV su disco (non un valore
    fisso): sotto ~1MB è quasi certamente ancora uno smoke test a pochi
    sensori, non il run completo a 14."""
    path = SOURCES[(METHOD_QUANTILE, APPROCCIO_RESIDUO)]["metrics"]
    try:
        return os.path.getsize(path) < 1_000_000
    except OSError:
        return False


def copertura_famiglie():
    """Una riga per famiglia prevista: presente o mancante, righe e sensori
    distinti nei preds, percorso.

    Sostituisce il vecchio controllo sulla dimensione del file con un dato
    diretto — quanti sensori ci sono davvero — e lo estende a tutte le
    famiglie: con i run lanciati un po' alla volta, una famiglia ferma allo
    smoke test (pochi sensori) ha numeri non confrontabili con le altre, e va
    vista subito.
    """
    righe = []
    for method in METHODS:
        for approccio in APPROCCI:
            src = SOURCES[(method, approccio)]
            presente = (method, approccio) in FAMIGLIE_DISPONIBILI
            n_righe = n_sensori = np.nan
            if presente:
                preds = pd.read_csv(src["preds"], usecols=["sensor"])
                n_righe = len(preds)
                n_sensori = preds["sensor"].nunique()
            righe.append({
                "tipo": tipo_label(method, approccio),
                "metodo": method, "approccio": approccio,
                "stato": "presente" if presente else "MANCANTE",
                "n_righe_preds": n_righe, "n_sensori": n_sensori,
                "percorso": os.path.relpath(src["preds"], PIPE),
            })
    df = pd.DataFrame(righe)
    presenti = df[df["stato"] == "presente"]
    if not presenti.empty:
        massimo = presenti["n_sensori"].max()
        # meno sensori delle altre = quasi certamente ancora uno smoke test
        df["nota"] = np.where((df["stato"] == "presente") & (df["n_sensori"] < massimo),
                              "probabile smoke test: meno sensori delle altre famiglie", "")
    else:
        df["nota"] = ""
    return df


def write_copertura_sheet(wb, copertura, best_feature_set):
    ws = wb.create_sheet("Copertura")
    ws.append(["Da quali famiglie è costruito questo report"])
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([])
    n_ok = int((copertura["stato"] == "presente").sum())
    ws.cell(row=ws.max_row + 1, column=1,
            value=f"{n_ok} famiglie su {len(copertura)} hanno i risultati. Le famiglie MANCANTI non "
                  f"compaiono in nessun altro foglio: la pipeline corrispondente non ha ancora prodotto "
                  f"i CSV. Rieseguendo questo script a run finiti rientrano da sole, senza modifiche al "
                  f"codice.").font = Font(italic=True)
    ws.append([])

    headers = ["Famiglia", "Stato", "Feature-set scelto", "Righe preds", "Sensori", "Percorso", "Nota"]
    ws.append(headers)
    _style_header_row(ws, ws.max_row, len(headers))

    for _, row in copertura.iterrows():
        fset = best_feature_set.get(row["tipo"], "")
        ws.append([row["tipo"], row["stato"], fset,
                   "" if pd.isna(row["n_righe_preds"]) else int(row["n_righe_preds"]),
                   "" if pd.isna(row["n_sensori"]) else int(row["n_sensori"]),
                   row["percorso"], row["nota"]])
        rr = ws.max_row
        for c in range(1, len(headers) + 1):
            cell = ws.cell(row=rr, column=c)
            cell.border = BORDER
            if row["stato"] == "presente":
                cell.fill = TIPO_FILL.get(row["tipo"])
            else:
                cell.fill = WORSE_FILL
        if row["nota"]:
            ws.cell(row=rr, column=len(headers)).font = Font(bold=True, color="CC0000")

    for i, w in enumerate([28, 12, 20, 12, 10, 62, 52]):
        ws.column_dimensions[get_column_letter(i + 1)].width = w
    return ws


def write_summary_sheet(wb, combined, best_feature_set):
    ws = wb.create_sheet("Riepilogo", 0)
    ws.append([f"Confronto unificato metodi di incertezza (Optuna): "
               f"{', '.join(METODI_DISPONIBILI)} — diretta vs residuo Penman"])
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([])

    ws.cell(row=ws.max_row + 1, column=1,
            value=f"Nota metodologica: solo pipeline Optuna, la griglia esaustiva resta in "
                  f"confronto_incertezza.xlsx, invariato. Le {len(TIPO_ORDER)} famiglie sotto sono "
                  f"{len(METODI_DISPONIBILI)} metodi, ciascuno nella variante diretta (target "
                  "assoluto) e residuo (target − penman_pred, ricostruito prima di calcolare qualunque "
                  "metrica — coverage/sharpness/MAE/sigma sono quindi sempre in scala assoluta, "
                  "confrontabili tra le due varianti). Il foglio \"Copertura\" dice quali famiglie "
                  "hanno i risultati e quali mancano ancora; il foglio \"Diretta vs Residuo\" fa il "
                  "confronto numerico fra le due varianti dello stesso metodo.").font = Font(italic=True)

    if _quantile_residuo_is_smoke_test():
        ws.cell(row=ws.max_row + 1, column=1,
                value="ATTENZIONE: quantile_regression_residual_componibile Optuna è ancora a livello di "
                      "smoke test (pochi sensori) al momento di questo report — i suoi numeri non sono "
                      "ancora comparabili in scala con le altre famiglie (run completo, 14 sensori) finché "
                      "non viene rilanciato per intero.").font = Font(italic=True, bold=True, color="CC0000")
    ws.append([])

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

    widths = [28, 26, 16, 12, 10, 12, 12]
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
    body("SIGMA (solo MC-Dropout e Deep Ensemble): la deviazione standard totale dichiarata dal "
         "modello (total_std = radice di epistemica² + aleatoria²). L'incertezza EPISTEMICA riflette "
         "quanto il modello è incerto per mancanza di dati/conoscenza (si riduce con più dati); "
         "l'incertezza ALEATORIA riflette il rumore intrinseco nei dati stessi (non si riduce con più "
         "dati). Quantile Regression non assume una gaussiana e quindi non ha un sigma letterale — "
         "nelle tabelle di questo report le sue celle sigma/epistemica/aleatoria restano vuote, non è "
         "un errore.")
    blank()
    blank()

    title("Come si calcolano")
    blank()
    body("MC-Dropout e Deep Ensemble assumono che l'errore segua una gaussiana N(mu, sigma^2), "
         "dove mu è la previsione puntuale e sigma (\"total_std\") è la deviazione standard totale "
         "dichiarata (epistemica + aleatoria). L'intervallo al 90% è [mu - z·sigma, mu + z·sigma] "
         "con z≈1.645. coverage_90 = quante volte il vero cade in quell'intervallo; sharpness_90 = "
         "2·z·sigma mediato su tutte le righe. Formule esatte in "
         "uncertainty_ml/eval_uncertainty.py: coverage_at_level() e sharpness_at_level().")
    blank()
    body("Quantile Regression non assume nessuna forma: l'intervallo al 90% è preso DIRETTAMENTE "
         "dai percentili che il modello predice, [Q05, Q95]. coverage_90 = quante volte il vero "
         "cade tra Q05 e Q95; sharpness_90 = Q95 - Q05 mediato su tutte le righe. Formule esatte in "
         "quantile_regression_ml/eval_quantile.py: coverage_from_quantiles() e "
         "sharpness_from_quantiles().")
    blank()
    blank()

    title("Come si interpretano insieme (non una sola delle due)")
    blank()
    body("Il confronto onesto tra due modelli guarda SEMPRE la coppia coverage+sharpness "
         "insieme, mai la sharpness da sola: un modello A con sharpness più piccola di un "
         "modello B è \"migliore\" SOLO SE la sua coverage è altrettanto vicina (o più vicina) "
         "al livello nominale.")
    blank()
    blank()

    title("Perché il confronto Diretta vs Residuo è lecito (e cosa aspettarsi)")
    blank()
    body("penman_pred è un valore FISSO, deterministico (uscita di una simulazione fisica, nessuna "
         "componente aleatoria). Sommare una costante a una gaussiana sposta solo la media, MAI la "
         "varianza — quindi epistemic_std/aleatoric_std/total_std del modello residuo, dopo la "
         "ricostruzione, sono nella STESSA scala e STESSA definizione operativa di quelle del modello "
         "diretto (stessa unità: punti % di umidità, stessa domanda: quanto è incerta la previsione "
         "ASSOLUTA). Il confronto nel foglio \"Diretta vs Residuo\" è quindi strutturalmente lecito, "
         "non un errore di mele con pere.")
    blank()
    body("NON aspettarti però che i due sigma siano numericamente uguali o simili: le due pipeline "
         "spiegano target diversi (il target assoluto contro target − penman_pred). Se la fisica "
         "Penman-Monteith cattura già una parte sistematica del comportamento, il residuo ha "
         "INTRINSECAMENTE meno varianza da spiegare — un sigma residuo più piccolo non è un artefatto "
         "della ricostruzione, è il segnale utile che questo confronto vuole catturare: partire dalla "
         "fisica Penman riduce davvero l'incertezza residua, o no? Vedi "
         "confronto/report_incertezza_dettagliato.md, domanda 3, per la spiegazione estesa.")

    return ws


# ═══════════════════════════════════════════════════════════════════════════
# Sheet "Confronto metodi (evento vs non-evento)"
# ═══════════════════════════════════════════════════════════════════════════

def write_method_comparison_sheet(wb, event_comparison):
    ws = wb.create_sheet("Confronto evento-non evento")
    ws.cell(row=1, column=1,
           value="Le stesse metriche comuni (MAE, coverage_90, sharpness_90) più il sigma "
                 "(totale/epistemica/aleatoria, dove disponibile — vuoto per Quantile Regression), "
                 "calcolate separatamente sui giorni con evento acqua (pioggia o irrigazione) e sui "
                 "giorni senza — test split, feature-set migliore di ciascuna famiglia "
                 "(Optuna, diretta e residuo).").font = Font(italic=True)
    ws.append([])

    col_headers = ["Famiglia", "Feature-set", "Orizzonte",
                   "n (evento)", "MAE (evento)", "coverage_90 (evento)", "sharpness_90 (evento)",
                   "sigma (evento)", "epistemica (evento)", "aleatoria (evento)",
                   "n (non evento)", "MAE (non evento)", "coverage_90 (non evento)", "sharpness_90 (non evento)",
                   "sigma (non evento)", "epistemica (non evento)", "aleatoria (non evento)"]
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
                row["sigma_evento"], row["epistemica_evento"], row["aleatoria_evento"],
                row["n_non_evento"], row["MAE_non_evento"], row["coverage_90_non_evento"], row["sharpness_90_non_evento"],
                row["sigma_non_evento"], row["epistemica_non_evento"], row["aleatoria_non_evento"],
            ]
            ws.append([None if pd.isna(v) else v for v in values])
            rr = ws.max_row
            for c in range(1, len(col_headers) + 1):
                cell = ws.cell(row=rr, column=c)
                cell.border = BORDER
                cell.fill = TIPO_FILL.get(tipo)
            if row["_horizon_key"] == ALL_KEY:
                for c in range(1, len(col_headers) + 1):
                    ws.cell(row=rr, column=c).font = Font(bold=True)

    widths = [26, 14, 14, 11, 13, 18, 18, 11, 14, 14, 14, 15, 20, 20, 14, 16, 16]
    for i, w in enumerate(widths):
        ws.column_dimensions[get_column_letter(i + 1)].width = w
    ws.freeze_panes = "A4"
    return ws


# ═══════════════════════════════════════════════════════════════════════════
# Sheet "Per orizzonte" — dettaglio t+1..t+7 + Complessivo, per ogni famiglia
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
# Sheet "Diretta vs Residuo"
# ═══════════════════════════════════════════════════════════════════════════

def write_diretta_vs_residuo_sheet(wb, diretta_vs_residuo):
    ws = wb.create_sheet("Diretta vs Residuo")
    ws.cell(row=1, column=1,
           value="Confronto diretto tra la variante DIRETTA (target assoluto) e la variante RESIDUO "
                 "(target − penman_pred, ricostruito) di ciascun metodo — stesso feature-set (il "
                 "migliore per il residuo), test set, Complessivo 7gg. Verde = residuo migliore, "
                 "rosso = diretta migliore (soglia 1% per evitare falsi verdetti sul rumore). Vedi "
                 "report_incertezza_dettagliato.md §3-4 per perché il confronto è lecito, anche se i "
                 "due modelli imparano target diversi.").font = Font(italic=True)
    ws.append([])

    metric_labels = [
        ("primary_col", "Punteggio primario (NLL/pinball)", True),
        ("MAE", "MAE", True),
        ("R2", "R²", False),
        ("coverage_90", "coverage_90 (nominale 0.90)", None),
        ("sharpness_90", "sharpness_90", True),
        ("mean_epistemic_std", "epistemica media", True),
        ("mean_aleatoric_std", "aleatoria media", True),
    ]

    headers = ["Metodo", "Feature-set", "Metrica", "Diretta", "Residuo", "Δ (Residuo - Diretta)", "Verdetto"]
    hdr_row = ws.max_row + 1
    for i, h in enumerate(headers, start=1):
        ws.cell(row=hdr_row, column=i, value=h)
    _style_header_row(ws, hdr_row, len(headers))

    for _, comp_row in diretta_vs_residuo.iterrows():
        metodo = comp_row["metodo"]
        fset = comp_row["feature_set"]
        primary_col = comp_row["primary_col"]

        for col_key, label, lower_is_better in metric_labels:
            actual_col = primary_col if col_key == "primary_col" else col_key
            diretta_val = comp_row.get(f"diretta_{actual_col}", np.nan)
            residuo_val = comp_row.get(f"residuo_{actual_col}", np.nan)
            if pd.isna(diretta_val) or pd.isna(residuo_val):
                continue
            delta = residuo_val - diretta_val

            if actual_col == "coverage_90":
                verdict = "meglio Residuo" if abs(residuo_val - 0.90) < abs(diretta_val - 0.90) - 1e-6 else (
                    "meglio Diretta" if abs(diretta_val - 0.90) < abs(residuo_val - 0.90) - 1e-6 else "pari")
            elif lower_is_better is None:
                verdict = "pari"
            elif lower_is_better:
                verdict = "meglio Residuo" if delta < -0.01 * abs(diretta_val) else ("meglio Diretta" if delta > 0.01 * abs(diretta_val) else "pari")
            else:
                verdict = "meglio Residuo" if delta > 0.01 * abs(diretta_val) else ("meglio Diretta" if delta < -0.01 * abs(diretta_val) else "pari")

            display_label = label if actual_col != "primary_col" else f"{label} ({primary_col})"
            ws.append([metodo, fset, display_label, round(diretta_val, 4), round(residuo_val, 4), round(delta, 4), verdict])
            rr = ws.max_row
            for c in range(1, len(headers) + 1):
                cell = ws.cell(row=rr, column=c)
                cell.border = BORDER
                if c <= 2:
                    cell.fill = METHOD_FILL.get(metodo)
            verdict_cell = ws.cell(row=rr, column=7)
            if verdict == "meglio Residuo":
                verdict_cell.fill = BETTER_FILL
            elif verdict == "meglio Diretta":
                verdict_cell.fill = WORSE_FILL

    widths = [18, 14, 30, 12, 12, 18, 14]
    for i, w in enumerate(widths):
        ws.column_dimensions[get_column_letter(i + 1)].width = w
    ws.freeze_panes = "A3"
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

    widths = [26, 14, 16, 14, 10, 10, 8, 12, 12, 12, 12]
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
    print("Migliori feature-set per famiglia (Optuna, diretta+residuo):", best_feature_set)

    print("Calcolo confronto metodi evento vs non-evento (righe di *_preds.csv filtrate per has_water_event)...")
    event_comparison = compute_event_vs_nonevent(best_feature_set)

    print("Calcolo confronto Diretta vs Residuo (stesso feature-set, quello scelto dal residuo)...")
    diretta_vs_residuo = compute_diretta_vs_residuo(combined, best_feature_set)

    copertura = copertura_famiglie()
    n_ok = int((copertura["stato"] == "presente").sum())
    print(f"Famiglie con risultati: {n_ok}/{len(copertura)}")
    for _, r in copertura[copertura["stato"] != "presente"].iterrows():
        print(f"  MANCANTE: {r['tipo']}  ({r['percorso']})")

    wb = Workbook()
    wb.remove(wb.active)
    write_summary_sheet(wb, combined, best_feature_set)
    write_copertura_sheet(wb, copertura, best_feature_set)
    write_guida_sheet(wb)
    write_method_comparison_sheet(wb, event_comparison)
    write_by_horizon_sheet(wb, combined, best_feature_set)
    write_diretta_vs_residuo_sheet(wb, diretta_vs_residuo)
    write_ranking_sheet(wb, combined)
    write_full_data_sheet(wb, combined)

    wb.save(OUT_PATH)
    print(f"Salvato: {OUT_PATH}")
    print(f"Righe totali combinate: {len(combined)}")


if __name__ == "__main__":
    main()