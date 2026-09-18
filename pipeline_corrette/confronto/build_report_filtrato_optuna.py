"""
build_report_componibile_filtrato_optuna.py
─────────────────────────────────────────────────────────────────────────────
Come build_report_componibile_filtrato.py (stessa identica metodologia,
stessa struttura — vedi quel file per la spiegazione completa), ma "ML puro"
e "Penman + residuo" vengono dalle versioni OPTUNA delle due pipeline invece
che dalla griglia esaustiva:
  - ML puro (filtrato, Optuna):
    multioutput_ml/results_comparabile_penman_optuna/experiments_comparabile_penman_optuna_preds.csv
  - Penman puro (INVARIATO, nessuna variante Optuna esiste per la fisica):
    penman_baseline_componibile/results/penman_preds.csv
  - Penman + residuo (solo feature, filtrato, Optuna):
    penman_residual_componibile/results_solo_feature_optuna/penman_residual_solo_feature_optuna_preds.csv

Entrambe le pipeline Optuna condividono lo stesso filtro di comparabilità
Penman applicato PRIMA del training (identico a quello delle rispettive
versioni a griglia) — la logica di "finestra comune" sotto resta quindi,
come nel file griglia, una rete di sicurezza/verifica, non una restrizione
attesa.

Output: confronto/confronto_filtrato_optuna.xlsx
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(__file__)
PIPE = os.path.join(HERE, "..")            # pipeline_corrette/
CORE = os.path.join(PIPE, "core")

# feature_groups.py sta in pipeline_corrette/core/, non nella radice della
# pipeline: CORE va aggiunto al path PRIMA di importarlo.
sys.path.insert(0, os.path.join(PIPE))
sys.path.insert(0, CORE)
from feature_groups import THEMATIC_FEATURE_SETS  # noqa: E402

ML_PREDS = os.path.join(PIPE, "multioutput", "results_comparabile_penman_optuna", "experiments_comparabile_penman_optuna_preds.csv")
PENMAN_PREDS = os.path.join(PIPE, "penman_baseline", "results", "penman_preds.csv")
RESIDUAL_PREDS = os.path.join(PIPE, "penman_residual", "results_solo_feature_optuna", "penman_residual_solo_feature_optuna_preds.csv")

OUT_PATH = os.path.join(HERE, "confronto_filtrato_optuna.xlsx")

FSET_ORDER = list(THEMATIC_FEATURE_SETS.keys())
MODEL_ORDER = ["Ridge", "RF", "MLP", "LSTM"]
HORIZONS = [f"t+{h}" for h in range(1, 8)]
ALL_LABEL = "Complessivo (7gg)"
TIPO_ORDER = ["Penman puro", "ML puro (componibile, filtrato, Optuna)", "Penman + ML (residuo, solo feature, filtrato, Optuna)"]


# ═══════════════════════════════════════════════════════════════════════════
# Caricamento + finestra comune di giorni (sensore, data)
# ═══════════════════════════════════════════════════════════════════════════

def _full_week_dayset(df, model, fset=None):
    """(sensore, data) per cui questo modello (con questo feature-set, se
    dato) ha una previsione VALIDA (y_pred non-NaN) per tutti e 7 gli
    orizzonti."""
    sub = df[df["model"] == model]
    if fset is not None:
        sub = sub[sub["feature_set"] == fset]
    sub = sub.dropna(subset=["y_pred"])
    g = sub.groupby(["sensor", "date"])["horizon"].nunique()
    return set(g[g == 7].index)


def load_preds():
    ml = pd.read_csv(ML_PREDS)
    ml = ml[ml["split"] == "test"].copy()

    pen = pd.read_csv(PENMAN_PREDS)
    pen = pen[pen["split"] == "test"].copy()
    pen["feature_set"] = "--"

    res = pd.read_csv(RESIDUAL_PREDS)
    res = res[res["split"] == "test"].copy()

    return ml, pen, res


def compute_dayset_info(ml, pen, res):
    # Giorni canonici di ML puro filtrato: per costruzione identici per ogni
    # feature-set/modello non-LSTM (split condiviso) — Ridge come
    # riferimento, primo feature-set disponibile.
    reference_fset = FSET_ORDER[0] if FSET_ORDER[0] in ml["feature_set"].unique() else ml["feature_set"].iloc[0]
    ml_days = _full_week_dayset(ml, "Ridge", reference_fset)

    # Giorni con previsione Penman VALIDA (la catena fisica arriva in fondo).
    pen_days = _full_week_dayset(pen, "Penman-Monteith")

    # Giorni usati dal residuo (atteso: coincidenza pressoché totale con
    # ml_days e pen_days, dato che tutte e tre le fonti condividono a monte
    # lo stesso filtro Penman applicato PRIMA del training).
    res_days = _full_week_dayset(res, "Ridge", reference_fset)

    common_days = ml_days & pen_days & res_days

    counts = {
        "ML puro (componibile, filtrato, Optuna) — copertura nativa": len(ml_days),
        "Penman puro — copertura nativa": len(pen_days),
        "Penman + residuo (solo feature, filtrato, Optuna) — copertura nativa": len(res_days),
        "Finestra comune usata nel confronto": len(common_days),
    }
    return ml_days, pen_days, res_days, common_days, counts


def per_sensor_coverage(ml_days, pen_days, res_days, common_days):
    sensors = sorted(set(s for s, d in ml_days | pen_days | res_days))
    rows = []
    for sensor in sensors:
        rows.append({
            "sensore": sensor,
            "ML puro (filtrato, Optuna) — copertura nativa": sum(1 for s, d in ml_days if s == sensor),
            "Penman puro — copertura nativa": sum(1 for s, d in pen_days if s == sensor),
            "Penman + residuo (filtrato, Optuna) — copertura nativa": sum(1 for s, d in res_days if s == sensor),
            "Finestra comune usata nel confronto": sum(1 for s, d in common_days if s == sensor),
        })
    return pd.DataFrame(rows)


def _metrics_core(yt, yp):
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 5:
        return None
    try:
        r, _ = pearsonr(yt, yp)
    except Exception:
        r = np.nan
    return {
        "MAE": round(float(mean_absolute_error(yt, yp)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        "R2": round(float(r2_score(yt, yp)), 4),
        "r": round(float(r), 4) if pd.notna(r) else np.nan,
        "n": int(len(yt)),
    }


def recompute_metrics(df, tipo, common_days, has_fset):
    df = df.copy()
    df["_key"] = list(zip(df["sensor"], df["date"]))
    df = df[df["_key"].isin(common_days)]
    if not has_fset:
        df["feature_set"] = "--"

    rows = []
    group_cols = ["model", "feature_set"] if has_fset else ["model"]
    for keys, g in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        kwargs = dict(zip(group_cols, keys))
        fset = kwargs.get("feature_set", "--")
        model = kwargs["model"]

        for h in HORIZONS:
            gh = g[g["horizon"] == h]
            m = _metrics_core(gh["y_true"].values.astype(float), gh["y_pred"].values.astype(float))
            if m is None:
                continue
            rows.append({"tipo": tipo, "model": model, "feature_set": fset, "horizon": h, **m})

        m_all = _metrics_core(g["y_true"].values.astype(float), g["y_pred"].values.astype(float))
        if m_all is not None:
            rows.append({"tipo": tipo, "model": model, "feature_set": fset, "horizon": ALL_LABEL, **m_all})

    return pd.DataFrame(rows)


def load_all():
    ml, pen, res = load_preds()
    ml_days, pen_days, res_days, common_days, counts = compute_dayset_info(ml, pen, res)
    coverage = per_sensor_coverage(ml_days, pen_days, res_days, common_days)

    df_ml = recompute_metrics(ml, "ML puro (componibile, filtrato, Optuna)", common_days, has_fset=True)
    df_pen = recompute_metrics(pen, "Penman puro", common_days, has_fset=False)
    df_res = recompute_metrics(res, "Penman + ML (residuo, solo feature, filtrato, Optuna)", common_days, has_fset=True)

    combined = pd.concat([df_pen, df_ml, df_res], ignore_index=True)
    return combined, counts, len(common_days), coverage


def _row_sort_key(row):
    tipo_order = {t: i for i, t in enumerate(TIPO_ORDER)}
    fset_order = {f: i for i, f in enumerate(FSET_ORDER)}
    model_order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return (
        tipo_order.get(row["tipo"], 9),
        fset_order.get(row["feature_set"], -1 if row["feature_set"] == "--" else 99),
        model_order.get(row["model"], 0),
        row["model"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Stile (stessa palette di build_report_componibile_filtrato.py, per coerenza visiva)
# ═══════════════════════════════════════════════════════════════════════════

HEADER_FILL = PatternFill("solid", fgColor="2F5233")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TITLE_FONT = Font(bold=True, size=13)
SECTION_FILL = {
    "Penman puro":                                                    PatternFill("solid", fgColor="FCE4D6"),
    "ML puro (componibile, filtrato, Optuna)":                         PatternFill("solid", fgColor="C9DAF8"),
    "Penman + ML (residuo, solo feature, filtrato, Optuna)":           PatternFill("solid", fgColor="FCE5CD"),
}
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
BEST_FILL = PatternFill("solid", fgColor="B6D7A8")


def _style_header_row(ws, row_idx, n_cols):
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row_idx, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center")


# ═══════════════════════════════════════════════════════════════════════════
# Sheet "Per feature-set (MAE/R2)"
# ═══════════════════════════════════════════════════════════════════════════

def write_by_feature_set_sheet(wb, title, df_all, value_col, better="low"):
    ws = wb.create_sheet(title)
    col_headers = ["Tipo", "Modello"] + HORIZONS + [ALL_LABEL]
    n_cols = len(col_headers)
    color_cols = list(range(3, 3 + len(HORIZONS) + 1))

    fset_blocks = ["--"] + FSET_ORDER
    row_cursor = 1

    for fset in fset_blocks:
        block = df_all[df_all["feature_set"] == fset].copy()
        if block.empty:
            continue
        label = "Penman puro (nessun feature-set)" if fset == "--" else f"Feature-set: {fset}"
        ws.cell(row=row_cursor, column=1, value=label).font = TITLE_FONT
        row_cursor += 1

        hdr_row = row_cursor
        for i, h in enumerate(col_headers, start=1):
            ws.cell(row=hdr_row, column=i, value=h)
        _style_header_row(ws, hdr_row, n_cols)
        row_cursor += 1

        block["_sort"] = block.apply(_row_sort_key, axis=1)
        block = block.sort_values("_sort")
        pivot = block.pivot_table(index=["tipo", "model"], columns="horizon",
                                  values=value_col, aggfunc="first")
        order = block.drop_duplicates(subset=["tipo", "model"])[["tipo", "model"]]
        pivot = pivot.reindex(columns=HORIZONS + [ALL_LABEL])

        data_start = row_cursor
        for _, r in order.iterrows():
            tipo, model = r["tipo"], r["model"]
            vals = [tipo, model]
            for h in HORIZONS + [ALL_LABEL]:
                v = pivot.loc[(tipo, model), h] if (tipo, model) in pivot.index else np.nan
                vals.append(None if pd.isna(v) else round(float(v), 4))
            ws.append(vals)
            rr = ws.max_row
            fill = SECTION_FILL.get(tipo)
            for c in range(1, n_cols + 1):
                cell = ws.cell(row=rr, column=c)
                cell.border = BORDER
                if c <= 2 and fill:
                    cell.fill = fill
            row_cursor += 1
        data_end = row_cursor - 1

        if data_end >= data_start:
            for col in color_cols:
                col_letter = get_column_letter(col)
                rng = f"{col_letter}{data_start}:{col_letter}{data_end}"
                colors = ("63BE7B", "FFEB84", "F8696B") if better == "low" else ("F8696B", "FFEB84", "63BE7B")
                rule = ColorScaleRule(start_type="min", start_color=colors[0],
                                      mid_type="percentile", mid_value=50, mid_color=colors[1],
                                      end_type="max", end_color=colors[2])
                ws.conditional_formatting.add(rng, rule)

        row_cursor += 1

    ws.freeze_panes = "C2"
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 10
    for i in range(len(HORIZONS) + 1):
        ws.column_dimensions[get_column_letter(3 + i)].width = 12
    return ws


# ═══════════════════════════════════════════════════════════════════════════
# Sheet "Riepilogo"
# ═══════════════════════════════════════════════════════════════════════════

def write_summary_sheet(wb, df_all, counts, n_common, coverage):
    ws = wb.create_sheet("Riepilogo", 0)
    ws.append(["Confronto pipeline componibili FILTRATE su Penman (Optuna): Penman puro vs ML puro vs Penman+ML residuo (solo feature)"])
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([])

    ws.cell(row=ws.max_row + 1, column=1,
            value="Nota metodologica: come confronto_filtrato.xlsx, ma ML puro e il "
                  "residuo vengono dalle versioni OPTUNA delle rispettive pipeline (ricerca "
                  "iperparametri con TPESampler invece di griglia esaustiva — vedi "
                  "run_multioutput_componibile_comparabile_penman_optuna.py e "
                  "run_penman_residual_componibile_solo_feature_optuna.py), non dalla griglia. "
                  "Entrambe restano allenate SOLO sul sotto-insieme di righe in cui la catena fisica "
                  "Penman arriva in fondo per tutti e 7 gli orizzonti (filtro applicato PRIMA del "
                  "training). Il residuo usa solo le feature componibili (nessuna colonna fisica "
                  "Penman aggiuntiva). La finestra comune qui sotto resta calcolata come rete di "
                  "sicurezza: ci si aspetta che coincida quasi esattamente con la copertura nativa di "
                  "ciascuna fonte, dato che condividono già a monte lo stesso filtro Penman.").font = Font(italic=True)
    ws.cell(row=ws.max_row + 1, column=1,
            value=f"Copertura nativa: {counts['ML puro (componibile, filtrato, Optuna) — copertura nativa']} giorni ML puro, "
                  f"{counts['Penman puro — copertura nativa']} giorni Penman puro, "
                  f"{counts['Penman + residuo (solo feature, filtrato, Optuna) — copertura nativa']} giorni residuo "
                  f"→ finestra comune usata nel confronto = {n_common} giorni.").font = Font(italic=True)
    ws.append([])

    ws.cell(row=ws.max_row + 1, column=1, value="Copertura per sensore").font = Font(bold=True, size=12)
    cov_headers = list(coverage.columns)
    ws.append(cov_headers)
    _style_header_row(ws, ws.max_row, len(cov_headers))
    for _, r in coverage.iterrows():
        ws.append([r[c] for c in cov_headers])
        rr = ws.max_row
        for c in range(1, len(cov_headers) + 1):
            ws.cell(row=rr, column=c).border = BORDER
    ws.append([])
    ws.append([])

    ws.cell(row=ws.max_row + 1, column=1, value="Il migliore per ciascun orizzonte").font = Font(bold=True, size=12)
    headers = ["Orizzonte", "Migliore assoluto (tipo / fset / modello)", "MAE", "R2", "n"]
    ws.append(headers)
    _style_header_row(ws, ws.max_row, len(headers))
    for h in HORIZONS + [ALL_LABEL]:
        sub = df_all[df_all["horizon"] == h].dropna(subset=["MAE"])
        if sub.empty:
            continue
        best = sub.loc[sub["MAE"].idxmin()]
        ws.append([h, f"{best['tipo']} / {best['feature_set']} / {best['model']}",
                  round(float(best["MAE"]), 4),
                  round(float(best["R2"]), 4) if pd.notna(best["R2"]) else None,
                  int(best["n"]) if pd.notna(best["n"]) else None])
        for c in range(1, len(headers) + 1):
            ws.cell(row=ws.max_row, column=c).border = BORDER
        if h == ALL_LABEL:
            for c in range(1, len(headers) + 1):
                ws.cell(row=ws.max_row, column=c).fill = BEST_FILL
    ws.append([])
    ws.append([])

    ws.cell(row=ws.max_row + 1, column=1,
           value="Tabella di confronto totale (ordinata per MAE complessivo crescente)").font = Font(bold=True, size=12)
    headers2 = ["#", "Tipo", "Feature-set", "Modello", "MAE (complessivo)", "RMSE", "R2", "r", "n"]
    ws.append(headers2)
    _style_header_row(ws, ws.max_row, len(headers2))

    total = df_all[df_all["horizon"] == ALL_LABEL].dropna(subset=["MAE"]).sort_values("MAE").reset_index(drop=True)
    for i, row in total.iterrows():
        ws.append([i + 1, row["tipo"], row["feature_set"], row["model"],
                  round(float(row["MAE"]), 4), round(float(row["RMSE"]), 4) if pd.notna(row["RMSE"]) else None,
                  round(float(row["R2"]), 4) if pd.notna(row["R2"]) else None,
                  round(float(row["r"]), 4) if pd.notna(row["r"]) else None,
                  int(row["n"]) if pd.notna(row["n"]) else None])
        rr = ws.max_row
        fill = SECTION_FILL.get(row["tipo"])
        for c in range(1, len(headers2) + 1):
            cell = ws.cell(row=rr, column=c)
            cell.border = BORDER
            if c <= 4 and fill:
                cell.fill = fill
        if i == 0:
            for c in range(1, len(headers2) + 1):
                ws.cell(row=rr, column=c).font = Font(bold=True)

    widths = [5, 34, 26, 10, 16, 10, 8, 8, 8]
    for i, w in enumerate(widths):
        ws.column_dimensions[get_column_letter(i + 1)].width = w
    return ws


def write_full_data_sheet(wb, combined):
    ws = wb.create_sheet("Dati completi")
    cols = combined.columns.tolist()
    ws.append(cols)
    _style_header_row(ws, 1, len(cols))
    for _, row in combined.iterrows():
        ws.append([row[c] for c in cols])
    ws.freeze_panes = "A2"
    for i, c in enumerate(cols):
        ws.column_dimensions[get_column_letter(i + 1)].width = max(10, min(30, len(c) + 4))
    return ws


def main():
    df_all, counts, n_common, coverage = load_all()
    print("Copertura:", counts)
    print(coverage.to_string(index=False))

    wb = Workbook()
    wb.remove(wb.active)
    write_summary_sheet(wb, df_all, counts, n_common, coverage)
    write_by_feature_set_sheet(wb, "Per feature-set (MAE)", df_all, "MAE", better="low")
    write_by_feature_set_sheet(wb, "Per feature-set (R2)", df_all, "R2", better="high")
    write_full_data_sheet(wb, df_all)

    wb.save(OUT_PATH)
    print(f"Salvato: {OUT_PATH}")
    print(f"Righe totali combinate: {len(df_all)}")


if __name__ == "__main__":
    main()