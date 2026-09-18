# ─────────────────────────────────────────────────────────────────────────────
# COPIA in pipeline_corrette/dataset_aggiornato/ di pipeline_corrette/penman_baseline/run_penman_osservato.py.
# Identica all'originale tranne tre punti: CORE sale di un livello (la copia
# sta una cartella più in basso), il dataset di default è
# paths.DATASET_AGGIORNATO (quello rigenerato con i periodi 2026 dei sensori
# 3, 4, 8, 20) invece di DEFAULT_DATASET, e il dataset in uso viene stampato
# all'avvio. I risultati finiscono nelle results*/ di QUESTA cartella, separati
# da quelli sul dataset vecchio.
# ─────────────────────────────────────────────────────────────────────────────

"""
run_penman_componibile.py
─────────────────────────────────────────────────────────────────────────────
Come penman_baseline/run_penman.py (simulazione Penman-Monteith iterativa
t+1..t+7, FAO-56), ma con due differenze:

  1. Usa l'irrigazione RICOSTRUITA
     (irrigazione_mancante/irrigazione_ricostruita.csv: log originale + 8
     eventi ricostruiti dall'analisi dei salti di umidità senza pioggia)
     invece del solo log originale — più segnale di irrigazione nel
     bilancio idrico simulato.

  2. Le predizioni sono etichettate con lo split CANONICO CONDIVISO
     (ai_feature_componibile/dati_ricostruiti_util.py — lo stesso split di
     multioutput_ml/ e penman_residual_componibile/), non con uno split
     ricalcolato indipendentemente PER ORIZZONTE come fa
     penman_baseline/run_penman.py originale. Quella indipendenza per
     orizzonte è la causa root del problema di allineamento "n" già
     risolto A VALLE (in confronto/build_report.py, restringendo
     all'insieme canonico dopo il fatto). Qui si risolve ALLA RADICE:
     Penman-Monteith non ha nulla da allenare, quindi non ha bisogno di un
     proprio split — riusa direttamente quello canonico, calcolato una
     volta sola e condiviso con le altre due pipeline "componibili".

Riuso in sola lettura (nessuna modifica ai file sorgenti):
  - penman_baseline/run_penman.py: run_penman_multi_horizon_for_sensor,
    enrich_all_with_penman — fisica INVARIATA, cambia solo quali dati
    (irrigazione ricostruita) e quale split (canonico condiviso) le
    vengono dati intorno.
  - modelli singoli/data_driven_ml/utils.py: HORIZONS, SAT_OVERRIDE,
    load_colture, load_sensori, apply_soil_overrides, _metric_dict.
  - modelli singoli/data_driven_ml/run_experiments.py: DEFAULT_DATASET,
    DEFAULT_SENSORS, TRAIN_FRAC, VAL_FRAC.
  - ai_feature_componibile/dati_ricostruiti_util.py (stessa cartella padre):
    build_all_datasets_with_reconstructed_irrigation, build_canonical_split,
    compute_split_composition.

colture.csv copiato localmente (stessa prassi già seguita per le altre
pipeline Penman del progetto).

Output in results/:
  penman_metrics.csv     — per orizzonte-o-ALL x split
  penman_preds.csv       — predizioni giorno per giorno, etichettate con lo
                           split canonico (solo per i (sensore,giorno) che
                           rientrano nell'insieme canonico — gli altri non
                           vengono riportati, nessuna etichetta inventata)
  split_composition.csv  — giorni di pioggia/irrigazione per split

Utilizzo:
    python -u run_penman_componibile.py
    python -u run_penman_componibile.py --sensors-filter EM-500-9 EM-500-12
"""

# ═══════════════════════════════════════════════════════════════════════════
# ATTENZIONE — QUESTO E' UN ORACOLO, NON UNA PIPELINE UTILIZZABILE.
# Al passo k usa la pioggia REALMENTE CADUTA il giorno D+k (ws_rainfall_total,
# stazione meteo) al posto della previsione: guarda il futuro, informazione non
# disponibile al momento della previsione. Serve a misurare il limite superiore
# della fisica di Penman se la previsione di pioggia fosse perfetta — va sempre
# etichettato come tale nei confronti, mai messo alla pari delle altre pipeline.
# La controparte deployabile e' run_penman_previsto.py.
# ═══════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Versione CORRETTA e INDIPENDENTE (pipeline_corrette/).
# Rispetto all'originale in ai_feature_componibile/ cambiano solo gli import:
# tutto il motore arriva da pipeline_corrette/core/, che contiene il fix di
# aggregazione delle previsioni — fc{h}_rain e fc{h}_et0_fao_evapotranspiration
# ora sono SOMMATE sulle ~24 letture orarie invece che mediate (prima erano
# ~23-24x troppo piccole). Nessun import da modelli singoli/,
# ai_feature_componibile/, penman_baseline/, uncertainty_ml/, deep_ensemble_ml/,
# quantile_regression_ml/, data_driven_ml_multioutput/.
# ─────────────────────────────────────────────────────────────────────────────


import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
CORE = os.path.join(HERE, "..", "..", "core")         # pipeline_corrette/core

sys.path.insert(0, CORE)

from dataset_builder import (  # noqa: E402
    build_all_datasets_with_reconstructed_irrigation,
    build_canonical_split,
    compute_split_composition,
    simplified_metrics,
)

from data_utils import (  # noqa: E402
    HORIZONS, SAT_OVERRIDE,
    load_colture, load_sensori, apply_soil_overrides,
)
from paths import (DATASET_AGGIORNATO as DEFAULT_DATASET, DEFAULT_SENSORS,  # noqa: E402
                   DEFAULT_IRRIGAZIONE_RICOSTRUITA, DEFAULT_COLTURE,
                   TRAIN_FRAC, VAL_FRAC, require_data_files)
from penman import run_penman_multi_horizon_for_sensor, enrich_all_with_penman  # noqa: E402  (penman_baseline/, invariato)

OUT_DIR = os.path.join(HERE, "results_osservato")

TARGET_COLS = [f"target_t{h}" for h in HORIZONS]
MODEL_NAME = "Penman-Monteith (pioggia OSSERVATA, oracolo)"
OBSERVED_RAIN_COL = "ws_rainfall_total"


# ═══════════════════════════════════════════════════════════════════════════
# Predizioni in formato lungo, etichettate con lo split canonico
# ═══════════════════════════════════════════════════════════════════════════

def build_predictions_long(datasets_penman, split_lookup):
    """
    Costruisce (date, sensor, model, horizon, split, y_true, y_pred,
    residual) per tutti i (sensore, giorno, orizzonte) dove il target
    esiste. Un giorno viene etichettato con lo split canonico SE quel
    (sensore, giorno) fa parte dell'insieme canonico — altrimenti viene
    scartato (non gli si inventa un'etichetta): la fisica può aver
    calcolato una previsione anche per giorni fuori dal canonico (es. molto
    vicini all'inizio/fine serie), ma qui contano solo i giorni che ML puro
    userebbe per il confronto.
    """
    frames = []
    for sensor, ds in datasets_penman.items():
        ds = ds.copy()
        ds["date"] = pd.to_datetime(ds["date"]).dt.strftime("%Y-%m-%d")
        ds["sensor"] = sensor
        keep_cols = ["date", "sensor"] + [f"target_t{h}" for h in HORIZONS] + \
                   [f"penman_pred_t{h}" for h in HORIZONS if f"penman_pred_t{h}" in ds.columns] + \
                   [c for c in ("has_rain", "has_irr") if c in ds.columns]
        keep_cols = [c for c in keep_cols if c in ds.columns]
        frames.append(ds[keep_cols])

    if not frames:
        return pd.DataFrame()
    all_data = pd.concat(frames, ignore_index=True)

    horizon_blocks = []
    for h in HORIZONS:
        target_col, pred_col = f"target_t{h}", f"penman_pred_t{h}"
        if target_col not in all_data.columns:
            continue

        block_cols = ["date", "sensor", target_col]
        if pred_col in all_data.columns:
            block_cols.append(pred_col)
        for event_col in ("has_rain", "has_irr"):
            if event_col in all_data.columns:
                block_cols.append(event_col)
        block = all_data[block_cols].copy()
        block = block.rename(columns={target_col: "y_true"})
        block["y_pred"] = block[pred_col] if pred_col in block.columns else np.nan
        block = block.dropna(subset=["y_true"])

        block["split"] = [split_lookup.get((sensor, date)) for sensor, date in zip(block["sensor"], block["date"])]
        block = block[block["split"].notna()]

        block["model"] = MODEL_NAME
        block["horizon"] = f"t+{h}"
        block["residual"] = block["y_pred"] - block["y_true"]
        block["y_true"] = block["y_true"].round(4)
        block["y_pred"] = block["y_pred"].round(4)
        block["residual"] = block["residual"].round(4)
        if "has_rain" not in block.columns:
            block["has_rain"] = False
        if "has_irr" not in block.columns:
            block["has_irr"] = False

        horizon_blocks.append(block[["date", "sensor", "model", "horizon", "split",
                                     "y_true", "y_pred", "residual", "has_rain", "has_irr"]])

    if not horizon_blocks:
        return pd.DataFrame()
    return pd.concat(horizon_blocks, ignore_index=True)


def compute_metrics(preds_df):
    """MAE/RMSE/R2 complessivi + MAE nei giorni con evento (pioggia o
    irrigazione, ev_) e nei giorni di "discesa" senza evento (dc_) — vedi
    dati_ricostruiti_util.simplified_metrics."""
    rows = []
    for split in ("train", "val", "test"):
        split_df = preds_df[preds_df["split"] == split]

        for h in HORIZONS:
            h_label = f"t+{h}"
            sub = split_df[(split_df["horizon"] == h_label) & split_df["y_pred"].notna()]
            if sub.empty:
                continue
            ev_mask = (sub["has_rain"].fillna(False) | sub["has_irr"].fillna(False)).values
            dc_mask = ~ev_mask
            m = simplified_metrics(sub["y_true"].values, sub["y_pred"].values, ev_mask=ev_mask, dc_mask=dc_mask)
            rows.append({"model": MODEL_NAME, "horizon": h_label, "split": split, **m})

        sub_all = split_df[split_df["y_pred"].notna()]
        if not sub_all.empty:
            ev_mask_all = (sub_all["has_rain"].fillna(False) | sub_all["has_irr"].fillna(False)).values
            dc_mask_all = ~ev_mask_all
            m = simplified_metrics(sub_all["y_true"].values, sub_all["y_pred"].values,
                                   ev_mask=ev_mask_all, dc_mask=dc_mask_all)
            rows.append({"model": MODEL_NAME, "horizon": "ALL", "split": split, **m})

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--sensors", default=DEFAULT_SENSORS)
    parser.add_argument("--colture", default=DEFAULT_COLTURE)
    parser.add_argument("--irr", default=DEFAULT_IRRIGAZIONE_RICOSTRUITA)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--alpha-rain", type=float, default=0.50)
    parser.add_argument("--alpha-irr", type=float, default=0.70)
    parser.add_argument("--fc", type=float, default=None)
    parser.add_argument("--wp", type=float, default=None)
    parser.add_argument("--sat", type=float, default=SAT_OVERRIDE)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--sensors-filter", nargs="+", default=None)
    args = parser.parse_args()
    require_data_files(args.dataset, args.sensors, args.irr,
                       getattr(args, "colture", None))
    print(f"[dataset_aggiornato] dataset in uso: {os.path.abspath(args.dataset)}")

    os.makedirs(args.out, exist_ok=True)

    print("Carico configurazione agronomica (colture.csv, sensori-corretti.csv)...")
    colture_df = load_colture(args.colture)
    colture_df = apply_soil_overrides(colture_df, fc=args.fc, wp=args.wp, sat=args.sat)
    sensori_df = load_sensori(args.sensors)

    print("\nCostruisco i dataset per sensore (irrigazione ricostruita)...")
    datasets = build_all_datasets_with_reconstructed_irrigation(
        args.dataset, args.sensors, args.irr, sensors_filter=args.sensors_filter)

    print("\nCostruisco lo split canonico condiviso (stesso di multioutput_ml/ "
          "e penman_residual_componibile/)...")
    canonical_split = build_canonical_split(datasets, TARGET_COLS, args.train_frac, args.val_frac)
    if canonical_split.empty:
        raise RuntimeError("Split canonico vuoto: controlla i dati/sensori in ingresso.")

    split_lookup = {
        (row["device"], pd.Timestamp(row["date"]).strftime("%Y-%m-%d")): row["_split"]
        for _, row in canonical_split[["device", "date", "_split"]].iterrows()
    }
    n_train = int((canonical_split["_split"] == "train").sum())
    n_val = int((canonical_split["_split"] == "val").sum())
    n_test = int((canonical_split["_split"] == "test").sum())
    print(f"  n_train={n_train} n_val={n_val} n_test={n_test}")

    split_composition = compute_split_composition(canonical_split)

    print(f"\nSimulo Penman-Monteith iterativo t+1..t+{max(HORIZONS)}...")
    datasets_penman = enrich_all_with_penman(datasets, sensori_df, colture_df, args.alpha_rain, args.alpha_irr,
                                             observed_rain_col=OBSERVED_RAIN_COL)
    n_ok = sum(1 for ds in datasets_penman.values() if "penman_pred_t1" in ds.columns
              and ds["penman_pred_t1"].notna().any())
    print(f"  Sensori con predizioni Penman: {n_ok}/{len(datasets_penman)}")

    preds_df = build_predictions_long(datasets_penman, split_lookup)
    metrics_df = compute_metrics(preds_df)

    mm_path = os.path.join(args.out, "penman_metrics.csv")
    mp_path = os.path.join(args.out, "penman_preds.csv")
    sc_path = os.path.join(args.out, "split_composition.csv")
    metrics_df.to_csv(mm_path, index=False)
    preds_df.to_csv(mp_path, index=False)
    split_composition.to_csv(sc_path, index=False)

    print(f"\nSalvato: {mm_path} ({len(metrics_df)} righe)")
    print(f"Salvato: {mp_path} ({len(preds_df)} righe)")
    print(f"Salvato: {sc_path} ({len(split_composition)} righe)")

    print("\nCopertura Penman sull'insieme canonico, per split (t+1, quanti "
          "giorni canonici hanno una previsione Penman VALIDA — la catena "
          "fisica non arriva ovunque, mostrato in modo trasparente):")
    for split in ("train", "val", "test"):
        n_canon = int((canonical_split["_split"] == split).sum())
        n_valid_t1 = preds_df[(preds_df["split"] == split) & (preds_df["horizon"] == "t+1") &
                              preds_df["y_pred"].notna()].shape[0]
        print(f"  {split:5s}: {n_valid_t1}/{n_canon} giorni")


if __name__ == "__main__":
    main()
