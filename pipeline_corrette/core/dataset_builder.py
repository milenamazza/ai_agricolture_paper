"""
dati_ricostruiti_util.py
─────────────────────────────────────────────────────────────────────────────
Utility condivise dalle pipeline "componibili" di questa cartella
(multioutput_ml/, penman_baseline_componibile/, penman_residual_componibile/):

  - Caricamento dati con l'irrigazione RICOSTRUITA (log originale + eventi
    ricostruiti dall'analisi dei salti di umidità senza pioggia, vedi
    irrigazione_mancante/) invece del solo log originale.
  - Pool + split CANONICO: calcolato UNA VOLTA con questo modulo, riusato
    identico da tutte e tre le pipeline — è quello che garantisce (per
    costruzione, non per verifica a posteriori) che ML puro, Penman puro e
    Penman+residuo condividano esattamente lo stesso insieme di
    (sensore, giorno) e la stessa etichetta train/val/test.

Perché è possibile: il pool scarta le righe SOLO in base ai target
(target_t1..t7 tutti non-NaN), mai in base alle feature né alle colonne
fisiche Penman (che vengono aggiunte per arricchimento, dopo, senza
cambiare quali righe hanno target validi). L'insieme di (sensore, giorno)
selezionato non dipende quindi da quale feature-set si usa, né dal fatto
che i dati siano già stati arricchiti con Penman o no — motivo per cui lo
stesso identico split canonico si può calcolare una sola volta e riusare
ovunque, prima o dopo l'arricchimento Penman, con qualunque feature-set.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Caricamento dati con irrigazione ricostruita
# ═══════════════════════════════════════════════════════════════════════════

def load_irrigation_reconstructed(path: str) -> pd.DataFrame:
    """
    Carica irrigazione_mancante/irrigazione_ricostruita.csv — schema lungo
    device_name, date, irrigation_mm[, fonte], già nella forma che
    build_sensor_dataset si aspetta (lo stesso schema che produce
    load_irrigation_wide a partire dal vecchio file largo): nessuna
    conversione necessaria oltre al parsing della data. Contiene sia gli
    eventi originali (fonte=log_originale) sia quelli ricostruiti
    dall'analisi dei salti di umidità senza pioggia (fonte=ricostruita).
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["device_name", "date", "irrigation_mm"])

    irrigation = pd.read_csv(path)
    irrigation["date"] = pd.to_datetime(irrigation["date"]).dt.normalize()
    # Somma eventuali righe duplicate sullo stesso (sensore, giorno) invece
    # di scartarle silenziosamente.
    irrigation = (irrigation.groupby(["device_name", "date"], as_index=False)["irrigation_mm"]
                  .sum())
    return irrigation


def build_all_datasets_with_reconstructed_irrigation(dataset_path, sensors_csv, irr_path,
                                                      sensors_filter=None):
    """
    Stessa identica struttura di
    modelli singoli/data_driven_ml/run_experiments.py::build_all_datasets_clean,
    con un solo cambiamento: il caricamento dell'irrigazione usa
    load_irrigation_reconstructed invece di load_irrigation_wide, per
    includere gli eventi ricostruiti. Le funzioni di basso livello
    (load_and_clean, pivot_em500, pivot_ws, ecc.) sono importate in sola
    lettura dal chiamante e passate qui per evitare un giro di sys.path
    aggiuntivo in questo modulo condiviso.
    """
    from load_clean_dataset import load_and_clean
    from data_utils import (
        MIN_SAMPLES, pivot_em500, pivot_ws, pivot_forecast,
        build_sensor_ws_map, build_sensor_dataset,
        compute_ws_daily_stats, add_hourly_forecast, add_derived_features,
        compute_daily_median_rain, add_event_flags,
        build_crop_map, add_crop_feature,
    )

    df = load_and_clean(dataset_path)

    print("\nPivot EM-500 / WS / forecast...")
    em_pivot = pivot_em500(df)
    ws_pivot = pivot_ws(df)
    fc_pivots = pivot_forecast(df)

    irrigation = load_irrigation_reconstructed(irr_path)
    print(f"  Irrigazioni (originali + ricostruite): {len(irrigation)} eventi")

    ws_map = build_sensor_ws_map(df, None)

    sensors = sorted(em_pivot["device"].unique())
    if sensors_filter:
        sensors = [s for s in sensors if s in sensors_filter]

    datasets = {}
    for sensor in sensors:
        periods = ws_map.get(sensor, [])
        ds = build_sensor_dataset(sensor, em_pivot, ws_pivot, periods, fc_pivots, irrigation)
        if "target_t1" not in ds.columns or ds["target_t1"].notna().sum() < MIN_SAMPLES:
            continue
        ds["device"] = sensor
        datasets[sensor] = ds

    print(f"Dataset base pronti: {len(datasets)} sensori")

    print("Statistiche WS, forecast orario, feature derivate, eventi, crop encoding...")
    ws_stats = compute_ws_daily_stats(df)
    rain_med = compute_daily_median_rain(df)
    datasets = add_hourly_forecast(datasets, df)
    datasets = add_derived_features(datasets, ws_stats, {s: list(p) for s, p in ws_map.items()})
    if not rain_med.empty:
        for sensor in datasets:
            datasets[sensor] = datasets[sensor].merge(rain_med, on="date", how="left")
            datasets[sensor] = add_event_flags(datasets[sensor])
    else:
        for sensor in datasets:
            datasets[sensor] = add_event_flags(datasets[sensor])

    if sensors_csv and os.path.exists(sensors_csv):
        crop_map = build_crop_map(sensors_csv)
        datasets = add_crop_feature(datasets, crop_map)

    return datasets


# ═══════════════════════════════════════════════════════════════════════════
# Pool + split CANONICO (senza restringere le colonne — riusabile da
# qualunque pipeline a valle, con o senza colonne fisiche Penman)
# ═══════════════════════════════════════════════════════════════════════════

def pool_all_columns_multi(datasets, target_columns):
    """
    Come pool_datasets_multi (usato nelle altre pipeline "multioutput" del
    progetto) ma SENZA restringere le colonne a un feature-set specifico:
    mantiene TUTTE le colonne disponibili in ciascun dataset per-sensore
    (feature ML, colonne fisiche Penman se già presenti, flag evento...).
    Scarta le righe solo se non tutti i target_columns sono non-NaN — mai in
    base alle feature. Questo è ciò che rende possibile calcolare lo split
    canonico una sola volta e riusarlo da qualunque pipeline a valle,
    qualunque sia il sotto-insieme di colonne che le serve.
    """
    pooled_frames = []
    for sensor_name, sensor_df in datasets.items():
        if any(target_col not in sensor_df.columns for target_col in target_columns):
            continue
        sensor_subset = sensor_df.dropna(subset=target_columns).copy()
        if "device" not in sensor_subset.columns:
            sensor_subset["device"] = sensor_name
        if not sensor_subset.empty:
            pooled_frames.append(sensor_subset)

    if not pooled_frames:
        return pd.DataFrame()

    return pd.concat(pooled_frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def build_canonical_split(datasets, target_columns, train_frac, val_frac):
    """
    Pool + split UNA VOLTA SOLA, su TUTTE le colonne disponibili. Ritorna un
    unico DataFrame con una colonna '_split' in {'train','val','test'} —
    stesso schema già in uso nelle altre pipeline del progetto
    (build_pooled_split). Questo è l'oggetto che va CONDIVISO (non
    ricalcolato) da tutte le pipeline "componibili": ogni feature-set/
    modello seleziona le proprie colonne da questo stesso DataFrame già
    diviso, mai un nuovo pool/split.
    """
    from data_utils import sensor_train_val_test

    pooled = pool_all_columns_multi(datasets, target_columns)
    if pooled.empty:
        return pooled

    train_df, val_df, test_df = sensor_train_val_test(pooled, train_frac=train_frac, val_frac=val_frac)
    train_df = train_df.copy(); train_df["_split"] = "train"
    val_df = val_df.copy(); val_df["_split"] = "val"
    test_df = test_df.copy(); test_df["_split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values("date").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Composizione pioggia/irrigazione per split (train/val/test, aggregato + per sensore)
# ═══════════════════════════════════════════════════════════════════════════

def _composition_row(split_name, sensor_label, df_slice):
    n_days = len(df_slice)
    n_rain = int(df_slice.get("has_rain", pd.Series(False, index=df_slice.index)).fillna(False).sum())
    n_irr = int(df_slice.get("has_irr", pd.Series(False, index=df_slice.index)).fillna(False).sum())
    n_water = int(df_slice.get("has_water_event", pd.Series(False, index=df_slice.index)).fillna(False).sum())
    return {
        "split": split_name, "sensore": sensor_label, "n_giorni": n_days,
        "n_giorni_pioggia": n_rain, "pct_pioggia": round(100 * n_rain / n_days, 2) if n_days else 0.0,
        "n_giorni_irrigazione": n_irr, "pct_irrigazione": round(100 * n_irr / n_days, 2) if n_days else 0.0,
        "n_giorni_evento_acqua": n_water, "pct_evento_acqua": round(100 * n_water / n_days, 2) if n_days else 0.0,
    }


def compute_split_composition(canonical_split_df):
    """canonical_split_df: il DataFrame ritornato da build_canonical_split
    (con colonna '_split'). Ritorna righe aggregate + per sensore per
    ciascuna porzione train/val/test."""
    rows = []
    for split_name in ("train", "val", "test"):
        split_df = canonical_split_df[canonical_split_df["_split"] == split_name]
        rows.append(_composition_row(split_name, "TUTTI", split_df))
        for sensor_name, sensor_group in split_df.groupby("device"):
            rows.append(_composition_row(split_name, sensor_name, sensor_group))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Metriche snelle: solo MAE/RMSE/R2 complessivi + MAE nei giorni con evento
# (pioggia o irrigazione) e nei giorni di "discesa" (senza evento) — un
# sottoinsieme leggibile delle tante metriche di utils._metric_dict (che
# resta invariata, condivisa da tutto il resto del progetto: qui si affianca
# una versione più snella, non la si sostituisce ovunque).
# ═══════════════════════════════════════════════════════════════════════════

def simplified_metrics(y_true, y_pred, ev_mask=None, dc_mask=None):
    """
    y_true, y_pred: array 1D della stessa lunghezza.
    ev_mask: booleano, True nei giorni con pioggia o irrigazione (evento).
    dc_mask: booleano, True nei giorni senza ("discesa" — drying/dry conditions).
    Entrambi opzionali e allineati per indice a y_true/y_pred (stessa
    lunghezza, PRIMA di scartare i NaN — l'allineamento con i NaN scartati
    viene gestito qui dentro).

    Ritorna un dizionario piatto: n, MAE, RMSE, R2 (+ ev_n, ev_MAE se
    ev_mask fornito; dc_n, dc_MAE se dc_mask fornito).
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[valid], y_pred[valid]

    result = {"n": int(len(yt))}
    if len(yt) >= 2:
        result["MAE"] = round(float(mean_absolute_error(yt, yp)), 4)
        result["RMSE"] = round(float(np.sqrt(mean_squared_error(yt, yp))), 4)
        result["R2"] = round(float(r2_score(yt, yp)), 4)
    else:
        result["MAE"] = result["RMSE"] = result["R2"] = float("nan")

    if ev_mask is not None:
        ev_valid = np.asarray(ev_mask)[valid]
        yt_ev, yp_ev = yt[ev_valid], yp[ev_valid]
        result["ev_n"] = int(len(yt_ev))
        result["ev_MAE"] = round(float(mean_absolute_error(yt_ev, yp_ev)), 4) if len(yt_ev) > 0 else float("nan")

    if dc_mask is not None:
        dc_valid = np.asarray(dc_mask)[valid]
        yt_dc, yp_dc = yt[dc_valid], yp[dc_valid]
        result["dc_n"] = int(len(yt_dc))
        result["dc_MAE"] = round(float(mean_absolute_error(yt_dc, yp_dc)), 4) if len(yt_dc) > 0 else float("nan")

    return result
