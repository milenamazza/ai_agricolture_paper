# ─────────────────────────────────────────────────────────────────────────────
# COPIA in pipeline_corrette/dataset_aggiornato/ di pipeline_corrette/penman_residual/run_residual_solo_feature.py.
# Identica all'originale tranne tre punti: CORE sale di un livello (la copia
# sta una cartella più in basso), il dataset di default è
# paths.DATASET_AGGIORNATO (quello rigenerato con i periodi 2026 dei sensori
# 3, 4, 8, 20) invece di DEFAULT_DATASET, e il dataset in uso viene stampato
# all'avvio. I risultati finiscono nelle results*/ di QUESTA cartella, separati
# da quelli sul dataset vecchio.
# ─────────────────────────────────────────────────────────────────────────────

"""
run_penman_residual_componibile_solo_feature.py
─────────────────────────────────────────────────────────────────────────────
Come run_penman_residual_componibile.py (Ridge/RF/MLP/LSTM predicono il
residuo target_t{h}-penman_pred_t{h} per tutti e 7 gli orizzonti insieme —
vedi quel file per la spiegazione completa), ma le feature ML sono SOLO
quelle componibili (feature_groups_util) — NESSUNA colonna fisica Penman
(penman_phys_cols) né penman_residual_lag1 come input al modello.

Perché: le 3 pipeline di incertezza sul residuo
(uncertainty_residual_componibile, deep_ensemble_residual_componibile,
quantile_regression_residual_componibile) usano solo feature componibili
per scelta esplicita — Penman entra lì solo per costruire il target e
ricostruire la previsione assoluta, mai come input. Questa variante allinea
Ridge/RF/MLP/LSTM allo stesso principio, per restare comparabili: stesse
feature, stesso identico sotto-insieme (sensore, giorno).

Per la stessa ragione di comparabilità, l'arricchimento Penman qui usa
enrich_all_with_penman (la fisica "base" di penman_baseline/run_penman.py,
la STESSA già usata nelle 3 pipeline di incertezza sul residuo) invece di
enrich_all_with_penman_ext (la fisica "estesa" di modelli singoli/
penman_residual/run_penman_residual.py, pensata apposta per fornire colonne
aggiuntive come feature — qui non più necessaria dato che quelle colonne non
sono più usate come feature). Risultato: stesso identico penman_pred_t{h} e
quindi stesso identico sotto-insieme filtrato delle 3 pipeline di incertezza
sul residuo.

Riuso in sola lettura (nessuna modifica ai file sorgenti):
  - penman_baseline/run_penman.py: enrich_all_with_penman — fisica
    INVARIATA, stessa già riusata nelle 3 pipeline di incertezza sul
    residuo.
  - data_driven_ml_multioutput/deep_models_multi.py: import in sola
    lettura, INVARIATO.
  - modelli singoli/data_driven_ml/utils.py,run_experiments.py: utility di
    base, DEFAULT_DATASET, DEFAULT_SENSORS, TRAIN_FRAC, VAL_FRAC.
  - ai_feature_componibile/{feature_groups_util.py,dati_ricostruiti_util.py}.

Output in results_solo_feature/ (cartella separata da results/, per non
confondere con la versione "con più feature"):
  penman_residual_solo_feature_metrics.csv
  penman_residual_solo_feature_preds.csv
  penman_residual_solo_feature_hyperparameters.csv
  split_composition.csv

Utilizzo:
    python -u run_penman_residual_componibile_solo_feature.py
    python -u run_penman_residual_componibile_solo_feature.py --sensors-filter EM-500-9 EM-500-12 --fsets ACQUA TERRA
"""

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
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
CORE = os.path.join(HERE, "..", "..", "core")         # pipeline_corrette/core

sys.path.insert(0, CORE)

from feature_groups import THEMATIC_FEATURE_SETS, compose_feature_set  # noqa: E402
from dataset_builder import (  # noqa: E402
    build_all_datasets_with_reconstructed_irrigation,
    build_canonical_split,
    compute_split_composition,
    simplified_metrics,
)

from data_utils import (  # noqa: E402
    MIN_SAMPLES, SEED, HORIZONS, SAT_OVERRIDE,
    load_colture, load_sensori, apply_soil_overrides,
    _make_sample_weights, _tune_val,
    GRID_RIDGE, GRID_RF,
)
from paths import (DATASET_AGGIORNATO as DEFAULT_DATASET, DEFAULT_SENSORS,  # noqa: E402
                   DEFAULT_IRRIGAZIONE_RICOSTRUITA, DEFAULT_COLTURE,
                   TRAIN_FRAC, VAL_FRAC, require_data_files)
from penman import enrich_all_with_penman  # noqa: E402  (import in sola lettura, invariato)
import deep_models_multi as dmm  # noqa: E402  (import in sola lettura, invariato)

OUT_DIR = os.path.join(HERE, "results_solo_feature")

EVENT_WEIGHT = 3
LOOKBACK = 14
TARGET_COLS = [f"target_t{h}" for h in HORIZONS]
PRED_COLS = [f"penman_pred_t{h}" for h in HORIZONS]
FSETS_DEFAULT = list(THEMATIC_FEATURE_SETS.keys())

NN_FIXED  = {"batch_size": 32, "patience": 50, "epochs": 300}
GRID_MLP  = {"hidden": [(32,), (64, 32), (128, 64, 32), (32, 32)], "lr": [1e-2, 1e-3, 5e-4, 1e-4]}
GRID_LSTM = {"hidden": [16, 32, 64], "lr": [1e-3, 5e-4]}


# ═══════════════════════════════════════════════════════════════════════════
# Feature: SOLO ML componibili, nessuna colonna fisica Penman
# ═══════════════════════════════════════════════════════════════════════════

def residual_columns_for(fset_name):
    """Solo le feature ML del gruppo componibile (su tutti gli orizzonti) —
    a differenza di run_penman_residual_componibile.py, NESSUNA colonna
    fisica Penman né penman_residual_lag1: Penman entra solo a costruire il
    target e a ricostruire la previsione assoluta, mai come feature."""
    group_names = THEMATIC_FEATURE_SETS[fset_name]
    ml_columns = set()
    for h in HORIZONS:
        ml_columns.update(compose_feature_set(*group_names, horizon=h))
    return sorted(ml_columns)


def impute3(X_tr, X_val, X_te):
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X_tr), imputer.transform(X_val), imputer.transform(X_te)


def prep_residual_multi(df_split, feature_columns, target_columns, pred_columns):
    """Scarta le righe senza tutti i penman_pred_t{h} validi (nessun dato
    inventato) — applicato allo STESSO split canonico, non un nuovo split.
    Ritorna X grezzo, residuo, target assoluto, predizione Penman, allineati."""
    sub = df_split.dropna(subset=target_columns + pred_columns).reset_index(drop=True)
    available = [c for c in feature_columns if c in sub.columns]
    X = sub[available].apply(pd.to_numeric, errors="coerce").values
    Y_true = sub[target_columns].values.astype(float)
    Y_penman = sub[pred_columns].values.astype(float)
    Y_res = Y_true - Y_penman
    return X, Y_res, Y_true, Y_penman, sub, available


# ═══════════════════════════════════════════════════════════════════════════
# Metriche: per orizzonte + complessivo ("ALL"), sulla previsione assoluta ricostruita
# ═══════════════════════════════════════════════════════════════════════════

def _record_multi(metric_rows, pred_rows, model_name, fset_name, split_name,
                  Y_true_abs, Y_pred_abs, ev_mask, dc_mask, dates, devices, event_weight):
    n_h = len(HORIZONS)
    for i, h in enumerate(HORIZONS):
        yt, yp = Y_true_abs[:, i], Y_pred_abs[:, i]
        m = simplified_metrics(yt, yp, ev_mask=ev_mask, dc_mask=dc_mask)
        metric_rows.append({"model": model_name, "feature_set": fset_name,
                            "horizon": f"t+{h}", "split": split_name,
                            "event_weight": event_weight, **m})
        for d, dev, a, b in zip(dates, devices, yt, yp):
            pred_rows.append({"date": d, "sensor": dev, "model": model_name,
                              "feature_set": fset_name, "horizon": f"t+{h}", "split": split_name,
                              "y_true": round(float(a), 4), "y_pred": round(float(b), 4),
                              "residual": round(float(b - a), 4)})

    yt_all, yp_all = Y_true_abs.ravel(), Y_pred_abs.ravel()
    ev_all = np.repeat(ev_mask, n_h) if ev_mask is not None else None
    dc_all = np.repeat(dc_mask, n_h) if dc_mask is not None else None
    m = simplified_metrics(yt_all, yp_all, ev_mask=ev_all, dc_mask=dc_all)
    metric_rows.append({"model": model_name, "feature_set": fset_name,
                        "horizon": "ALL", "split": split_name,
                        "event_weight": event_weight, **m})
    if split_name == "test":
        print(f"  {model_name:6s} ALL(7gg) [test]  MAE={m['MAE']:.4f}  R2={m['R2']:.4f}")


def _record_hparams(hparam_rows, model_name, fset_name, params, val_score, n_features):
    hparam_rows.append({
        "model": model_name, "feature_set": fset_name, "n_features": n_features,
        "val_MAE_res": round(float(val_score), 4) if val_score is not None else np.nan,
        "params": json.dumps(params, default=str, sort_keys=True),
    })


def predict_and_record_all_splits(metric_rows, pred_rows, model_name, fset_name,
                                  predict_res_fn, event_weight, split_infos):
    """
    split_infos: lista di tuple (split_name, X_imp, Y_penman, Y_true_abs,
    dates, devices, ev_mask, dc_mask) — una per train/val/test.
    predict_res_fn(X_imp) predice il RESIDUO; qui si ricostruisce la
    previsione assoluta (residuo + Penman) e si registra per ciascuno split.
    """
    for split_name, X_imp, Y_penman, Y_true_abs, dates, devices, ev_mask, dc_mask in split_infos:
        Y_pred_abs = predict_res_fn(X_imp) + Y_penman
        _record_multi(metric_rows, pred_rows, model_name, fset_name, split_name,
                     Y_true_abs, Y_pred_abs, ev_mask, dc_mask, dates, devices, event_weight)


# ═══════════════════════════════════════════════════════════════════════════
# Loop principale
# ═══════════════════════════════════════════════════════════════════════════

def run_experiments(datasets_penman, fsets, run_mlp=True, run_lstm=True,
                    event_weight=EVENT_WEIGHT, lookback=LOOKBACK,
                    train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    metric_rows, pred_rows, hparam_rows = [], [], []
    t_start = time.time()

    def elapsed_str():
        s = time.time() - t_start
        return f"{int(s // 60):02d}:{int(s % 60):02d}"

    print("\nCostruisco lo split canonico (una sola volta, sui dati già "
          "arricchiti con Penman — stesso identico insieme di multioutput_ml/ "
          "e penman_baseline_componibile/)...")
    canonical_split = build_canonical_split(datasets_penman, TARGET_COLS, train_frac, val_frac)
    if canonical_split.empty:
        raise RuntimeError("Split canonico vuoto: controlla i dati/sensori in ingresso.")

    train_split = canonical_split[canonical_split["_split"] == "train"]
    val_split = canonical_split[canonical_split["_split"] == "val"]
    test_split = canonical_split[canonical_split["_split"] == "test"]
    print(f"  n_train={len(train_split)} n_val={len(val_split)} n_test={len(test_split)}  "
          f"(canonico, prima del filtro Penman per feature-set)")

    split_composition = compute_split_composition(canonical_split)

    for fi, fset_name in enumerate(fsets, start=1):
        print(f"\n[{elapsed_str()}] [feature-set {fi}/{len(fsets)}: {fset_name}]", flush=True)

        feature_columns = residual_columns_for(fset_name)

        X_tr, Y_tr_res, Y_tr_abs, Y_tr_pm, tr_df, avail = prep_residual_multi(
            train_split, feature_columns, TARGET_COLS, PRED_COLS)
        X_val, Y_val_res, Y_val_abs, Y_val_pm, va_df, _ = prep_residual_multi(
            val_split, feature_columns, TARGET_COLS, PRED_COLS)
        X_te, Y_te_res, Y_te_abs, Y_te_pm, te_df, _ = prep_residual_multi(
            test_split, feature_columns, TARGET_COLS, PRED_COLS)

        if len(X_tr) < MIN_SAMPLES or len(X_val) < MIN_SAMPLES or len(X_te) < MIN_SAMPLES:
            print(f"  split insufficiente dopo filtro Penman (train={len(X_tr)}, "
                  f"val={len(X_val)}, test={len(X_te)}), salto.")
            continue

        X_tr_imp, X_val_imp, X_te_imp = impute3(X_tr, X_val, X_te)
        sw = _make_sample_weights(tr_df, event_weight)

        def _event_masks(split_df):
            has_rain = split_df.get("has_rain", pd.Series(False, index=split_df.index)).fillna(False).values.astype(bool)
            has_irr = split_df.get("has_irr", pd.Series(False, index=split_df.index)).fillna(False).values.astype(bool)
            return (has_rain | has_irr), ~(has_rain | has_irr)

        ev_mask_tr, dc_mask_tr = _event_masks(tr_df)
        ev_mask_val, dc_mask_val = _event_masks(va_df)
        ev_mask_te, dc_mask_te = _event_masks(te_df)

        print(f"  n_train={len(X_tr)} n_val={len(X_val)} n_test={len(X_te)}  n_features={len(avail)}  "
              f"(canonico: train={len(train_split)}/val={len(val_split)}/test={len(test_split)}, "
              f"ridotto dal filtro Penman — atteso, mostrato per trasparenza)")

        split_infos = [
            ("train", X_tr_imp, Y_tr_pm, Y_tr_abs, tr_df["date"].values, tr_df["device"].values, ev_mask_tr, dc_mask_tr),
            ("val", X_val_imp, Y_val_pm, Y_val_abs, va_df["date"].values, va_df["device"].values, ev_mask_val, dc_mask_val),
            ("test", X_te_imp, Y_te_pm, Y_te_abs, te_df["date"].values, te_df["device"].values, ev_mask_te, dc_mask_te),
        ]

        # --- Ridge (multi-output nativo, target = residuo) ---------------------
        model, best_params, val_mae_res = _tune_val(
            Ridge, GRID_RIDGE, X_tr_imp, Y_tr_res, X_val_imp, Y_val_res, sample_weight=sw)
        predict_and_record_all_splits(metric_rows, pred_rows, "Ridge", fset_name,
                                      model.predict, event_weight, split_infos)
        _record_hparams(hparam_rows, "Ridge", fset_name, best_params, val_mae_res, len(avail))

        # --- RF (multi-output nativo, target = residuo) -------------------------
        model, best_params, val_mae_res = _tune_val(
            RandomForestRegressor, GRID_RF, X_tr_imp, Y_tr_res, X_val_imp, Y_val_res,
            sample_weight=sw, random_state=SEED, n_jobs=-1)
        predict_and_record_all_splits(metric_rows, pred_rows, "RF", fset_name,
                                      model.predict, event_weight, split_infos)
        _record_hparams(hparam_rows, "RF", fset_name, best_params, val_mae_res, len(avail))

        # --- MLP multi-output (target = residuo) --------------------------------
        if run_mlp:
            (model, scaler, y_stats), best_params, val_mae_res = dmm.tune_mlp_multi_val(
                X_tr_imp, Y_tr_res, X_val_imp, Y_val_res, GRID_MLP, sample_weight=sw, nn_fixed=NN_FIXED)
            predict_res_fn = lambda X: dmm.predict_mlp_multi(model, scaler, y_stats, X)  # noqa: E731
            predict_and_record_all_splits(metric_rows, pred_rows, "MLP", fset_name,
                                          predict_res_fn, event_weight, split_infos)
            _record_hparams(hparam_rows, "MLP", fset_name, {**best_params, **NN_FIXED}, val_mae_res, len(avail))

        # --- LSTM multi-output (sequenze sul residuo, sullo stesso canonico) ---
        if run_lstm:
            pooled_seq = canonical_split.copy()
            res_cols = []
            for h in HORIZONS:
                rc = f"_residual_t{h}"
                pooled_seq[rc] = pooled_seq[f"target_t{h}"] - pooled_seq[f"penman_pred_t{h}"]
                res_cols.append(rc)

            X_seq, Y_seq_res, meta = dmm.build_sequences_multi(
                pooled_seq, avail, res_cols, split_col="_split", lookback=lookback,
                extra_cols=["has_rain", "has_irr"] + PRED_COLS + TARGET_COLS)
            if len(X_seq) > 0:
                train_mask = (meta["split"] == "train").values
                val_mask = (meta["split"] == "val").values
                test_mask = (meta["split"] == "test").values
                n_tr, n_va, n_te = train_mask.sum(), val_mask.sum(), test_mask.sum()
                if n_tr >= MIN_SAMPLES and n_va >= MIN_SAMPLES and n_te >= MIN_SAMPLES:
                    (model, scaler, y_stats), best_params, val_mae_res = dmm.tune_lstm_multi_val(
                        X_seq[train_mask], Y_seq_res[train_mask], X_seq[val_mask], Y_seq_res[val_mask],
                        GRID_LSTM, nn_fixed=NN_FIXED)

                    lstm_split_infos = []
                    for split_name, split_mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
                        meta_split = meta[split_mask].reset_index(drop=True)
                        Y_penman_split = meta_split[PRED_COLS].values.astype(float)
                        Y_true_abs_split = meta_split[TARGET_COLS].values.astype(float)
                        ev_seq = (meta_split.get("has_rain", pd.Series(False, index=meta_split.index)).fillna(False) |
                                 meta_split.get("has_irr", pd.Series(False, index=meta_split.index)).fillna(False)).values
                        dc_seq = ~ev_seq
                        lstm_split_infos.append((
                            split_name, X_seq[split_mask], Y_penman_split, Y_true_abs_split,
                            meta_split["date"].values, meta_split["device"].values, ev_seq, dc_seq,
                        ))

                    predict_res_fn = lambda X: dmm.predict_lstm_multi(model, scaler, y_stats, X)  # noqa: E731
                    predict_and_record_all_splits(metric_rows, pred_rows, "LSTM", fset_name,
                                                  predict_res_fn, event_weight, lstm_split_infos)
                    _record_hparams(hparam_rows, "LSTM", fset_name,
                                   {**best_params, **NN_FIXED, "lookback": lookback}, val_mae_res, len(avail))
                    print(f"  LSTM sequenze: train={n_tr} val={n_va} test={n_te}")
                else:
                    print(f"  LSTM: sequenze insufficienti (train={n_tr}, val={n_va}, test={n_te}), salto.")
            else:
                print("  LSTM: nessuna sequenza valida, salto.")

    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), pd.DataFrame(hparam_rows), split_composition


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
    parser.add_argument("--event-weight", type=float, default=EVENT_WEIGHT)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--fsets", nargs="+", default=FSETS_DEFAULT, choices=FSETS_DEFAULT)
    parser.add_argument("--sensors-filter", nargs="+", default=None)
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--no-lstm", action="store_true")
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

    print(f"\nSimulo Penman-Monteith iterativo t+1..t+{max(HORIZONS)} (fisica base, "
          f"nessuna colonna extra: qui Penman non entra come feature)...")
    datasets_penman = enrich_all_with_penman(datasets, sensori_df, colture_df, args.alpha_rain, args.alpha_irr)
    n_ok = sum(1 for ds in datasets_penman.values() if "penman_pred_t1" in ds.columns
              and ds["penman_pred_t1"].notna().any())
    print(f"  Sensori con predizioni Penman: {n_ok}/{len(datasets_penman)}")

    metrics_df, preds_df, hparams_df, split_composition = run_experiments(
        datasets_penman, fsets=args.fsets,
        run_mlp=not args.no_mlp, run_lstm=not args.no_lstm,
        event_weight=args.event_weight, lookback=args.lookback,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )

    mm_path = os.path.join(args.out, "penman_residual_solo_feature_metrics.csv")
    mp_path = os.path.join(args.out, "penman_residual_solo_feature_preds.csv")
    hp_path = os.path.join(args.out, "penman_residual_solo_feature_hyperparameters.csv")
    sc_path = os.path.join(args.out, "split_composition.csv")
    metrics_df.to_csv(mm_path, index=False)
    preds_df.to_csv(mp_path, index=False)
    hparams_df.to_csv(hp_path, index=False)
    split_composition.to_csv(sc_path, index=False)

    print(f"\nSalvato: {mm_path} ({len(metrics_df)} righe)")
    print(f"Salvato: {mp_path} ({len(preds_df)} righe)")
    print(f"Salvato: {hp_path} ({len(hparams_df)} righe)")
    print(f"Salvato: {sc_path} ({len(split_composition)} righe)")


if __name__ == "__main__":
    main()
