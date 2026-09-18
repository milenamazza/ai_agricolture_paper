"""
run_penman_residual_componibile_solo_feature_optuna.py
─────────────────────────────────────────────────────────────────────────────
Stessa pipeline di run_penman_residual_componibile_solo_feature.py (target =
residuo target_t{h}-penman_pred_t{h}, SOLO feature componibili, stesso split
canonico condiviso — vedi quel file per la spiegazione completa), ma la
ricerca degli iperparametri per TUTTI E QUATTRO i modelli (Ridge, RF, MLP,
LSTM) usa OPTUNA invece della griglia esaustiva ParameterGrid — stesso
identico principio di run_multioutput_componibile_comparabile_penman_optuna.py
(vedi quel file per la spiegazione dettagliata di ciascuno spazio di
ricerca): continuo dove l'asse della griglia è scalare/ordinale, categorico
dove è strutturale. Per MLP e LSTM, batch_size è anch'esso cercato da Optuna
(BATCH_SIZE_CHOICES) — nella griglia era fisso, qui si apre come dimensione
aggiuntiva. event_weight è anch'esso cercato da Optuna per Ridge/RF/MLP
(EVENT_WEIGHT_RANGE), sample_weight ricalcolato a ogni prova con
l'event_weight suggerito — LSTM resta l'eccezione (non usa sample_weight
nemmeno nella griglia originale, il valore registrato nei CSV per LSTM resta
quello fisso da CLI, solo come etichetta). Refit finale del modello vincente
con study.best_params dopo ogni study Optuna.

Unica differenza dalla ricerca Optuna "diretta": gli obiettivi Optuna qui
minimizzano la MAE sul RESIDUO (Y_val_res), non sul target assoluto — la
ricostruzione (pred_res + Y_penman) avviene DOPO, esattamente come nella
griglia, in predict_and_record_all_splits.

Riuso in sola lettura (nessuna modifica ai file sorgenti):
  - penman_baseline/run_penman.py: enrich_all_with_penman — fisica INVARIATA.
  - data_driven_ml_multioutput/deep_models_multi.py: import in sola lettura,
    INVARIATO — si usano fit_mlp_multi/predict_mlp_multi/fit_lstm_multi/
    predict_lstm_multi/build_sequences_multi direttamente.
  - modelli singoli/data_driven_ml/utils.py,run_experiments.py: utility di
    base, DEFAULT_DATASET, DEFAULT_SENSORS, TRAIN_FRAC, VAL_FRAC.
  - ai_feature_componibile/{feature_groups_util.py,dati_ricostruiti_util.py}.

Output in results_solo_feature_optuna/ (cartella separata, per non
confondere con la griglia esaustiva):
  penman_residual_solo_feature_optuna_metrics.csv
  penman_residual_solo_feature_optuna_preds.csv
  penman_residual_solo_feature_optuna_hyperparameters.csv
  split_composition.csv

Utilizzo:
    python -u run_penman_residual_componibile_solo_feature_optuna.py
    python -u run_penman_residual_componibile_solo_feature_optuna.py --sensors-filter EM-500-9 EM-500-12 --fsets ACQUA TERRA --no-lstm --n-trials 3
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
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)   # silenzia il log verboso di Optuna, si stampa un riassunto per conto nostro

HERE = os.path.dirname(__file__)
CORE = os.path.join(HERE, "..", "core")               # pipeline_corrette/core

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
    _make_sample_weights,
)
from paths import (DEFAULT_DATASET, DEFAULT_SENSORS,  # noqa: E402
                   DEFAULT_IRRIGAZIONE_RICOSTRUITA, DEFAULT_COLTURE,
                   TRAIN_FRAC, VAL_FRAC, require_data_files)
from penman import enrich_all_with_penman  # noqa: E402  (import in sola lettura, invariato)
import deep_models_multi as dmm  # noqa: E402  (import in sola lettura, invariato)

OUT_DIR = os.path.join(HERE, "results_solo_feature_optuna")

EVENT_WEIGHT = 3
LOOKBACK = 14
TARGET_COLS = [f"target_t{h}" for h in HORIZONS]
PRED_COLS = [f"penman_pred_t{h}" for h in HORIZONS]
FSETS_DEFAULT = list(THEMATIC_FEATURE_SETS.keys())

NN_FIXED = {"patience": 20, "epochs": 300}         # batch_size non più fisso: cercato da Optuna
N_TRIALS_DEFAULT = 100
OPTUNA_SEED = 42

# Spazio di ricerca Optuna — identico a
# run_multioutput_componibile_comparabile_penman_optuna.py (stessi bound
# delle griglie GRID_RIDGE/GRID_RF/GRID_MLP/GRID_LSTM di questo file), più
# batch_size (MLP/LSTM) come dimensione aggiuntiva non presente nella griglia.
RIDGE_ALPHA_RANGE = (0.01, 100.0)
RF_N_ESTIMATORS_RANGE = (100, 200)
RF_MAX_DEPTH_CHOICES = [4, 8, None]
RF_MIN_SAMPLES_LEAF_RANGE = (2, 5)
HIDDEN_CHOICES = {
    "32": (32,),
    "64_32": (64, 32),
    "128_64_32": (128, 64, 32),
    "32_32": (32, 32),
}
MLP_LR_RANGE = (1e-5, 1e-2)
LSTM_HIDDEN_RANGE = (16, 64)
LSTM_LR_RANGE = (5e-4, 1e-3)
BATCH_SIZE_CHOICES = [16, 32, 64, 128]             # nuovo asse (MLP/LSTM), stessa lista degli script di incertezza
EVENT_WEIGHT_RANGE = (1.0, 10.0)                   # nuovo asse (Ridge/RF/MLP), stessi bound del precedente di incertezza


# ═══════════════════════════════════════════════════════════════════════════
# Feature: SOLO ML componibili, nessuna colonna fisica Penman
# ═══════════════════════════════════════════════════════════════════════════

def residual_columns_for(fset_name):
    """Solo le feature ML del gruppo componibile (su tutti gli orizzonti) —
    NESSUNA colonna fisica Penman né penman_residual_lag1: Penman entra solo
    a costruire il target e a ricostruire la previsione assoluta, mai come
    feature."""
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
# Ricerca iperparametri con Optuna — un helper per modello, sostituisce
# _tune_val(...)/dmm.tune_mlp_multi_val(...)/dmm.tune_lstm_multi_val(...).
# Obiettivo: MAE sul RESIDUO (Y_val_res) — la ricostruzione assoluta avviene
# dopo, in predict_and_record_all_splits.
# ═══════════════════════════════════════════════════════════════════════════

def tune_ridge_optuna(X_tr, Y_tr_res, X_val, Y_val_res, train_df, n_trials):
    def objective(trial):
        alpha = trial.suggest_float("alpha", *RIDGE_ALPHA_RANGE, log=True)
        event_weight = trial.suggest_float("event_weight", *EVENT_WEIGHT_RANGE)
        sw = _make_sample_weights(train_df, event_weight)
        model = Ridge(alpha=alpha)
        model.fit(X_tr, Y_tr_res, sample_weight=sw)
        return float(mean_absolute_error(Y_val_res, model.predict(X_val)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    sw = _make_sample_weights(train_df, best_params["event_weight"])
    model = Ridge(alpha=best_params["alpha"])
    model.fit(X_tr, Y_tr_res, sample_weight=sw)
    return model, best_params, study.best_value


def tune_rf_optuna(X_tr, Y_tr_res, X_val, Y_val_res, train_df, n_trials):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", *RF_N_ESTIMATORS_RANGE)
        max_depth = trial.suggest_categorical("max_depth", RF_MAX_DEPTH_CHOICES)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", *RF_MIN_SAMPLES_LEAF_RANGE)
        event_weight = trial.suggest_float("event_weight", *EVENT_WEIGHT_RANGE)
        sw = _make_sample_weights(train_df, event_weight)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=SEED, n_jobs=-1)
        model.fit(X_tr, Y_tr_res, sample_weight=sw)
        return float(mean_absolute_error(Y_val_res, model.predict(X_val)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    sw = _make_sample_weights(train_df, best_params["event_weight"])
    model = RandomForestRegressor(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],
                                  min_samples_leaf=best_params["min_samples_leaf"],
                                  random_state=SEED, n_jobs=-1)
    model.fit(X_tr, Y_tr_res, sample_weight=sw)
    return model, best_params, study.best_value


def tune_mlp_optuna(X_tr, Y_tr_res, X_val, Y_val_res, train_df, n_trials):
    def objective(trial):
        hidden_key = trial.suggest_categorical("hidden", list(HIDDEN_CHOICES.keys()))
        lr = trial.suggest_float("lr", *MLP_LR_RANGE, log=True)
        batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
        event_weight = trial.suggest_float("event_weight", *EVENT_WEIGHT_RANGE)
        sw = _make_sample_weights(train_df, event_weight)
        model, scaler, y_stats = dmm.fit_mlp_multi(
            X_tr, Y_tr_res, X_val, Y_val_res, sample_weight=sw,
            hidden=HIDDEN_CHOICES[hidden_key], lr=lr, batch_size=batch_size, **NN_FIXED)
        pred = dmm.predict_mlp_multi(model, scaler, y_stats, X_val)
        return float(mean_absolute_error(Y_val_res, pred))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    hidden = HIDDEN_CHOICES[best_params["hidden"]]
    sw = _make_sample_weights(train_df, best_params["event_weight"])
    bundle = dmm.fit_mlp_multi(X_tr, Y_tr_res, X_val, Y_val_res, sample_weight=sw,
                               hidden=hidden, lr=best_params["lr"], batch_size=best_params["batch_size"], **NN_FIXED)
    clean_params = {"hidden": hidden, "lr": best_params["lr"], "batch_size": best_params["batch_size"],
                    "event_weight": best_params["event_weight"]}
    return bundle, clean_params, study.best_value


def tune_lstm_optuna(X_tr, Y_tr_res, X_val, Y_val_res, n_trials):
    def objective(trial):
        hidden = trial.suggest_int("hidden", *LSTM_HIDDEN_RANGE)
        lr = trial.suggest_float("lr", *LSTM_LR_RANGE, log=True)
        batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
        model, scaler, y_stats = dmm.fit_lstm_multi(
            X_tr, Y_tr_res, X_val, Y_val_res, hidden=hidden, lr=lr, batch_size=batch_size, **NN_FIXED)
        pred = dmm.predict_lstm_multi(model, scaler, y_stats, X_val)
        return float(mean_absolute_error(Y_val_res, pred))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    bundle = dmm.fit_lstm_multi(X_tr, Y_tr_res, X_val, Y_val_res,
                                hidden=best_params["hidden"], lr=best_params["lr"],
                                batch_size=best_params["batch_size"], **NN_FIXED)
    return bundle, best_params, study.best_value


# ═══════════════════════════════════════════════════════════════════════════
# Metriche: per orizzonte + complessivo ("ALL"), sulla previsione assoluta
# ricostruita — IDENTICA a run_penman_residual_componibile_solo_feature.py.
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

def run_experiments(datasets_penman, fsets, n_trials, run_mlp=True, run_lstm=True,
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
    print(f"  ricerca Optuna: {n_trials} prove per modello (Ridge/RF/MLP/LSTM), per feature-set")

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

        # --- Ridge (Optuna, target = residuo) -----------------------------------
        model, best_params, val_mae_res = tune_ridge_optuna(
            X_tr_imp, Y_tr_res, X_val_imp, Y_val_res, tr_df, n_trials)
        predict_and_record_all_splits(metric_rows, pred_rows, "Ridge", fset_name,
                                      model.predict, best_params["event_weight"], split_infos)
        _record_hparams(hparam_rows, "Ridge", fset_name, best_params, val_mae_res, len(avail))
        print(f"  Ridge  migliori iperparametri: {best_params}  (val_MAE_res={val_mae_res:.4f})")

        # --- RF (Optuna, target = residuo) ---------------------------------------
        model, best_params, val_mae_res = tune_rf_optuna(
            X_tr_imp, Y_tr_res, X_val_imp, Y_val_res, tr_df, n_trials)
        predict_and_record_all_splits(metric_rows, pred_rows, "RF", fset_name,
                                      model.predict, best_params["event_weight"], split_infos)
        _record_hparams(hparam_rows, "RF", fset_name, best_params, val_mae_res, len(avail))
        print(f"  RF     migliori iperparametri: {best_params}  (val_MAE_res={val_mae_res:.4f})")

        # --- MLP multi-output (Optuna, target = residuo) --------------------------
        if run_mlp:
            (model, scaler, y_stats), best_params, val_mae_res = tune_mlp_optuna(
                X_tr_imp, Y_tr_res, X_val_imp, Y_val_res, tr_df, n_trials)
            predict_res_fn = lambda X: dmm.predict_mlp_multi(model, scaler, y_stats, X)  # noqa: E731
            predict_and_record_all_splits(metric_rows, pred_rows, "MLP", fset_name,
                                          predict_res_fn, best_params["event_weight"], split_infos)
            _record_hparams(hparam_rows, "MLP", fset_name, {**best_params, **NN_FIXED}, val_mae_res, len(avail))
            print(f"  MLP    migliori iperparametri: {best_params}  (val_MAE_res={val_mae_res:.4f})")

        # --- LSTM multi-output (sequenze sul residuo, sullo stesso canonico, Optuna) ---
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
                    (model, scaler, y_stats), best_params, val_mae_res = tune_lstm_optuna(
                        X_seq[train_mask], Y_seq_res[train_mask], X_seq[val_mask], Y_seq_res[val_mask], n_trials)

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
                    print(f"  LSTM   migliori iperparametri: {best_params}  (val_MAE_res={val_mae_res:.4f})")
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
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                        help="Numero di prove Optuna per modello, per feature-set")
    args = parser.parse_args()
    require_data_files(args.dataset, args.sensors, args.irr,
                       getattr(args, "colture", None))

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
        datasets_penman, fsets=args.fsets, n_trials=args.n_trials,
        run_mlp=not args.no_mlp, run_lstm=not args.no_lstm,
        event_weight=args.event_weight, lookback=args.lookback,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )

    mm_path = os.path.join(args.out, "penman_residual_solo_feature_optuna_metrics.csv")
    mp_path = os.path.join(args.out, "penman_residual_solo_feature_optuna_preds.csv")
    hp_path = os.path.join(args.out, "penman_residual_solo_feature_optuna_hyperparameters.csv")
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
