# ─────────────────────────────────────────────────────────────────────────────
# COPIA in pipeline_corrette/dataset_aggiornato/ di pipeline_corrette/multioutput/run_multioutput_optuna.py.
# Identica all'originale tranne tre punti: CORE sale di un livello (la copia
# sta una cartella più in basso), il dataset di default è
# paths.DATASET_AGGIORNATO (quello rigenerato con i periodi 2026 dei sensori
# 3, 4, 8, 20) invece di DEFAULT_DATASET, e il dataset in uso viene stampato
# all'avvio. I risultati finiscono nelle results*/ di QUESTA cartella, separati
# da quelli sul dataset vecchio.
# ─────────────────────────────────────────────────────────────────────────────

"""
run_multioutput_componibile_comparabile_penman_optuna.py
─────────────────────────────────────────────────────────────────────────────
Stessa pipeline di run_multioutput_componibile_comparabile_penman.py (stesse
feature componibili, stesso filtro di comparabilità con Penman, stesso split
canonico condiviso — vedi quel file per la spiegazione completa), ma la
ricerca degli iperparametri per TUTTI E QUATTRO i modelli (Ridge, RF, MLP,
LSTM) usa OPTUNA invece della griglia esaustiva ParameterGrid.

Perché: prima volta in questo progetto che Optuna viene applicato a
Ridge/RF (i 6 script *_optuna.py già esistenti tunano tutti una sola
architettura neurale — MLP Gaussiano o Quantile). Qui, per restare fedeli e
prevedibili, la ricerca Optuna replica gli assi già esplorati dalla
rispettiva griglia esistente (GRID_RIDGE, GRID_RF, GRID_MLP, GRID_LSTM):
continua dove l'asse è scalare/ordinale (alpha, n_estimators,
min_samples_leaf, lr, hidden LSTM), categorica dove è strutturale (max_depth
di RF include None; hidden di MLP è la FORMA della rete, non un numero
singolo). In più, per MLP e LSTM, anche batch_size è cercato da Optuna
(BATCH_SIZE_CHOICES, stessa lista di valori già usata negli script di
incertezza) — nella griglia era fisso, qui si apre come dimensione
aggiuntiva su richiesta esplicita. Uno study Optuna indipendente per
(modello, feature-set), stesso TPESampler(seed=42) degli altri script
Optuna del progetto; ogni prova allena su train e valuta MAE su val, poi si
riallena UNA volta il modello vincente con study.best_params (stesso schema
di refit finale già usato per MLP/LSTM negli script di incertezza,
qui esteso anche a Ridge/RF).

event_weight è anch'esso cercato da Optuna per Ridge/RF/MLP (EVENT_WEIGHT_RANGE,
stessi bound del precedente di incertezza) — sample_weight ricalcolato a
ogni prova con l'event_weight suggerito in quella prova, refit finale con
il valore vincente. LSTM resta l'eccezione: non usa sample_weight nemmeno
nella griglia originale (dmm.tune_lstm_multi_val non lo riceve), quindi non
ha un event_weight da cercare — per LSTM il valore registrato nei CSV resta
quello fisso da CLI (--event-weight, default 3), solo come etichetta,
esattamente come nella griglia.

Comparabilità con Penman: IDENTICA a run_multioutput_componibile_comparabile_penman.py
— training/valutazione sul sotto-insieme del canonico dove Penman-Monteith
produce una previsione valida per TUTTI i 7 orizzonti.

Riuso in sola lettura (nessuna modifica ai file sorgenti):
  - penman_baseline/run_penman.py: enrich_all_with_penman — fisica INVARIATA.
  - data_driven_ml_multioutput/deep_models_multi.py: import in sola lettura,
    INVARIATO — si usano fit_mlp_multi/predict_mlp_multi/fit_lstm_multi/
    predict_lstm_multi/build_sequences_multi direttamente (non
    tune_mlp_multi_val/tune_lstm_multi_val, che fanno ParameterGrid).
  - modelli singoli/data_driven_ml/utils.py: MIN_SAMPLES, SEED, HORIZONS,
    SAT_OVERRIDE, load_colture, load_sensori, apply_soil_overrides,
    _make_sample_weights, GRID_RIDGE, GRID_RF (solo per leggerne i bound).
  - modelli singoli/data_driven_ml/run_experiments.py: DEFAULT_DATASET,
    DEFAULT_SENSORS, TRAIN_FRAC, VAL_FRAC.
  - ai_feature_componibile/{feature_groups_util.py,dati_ricostruiti_util.py}.

Output in results_comparabile_penman_optuna/ (cartella separata, per non
confondere con la griglia esaustiva):
  experiments_comparabile_penman_optuna_metrics.csv
  experiments_comparabile_penman_optuna_preds.csv
  experiments_comparabile_penman_optuna_hyperparameters.csv
  split_composition.csv

Utilizzo:
    python -u run_multioutput_componibile_comparabile_penman_optuna.py
    python -u run_multioutput_componibile_comparabile_penman_optuna.py --sensors-filter EM-500-9 EM-500-12 --fsets ACQUA TERRA --no-lstm --n-trials 3
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
    _make_sample_weights,
)
from paths import (DATASET_AGGIORNATO as DEFAULT_DATASET, DEFAULT_SENSORS,  # noqa: E402
                   DEFAULT_IRRIGAZIONE_RICOSTRUITA, DEFAULT_COLTURE,
                   TRAIN_FRAC, VAL_FRAC, require_data_files)
from penman import enrich_all_with_penman  # noqa: E402  (import in sola lettura, invariato)
import deep_models_multi as dmm  # noqa: E402  (import in sola lettura, invariato)

OUT_DIR = os.path.join(HERE, "results_comparabile_penman_optuna")

EVENT_WEIGHT = 3
LOOKBACK = 14
TARGET_COLS = [f"target_t{h}" for h in HORIZONS]
PENMAN_PRED_COLS = [f"penman_pred_t{h}" for h in HORIZONS]
FSETS_DEFAULT = list(THEMATIC_FEATURE_SETS.keys())

NN_FIXED = {"patience": 50, "epochs": 300}         # batch_size non più fisso: cercato da Optuna
N_TRIALS_DEFAULT = 20
OPTUNA_SEED = 42

# Spazio di ricerca Optuna — replica gli assi delle griglie esistenti
# (GRID_RIDGE/GRID_RF/GRID_MLP/GRID_LSTM), continui dove l'asse è
# scalare/ordinale, categorici dove è strutturale, più batch_size (MLP/LSTM)
# come dimensione aggiuntiva non presente nella griglia originale.
RIDGE_ALPHA_RANGE = (0.01, 100.0)                 # log-uniforme, stessi bound di GRID_RIDGE
RF_N_ESTIMATORS_RANGE = (100, 200)                # stessi bound di GRID_RF
RF_MAX_DEPTH_CHOICES = [4, 8, None]                # resta categorico (None incluso)
RF_MIN_SAMPLES_LEAF_RANGE = (2, 5)                # stessi bound di GRID_RF
HIDDEN_CHOICES = {                                 # stesse 4 forme di GRID_MLP
    "32": (32,),
    "64_32": (64, 32),
    "128_64_32": (128, 64, 32),
    "32_32": (32, 32),
}
MLP_LR_RANGE = (1e-4, 1e-2)                        # log-uniforme, copre GRID_MLP
LSTM_HIDDEN_RANGE = (16, 64)                       # int continuo, copre GRID_LSTM
LSTM_LR_RANGE = (5e-4, 1e-3)                       # log-uniforme, copre GRID_LSTM
BATCH_SIZE_CHOICES = [16, 32, 64, 128]             # nuovo asse (MLP/LSTM), stessa lista degli script di incertezza
EVENT_WEIGHT_RANGE = (1.0, 10.0)                   # nuovo asse (Ridge/RF/MLP), stessi bound del precedente di incertezza


def impute_train_val_test(X_train, X_val, X_test):
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X_train), imputer.transform(X_val), imputer.transform(X_test)


# ═══════════════════════════════════════════════════════════════════════════
# Ricerca iperparametri con Optuna — un helper per modello, sostituisce
# _tune_val(...)/dmm.tune_mlp_multi_val(...)/dmm.tune_lstm_multi_val(...)
# ═══════════════════════════════════════════════════════════════════════════

def tune_ridge_optuna(X_tr, Y_tr, X_val, Y_val, train_df, n_trials):
    def objective(trial):
        alpha = trial.suggest_float("alpha", *RIDGE_ALPHA_RANGE, log=True)
        event_weight = trial.suggest_float("event_weight", *EVENT_WEIGHT_RANGE)
        sw = _make_sample_weights(train_df, event_weight)
        model = Ridge(alpha=alpha)
        model.fit(X_tr, Y_tr, sample_weight=sw)
        return float(mean_absolute_error(Y_val, model.predict(X_val)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    sw = _make_sample_weights(train_df, best_params["event_weight"])
    model = Ridge(alpha=best_params["alpha"])
    model.fit(X_tr, Y_tr, sample_weight=sw)
    return model, best_params, study.best_value


def tune_rf_optuna(X_tr, Y_tr, X_val, Y_val, train_df, n_trials):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", *RF_N_ESTIMATORS_RANGE)
        max_depth = trial.suggest_categorical("max_depth", RF_MAX_DEPTH_CHOICES)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", *RF_MIN_SAMPLES_LEAF_RANGE)
        event_weight = trial.suggest_float("event_weight", *EVENT_WEIGHT_RANGE)
        sw = _make_sample_weights(train_df, event_weight)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=SEED, n_jobs=-1)
        model.fit(X_tr, Y_tr, sample_weight=sw)
        return float(mean_absolute_error(Y_val, model.predict(X_val)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    sw = _make_sample_weights(train_df, best_params["event_weight"])
    model = RandomForestRegressor(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],
                                  min_samples_leaf=best_params["min_samples_leaf"],
                                  random_state=SEED, n_jobs=-1)
    model.fit(X_tr, Y_tr, sample_weight=sw)
    return model, best_params, study.best_value


def tune_mlp_optuna(X_tr, Y_tr, X_val, Y_val, train_df, n_trials):
    def objective(trial):
        hidden_key = trial.suggest_categorical("hidden", list(HIDDEN_CHOICES.keys()))
        lr = trial.suggest_float("lr", *MLP_LR_RANGE, log=True)
        batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
        event_weight = trial.suggest_float("event_weight", *EVENT_WEIGHT_RANGE)
        sw = _make_sample_weights(train_df, event_weight)
        model, scaler, y_stats = dmm.fit_mlp_multi(
            X_tr, Y_tr, X_val, Y_val, sample_weight=sw,
            hidden=HIDDEN_CHOICES[hidden_key], lr=lr, batch_size=batch_size, **NN_FIXED)
        pred = dmm.predict_mlp_multi(model, scaler, y_stats, X_val)
        return float(mean_absolute_error(Y_val, pred))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    hidden = HIDDEN_CHOICES[best_params["hidden"]]
    sw = _make_sample_weights(train_df, best_params["event_weight"])
    bundle = dmm.fit_mlp_multi(X_tr, Y_tr, X_val, Y_val, sample_weight=sw,
                               hidden=hidden, lr=best_params["lr"], batch_size=best_params["batch_size"], **NN_FIXED)
    clean_params = {"hidden": hidden, "lr": best_params["lr"], "batch_size": best_params["batch_size"],
                    "event_weight": best_params["event_weight"]}
    return bundle, clean_params, study.best_value


def tune_lstm_optuna(X_tr, Y_tr, X_val, Y_val, n_trials):
    def objective(trial):
        hidden = trial.suggest_int("hidden", *LSTM_HIDDEN_RANGE)
        lr = trial.suggest_float("lr", *LSTM_LR_RANGE, log=True)
        batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
        model, scaler, y_stats = dmm.fit_lstm_multi(
            X_tr, Y_tr, X_val, Y_val, hidden=hidden, lr=lr, batch_size=batch_size, **NN_FIXED)
        pred = dmm.predict_lstm_multi(model, scaler, y_stats, X_val)
        return float(mean_absolute_error(Y_val, pred))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    bundle = dmm.fit_lstm_multi(X_tr, Y_tr, X_val, Y_val,
                                hidden=best_params["hidden"], lr=best_params["lr"],
                                batch_size=best_params["batch_size"], **NN_FIXED)
    return bundle, best_params, study.best_value


# ═══════════════════════════════════════════════════════════════════════════
# Metriche: per orizzonte + complessivo ("ALL"), per OGNI split — IDENTICA a
# run_multioutput_componibile_comparabile_penman.py.
# ═══════════════════════════════════════════════════════════════════════════

def _record_multi(metric_rows, pred_rows, model_name, fset_name, split_name,
                  Y_true, Y_pred, ev_mask, dc_mask, dates, devices, event_weight):
    n_horizons = len(HORIZONS)
    for i, h in enumerate(HORIZONS):
        yt, yp = Y_true[:, i], Y_pred[:, i]
        m = simplified_metrics(yt, yp, ev_mask=ev_mask, dc_mask=dc_mask)
        metric_rows.append({"model": model_name, "feature_set": fset_name,
                            "horizon": f"t+{h}", "split": split_name,
                            "event_weight": event_weight, **m})
        for d, dev, a, b in zip(dates, devices, yt, yp):
            pred_rows.append({"date": d, "sensor": dev, "model": model_name,
                              "feature_set": fset_name, "horizon": f"t+{h}", "split": split_name,
                              "y_true": round(float(a), 4), "y_pred": round(float(b), 4),
                              "residual": round(float(b - a), 4)})

    yt_all, yp_all = Y_true.ravel(), Y_pred.ravel()
    ev_all = np.repeat(ev_mask, n_horizons) if ev_mask is not None else None
    dc_all = np.repeat(dc_mask, n_horizons) if dc_mask is not None else None
    m = simplified_metrics(yt_all, yp_all, ev_mask=ev_all, dc_mask=dc_all)
    metric_rows.append({"model": model_name, "feature_set": fset_name,
                        "horizon": "ALL", "split": split_name,
                        "event_weight": event_weight, **m})
    if split_name == "test":
        print(f"  {model_name:6s} ALL(7gg) [test]  MAE={m['MAE']:.4f}  R2={m['R2']:.4f}")


def _record_hparams(hparam_rows, model_name, fset_name, params, val_score, n_features):
    hparam_rows.append({
        "model": model_name, "feature_set": fset_name, "n_features": n_features,
        "val_MAE": round(float(val_score), 4) if val_score is not None else np.nan,
        "params": json.dumps(params, default=str, sort_keys=True),
    })


def predict_and_record_all_splits(metric_rows, pred_rows, model_name, fset_name,
                                  predict_fn, event_weight, split_infos):
    """
    split_infos: lista di tuple (split_name, X_imp, Y_true, dates, devices,
    ev_mask, dc_mask) — una per train/val/test. Predice con predict_fn su
    ciascuno split e registra le metriche/predizioni per tutti e tre, non
    solo per il test.
    """
    for split_name, X_imp, Y_true, dates, devices, ev_mask, dc_mask in split_infos:
        Y_pred = predict_fn(X_imp)
        _record_multi(metric_rows, pred_rows, model_name, fset_name, split_name,
                     Y_true, Y_pred, ev_mask, dc_mask, dates, devices, event_weight)


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

    print("\nCostruisco lo split canonico (sui dati già arricchiti con Penman — "
          "stesso split di multioutput_ml/, penman_baseline_componibile/ e "
          "penman_residual_componibile/)...")
    canonical_split = build_canonical_split(datasets_penman, TARGET_COLS, train_frac, val_frac)
    if canonical_split.empty:
        raise RuntimeError("Split canonico vuoto: controlla i dati/sensori in ingresso.")

    train_df = canonical_split[canonical_split["_split"] == "train"].reset_index(drop=True)
    val_df = canonical_split[canonical_split["_split"] == "val"].reset_index(drop=True)
    test_df = canonical_split[canonical_split["_split"] == "test"].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Split canonico vuoto: controlla i dati/sensori in ingresso.")

    split_composition = compute_split_composition(canonical_split)

    # --- Filtro di comparabilità con Penman ---------------------------------
    # Stesso identico filtro di run_multioutput_componibile_comparabile_penman.py:
    # scarta le righe senza TUTTI i penman_pred_t{h} validi. Applicato sia
    # alle 3 slice sia al canonical_split completo (serve anche all'LSTM).
    n_train_canon, n_val_canon, n_test_canon = len(train_df), len(val_df), len(test_df)
    train_df = train_df.dropna(subset=PENMAN_PRED_COLS).reset_index(drop=True)
    val_df = val_df.dropna(subset=PENMAN_PRED_COLS).reset_index(drop=True)
    test_df = test_df.dropna(subset=PENMAN_PRED_COLS).reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Split vuoto dopo il filtro Penman: controlla dati/sensori.")

    canonical_split = canonical_split.dropna(subset=PENMAN_PRED_COLS).reset_index(drop=True)

    print(f"  n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)}  "
          f"(canonico: train={n_train_canon}/val={n_val_canon}/test={n_test_canon}, ridotto dal filtro "
          f"Penman — atteso, mostrato per trasparenza)")
    print("  STESSO train_df/val_df/test_df riusato per ogni feature-set, nessun nuovo pool/split")
    print(f"  ricerca Optuna: {n_trials} prove per modello (Ridge/RF/MLP/LSTM), per feature-set")

    Y_train_full = train_df[TARGET_COLS].values.astype(float)
    Y_val_full = val_df[TARGET_COLS].values.astype(float)
    Y_test_full = test_df[TARGET_COLS].values.astype(float)

    def _event_masks(split_df):
        has_rain = split_df.get("has_rain", pd.Series(False, index=split_df.index)).fillna(False).values.astype(bool)
        has_irr = split_df.get("has_irr", pd.Series(False, index=split_df.index)).fillna(False).values.astype(bool)
        return (has_rain | has_irr), ~(has_rain | has_irr)

    ev_mask_train, dc_mask_train = _event_masks(train_df)
    ev_mask_val, dc_mask_val = _event_masks(val_df)
    ev_mask_test, dc_mask_test = _event_masks(test_df)
    dates_train, devices_train = train_df["date"].values, train_df["device"].values
    dates_val, devices_val = val_df["date"].values, val_df["device"].values
    dates_test, devices_test = test_df["date"].values, test_df["device"].values

    for fi, fset_name in enumerate(fsets, start=1):
        print(f"\n[{elapsed_str()}] [feature-set {fi}/{len(fsets)}: {fset_name}]", flush=True)

        group_names = THEMATIC_FEATURE_SETS[fset_name]
        columns_per_horizon = {h: compose_feature_set(*group_names, horizon=h) for h in HORIZONS}
        available_features = sorted({c for h in HORIZONS for c in columns_per_horizon[h] if c in train_df.columns})

        X_train = train_df[available_features].apply(pd.to_numeric, errors="coerce").values
        X_val = val_df[available_features].apply(pd.to_numeric, errors="coerce").values
        X_test = test_df[available_features].apply(pd.to_numeric, errors="coerce").values
        X_train_imp, X_val_imp, X_test_imp = impute_train_val_test(X_train, X_val, X_test)

        print(f"  n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)}  "
              f"n_features={len(available_features)}  (gruppi: {', '.join(group_names)})")

        split_infos = [
            ("train", X_train_imp, Y_train_full, dates_train, devices_train, ev_mask_train, dc_mask_train),
            ("val", X_val_imp, Y_val_full, dates_val, devices_val, ev_mask_val, dc_mask_val),
            ("test", X_test_imp, Y_test_full, dates_test, devices_test, ev_mask_test, dc_mask_test),
        ]

        # --- Ridge (Optuna) -----------------------------------------------------
        model, best_params, val_mae = tune_ridge_optuna(
            X_train_imp, Y_train_full, X_val_imp, Y_val_full, train_df, n_trials)
        predict_and_record_all_splits(metric_rows, pred_rows, "Ridge", fset_name,
                                      model.predict, best_params["event_weight"], split_infos)
        _record_hparams(hparam_rows, "Ridge", fset_name, best_params, val_mae, len(available_features))
        print(f"  Ridge  migliori iperparametri: {best_params}  (val_MAE={val_mae:.4f})")

        # --- RF (Optuna) ----------------------------------------------------------
        model, best_params, val_mae = tune_rf_optuna(
            X_train_imp, Y_train_full, X_val_imp, Y_val_full, train_df, n_trials)
        predict_and_record_all_splits(metric_rows, pred_rows, "RF", fset_name,
                                      model.predict, best_params["event_weight"], split_infos)
        _record_hparams(hparam_rows, "RF", fset_name, best_params, val_mae, len(available_features))
        print(f"  RF     migliori iperparametri: {best_params}  (val_MAE={val_mae:.4f})")

        # --- MLP multi-output (Optuna) --------------------------------------------
        if run_mlp:
            (model, scaler, y_stats), best_params, val_mae = tune_mlp_optuna(
                X_train_imp, Y_train_full, X_val_imp, Y_val_full, train_df, n_trials)
            predict_fn = lambda X: dmm.predict_mlp_multi(model, scaler, y_stats, X)  # noqa: E731
            predict_and_record_all_splits(metric_rows, pred_rows, "MLP", fset_name,
                                          predict_fn, best_params["event_weight"], split_infos)
            _record_hparams(hparam_rows, "MLP", fset_name, {**best_params, **NN_FIXED}, val_mae, len(available_features))
            print(f"  MLP    migliori iperparametri: {best_params}  (val_MAE={val_mae:.4f})")

        # --- LSTM multi-output (sequenze, stesso split canonico già filtrato) ---
        if run_lstm:
            X_seq, Y_seq, meta = dmm.build_sequences_multi(
                canonical_split, available_features, TARGET_COLS, split_col="_split",
                lookback=lookback, extra_cols=["has_rain", "has_irr"])

            if len(X_seq) > 0:
                train_mask = (meta["split"] == "train").values
                val_mask = (meta["split"] == "val").values
                test_mask = (meta["split"] == "test").values
                n_tr, n_va, n_te = train_mask.sum(), val_mask.sum(), test_mask.sum()
                if n_tr >= MIN_SAMPLES and n_va >= MIN_SAMPLES and n_te >= MIN_SAMPLES:
                    (model, scaler, y_stats), best_params, val_mae = tune_lstm_optuna(
                        X_seq[train_mask], Y_seq[train_mask], X_seq[val_mask], Y_seq[val_mask], n_trials)

                    lstm_split_infos = []
                    for split_name, split_mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
                        meta_split = meta[split_mask].reset_index(drop=True)
                        ev_seq = (meta_split.get("has_rain", pd.Series(False, index=meta_split.index)).fillna(False) |
                                 meta_split.get("has_irr", pd.Series(False, index=meta_split.index)).fillna(False)).values
                        dc_seq = ~ev_seq
                        lstm_split_infos.append((
                            split_name, X_seq[split_mask], Y_seq[split_mask],
                            meta_split["date"].values, meta_split["device"].values, ev_seq, dc_seq,
                        ))

                    predict_fn = lambda X: dmm.predict_lstm_multi(model, scaler, y_stats, X)  # noqa: E731
                    predict_and_record_all_splits(metric_rows, pred_rows, "LSTM", fset_name,
                                                  predict_fn, event_weight, lstm_split_infos)
                    _record_hparams(hparam_rows, "LSTM", fset_name,
                                   {**best_params, **NN_FIXED, "lookback": lookback}, val_mae, len(available_features))
                    print(f"  LSTM sequenze: train={n_tr} val={n_va} test={n_te} "
                          f"(inferiore a n_test={len(test_df)}: vincolo di finestra a {lookback} giorni consecutivi, atteso)")
                    print(f"  LSTM   migliori iperparametri: {best_params}  (val_MAE={val_mae:.4f})")
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
    print(f"[dataset_aggiornato] dataset in uso: {os.path.abspath(args.dataset)}")

    os.makedirs(args.out, exist_ok=True)

    datasets = build_all_datasets_with_reconstructed_irrigation(
        args.dataset, args.sensors, args.irr, sensors_filter=args.sensors_filter)

    print("\nCarico configurazione agronomica (colture.csv, sensori-corretti.csv) e arricchisco con "
          "Penman-Monteith — serve solo a determinare quali (sensore, giorno) sono comparabili con "
          "Penman, le feature ML restano quelle componibili...")
    colture_df = load_colture(args.colture)
    colture_df = apply_soil_overrides(colture_df, fc=args.fc, wp=args.wp, sat=args.sat)
    sensori_df = load_sensori(args.sensors)
    datasets_penman = enrich_all_with_penman(datasets, sensori_df, colture_df, args.alpha_rain, args.alpha_irr)

    metrics_df, preds_df, hparams_df, split_composition = run_experiments(
        datasets_penman, fsets=args.fsets, n_trials=args.n_trials,
        run_mlp=not args.no_mlp, run_lstm=not args.no_lstm,
        event_weight=args.event_weight, lookback=args.lookback,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )

    mm_path = os.path.join(args.out, "experiments_comparabile_penman_optuna_metrics.csv")
    mp_path = os.path.join(args.out, "experiments_comparabile_penman_optuna_preds.csv")
    hp_path = os.path.join(args.out, "experiments_comparabile_penman_optuna_hyperparameters.csv")
    sc_path = os.path.join(args.out, "split_composition.csv")

    metrics_df.to_csv(mm_path, index=False)
    preds_df.to_csv(mp_path, index=False)
    hparams_df.to_csv(hp_path, index=False)
    split_composition.to_csv(sc_path, index=False)

    print(f"\nSalvato: {mm_path} ({len(metrics_df)} righe)")
    print(f"Salvato: {mp_path} ({len(preds_df)} righe)")
    print(f"Salvato: {hp_path} ({len(hparams_df)} righe)")
    print(f"Salvato: {sc_path} ({len(split_composition)} righe)")

    print("\nComposizione pioggia/irrigazione per split (aggregato):")
    print(split_composition[split_composition["sensore"] == "TUTTI"].to_string(index=False))


if __name__ == "__main__":
    main()
