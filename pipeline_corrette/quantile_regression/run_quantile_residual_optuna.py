"""
run_quantile_residual_componibile_optuna.py
─────────────────────────────────────────────────────────────────────────────
Come run_quantile_residual_componibile.py (Quantile Regression che predice i
5 percentili del RESIDUO target-penman_pred invece che dell'assoluto, poi
ricostruisce sommando penman_pred a ogni percentile — vedi
uncertainty_residual_componibile/run_uncertainty_residual_componibile.py per
la giustificazione matematica completa), ma la ricerca degli iperparametri
usa OPTUNA invece della griglia esaustiva — stessa motivazione di
run_quantile_componibile_optuna.py.

Il punteggio di ogni prova è la pinball loss di validation, calcolata in
spazio RESIDUO (il target su cui il modello viene davvero allenato).

Il filtro sul sottoinsieme comparabile con Penman qui è STRUTTURALMENTE
necessario: senza penman_pred_t{h} non esiste il target residuo.

Output in results_optuna/:
  quantile_residual_metrics.csv, quantile_residual_preds.csv,
  quantile_residual_hyperparameters.csv, split_composition.csv

Utilizzo:
    python -u run_quantile_residual_componibile_optuna.py
    python -u run_quantile_residual_componibile_optuna.py --sensors-filter EM-500-9 EM-500-12 --fsets ACQUA TERRA --n-trials 15
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
)

from data_utils import (  # noqa: E402
    MIN_SAMPLES, HORIZONS, SAT_OVERRIDE,
    load_colture, load_sensori, apply_soil_overrides,
    _metric_dict, _make_sample_weights,
)
from paths import (DEFAULT_DATASET, DEFAULT_SENSORS,  # noqa: E402
                   DEFAULT_IRRIGAZIONE_RICOSTRUITA, DEFAULT_COLTURE,
                   TRAIN_FRAC, VAL_FRAC, require_data_files)
from penman import enrich_all_with_penman  # noqa: E402  (import in sola lettura, invariato)
import deep_models_quantile as dmq  # noqa: E402  (import in sola lettura, invariato)
import eval_quantile as eq          # noqa: E402  (import in sola lettura, invariato)

OUT_DIR = os.path.join(HERE, "results_residual_optuna")   # diretta e residuo condividono la cartella: output separati

TARGET_COLS = [f"target_t{h}" for h in HORIZONS]
PENMAN_PRED_COLS = [f"penman_pred_t{h}" for h in HORIZONS]
FSETS_DEFAULT = list(THEMATIC_FEATURE_SETS.keys())
MODEL_NAME = "MLP-QuantileRegression-Residual-Optuna"
QUANTILE_LEVELS = dmq.QUANTILE_LEVELS   # (0.05, 0.25, 0.50, 0.75, 0.95)
MEDIAN_INDEX = QUANTILE_LEVELS.index(0.50)
N_TRIALS_DEFAULT = 100
OPTUNA_SEED = 42

NN_FIXED_QUANTILE = {"patience": 30, "epochs": 300}   # batch_size lo sceglie Optuna, non è più fisso qui

HIDDEN_CHOICES = {
    "32": (32,),
    "64_32": (64, 32),
    "128_64_32": (128, 64, 32),
    "32_32": (32, 32),
    "256_128_64_32": (256, 128, 64, 32),
}
BATCH_SIZE_CHOICES = [16, 32, 64, 128]
DROPOUT_RANGE = (0.05, 0.4)
LR_RANGE = (1e-4, 1e-2)          # campionato log-uniforme
EVENT_WEIGHT_RANGE = (1.0, 10.0)


def impute_train_val_test(X_train, X_val, X_test):
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X_train), imputer.transform(X_val), imputer.transform(X_test)


# ═══════════════════════════════════════════════════════════════════════════
# Ricerca iperparametri con Optuna — obiettivo calcolato in spazio RESIDUO
# ═══════════════════════════════════════════════════════════════════════════

def make_objective(train_df, X_train_imp, Y_train_res, X_val_imp, Y_val_res):
    def objective(trial):
        hidden_key = trial.suggest_categorical("hidden", list(HIDDEN_CHOICES.keys()))
        dropout = trial.suggest_float("dropout", *DROPOUT_RANGE)
        lr = trial.suggest_float("lr", *LR_RANGE, log=True)
        batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
        event_weight = trial.suggest_float("event_weight", *EVENT_WEIGHT_RANGE)

        sample_weight = _make_sample_weights(train_df, event_weight)
        _, _, _, val_pinball = dmq.fit_mlp_quantile_multi(
            X_train_imp, Y_train_res, X_val_imp, Y_val_res,
            sample_weight=sample_weight,
            hidden=HIDDEN_CHOICES[hidden_key], dropout=dropout, lr=lr, batch_size=batch_size,
            patience=NN_FIXED_QUANTILE["patience"], epochs=NN_FIXED_QUANTILE["epochs"],
        )
        return val_pinball

    return objective


def tune_with_optuna(train_df, X_train_imp, Y_train_res, X_val_imp, Y_val_res, n_trials):
    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def trial_callback(study, trial):
        print(f"    [Optuna {trial.number + 1}/{n_trials}] val_pinball={trial.value:.4f}  "
              f"hidden={trial.params['hidden']} lr={trial.params['lr']:.5f} "
              f"dropout={trial.params['dropout']:.3f} batch={trial.params['batch_size']} "
              f"event_weight={trial.params['event_weight']:.2f}")

    objective = make_objective(train_df, X_train_imp, Y_train_res, X_val_imp, Y_val_res)
    study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])

    best_params = dict(study.best_params)
    best_params["hidden"] = HIDDEN_CHOICES[best_params["hidden"]]
    return best_params, study.best_value


# ═══════════════════════════════════════════════════════════════════════════
# Bookkeeping — IDENTICA a run_quantile_residual_componibile.py.
# ═══════════════════════════════════════════════════════════════════════════

def record_results(metric_rows, pred_rows, fset_name, split_name,
                   Y_true, quantile_predictions, dates, devices,
                   has_rain, has_irr, has_water_event):
    for horizon_index, horizon in enumerate(HORIZONS):
        y_true_h = Y_true[:, horizon_index]
        quantiles_h = quantile_predictions[:, horizon_index, :]   # (n_rows, n_quantiles)
        median_h = quantiles_h[:, MEDIAN_INDEX]

        point_metrics = _metric_dict(y_true_h, median_h)
        quantile_metrics = eq.quantile_metric_dict(y_true_h, quantiles_h, QUANTILE_LEVELS)

        metric_rows.append({
            "model": MODEL_NAME, "feature_set": fset_name,
            "horizon": f"t+{horizon}", "split": split_name,
            **point_metrics, **quantile_metrics,
        })

        for row_index in range(len(y_true_h)):
            pred_row = {
                "date": dates[row_index], "sensor": devices[row_index],
                "model": MODEL_NAME, "feature_set": fset_name,
                "horizon": f"t+{horizon}", "split": split_name,
                "y_true": round(float(y_true_h[row_index]), 4),
            }
            for quantile_index, quantile_level in enumerate(QUANTILE_LEVELS):
                column_name = f"q{int(round(quantile_level * 100)):02d}"
                pred_row[column_name] = round(float(quantiles_h[row_index, quantile_index]), 4)
            pred_row["has_rain"] = bool(has_rain[row_index])
            pred_row["has_irr"] = bool(has_irr[row_index])
            pred_row["has_water_event"] = bool(has_water_event[row_index])
            pred_rows.append(pred_row)

    # Aggregato "ALL": tutti e 7 gli orizzonti concatenati in un solo vettore.
    y_true_all = Y_true.ravel()
    # NIENTE transpose: stesso motivo di run_quantile_componibile.py.
    quantiles_all = quantile_predictions.reshape(-1, len(QUANTILE_LEVELS))
    median_all = quantiles_all[:, MEDIAN_INDEX]

    point_metrics_all = _metric_dict(y_true_all, median_all)
    quantile_metrics_all = eq.quantile_metric_dict(y_true_all, quantiles_all, QUANTILE_LEVELS)

    metric_rows.append({
        "model": MODEL_NAME, "feature_set": fset_name,
        "horizon": "ALL", "split": split_name,
        **point_metrics_all, **quantile_metrics_all,
    })

    print(f"    [{split_name:5s}] ALL(7gg)  MAE={point_metrics_all['MAE']:.4f}  "
          f"pinball_loss={quantile_metrics_all['pinball_loss']:.4f}  "
          f"coverage_90={quantile_metrics_all['coverage_90']:.3f}  "
          f"sharpness_90={quantile_metrics_all['sharpness_90']:.4f}")


def record_hyperparameters(hparam_rows, fset_name, best_params, val_pinball, n_features):
    hparam_rows.append({
        "model": MODEL_NAME, "feature_set": fset_name,
        "n_features": n_features,
        "val_pinball_loss": round(float(val_pinball), 4),
        "params": json.dumps(best_params, default=str, sort_keys=True),
    })


# ═══════════════════════════════════════════════════════════════════════════
# Loop principale
# ═══════════════════════════════════════════════════════════════════════════

def run_quantile_experiments(datasets_penman, fsets, n_trials,
                             train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    metric_rows = []
    pred_rows = []
    hparam_rows = []

    start_time = time.time()

    def elapsed_time_str():
        seconds = time.time() - start_time
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    print("\nCostruisco lo split canonico (sui dati già arricchiti con Penman)...")
    canonical_split = build_canonical_split(datasets_penman, TARGET_COLS, train_frac, val_frac)
    if canonical_split.empty:
        raise RuntimeError("Split canonico vuoto: controlla i dati/sensori in ingresso.")

    train_df = canonical_split[canonical_split["_split"] == "train"].reset_index(drop=True)
    val_df = canonical_split[canonical_split["_split"] == "val"].reset_index(drop=True)
    test_df = canonical_split[canonical_split["_split"] == "test"].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Split canonico vuoto: controlla i dati/sensori in ingresso.")

    split_composition = compute_split_composition(canonical_split)

    # --- Filtro: senza penman_pred_t{h} non esiste il target residuo -------
    n_train_canon, n_val_canon, n_test_canon = len(train_df), len(val_df), len(test_df)
    train_df = train_df.dropna(subset=PENMAN_PRED_COLS).reset_index(drop=True)
    val_df = val_df.dropna(subset=PENMAN_PRED_COLS).reset_index(drop=True)
    test_df = test_df.dropna(subset=PENMAN_PRED_COLS).reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Split vuoto dopo il filtro Penman: controlla dati/sensori.")

    print(f"  n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)}  "
          f"(canonico: train={n_train_canon}/val={n_val_canon}/test={n_test_canon}, ridotto dal filtro "
          f"Penman — necessario per costruire il target residuo)")
    print("  STESSO train_df/val_df/test_df riusato per ogni feature-set, nessun nuovo pool/split")

    Y_train_abs = train_df[TARGET_COLS].values.astype(float)
    Y_val_abs = val_df[TARGET_COLS].values.astype(float)
    Y_test_abs = test_df[TARGET_COLS].values.astype(float)

    Y_train_penman = train_df[PENMAN_PRED_COLS].values.astype(float)
    Y_val_penman = val_df[PENMAN_PRED_COLS].values.astype(float)
    Y_test_penman = test_df[PENMAN_PRED_COLS].values.astype(float)

    Y_train_res = Y_train_abs - Y_train_penman
    Y_val_res = Y_val_abs - Y_val_penman

    for fset_index, fset_name in enumerate(fsets, start=1):
        print(f"\n[{elapsed_time_str()}] [feature-set {fset_index}/{len(fsets)}: {fset_name}]", flush=True)

        group_names = THEMATIC_FEATURE_SETS[fset_name]
        columns_per_horizon = {h: compose_feature_set(*group_names, horizon=h) for h in HORIZONS}
        available_features = sorted({c for h in HORIZONS for c in columns_per_horizon[h] if c in train_df.columns})

        X_train = train_df[available_features].apply(pd.to_numeric, errors="coerce").values
        X_val = val_df[available_features].apply(pd.to_numeric, errors="coerce").values
        X_test = test_df[available_features].apply(pd.to_numeric, errors="coerce").values

        X_train_imp, X_val_imp, X_test_imp = impute_train_val_test(X_train, X_val, X_test)

        print(f"  n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)}  "
              f"n_features={len(available_features)}  (gruppi: {', '.join(group_names)}, target=residuo)")
        print(f"  ricerca Optuna: {n_trials} prove (hidden/lr/dropout/batch_size/event_weight)")

        # --- Ricerca Optuna via validation pinball loss (target = residuo) -
        best_params, best_val_pinball = tune_with_optuna(
            train_df, X_train_imp, Y_train_res, X_val_imp, Y_val_res, n_trials,
        )
        print(f"  migliori iperparametri: {best_params}  (val_pinball_loss={best_val_pinball:.4f}, su residuo)")
        record_hyperparameters(hparam_rows, fset_name, best_params, best_val_pinball, len(available_features))

        # --- Riallena UNA volta con i migliori iperparametri (su residuo) --
        sample_weight = _make_sample_weights(train_df, best_params["event_weight"])
        best_model, feature_scaler, y_stats, _ = dmq.fit_mlp_quantile_multi(
            X_train_imp, Y_train_res, X_val_imp, Y_val_res,
            sample_weight=sample_weight,
            hidden=best_params["hidden"], dropout=best_params["dropout"],
            lr=best_params["lr"], batch_size=best_params["batch_size"],
            patience=NN_FIXED_QUANTILE["patience"], epochs=NN_FIXED_QUANTILE["epochs"],
        )

        # --- Predizione (residuo) + ricostruzione assoluta, su train/val/test -
        splits_to_evaluate = [
            ("train", train_df, X_train_imp, Y_train_abs, Y_train_penman),
            ("val", val_df, X_val_imp, Y_val_abs, Y_val_penman),
            ("test", test_df, X_test_imp, Y_test_abs, Y_test_penman),
        ]
        for split_name, split_df, X_split_imp, Y_split_abs, Y_split_penman in splits_to_evaluate:
            quantile_predictions_res = dmq.predict_mlp_quantile_multi(best_model, feature_scaler, y_stats, X_split_imp)
            quantile_predictions_abs = quantile_predictions_res + Y_split_penman[:, :, np.newaxis]
            record_results(
                metric_rows, pred_rows, fset_name, split_name,
                Y_split_abs, quantile_predictions_abs,
                split_df["date"].values, split_df["device"].values,
                split_df["has_rain"].values, split_df["has_irr"].values,
                split_df["has_water_event"].values,
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), pd.DataFrame(hparam_rows), split_composition


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--sensors", default=DEFAULT_SENSORS)
    parser.add_argument("--irr", default=DEFAULT_IRRIGAZIONE_RICOSTRUITA)
    parser.add_argument("--colture", default=DEFAULT_COLTURE)
    parser.add_argument("--alpha-rain", type=float, default=0.50)
    parser.add_argument("--alpha-irr", type=float, default=0.70)
    parser.add_argument("--fc", type=float, default=None)
    parser.add_argument("--wp", type=float, default=None)
    parser.add_argument("--sat", type=float, default=SAT_OVERRIDE)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--fsets", nargs="+", default=FSETS_DEFAULT, choices=FSETS_DEFAULT)
    parser.add_argument("--sensors-filter", nargs="+", default=None,
                        help="Limita ai sensori indicati, es. EM-500-9 EM-500-12 (per test rapidi)")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                        help="Numero di prove Optuna per feature-set")
    args = parser.parse_args()
    require_data_files(args.dataset, args.sensors, args.irr,
                       getattr(args, "colture", None))

    os.makedirs(args.out, exist_ok=True)

    datasets = build_all_datasets_with_reconstructed_irrigation(
        args.dataset, args.sensors, args.irr, sensors_filter=args.sensors_filter)

    print("\nCarico configurazione agronomica (colture.csv, sensori-corretti.csv) e arricchisco con "
          "Penman-Monteith — qui serve a costruire il target residuo, non entra come feature ML...")
    colture_df = load_colture(args.colture)
    colture_df = apply_soil_overrides(colture_df, fc=args.fc, wp=args.wp, sat=args.sat)
    sensori_df = load_sensori(args.sensors)
    datasets_penman = enrich_all_with_penman(datasets, sensori_df, colture_df, args.alpha_rain, args.alpha_irr)

    metrics_df, preds_df, hparams_df, split_composition = run_quantile_experiments(
        datasets_penman, fsets=args.fsets, n_trials=args.n_trials,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )

    metrics_path = os.path.join(args.out, "quantile_residual_metrics.csv")
    preds_path = os.path.join(args.out, "quantile_residual_preds.csv")
    hparams_path = os.path.join(args.out, "quantile_residual_hyperparameters.csv")
    sc_path = os.path.join(args.out, "split_composition.csv")

    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    hparams_df.to_csv(hparams_path, index=False)
    split_composition.to_csv(sc_path, index=False)

    print(f"\nSalvato: {metrics_path} ({len(metrics_df)} righe)")
    print(f"Salvato: {preds_path} ({len(preds_df)} righe)")
    print(f"Salvato: {hparams_path} ({len(hparams_df)} righe)")
    print(f"Salvato: {sc_path} ({len(split_composition)} righe)")


if __name__ == "__main__":
    main()
