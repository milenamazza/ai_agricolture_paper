"""
run_quantile_componibile.py
─────────────────────────────────────────────────────────────────────────────
Come quantile_regression_ml/run_quantile.py (un solo modello MLP
multi-output per feature-set, che predice direttamente 5 percentili — 5°,
25°, 50°, 75°, 95° — della distribuzione di moisture per ciascuno dei 7
orizzonti, via pinball loss — vedi deep_models_quantile.py per la
matematica, INVARIATA qui), ma:

  1. Le feature vengono dai 7 feature-set COMPONIBILI di
     ai_feature_componibile/feature_groups_util.py invece di FSET_MAP.
  2. L'irrigazione viene da irrigazione_mancante/irrigazione_ricostruita.csv
     invece del solo log originale.
  3. Lo split è quello CANONICO CONDIVISO
     (ai_feature_componibile/dati_ricostruiti_util.py), calcolato UNA VOLTA
     SOLA (non per feature-set) — stesso identico insieme (sensore, giorno)
     di multioutput_ml/, penman_baseline_componibile/,
     penman_residual_componibile/, uncertainty_componibile/ e
     deep_ensemble_componibile/.

Riuso in sola lettura (nessuna modifica ai file sorgenti):
  - quantile_regression_ml/deep_models_quantile.py, eval_quantile.py:
    INVARIATI — la logica di training/pinball-loss/coverage non cambia.
  - modelli singoli/data_driven_ml/utils.py: MIN_SAMPLES, HORIZONS,
    _metric_dict, _make_sample_weights.
  - modelli singoli/data_driven_ml/run_experiments.py: DEFAULT_DATASET,
    DEFAULT_SENSORS, TRAIN_FRAC, VAL_FRAC.
  - ai_feature_componibile/{feature_groups_util.py,dati_ricostruiti_util.py}
    (stessa cartella padre).

Output in results/:
  quantile_metrics.csv         — metriche (punto + pinball/coverage/sharpness)
                                  per feature-set x orizzonte-o-ALL x split
  quantile_preds.csv           — previsioni giorno per giorno con tutti e 5
                                  i percentili
  quantile_hyperparameters.csv
  split_composition.csv

Utilizzo:
    python -u run_quantile_componibile.py
    python -u run_quantile_componibile.py --sensors-filter EM-500-9 EM-500-12 --fsets ACQUA TERRA
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

warnings.filterwarnings("ignore")

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
    MIN_SAMPLES, HORIZONS,
    _metric_dict, _make_sample_weights,
)
from paths import (DEFAULT_DATASET, DEFAULT_SENSORS,  # noqa: E402
                   DEFAULT_IRRIGAZIONE_RICOSTRUITA, DEFAULT_COLTURE,
                   TRAIN_FRAC, VAL_FRAC, require_data_files)
import deep_models_quantile as dmq  # noqa: E402  (import in sola lettura, invariato)
import eval_quantile as eq          # noqa: E402  (import in sola lettura, invariato)

OUT_DIR = os.path.join(HERE, "results")

EVENT_WEIGHT = 3
TARGET_COLS = [f"target_t{h}" for h in HORIZONS]
FSETS_DEFAULT = list(THEMATIC_FEATURE_SETS.keys())
MODEL_NAME = "MLP-QuantileRegression"
QUANTILE_LEVELS = dmq.QUANTILE_LEVELS   # (0.05, 0.25, 0.50, 0.75, 0.95)
MEDIAN_INDEX = QUANTILE_LEVELS.index(0.50)

NN_FIXED_QUANTILE = {"batch_size": 32, "patience": 50, "epochs": 300}
GRID_MLP_QUANTILE = {
    "hidden": [(32,), (64, 32), (128, 64, 32), (32, 32), (256, 128, 64, 32)],
    "lr": [1e-2, 1e-3, 5e-4, 1e-4],
    "dropout": [0.1, 0.2, 0.3],
}


def impute_train_val_test(X_train, X_val, X_test):
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X_train), imputer.transform(X_val), imputer.transform(X_test)


# ═══════════════════════════════════════════════════════════════════════════
# Bookkeeping: metriche + predizioni per (feature-set, split), per orizzonte
# + aggregato "ALL"
# ═══════════════════════════════════════════════════════════════════════════

def record_results(metric_rows, pred_rows, fset_name, split_name,
                   Y_true, quantile_predictions, dates, devices,
                   has_rain, has_irr, has_water_event):
    """
    quantile_predictions ha shape (n_rows, n_horizons, n_quantiles). Per ogni
    orizzonte si estrae la mediana come previsione puntuale (per MAE/RMSE/R²,
    confrontabile con le altre pipeline) e si calcolano pinball loss +
    coverage/sharpness sui 5 percentili.
    """
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
    # NIENTE transpose: quantile_predictions ha shape (n_rows, n_horizons,
    # n_quantiles) in ordine C (memoria contigua), esattamente come Y_true
    # (n_rows, n_horizons). Y_true.ravel() appiattisce riga-poi-orizzonte:
    # per restare allineati riga per riga con y_true_all, va appiattito qui
    # SOLO l'asse (n_rows, n_horizons) tenendo intatto l'asse dei quantili.
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

def run_quantile_experiments(datasets, fsets, event_weight=EVENT_WEIGHT,
                             train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    metric_rows = []
    pred_rows = []
    hparam_rows = []

    start_time = time.time()

    def elapsed_time_str():
        seconds = time.time() - start_time
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    print("\nCostruisco lo split canonico (una sola volta, modulo condiviso "
          "ai_feature_componibile/dati_ricostruiti_util.py — stesso split di "
          "multioutput_ml/, penman_baseline_componibile/, "
          "penman_residual_componibile/, uncertainty_componibile/ e "
          "deep_ensemble_componibile/)...")
    canonical_split = build_canonical_split(datasets, TARGET_COLS, train_frac, val_frac)
    if canonical_split.empty:
        raise RuntimeError("Split canonico vuoto: controlla i dati/sensori in ingresso.")

    train_df = canonical_split[canonical_split["_split"] == "train"].reset_index(drop=True)
    val_df = canonical_split[canonical_split["_split"] == "val"].reset_index(drop=True)
    test_df = canonical_split[canonical_split["_split"] == "test"].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Split canonico vuoto: controlla i dati/sensori in ingresso.")

    print(f"  n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)}  "
          f"— STESSO train_df/val_df/test_df riusato per ogni feature-set, nessun nuovo pool/split")

    split_composition = compute_split_composition(canonical_split)

    Y_train_full = train_df[TARGET_COLS].values.astype(float)
    Y_val_full = val_df[TARGET_COLS].values.astype(float)
    Y_test_full = test_df[TARGET_COLS].values.astype(float)

    for fset_index, fset_name in enumerate(fsets, start=1):
        print(f"\n[{elapsed_time_str()}] [feature-set {fset_index}/{len(fsets)}: {fset_name}]", flush=True)

        group_names = THEMATIC_FEATURE_SETS[fset_name]
        columns_per_horizon = {h: compose_feature_set(*group_names, horizon=h) for h in HORIZONS}
        available_features = sorted({c for h in HORIZONS for c in columns_per_horizon[h] if c in train_df.columns})

        X_train = train_df[available_features].apply(pd.to_numeric, errors="coerce").values
        X_val = val_df[available_features].apply(pd.to_numeric, errors="coerce").values
        X_test = test_df[available_features].apply(pd.to_numeric, errors="coerce").values

        X_train_imp, X_val_imp, X_test_imp = impute_train_val_test(X_train, X_val, X_test)
        sample_weight = _make_sample_weights(train_df, event_weight)

        print(f"  n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)}  "
              f"n_features={len(available_features)}  (gruppi: {', '.join(group_names)})")

        # --- Grid search via validation pinball loss -------------------------
        (best_model, feature_scaler, y_stats), best_params, best_val_pinball = dmq.tune_mlp_quantile_multi_val(
            X_train_imp, Y_train_full, X_val_imp, Y_val_full,
            GRID_MLP_QUANTILE, sample_weight=sample_weight, nn_fixed=NN_FIXED_QUANTILE,
        )
        print(f"  migliori iperparametri: {best_params}  (val_pinball_loss={best_val_pinball:.4f})")
        record_hyperparameters(hparam_rows, fset_name, best_params, best_val_pinball, len(available_features))

        # --- Predizione + metriche, su train/val/test ------------------------
        splits_to_evaluate = [
            ("train", train_df, X_train_imp, Y_train_full),
            ("val", val_df, X_val_imp, Y_val_full),
            ("test", test_df, X_test_imp, Y_test_full),
        ]
        for split_name, split_df, X_split_imp, Y_split in splits_to_evaluate:
            quantile_predictions = dmq.predict_mlp_quantile_multi(best_model, feature_scaler, y_stats, X_split_imp)
            record_results(
                metric_rows, pred_rows, fset_name, split_name,
                Y_split, quantile_predictions,
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
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--event-weight", type=float, default=EVENT_WEIGHT)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--fsets", nargs="+", default=FSETS_DEFAULT, choices=FSETS_DEFAULT)
    parser.add_argument("--sensors-filter", nargs="+", default=None,
                        help="Limita ai sensori indicati, es. EM-500-9 EM-500-12 (per test rapidi)")
    args = parser.parse_args()
    require_data_files(args.dataset, args.sensors, args.irr,
                       getattr(args, "colture", None))

    os.makedirs(args.out, exist_ok=True)

    datasets = build_all_datasets_with_reconstructed_irrigation(
        args.dataset, args.sensors, args.irr, sensors_filter=args.sensors_filter)

    metrics_df, preds_df, hparams_df, split_composition = run_quantile_experiments(
        datasets, fsets=args.fsets,
        event_weight=args.event_weight,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )

    metrics_path = os.path.join(args.out, "quantile_metrics.csv")
    preds_path = os.path.join(args.out, "quantile_preds.csv")
    hparams_path = os.path.join(args.out, "quantile_hyperparameters.csv")
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
