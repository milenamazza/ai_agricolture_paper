"""
run_ensemble_residual_componibile.py
─────────────────────────────────────────────────────────────────────────────
Come deep_ensemble_componibile/run_ensemble_componibile.py (N=5 reti MLP
indipendenti, epistemica = disaccordo tra reti, aleatoria = sigma dichiarata
— vedi deep_models_ensemble.py per la matematica, INVARIATA qui), ma il
modello predice il RESIDUO (target - penman_pred) invece del target
assoluto — vedi uncertainty_residual_componibile/run_uncertainty_residual_componibile.py
per la spiegazione completa e la giustificazione matematica della
ricostruzione (sommare una costante deterministica sposta solo la media,
mai le deviazioni standard).

Le feature ML restano ESCLUSIVAMENTE quelle componibili (feature_groups_util)
— NESSUNA colonna fisica Penman come input al modello. Penman entra solo per
costruire il target residuo e ricostruire la previsione assoluta.

Come nell'originale, il tuning usa UNA sola rete (seed=dme.BASE_SEED, non
l'intero ensemble) — stesso motivo (costo x5 altrimenti).

Riuso in sola lettura: deep_ensemble_ml/deep_models_ensemble.py,
uncertainty_ml/eval_uncertainty.py, penman_baseline/run_penman.py
(enrich_all_with_penman) — tutti INVARIATI.

Il filtro sul sottoinsieme comparabile con Penman qui è STRUTTURALMENTE
necessario: senza penman_pred_t{h} non esiste il target residuo.

Output in results/:
  ensemble_residual_metrics.csv
  ensemble_residual_preds.csv
  ensemble_residual_hyperparameters.csv
  split_composition.csv

Utilizzo:
    python -u run_ensemble_residual_componibile.py
    python -u run_ensemble_residual_componibile.py --sensors-filter EM-500-9 EM-500-12 --fsets ACQUA TERRA
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
    MIN_SAMPLES, HORIZONS, SAT_OVERRIDE,
    load_colture, load_sensori, apply_soil_overrides,
    _metric_dict, _make_sample_weights,
)
from paths import (DEFAULT_DATASET, DEFAULT_SENSORS,  # noqa: E402
                   DEFAULT_IRRIGAZIONE_RICOSTRUITA, DEFAULT_COLTURE,
                   TRAIN_FRAC, VAL_FRAC, require_data_files)
from penman import enrich_all_with_penman  # noqa: E402  (import in sola lettura, invariato)
import eval_uncertainty as eu              # noqa: E402  (import in sola lettura, invariato)
import deep_models_ensemble as dme         # noqa: E402  (import in sola lettura, invariato)

OUT_DIR = os.path.join(HERE, "results_residual")   # diretta e residuo condividono la cartella: output separati

EVENT_WEIGHT = 3
TARGET_COLS = [f"target_t{h}" for h in HORIZONS]
PENMAN_PRED_COLS = [f"penman_pred_t{h}" for h in HORIZONS]
FSETS_DEFAULT = list(THEMATIC_FEATURE_SETS.keys())
MODEL_NAME = "MLP-Gaussian-DeepEnsemble-Residual"

NN_FIXED_ENSEMBLE = {"batch_size": 32, "patience": 50, "epochs": 300}
GRID_MLP_ENSEMBLE = {
    "hidden": [(32,), (64, 32), (128, 64, 32), (32, 32), (256, 128, 64, 32)],
    "lr": [1e-2, 1e-3, 5e-4, 1e-4],
    "dropout": [0.1, 0.2, 0.3],
}
N_ENSEMBLE_MEMBERS = 5


def impute_train_val_test(X_train, X_val, X_test):
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X_train), imputer.transform(X_val), imputer.transform(X_test)


# ═══════════════════════════════════════════════════════════════════════════
# Bookkeeping — IDENTICA a run_ensemble_componibile.py. Riceve sempre valori
# in spazio ASSOLUTO (già ricostruiti sommando penman_pred).
# ═══════════════════════════════════════════════════════════════════════════

def record_results(metric_rows, pred_rows, fset_name, split_name,
                   Y_true, mu_pred, epistemic_std, aleatoric_std, total_std,
                   dates, devices, has_rain, has_irr, has_water_event):
    for horizon_index, horizon in enumerate(HORIZONS):
        y_true_h = Y_true[:, horizon_index]
        mu_h = mu_pred[:, horizon_index]
        epistemic_h = epistemic_std[:, horizon_index]
        aleatoric_h = aleatoric_std[:, horizon_index]
        total_h = total_std[:, horizon_index]

        point_metrics = _metric_dict(y_true_h, mu_h)
        uncertainty_metrics = eu.uncertainty_metric_dict(y_true_h, mu_h, epistemic_h, aleatoric_h, total_h)

        metric_rows.append({
            "model": MODEL_NAME, "feature_set": fset_name,
            "horizon": f"t+{horizon}", "split": split_name,
            **point_metrics, **uncertainty_metrics,
        })

        for date_value, device_value, y_true_v, mu_v, epi_v, ale_v, tot_v, rain_v, irr_v, water_v in zip(
            dates, devices, y_true_h, mu_h, epistemic_h, aleatoric_h, total_h,
            has_rain, has_irr, has_water_event,
        ):
            pred_rows.append({
                "date": date_value, "sensor": device_value,
                "model": MODEL_NAME, "feature_set": fset_name,
                "horizon": f"t+{horizon}", "split": split_name,
                "y_true": round(float(y_true_v), 4),
                "mu_pred": round(float(mu_v), 4),
                "epistemic_std": round(float(epi_v), 4),
                "aleatoric_std": round(float(ale_v), 4),
                "total_std": round(float(tot_v), 4),
                "has_rain": bool(rain_v), "has_irr": bool(irr_v), "has_water_event": bool(water_v),
            })

    y_true_all = Y_true.ravel()
    mu_all = mu_pred.ravel()
    epistemic_all = epistemic_std.ravel()
    aleatoric_all = aleatoric_std.ravel()
    total_all = total_std.ravel()

    point_metrics_all = _metric_dict(y_true_all, mu_all)
    uncertainty_metrics_all = eu.uncertainty_metric_dict(y_true_all, mu_all, epistemic_all, aleatoric_all, total_all)

    metric_rows.append({
        "model": MODEL_NAME, "feature_set": fset_name,
        "horizon": "ALL", "split": split_name,
        **point_metrics_all, **uncertainty_metrics_all,
    })

    print(f"    [{split_name:5s}] ALL(7gg)  MAE={point_metrics_all['MAE']:.4f}  "
          f"NLL={uncertainty_metrics_all['NLL']:.4f}  "
          f"coverage_90={uncertainty_metrics_all['coverage_90']:.3f}  "
          f"epistemic_std={uncertainty_metrics_all['mean_epistemic_std']:.4f}  "
          f"aleatoric_std={uncertainty_metrics_all['mean_aleatoric_std']:.4f}")


def record_hyperparameters(hparam_rows, fset_name, best_params, val_nll, n_features, n_members):
    hparam_rows.append({
        "model": MODEL_NAME, "feature_set": fset_name,
        "n_features": n_features, "n_ensemble_members": n_members,
        "val_NLL_single_member": round(float(val_nll), 4),
        "params": json.dumps(best_params, default=str, sort_keys=True),
    })


# ═══════════════════════════════════════════════════════════════════════════
# Loop principale
# ═══════════════════════════════════════════════════════════════════════════

def run_ensemble_experiments(datasets_penman, fsets, event_weight=EVENT_WEIGHT,
                             train_frac=TRAIN_FRAC, val_frac=VAL_FRAC,
                             n_members=N_ENSEMBLE_MEMBERS):
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
        sample_weight = _make_sample_weights(train_df, event_weight)

        print(f"  n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)}  "
              f"n_features={len(available_features)}  (gruppi: {', '.join(group_names)}, target=residuo)")

        # --- Tuning con UNA sola rete (seed di base), target = residuo -----
        best_params, best_val_nll = dme.tune_mlp_gaussian_multi_val(
            X_train_imp, Y_train_res, X_val_imp, Y_val_res,
            GRID_MLP_ENSEMBLE, sample_weight=sample_weight, nn_fixed=NN_FIXED_ENSEMBLE,
        )
        print(f"  migliori iperparametri (rete singola): {best_params}  (val_NLL={best_val_nll:.4f}, su residuo)")
        record_hyperparameters(hparam_rows, fset_name, best_params, best_val_nll,
                               len(available_features), n_members)

        # --- Training di N reti indipendenti con quegli iperparametri (residuo) -
        trained_members = []
        for member_index in range(n_members):
            member_seed = dme.BASE_SEED + member_index
            model, feature_scaler, y_stats, member_val_nll = dme.fit_mlp_gaussian_multi(
                X_train_imp, Y_train_res, X_val_imp, Y_val_res, seed=member_seed,
                sample_weight=sample_weight, **best_params, **NN_FIXED_ENSEMBLE,
            )
            trained_members.append((model, feature_scaler, y_stats))
            print(f"    membro {member_index + 1}/{n_members} (seed={member_seed})  "
                  f"val_NLL={member_val_nll:.4f}")

        # --- Predizione (residuo) + combinazione + ricostruzione assoluta --
        splits_to_evaluate = [
            ("train", train_df, X_train_imp, Y_train_abs, Y_train_penman),
            ("val", val_df, X_val_imp, Y_val_abs, Y_val_penman),
            ("test", test_df, X_test_imp, Y_test_abs, Y_test_penman),
        ]
        for split_name, split_df, X_split_imp, Y_split_abs, Y_split_penman in splits_to_evaluate:
            member_mu_list = []
            member_sigma_list = []
            for model, feature_scaler, y_stats in trained_members:
                mu_member, sigma_member = dme.predict_single_member(model, feature_scaler, y_stats, X_split_imp)
                member_mu_list.append(mu_member)
                member_sigma_list.append(sigma_member)

            mu_pred_res, epistemic_std, aleatoric_std, total_std = dme.combine_ensemble_predictions(
                member_mu_list, member_sigma_list,
            )
            mu_pred_abs = mu_pred_res + Y_split_penman
            record_results(
                metric_rows, pred_rows, fset_name, split_name,
                Y_split_abs, mu_pred_abs, epistemic_std, aleatoric_std, total_std,
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
    parser.add_argument("--event-weight", type=float, default=EVENT_WEIGHT)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--fsets", nargs="+", default=FSETS_DEFAULT, choices=FSETS_DEFAULT)
    parser.add_argument("--sensors-filter", nargs="+", default=None,
                        help="Limita ai sensori indicati, es. EM-500-9 EM-500-12 (per test rapidi)")
    parser.add_argument("--n-members", type=int, default=N_ENSEMBLE_MEMBERS,
                        help="Numero di reti indipendenti nell'ensemble")
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

    metrics_df, preds_df, hparams_df, split_composition = run_ensemble_experiments(
        datasets_penman, fsets=args.fsets,
        event_weight=args.event_weight,
        train_frac=args.train_frac, val_frac=args.val_frac,
        n_members=args.n_members,
    )

    metrics_path = os.path.join(args.out, "ensemble_residual_metrics.csv")
    preds_path = os.path.join(args.out, "ensemble_residual_preds.csv")
    hparams_path = os.path.join(args.out, "ensemble_residual_hyperparameters.csv")
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
