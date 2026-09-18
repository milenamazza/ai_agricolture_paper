"""
run_laplace_optuna.py
─────────────────────────────────────────────────────────────────────────────
APPROSSIMAZIONE DI LAPLACE (last-layer) sulle feature componibili, target
ASSOLUTO, ricerca degli iperparametri con Optuna.

Stessa pipeline dati di mc_dropout/run_mc_dropout_optuna.py — stesse feature
componibili, stessa irrigazione ricostruita, stesso split canonico condiviso,
stesso filtro di comparabilità con Penman — cambia SOLO il metodo con cui si
stima l'incertezza:

  MC-Dropout   T forward pass con neuroni spenti a caso
  Bayesian VI  N forward pass con i pesi campionati da una posteriore appresa
  Laplace      NESSUN campionamento: si allena una rete normale e poi si
               misura la curvatura della loss attorno al minimo, ottenendo
               l'incertezza epistemica in forma chiusa

La matematica sta in core/deep_models_laplace.py (import in sola lettura), che
chiude con la STESSA quadrupla (mu_pred, epistemic_std, aleatoric_std,
total_std) delle altre pipeline: i CSV prodotti qui hanno quindi lo schema
identico a quelli di mc_dropout/ e si confrontano riga per riga.

IPERPARAMETRI
Optuna cerca hidden, dropout, lr, batch_size, event_weight — gli stessi di
MC-Dropout (il dropout qui serve solo come regolarizzatore in training, NON
viene riacceso in predizione).

La precisione del prior NON è fra questi: cambiarla non richiede di
riaddestrare la rete, quindi si calibra dopo il training scandendo una
griglia logaritmica e tenendo il valore con la NLL di validation più bassa
(core.deep_models_laplace.calibra_prior_precision). Metterla in Optuna
costerebbe un training completo per valore senza guadagnare nulla. Il valore
scelto viene comunque salvato fra gli iperparametri, per tracciabilità.

Il punteggio che Optuna minimizza è la NLL di validation DOPO Laplace, cioè
con l'incertezza epistemica inclusa e la precisione del prior già calibrata:
si confronta il metodo completo, non la sola rete puntuale.

Output in results_optuna/:
  laplace_metrics.csv, laplace_preds.csv,
  laplace_hyperparameters.csv, split_composition.csv

Utilizzo:
    python -u run_laplace_optuna.py
    python -u run_laplace_optuna.py --sensors-filter EM-500-9 EM-500-12 --fsets ACQUA --n-trials 3
"""

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
import deep_models_laplace as dml          # noqa: E402  (import in sola lettura)
import eval_uncertainty as eu              # noqa: E402  (import in sola lettura, invariato)

OUT_DIR = os.path.join(HERE, "results_optuna")

TARGET_COLS = [f"target_t{h}" for h in HORIZONS]
FSETS_DEFAULT = list(THEMATIC_FEATURE_SETS.keys())
MODEL_NAME = "MLP-Laplace-Optuna"
N_TRIALS_DEFAULT = 100
OPTUNA_SEED = 42

NN_FIXED_UQ = {"patience": 30, "epochs": 300}   # batch_size lo sceglie Optuna

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
# Ricerca iperparametri con Optuna
# ═══════════════════════════════════════════════════════════════════════════

def make_objective(train_df, X_train_imp, Y_train_full, X_val_imp, Y_val_full):
    """Costruisce la funzione obiettivo per QUESTO feature-set. Ogni prova
    allena la rete MAP, ci costruisce sopra Laplace e ne calibra la precisione
    del prior: il valore ritornato è quindi la NLL di validation del metodo
    completo."""

    def objective(trial):
        hidden_key = trial.suggest_categorical("hidden", list(HIDDEN_CHOICES.keys()))
        dropout = trial.suggest_float("dropout", *DROPOUT_RANGE)
        lr = trial.suggest_float("lr", *LR_RANGE, log=True)
        batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
        event_weight = trial.suggest_float("event_weight", *EVENT_WEIGHT_RANGE)

        sample_weight = _make_sample_weights(train_df, event_weight)
        _, _, _, val_nll = dml.fit_laplace_multi(
            X_train_imp, Y_train_full, X_val_imp, Y_val_full,
            sample_weight=sample_weight,
            hidden=HIDDEN_CHOICES[hidden_key], dropout=dropout, lr=lr, batch_size=batch_size,
            patience=NN_FIXED_UQ["patience"], epochs=NN_FIXED_UQ["epochs"],
        )
        return val_nll

    return objective


def tune_with_optuna(train_df, X_train_imp, Y_train_full, X_val_imp, Y_val_full, n_trials):
    """Esegue lo study Optuna per un feature-set e ritorna (best_params_puliti,
    best_val_nll) — best_params_puliti ha già 'hidden' convertito da chiave
    stringa a tupla, pronto per essere ripassato a fit_laplace_multi."""
    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def trial_callback(study, trial):
        print(f"    [Optuna {trial.number + 1}/{n_trials}] val_NLL={trial.value:.4f}  "
              f"hidden={trial.params['hidden']} lr={trial.params['lr']:.5f} "
              f"dropout={trial.params['dropout']:.3f} batch={trial.params['batch_size']} "
              f"event_weight={trial.params['event_weight']:.2f}")

    objective = make_objective(train_df, X_train_imp, Y_train_full, X_val_imp, Y_val_full)
    study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])

    best_params = dict(study.best_params)
    best_params["hidden"] = HIDDEN_CHOICES[best_params["hidden"]]
    return best_params, study.best_value


# ═══════════════════════════════════════════════════════════════════════════
# Bookkeeping: metriche + predizioni per (feature-set, split), per orizzonte
# + aggregato "ALL" — stesso schema di mc_dropout/
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


def record_hyperparameters(hparam_rows, fset_name, best_params, val_nll, n_features):
    hparam_rows.append({
        "model": MODEL_NAME, "feature_set": fset_name,
        "n_features": n_features,
        "val_NLL": round(float(val_nll), 4),
        "params": json.dumps(best_params, default=str, sort_keys=True),
    })


# ═══════════════════════════════════════════════════════════════════════════
# Loop principale
# ═══════════════════════════════════════════════════════════════════════════

def run_uncertainty_experiments(datasets, fsets, n_trials,
                                train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    metric_rows = []
    pred_rows = []
    hparam_rows = []

    start_time = time.time()

    def elapsed_time_str():
        seconds = time.time() - start_time
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    print("\nCostruisco lo split canonico (una sola volta, stesso split delle altre "
          "pipeline di pipeline_corrette/)...")
    canonical_split = build_canonical_split(datasets, TARGET_COLS, train_frac, val_frac)
    if canonical_split.empty:
        raise RuntimeError("Split canonico vuoto: controlla i dati/sensori in ingresso.")

    train_df = canonical_split[canonical_split["_split"] == "train"].reset_index(drop=True)
    val_df = canonical_split[canonical_split["_split"] == "val"].reset_index(drop=True)
    test_df = canonical_split[canonical_split["_split"] == "test"].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Split canonico vuoto: controlla i dati/sensori in ingresso.")

    split_composition = compute_split_composition(canonical_split)

    # --- Filtro di comparabilità con Penman -------------------------------
    penman_pred_cols = [f"penman_pred_t{h}" for h in HORIZONS]
    n_train_canon, n_val_canon, n_test_canon = len(train_df), len(val_df), len(test_df)
    train_df = train_df.dropna(subset=penman_pred_cols).reset_index(drop=True)
    val_df = val_df.dropna(subset=penman_pred_cols).reset_index(drop=True)
    test_df = test_df.dropna(subset=penman_pred_cols).reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Split vuoto dopo il filtro di comparabilità Penman: controlla dati/sensori.")

    print(f"  n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)}  "
          f"(canonico: train={n_train_canon}/val={n_val_canon}/test={n_test_canon}, ridotto dal filtro "
          f"Penman per comparabilità — atteso, mostrato per trasparenza)")
    print("  STESSO train_df/val_df/test_df riusato per ogni feature-set, nessun nuovo pool/split")

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

        print(f"  n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)}  "
              f"n_features={len(available_features)}  (gruppi: {', '.join(group_names)})")
        print(f"  ricerca Optuna: {n_trials} prove (hidden/lr/dropout/batch_size/event_weight; "
              f"prior_precision calibrata dopo, su validation)")

        # --- Ricerca Optuna via validation NLL post-Laplace -----------------
        best_params, best_val_nll = tune_with_optuna(
            train_df, X_train_imp, Y_train_full, X_val_imp, Y_val_full, n_trials,
        )
        print(f"  migliori iperparametri: {best_params}  (val_NLL={best_val_nll:.4f})")

        # --- Riallena UNA volta con i migliori iperparametri trovati --------
        sample_weight = _make_sample_weights(train_df, best_params["event_weight"])
        laplace, feature_scaler, y_stats, _ = dml.fit_laplace_multi(
            X_train_imp, Y_train_full, X_val_imp, Y_val_full,
            sample_weight=sample_weight,
            hidden=best_params["hidden"], dropout=best_params["dropout"],
            lr=best_params["lr"], batch_size=best_params["batch_size"],
            patience=NN_FIXED_UQ["patience"], epochs=NN_FIXED_UQ["epochs"],
        )
        # la precisione del prior è scelta dentro fit_laplace_multi (non da
        # Optuna): si annota qui, dopo il riaddestramento, per tracciabilità
        best_params["prior_precision"] = laplace.prior_precision
        print(f"  prior_precision calibrata su validation: {laplace.prior_precision:.4g}")
        record_hyperparameters(hparam_rows, fset_name, best_params, best_val_nll, len(available_features))

        # --- Predizione in forma chiusa + metriche, su train/val/test -------
        splits_to_evaluate = [
            ("train", train_df, X_train_imp, Y_train_full),
            ("val", val_df, X_val_imp, Y_val_full),
            ("test", test_df, X_test_imp, Y_test_full),
        ]
        for split_name, split_df, X_split_imp, Y_split in splits_to_evaluate:
            mu_pred, epistemic_std, aleatoric_std, total_std = dml.laplace_predict(
                laplace, feature_scaler, y_stats, X_split_imp,
            )
            record_results(
                metric_rows, pred_rows, fset_name, split_name,
                Y_split, mu_pred, epistemic_std, aleatoric_std, total_std,
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
          "Penman-Monteith — serve solo a determinare quali (sensore, giorno) sono comparabili con "
          "Penman, le feature ML restano quelle componibili...")
    colture_df = load_colture(args.colture)
    colture_df = apply_soil_overrides(colture_df, fc=args.fc, wp=args.wp, sat=args.sat)
    sensori_df = load_sensori(args.sensors)
    datasets_penman = enrich_all_with_penman(datasets, sensori_df, colture_df, args.alpha_rain, args.alpha_irr)

    metrics_df, preds_df, hparams_df, split_composition = run_uncertainty_experiments(
        datasets_penman, fsets=args.fsets, n_trials=args.n_trials,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )

    metrics_path = os.path.join(args.out, "laplace_metrics.csv")
    preds_path = os.path.join(args.out, "laplace_preds.csv")
    hparams_path = os.path.join(args.out, "laplace_hyperparameters.csv")
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
