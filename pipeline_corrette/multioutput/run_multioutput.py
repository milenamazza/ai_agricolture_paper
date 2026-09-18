"""
run_multioutput_componibile_comparabile_penman.py
─────────────────────────────────────────────────────────────────────────────
Come run_multioutput_componibile.py (Ridge/RF/MLP/LSTM predicono il target
ASSOLUTO per feature-set componibile, tutti e 7 gli orizzonti insieme — vedi
quel file per la spiegazione completa), ma il training/valutazione avviene
sul sotto-insieme del canonico dove Penman-Monteith produce una previsione
valida per TUTTI i 7 orizzonti — stesso identico filtro già usato in
uncertainty_componibile_optuna, deep_ensemble_componibile_optuna,
quantile_regression_componibile_optuna e
penman_residual_componibile_solo_feature.

Perché: il confronto con Penman (confronto/build_report_componibile.py)
oggi restringe la finestra comune DOPO che ML puro e Penman sono stati
allenati/valutati separatamente sui rispettivi insiemi. Filtrando QUI, PRIMA
del training, il confronto con Penman non richiede più nessuna restrizione
a posteriori: questa pipeline vede fin dall'inizio solo i (sensore, giorno)
che Penman può effettivamente valutare.

Le feature ML restano ESATTAMENTE quelle componibili (feature_groups_util) —
NESSUNA colonna fisica Penman come input al modello: Penman entra qui solo
per determinare quali righe sono comparabili, mai come feature. Si usa
enrich_all_with_penman (la fisica "base" di penman_baseline/run_penman.py,
la STESSA già usata nelle pipeline di incertezza dirette Optuna) — non la
variante "_ext", che aggiunge colonne pensate per essere usate come feature
(qui non servono).

Riuso in sola lettura (nessuna modifica ai file sorgenti):
  - penman_baseline/run_penman.py: enrich_all_with_penman — fisica INVARIATA.
  - data_driven_ml_multioutput/deep_models_multi.py: import in sola lettura,
    INVARIATO — stessa architettura MLP/LSTM multi-output.
  - modelli singoli/data_driven_ml/utils.py: MIN_SAMPLES, SEED, HORIZONS,
    SAT_OVERRIDE, load_colture, load_sensori, apply_soil_overrides,
    _make_sample_weights, _tune_val, GRID_RIDGE, GRID_RF.
  - modelli singoli/data_driven_ml/run_experiments.py: DEFAULT_DATASET,
    DEFAULT_SENSORS, TRAIN_FRAC, VAL_FRAC.
  - ai_feature_componibile/{feature_groups_util.py,dati_ricostruiti_util.py}.

Output in results_comparabile_penman/ (cartella separata da results/, per
non confondere con la versione sull'intero canonico):
  experiments_comparabile_penman_metrics.csv
  experiments_comparabile_penman_preds.csv
  experiments_comparabile_penman_hyperparameters.csv
  split_composition.csv

Utilizzo:
    python -u run_multioutput_componibile_comparabile_penman.py
    python -u run_multioutput_componibile_comparabile_penman.py --sensors-filter EM-500-9 EM-500-12 --fsets ACQUA TERRA --no-lstm
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
    _make_sample_weights, _tune_val,
    GRID_RIDGE, GRID_RF,
)
from paths import (DEFAULT_DATASET, DEFAULT_SENSORS,  # noqa: E402
                   DEFAULT_IRRIGAZIONE_RICOSTRUITA, DEFAULT_COLTURE,
                   TRAIN_FRAC, VAL_FRAC, require_data_files)
from penman import enrich_all_with_penman  # noqa: E402  (import in sola lettura, invariato)
import deep_models_multi as dmm  # noqa: E402  (import in sola lettura, invariato)

OUT_DIR = os.path.join(HERE, "results_comparabile_penman")

EVENT_WEIGHT = 3
LOOKBACK = 14
TARGET_COLS = [f"target_t{h}" for h in HORIZONS]
PENMAN_PRED_COLS = [f"penman_pred_t{h}" for h in HORIZONS]
FSETS_DEFAULT = list(THEMATIC_FEATURE_SETS.keys())

NN_FIXED  = {"batch_size": 32, "patience": 50, "epochs": 300}
GRID_MLP  = {"hidden": [(32,), (64, 32), (128, 64, 32), (32, 32)], "lr": [1e-2, 1e-3, 5e-4, 1e-4]}
GRID_LSTM = {"hidden": [16, 32, 64], "lr": [1e-3, 5e-4]}


def impute_train_val_test(X_train, X_val, X_test):
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X_train), imputer.transform(X_val), imputer.transform(X_test)


# ═══════════════════════════════════════════════════════════════════════════
# Metriche: per orizzonte + complessivo ("ALL"), per OGNI split — IDENTICA a
# run_multioutput_componibile.py.
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

def run_experiments(datasets_penman, fsets, run_mlp=True, run_lstm=True,
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
    # Stesso identico filtro già usato nelle pipeline di incertezza dirette
    # Optuna e in penman_residual_componibile_solo_feature: scarta le righe
    # senza TUTTI i penman_pred_t{h} validi. Applicato sia alle 3 slice sia
    # al canonical_split completo (serve anche all'LSTM più sotto).
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
        sw = _make_sample_weights(train_df, event_weight)

        print(f"  n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)}  "
              f"n_features={len(available_features)}  (gruppi: {', '.join(group_names)})")

        split_infos = [
            ("train", X_train_imp, Y_train_full, dates_train, devices_train, ev_mask_train, dc_mask_train),
            ("val", X_val_imp, Y_val_full, dates_val, devices_val, ev_mask_val, dc_mask_val),
            ("test", X_test_imp, Y_test_full, dates_test, devices_test, ev_mask_test, dc_mask_test),
        ]

        # --- Ridge (multi-output nativo) -------------------------------------
        model, best_params, val_mae = _tune_val(
            Ridge, GRID_RIDGE, X_train_imp, Y_train_full, X_val_imp, Y_val_full, sample_weight=sw)
        predict_and_record_all_splits(metric_rows, pred_rows, "Ridge", fset_name,
                                      model.predict, event_weight, split_infos)
        _record_hparams(hparam_rows, "Ridge", fset_name, best_params, val_mae, len(available_features))

        # --- RF (multi-output nativo) -----------------------------------------
        model, best_params, val_mae = _tune_val(
            RandomForestRegressor, GRID_RF, X_train_imp, Y_train_full, X_val_imp, Y_val_full,
            sample_weight=sw, random_state=SEED, n_jobs=-1)
        predict_and_record_all_splits(metric_rows, pred_rows, "RF", fset_name,
                                      model.predict, event_weight, split_infos)
        _record_hparams(hparam_rows, "RF", fset_name, best_params, val_mae, len(available_features))

        # --- MLP multi-output ---------------------------------------------------
        if run_mlp:
            (model, scaler, y_stats), best_params, val_mae = dmm.tune_mlp_multi_val(
                X_train_imp, Y_train_full, X_val_imp, Y_val_full, GRID_MLP, sample_weight=sw, nn_fixed=NN_FIXED)
            predict_fn = lambda X: dmm.predict_mlp_multi(model, scaler, y_stats, X)  # noqa: E731
            predict_and_record_all_splits(metric_rows, pred_rows, "MLP", fset_name,
                                          predict_fn, event_weight, split_infos)
            _record_hparams(hparam_rows, "MLP", fset_name, {**best_params, **NN_FIXED}, val_mae, len(available_features))

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
                    (model, scaler, y_stats), best_params, val_mae = dmm.tune_lstm_multi_val(
                        X_seq[train_mask], Y_seq[train_mask], X_seq[val_mask], Y_seq[val_mask],
                        GRID_LSTM, nn_fixed=NN_FIXED)

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
        datasets_penman, fsets=args.fsets,
        run_mlp=not args.no_mlp, run_lstm=not args.no_lstm,
        event_weight=args.event_weight, lookback=args.lookback,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )

    mm_path = os.path.join(args.out, "experiments_comparabile_penman_metrics.csv")
    mp_path = os.path.join(args.out, "experiments_comparabile_penman_preds.csv")
    hp_path = os.path.join(args.out, "experiments_comparabile_penman_hyperparameters.csv")
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
