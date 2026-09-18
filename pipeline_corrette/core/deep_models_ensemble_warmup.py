"""
deep_models_ensemble_warmup.py
─────────────────────────────────────────────────────────────────────────────
Deep Ensemble con WARM-UP della media: ogni membro dell'ensemble viene
allenato in due fasi (prima solo mu con l'MSE, poi tutta la rete con la NLL
gaussiana) invece che con la NLL dalla prima epoca. Motivazione, dettagli
delle due fasi e cosa aspettarsi: vedi deep_models_uncertainty_warmup.py.

Tutto il resto è importato, non copiato, per garantire che rispetto a
deep_models_ensemble.py cambi SOLO il training:
  - da deep_models_ensemble.py: architettura del membro (MLPGaussianMulti),
    BASE_SEED, predizione di un membro e combinazione dei membri;
  - da deep_models_uncertainty_warmup.py: il ciclo a due fasi
    _train_loop_warmup, che accetta già un seed esplicito — è proprio ciò che
    distingue un membro dall'altro.

Diagnostica: dopo il fit, model.warmup_info contiene epoche e punteggi delle
due fasi per quel membro.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from sklearn.preprocessing import StandardScaler

from deep_models_ensemble import (  # noqa: F401  (import in sola lettura; alcuni nomi sono riesportati)
    BASE_SEED, _set_seed, MLPGaussianMulti,
    predict_single_member, combine_ensemble_predictions,
)
from deep_models_uncertainty_warmup import _train_loop_warmup  # noqa: E402


def fit_mlp_gaussian_multi(X_tr, Y_tr, X_val, Y_val, seed, sample_weight=None,
                           hidden=(64, 32), dropout=0.2, lr=1e-3,
                           batch_size=32, patience=30, epochs=300):
    """Stessa firma e stesso ritorno di deep_models_ensemble.fit_mlp_gaussian_multi
    — (model, feature_scaler, y_stats, best_val_nll), seed esplicito per membro
    — e stessa normalizzazione. Cambia solo il training."""
    _set_seed(seed)   # deve precedere la costruzione del modello: i pesi si inizializzano lì

    feature_scaler = StandardScaler().fit(X_tr)
    X_tr_scaled = feature_scaler.transform(X_tr)
    X_val_scaled = feature_scaler.transform(X_val)

    y_mean = Y_tr.mean(axis=0)
    y_std = Y_tr.std(axis=0) + 1e-8
    Y_tr_scaled = (Y_tr - y_mean) / y_std
    Y_val_scaled = (Y_val - y_mean) / y_std

    model = MLPGaussianMulti(
        n_features=X_tr.shape[1], n_outputs=Y_tr.shape[1],
        hidden=hidden, dropout=dropout,
    )
    model, best_val_nll = _train_loop_warmup(
        model, X_tr_scaled, Y_tr_scaled, X_val_scaled, Y_val_scaled, seed=seed,
        sample_weight=sample_weight, epochs=epochs, lr=lr,
        batch_size=batch_size, patience=patience,
    )

    y_stats = (y_mean, y_std)
    return model, feature_scaler, y_stats, best_val_nll
