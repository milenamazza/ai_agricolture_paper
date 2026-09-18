"""
deep_models_multi.py
─────────────────────────────────────────────────────────────────────────────
Modelli PyTorch MULTI-OUTPUT (MLP, LSTM): un solo modello predice
contemporaneamente tutti gli orizzonti (es. t+1..t+7) invece di un modello
separato per ciascuno. Modulo scritto da zero e autosufficiente (nessun
import da data_driven_ml/deep_models.py) per tenere questa cartella
compartimentata — stesso pattern/stesse scelte di determinismo, un po' di
duplicazione accettata a questo scopo.

- MLPRegressorMulti: stesse feature "flat" dei modelli baseline, testa
  finale a n_outputs invece di 1.
- LSTMRegressorMulti: sequenze costruite da build_sequences_multi() con una
  finestra di lookback giorni consecutivi per sensore. Finestre con anche un
  solo NaN (giorno mancante, feature mancante, o anche un solo dei target
  mancante) vengono scartate, nessun riempimento artificiale.

Determinismo: verificato empiricamente (doppio run bit-esatto) su GPU con
CUBLAS_WORKSPACE_CONFIG impostata prima di importare torch + cudnn
deterministico + generator fisso per lo shuffling del DataLoader. Se CUDA
non è disponibile si usa la CPU (deterministica di suo).
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOOKBACK_DEFAULT = 14

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
print(f"[deep_models_multi] device = {DEVICE}")


def _set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ═══════════════════════════════════════════════════════════════════════════
# Architetture multi-output
# ═══════════════════════════════════════════════════════════════════════════

class MLPRegressorMulti(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, hidden=(64, 32), dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, n_outputs)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)   # (batch, n_outputs) — nessuno squeeze, multi-output


class LSTMRegressorMulti(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, hidden: int = 32,
                num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_outputs)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])   # (batch, n_outputs)


# ═══════════════════════════════════════════════════════════════════════════
# Training loop generico multi-output
# ═══════════════════════════════════════════════════════════════════════════

def _train_loop_multi(model, X_tr, Y_tr, X_val, Y_val, sample_weight=None,
                      epochs=200, lr=1e-3, batch_size=64, patience=15):
    _set_seed()
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss(reduction="none")

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32)
    w_t = (torch.tensor(sample_weight, dtype=torch.float32).unsqueeze(-1)
           if sample_weight is not None else torch.ones(len(Y_tr_t), 1))
    gen = torch.Generator()
    gen.manual_seed(SEED)
    dl = DataLoader(TensorDataset(X_tr_t, Y_tr_t, w_t),
                    batch_size=batch_size, shuffle=True, generator=gen)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(DEVICE)

    best_state, best_val, bad_epochs = None, np.inf, 0
    for _ in range(epochs):
        model.train()
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            opt.zero_grad()
            loss = (loss_fn(model(xb), yb) * wb).mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_mae = torch.mean(torch.abs(model(X_val_t) - Y_val_t)).item()

        if val_mae < best_val - 1e-5:
            best_val, bad_epochs = val_mae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# ═══════════════════════════════════════════════════════════════════════════
# MLP multi-output — feature flat (già imputate)
# ═══════════════════════════════════════════════════════════════════════════

def fit_mlp_multi(X_tr, Y_tr, X_val, Y_val, sample_weight=None, hidden=(64, 32),
                  lr=1e-3, batch_size=64, patience=15, epochs=200):
    _set_seed()   # deve precedere la costruzione del modello
    scaler = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # normalizzazione target PER COLONNA (ogni orizzonte ha scala/varianza
    # diversa) — senza, la convergenza è troppo lenta rispetto alla scala di
    # moisture e l'early stopping ferma la rete su un output quasi costante.
    y_mean = Y_tr.mean(axis=0)
    y_std  = Y_tr.std(axis=0) + 1e-8
    Y_tr_s  = (Y_tr  - y_mean) / y_std
    Y_val_s = (Y_val - y_mean) / y_std

    model = MLPRegressorMulti(n_features=X_tr.shape[1], n_outputs=Y_tr.shape[1], hidden=hidden)
    model, val_mae = _train_loop_multi(model, X_tr_s, Y_tr_s, X_val_s, Y_val_s, sample_weight,
                                       epochs=epochs, lr=lr, batch_size=batch_size, patience=patience)
    return model, scaler, (y_mean, y_std)


def predict_mlp_multi(model, scaler, y_stats, X):
    model.eval()
    X_s = scaler.transform(X)
    with torch.no_grad():
        pred_s = model(torch.tensor(X_s, dtype=torch.float32).to(DEVICE))
    y_mean, y_std = y_stats
    return pred_s.cpu().numpy() * y_std + y_mean


# ═══════════════════════════════════════════════════════════════════════════
# LSTM multi-output — costruzione sequenze per sensore
# ═══════════════════════════════════════════════════════════════════════════

def build_sequences_multi(df: pd.DataFrame, feature_cols: list, target_cols: list,
                          split_col: str, lookback: int = LOOKBACK_DEFAULT,
                          extra_cols: list | None = None):
    """
    Come build_sequences ma con una LISTA di target (uscita multi-output).
    Ritorna X (n_seq, lookback, n_feat), Y (n_seq, n_targets), meta (DataFrame
    con date/device/split[/extra_cols] allineato riga per riga). Finestre con
    un giorno mancante, una feature NaN, o anche un solo dei target NaN
    vengono scartate — nessun riempimento artificiale.
    """
    extra_cols = extra_cols or []
    X_list, Y_list, meta_rows = [], [], []
    for device, g in df.groupby("device", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        dates   = pd.to_datetime(g["date"]).values
        feats   = g[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float).values
        targets = g[target_cols].apply(pd.to_numeric, errors="coerce").astype(float).values
        splits  = g[split_col].values if split_col in g.columns else None
        extras  = {c: g[c].values for c in extra_cols if c in g.columns}

        for i in range(lookback - 1, len(g)):
            window_dates = dates[i - lookback + 1: i + 1]
            diffs = np.diff(window_dates).astype("timedelta64[D]").astype(int)
            if not np.all(diffs == 1):
                continue
            window_feats = feats[i - lookback + 1: i + 1]
            if np.isnan(window_feats).any():
                continue
            y_row = targets[i]
            if np.isnan(y_row).any():
                continue
            if splits is not None and pd.isna(splits[i]):
                continue
            X_list.append(window_feats)
            Y_list.append(y_row)
            meta_rows.append({
                **{c: v[i] for c, v in extras.items()},
                "date": dates[i], "device": device,
                "split": splits[i] if splits is not None else None,
            })

    if not X_list:
        return (np.empty((0, lookback, len(feature_cols))),
                np.empty((0, len(target_cols))),
                pd.DataFrame(columns=["date", "device", "split"]))
    return np.stack(X_list), np.stack(Y_list), pd.DataFrame(meta_rows)


def fit_lstm_multi(X_tr, Y_tr, X_val, Y_val, sample_weight=None, hidden=32,
                   lr=1e-3, batch_size=64, patience=15, epochs=200):
    _set_seed()   # deve precedere la costruzione del modello
    n_t, lookback, n_feat = X_tr.shape
    scaler = StandardScaler().fit(X_tr.reshape(-1, n_feat))
    X_tr_s  = scaler.transform(X_tr.reshape(-1, n_feat)).reshape(n_t, lookback, n_feat)
    X_val_s = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape[0], lookback, n_feat)

    y_mean = Y_tr.mean(axis=0)
    y_std  = Y_tr.std(axis=0) + 1e-8
    Y_tr_s  = (Y_tr  - y_mean) / y_std
    Y_val_s = (Y_val - y_mean) / y_std

    model = LSTMRegressorMulti(n_features=n_feat, n_outputs=Y_tr.shape[1], hidden=hidden)
    model, val_mae = _train_loop_multi(model, X_tr_s, Y_tr_s, X_val_s, Y_val_s, sample_weight,
                                       epochs=epochs, lr=lr, batch_size=batch_size, patience=patience)
    return model, scaler, (y_mean, y_std)


def predict_lstm_multi(model, scaler, y_stats, X):
    n, lookback, n_feat = X.shape
    X_s = scaler.transform(X.reshape(-1, n_feat)).reshape(n, lookback, n_feat)
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(X_s, dtype=torch.float32).to(DEVICE))
    y_mean, y_std = y_stats
    return pred_s.cpu().numpy() * y_std + y_mean


# ═══════════════════════════════════════════════════════════════════════════
# Tuning deterministico via validation (stesso schema di run_experiments.py,
# generalizzato al caso multi-output)
# ═══════════════════════════════════════════════════════════════════════════

def tune_mlp_multi_val(X_tr, Y_tr, X_val, Y_val, grid, sample_weight=None, nn_fixed=None):
    nn_fixed = nn_fixed or {}
    best_score, best_params, best = np.inf, None, None
    for params in ParameterGrid(grid):
        model, scaler, y_stats = fit_mlp_multi(X_tr, Y_tr, X_val, Y_val,
                                               sample_weight=sample_weight, **params, **nn_fixed)
        pred = predict_mlp_multi(model, scaler, y_stats, X_val)
        score = float(mean_absolute_error(Y_val, pred))
        if score < best_score:
            best_score, best_params = score, params
            best = (model, scaler, y_stats)
    return best, best_params, best_score


def tune_lstm_multi_val(X_tr, Y_tr, X_val, Y_val, grid, nn_fixed=None):
    nn_fixed = nn_fixed or {}
    best_score, best_params, best = np.inf, None, None
    for params in ParameterGrid(grid):
        model, scaler, y_stats = fit_lstm_multi(X_tr, Y_tr, X_val, Y_val, **params, **nn_fixed)
        pred = predict_lstm_multi(model, scaler, y_stats, X_val)
        score = float(mean_absolute_error(Y_val, pred))
        if score < best_score:
            best_score, best_params = score, params
            best = (model, scaler, y_stats)
    return best, best_params, best_score
