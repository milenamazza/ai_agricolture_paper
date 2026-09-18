"""
deep_models_ensemble.py
─────────────────────────────────────────────────────────────────────────────
Deep Ensemble: un solo modello MLP multi-output (predice tutti e 7 gli
orizzonti insieme, come uncertainty_ml/), ma invece di un solo modello con
MC-Dropout attivo in predizione, si allenano N reti INDIPENDENTI (stessa
architettura, stesso training, ma pesi iniziali diversi — un seed diverso a
testa) e si combinano le loro previsioni.

Idea di fondo: ciascuna rete, allenata da sola sugli stessi dati, può
convergere a una soluzione leggermente diversa (i pesi iniziali influenzano
dove finisce la discesa del gradiente). Se le N reti sono tutte d'accordo su
un input, vuol dire che i dati di training determinavano bene la risposta
lì; se divergono, è un segnale che il modello "non sa bene" cosa rispondere
in quella zona — questa è la stima dell'incertezza EPISTEMICA.

Ogni singola rete della ensemble è comunque una testa doppia (mu, log_var)
allenata con la NLL gaussiana eteroscedastica, esattamente come in
uncertainty_ml/deep_models_uncertainty.py — quindi cattura ANCHE l'incertezza
ALEATORIA. Combinare ensembling (per l'epistemica) con una testa
eteroscedastica per rete (per l'aleatoria) è lo schema originale di
Lakshminarayanan, Pritzel, Blundell, "Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles" (NeurIPS 2017).

Perché questo file non importa semplicemente da uncertainty_ml/: l'unica
differenza rispetto a MLPGaussianMulti è che qui serve poter fissare un seed
DIVERSO per ciascun membro dell'ensemble (in uncertainty_ml/ il seed è
sempre lo stesso, fissato una volta per tutte per garantire determinismo
bit-esatto di un singolo modello) — una modifica che richiederebbe toccare
un file di un'altra cartella. Per restare compartimentati (nessuna modifica
a uncertainty_ml/), l'architettura è duplicata qui con questa piccola
estensione — stessa scelta già fatta nel progetto per deep_models_multi.py.
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
from sklearn.model_selection import ParameterGrid

BASE_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
print(f"[deep_models_ensemble] device = {DEVICE}")

LOG_VAR_MIN = -8.0
LOG_VAR_MAX = 8.0


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ═══════════════════════════════════════════════════════════════════════════
# Architettura: identica a MLPGaussianMulti di uncertainty_ml/ (stessa testa
# doppia mu/log_var, stesso clamp di sicurezza numerica — vedi quel file per
# la spiegazione completa del perché log_var e del perché il clamp)
# ═══════════════════════════════════════════════════════════════════════════

class MLPGaussianMulti(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, hidden=(64, 32), dropout: float = 0.2):
        super().__init__()

        trunk_layers = []
        input_dim = n_features
        for hidden_dim in hidden:
            trunk_layers.append(nn.Linear(input_dim, hidden_dim))
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

        trunk_output_dim = input_dim
        self.mu_head = nn.Linear(trunk_output_dim, n_outputs)
        self.logvar_head = nn.Linear(trunk_output_dim, n_outputs)

    def forward(self, x):
        shared_features = self.trunk(x)
        mu = self.mu_head(shared_features)
        log_var = self.logvar_head(shared_features)
        log_var = torch.clamp(log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX)
        return mu, log_var


def gaussian_nll_loss(mu, log_var, y_true, sample_weight=None):
    """Stessa NLL gaussiana eteroscedastica di uncertainty_ml/
    deep_models_uncertainty.py — vedi quel file per la spiegazione dettagliata
    dei due termini della formula."""
    inverse_variance = torch.exp(-log_var)
    squared_error = (y_true - mu) ** 2
    per_output_loss = 0.5 * log_var + 0.5 * inverse_variance * squared_error
    per_row_loss = per_output_loss.mean(dim=1)

    if sample_weight is not None:
        return (per_row_loss * sample_weight).mean()
    return per_row_loss.mean()


# ═══════════════════════════════════════════════════════════════════════════
# Training di UNA rete (un membro dell'ensemble), con seed esplicito
# ═══════════════════════════════════════════════════════════════════════════

def _train_loop_gaussian(model, X_tr, Y_tr, X_val, Y_val, seed, sample_weight=None,
                         epochs=300, lr=1e-3, batch_size=32, patience=30):
    """
    Identico allo schema di uncertainty_ml/ (stesso DataLoader con generator
    fisso, stesso early stopping sulla NLL di validation), ma il seed che
    determina sia l'inizializzazione dei pesi sia l'ordine di shuffling del
    DataLoader è un parametro esplicito, non una costante globale — è
    esattamente questo che rende ogni membro dell'ensemble diverso dagli
    altri, a parità di dati e di iperparametri.
    """
    _set_seed(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
    Y_tr_tensor = torch.tensor(Y_tr, dtype=torch.float32)
    if sample_weight is not None:
        weight_tensor = torch.tensor(sample_weight, dtype=torch.float32)
    else:
        weight_tensor = torch.ones(len(Y_tr_tensor))

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_tr_tensor, Y_tr_tensor, weight_tensor),
        batch_size=batch_size, shuffle=True, generator=generator,
    )

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(DEVICE)

    best_model_state = None
    best_val_nll = np.inf
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        for X_batch, Y_batch, weight_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            weight_batch = weight_batch.to(DEVICE)

            optimizer.zero_grad()
            mu_batch, log_var_batch = model(X_batch)
            loss = gaussian_nll_loss(mu_batch, log_var_batch, Y_batch, weight_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            mu_val, log_var_val = model(X_val_tensor)
            val_nll = gaussian_nll_loss(mu_val, log_var_val, Y_val_tensor).item()

        if val_nll < best_val_nll - 1e-5:
            best_val_nll = val_nll
            epochs_without_improvement = 0
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_nll


def fit_mlp_gaussian_multi(X_tr, Y_tr, X_val, Y_val, seed, sample_weight=None,
                           hidden=(64, 32), dropout=0.2, lr=1e-3,
                           batch_size=32, patience=30, epochs=300):
    """Come uncertainty_ml/fit_mlp_gaussian_multi, ma prende il seed come
    parametro esplicito invece di usare sempre la stessa costante globale —
    è quello che permette di allenare N membri diversi dell'ensemble."""
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
    model, best_val_nll = _train_loop_gaussian(
        model, X_tr_scaled, Y_tr_scaled, X_val_scaled, Y_val_scaled, seed,
        sample_weight=sample_weight, epochs=epochs, lr=lr,
        batch_size=batch_size, patience=patience,
    )

    y_stats = (y_mean, y_std)
    return model, feature_scaler, y_stats, best_val_nll


# ═══════════════════════════════════════════════════════════════════════════
# Predizione di UN membro (forward pass singolo, deterministico — a
# differenza di MC-Dropout qui NON si riattiva il dropout in predizione: la
# diversità viene dall'aver allenato reti diverse, non dal campionamento)
# ═══════════════════════════════════════════════════════════════════════════

def predict_single_member(model, feature_scaler, y_stats, X):
    """Un solo forward pass in modalità normale (model.eval(), dropout
    disattivo). Ritorna mu e sigma di QUESTO SOLO membro, in unità originali
    di moisture."""
    model.eval()
    X_scaled = feature_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        mu_scaled, log_var_scaled = model(X_tensor)

    y_mean, y_std = y_stats
    mu = mu_scaled.cpu().numpy() * y_std + y_mean
    sigma = np.sqrt(np.exp(log_var_scaled.cpu().numpy())) * y_std
    return mu, sigma


def combine_ensemble_predictions(member_mu_list, member_sigma_list):
    """
    Combina le previsioni di N membri dell'ensemble in un'unica previsione
    con incertezza scomposta, usando la stessa formula (matematicamente
    identica) già usata per combinare i T campioni MC-Dropout in
    uncertainty_ml/deep_models_uncertainty.py::mc_predict_mlp_gaussian_multi
    — cambia solo la fonte delle N coppie (mu, sigma): lì erano N forward
    pass della STESSA rete con dropout casuale, qui sono N reti DIVERSE.

    member_mu_list, member_sigma_list: liste di N array (n_rows, n_outputs).
    """
    mu_stack = np.stack(member_mu_list, axis=0)       # (N, n_rows, n_outputs)
    sigma_stack = np.stack(member_sigma_list, axis=0)  # (N, n_rows, n_outputs)

    mu_ensemble = mu_stack.mean(axis=0)
    epistemic_variance = mu_stack.var(axis=0)
    aleatoric_variance = (sigma_stack ** 2).mean(axis=0)
    total_variance = epistemic_variance + aleatoric_variance

    epistemic_std = np.sqrt(epistemic_variance)
    aleatoric_std = np.sqrt(aleatoric_variance)
    total_std = np.sqrt(total_variance)

    return mu_ensemble, epistemic_std, aleatoric_std, total_std


# ═══════════════════════════════════════════════════════════════════════════
# Tuning deterministico via validation NLL (una sola rete, seed=BASE_SEED —
# vedi run_ensemble.py per il motivo: allenare N reti per OGNI combinazione
# della griglia moltiplicherebbe il costo per N, non è la prassi standard)
# ═══════════════════════════════════════════════════════════════════════════

def tune_mlp_gaussian_multi_val(X_tr, Y_tr, X_val, Y_val, grid, sample_weight=None, nn_fixed=None):
    nn_fixed = nn_fixed or {}

    best_val_nll = np.inf
    best_params = None

    for params in ParameterGrid(grid):
        _, _, _, val_nll = fit_mlp_gaussian_multi(
            X_tr, Y_tr, X_val, Y_val, seed=BASE_SEED,
            sample_weight=sample_weight, **params, **nn_fixed,
        )
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_params = params

    return best_params, best_val_nll
