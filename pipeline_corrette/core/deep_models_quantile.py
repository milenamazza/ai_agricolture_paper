"""
deep_models_quantile.py
─────────────────────────────────────────────────────────────────────────────
Quantile Regression: un solo modello MLP multi-output (predice tutti e 7 gli
orizzonti insieme, come uncertainty_ml/ e deep_ensemble_ml/), ma invece di
assumere che l'errore sia gaussiano e prevedere (media, varianza), la rete
prevede DIRETTAMENTE alcuni percentili della distribuzione di moisture per
ciascun orizzonte — es. il 5°, 25°, 50° (mediana), 75°, 95° percentile.

Perché farlo: MC-Dropout e Deep Ensemble assumono ENTRAMBI che l'errore
attorno alla previsione sia distribuito a campana (gaussiano). La Quantile
Regression non fa questa assunzione — impara direttamente "sotto quale
valore cade il 5% delle osservazioni", "sotto quale valore cade il 95%",
ecc., qualunque sia la vera forma della distribuzione (che potrebbe essere
asimmetrica, con code più pesanti da un lato, ecc. — cosa abbastanza
plausibile per l'umidità del suolo, che non può superare la saturazione ma
può scendere molto in siccità).

Loss: la PINBALL LOSS (o "quantile loss"), una per ciascun percentile τ
richiesto:

    pinball_loss_τ(y_vero, y_previsto) = max( τ · (y_vero − y_previsto),
                                              (τ − 1) · (y_vero − y_previsto) )

È una loss ASIMMETRICA. Esempio con τ=0.90 (90° percentile):
  - se il modello SOTTOSTIMA (y_previsto < y_vero, quindi l'errore
    y_vero−y_previsto è positivo): si applica il primo termine,
    0.90 · errore — una sottostima costa 0.90 per ogni unità di errore.
  - se il modello SOVRASTIMA (errore negativo): si applica il secondo
    termine, (0.90−1) · errore = −0.10 · errore, che con errore negativo dà
    un numero positivo piccolo — una sovrastima costa solo 0.10 per unità.
Quindi per il 90° percentile sottostimare costa 9 volte più che sovrastimare
— ed è esattamente minimizzando questo squilibrio (9 a 1) che l'ottimo della
loss diventa proprio il 90° percentile vero della distribuzione: il modello
"conviene" spingere la previsione abbastanza in alto da sbagliare per difetto
solo il 10% delle volte. Lo stesso ragionamento, con pesi diversi, vale per
qualunque τ.

Che incertezza cattura: qui non c'è separazione epistemica/aleatoria. Un
intervallo di previsione si legge direttamente dalla distanza tra due
percentili previsti (es. intervallo al 90% = [Q(0.05), Q(0.95)]), e cattura
l'incertezza TOTALE, comunque sia composta.

Problema noto — "quantile crossing": nulla obbliga la rete a prevedere
Q(0.75) > Q(0.50) > Q(0.25) eccetera; soprattutto nelle prime fasi di
training può capitare che si "incrocino". Si applica un ordinamento post-hoc
(vedi predict_mlp_quantile_multi): sempre corretto, non altera la mediana,
sistema solo eventuali codee disordinate.
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

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
print(f"[deep_models_quantile] device = {DEVICE}")

# Percentili previsti da ogni modello. 0.50 dà la previsione puntuale
# (mediana); le coppie (0.25,0.75) e (0.05,0.95) danno rispettivamente un
# intervallo al 50% e al 90%.
QUANTILE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)


def _set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ═══════════════════════════════════════════════════════════════════════════
# Architettura: MLP multi-output, testa finale con n_horizons * n_quantili
# uscite, organizzate per orizzonte e poi per percentile
# ═══════════════════════════════════════════════════════════════════════════

class MLPQuantileMulti(nn.Module):
    def __init__(self, n_features: int, n_horizons: int, quantile_levels=QUANTILE_LEVELS,
                hidden=(64, 32), dropout: float = 0.2):
        super().__init__()

        trunk_layers = []
        input_dim = n_features
        for hidden_dim in hidden:
            trunk_layers.append(nn.Linear(input_dim, hidden_dim))
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

        self.n_horizons = n_horizons
        self.n_quantiles = len(quantile_levels)
        trunk_output_dim = input_dim
        self.quantile_head = nn.Linear(trunk_output_dim, n_horizons * self.n_quantiles)

    def forward(self, x):
        shared_features = self.trunk(x)
        flat_predictions = self.quantile_head(shared_features)
        # Riorganizza da (batch, n_horizons * n_quantiles) a
        # (batch, n_horizons, n_quantiles): una previsione per ciascuna
        # combinazione orizzonte/percentile.
        batch_size = x.shape[0]
        return flat_predictions.view(batch_size, self.n_horizons, self.n_quantiles)


# ═══════════════════════════════════════════════════════════════════════════
# Loss: somma delle pinball loss su tutti i percentili richiesti
# ═══════════════════════════════════════════════════════════════════════════

def pinball_loss(y_true, y_pred_quantiles, quantile_levels_tensor, sample_weight=None):
    """
    y_true:              (batch, n_horizons)
    y_pred_quantiles:     (batch, n_horizons, n_quantiles)
    quantile_levels_tensor: (n_quantiles,) — i valori di tau, es. [0.05,...,0.95]

    Per ciascun orizzonte e ciascun percentile calcola la pinball loss (vedi
    spiegazione nel docstring del modulo), poi media su percentili e
    orizzonti (così ogni riga pesa 1 indipendentemente da quanti orizzonti/
    percentili si stanno prevedendo), infine applica il sample_weight per
    riga se fornito.
    """
    y_true_expanded = y_true.unsqueeze(-1)   # (batch, n_horizons, 1) per il broadcasting
    error = y_true_expanded - y_pred_quantiles   # (batch, n_horizons, n_quantiles)

    over_estimate_cost = (quantile_levels_tensor - 1.0) * error
    under_estimate_cost = quantile_levels_tensor * error
    per_element_loss = torch.maximum(under_estimate_cost, over_estimate_cost)

    per_row_loss = per_element_loss.mean(dim=(1, 2))

    if sample_weight is not None:
        return (per_row_loss * sample_weight).mean()
    return per_row_loss.mean()


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def _train_loop_quantile(model, X_tr, Y_tr, X_val, Y_val, quantile_levels_tensor,
                         sample_weight=None, epochs=300, lr=1e-3, batch_size=32, patience=30):
    """Stesso schema delle altre pipeline (stesso seed, stesso DataLoader con
    generator fisso, stesso early stopping su una validation esplicita), ma
    la loss e il criterio di early stopping sono la pinball loss di
    validation invece della NLL gaussiana o della MAE."""
    _set_seed()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
    Y_tr_tensor = torch.tensor(Y_tr, dtype=torch.float32)
    if sample_weight is not None:
        weight_tensor = torch.tensor(sample_weight, dtype=torch.float32)
    else:
        weight_tensor = torch.ones(len(Y_tr_tensor))

    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(X_tr_tensor, Y_tr_tensor, weight_tensor),
        batch_size=batch_size, shuffle=True, generator=generator,
    )

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(DEVICE)

    best_model_state = None
    best_val_pinball = np.inf
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        for X_batch, Y_batch, weight_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            weight_batch = weight_batch.to(DEVICE)

            optimizer.zero_grad()
            quantile_predictions = model(X_batch)
            loss = pinball_loss(Y_batch, quantile_predictions, quantile_levels_tensor, weight_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_quantile_predictions = model(X_val_tensor)
            val_pinball = pinball_loss(Y_val_tensor, val_quantile_predictions, quantile_levels_tensor).item()

        if val_pinball < best_val_pinball - 1e-5:
            best_val_pinball = val_pinball
            epochs_without_improvement = 0
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_pinball


def fit_mlp_quantile_multi(X_tr, Y_tr, X_val, Y_val, sample_weight=None,
                           hidden=(64, 32), dropout=0.2, lr=1e-3,
                           batch_size=32, patience=30, epochs=300,
                           quantile_levels=QUANTILE_LEVELS):
    """Normalizza feature e target (media/std sul solo train, come nelle
    altre pipeline) e allena il modello. La normalizzazione del target è
    un'operazione affine (y_norm = (y - media) / std): i percentili di una
    variabile trasformata in modo affine crescente restano gli stessi
    percentili riscalati allo stesso modo, quindi si può tranquillamente
    prevedere in scala normalizzata e poi riconvertire ciascun percentile
    moltiplicando per std e sommando la media — nessuna approssimazione."""
    _set_seed()   # deve precedere la costruzione del modello: i pesi si inizializzano lì

    feature_scaler = StandardScaler().fit(X_tr)
    X_tr_scaled = feature_scaler.transform(X_tr)
    X_val_scaled = feature_scaler.transform(X_val)

    y_mean = Y_tr.mean(axis=0)
    y_std = Y_tr.std(axis=0) + 1e-8
    Y_tr_scaled = (Y_tr - y_mean) / y_std
    Y_val_scaled = (Y_val - y_mean) / y_std

    quantile_levels_tensor = torch.tensor(quantile_levels, dtype=torch.float32).to(DEVICE)

    model = MLPQuantileMulti(
        n_features=X_tr.shape[1], n_horizons=Y_tr.shape[1],
        quantile_levels=quantile_levels, hidden=hidden, dropout=dropout,
    )
    model, best_val_pinball = _train_loop_quantile(
        model, X_tr_scaled, Y_tr_scaled, X_val_scaled, Y_val_scaled, quantile_levels_tensor,
        sample_weight=sample_weight, epochs=epochs, lr=lr,
        batch_size=batch_size, patience=patience,
    )

    y_stats = (y_mean, y_std)
    return model, feature_scaler, y_stats, best_val_pinball


# ═══════════════════════════════════════════════════════════════════════════
# Predizione: singolo forward pass (nessun campionamento, l'incertezza qui
# viene dalla distanza tra i percentili, non da una ripetizione stocastica)
# ═══════════════════════════════════════════════════════════════════════════

def predict_mlp_quantile_multi(model, feature_scaler, y_stats, X):
    """
    Ritorna un array (n_rows, n_horizons, n_quantiles) in unità originali di
    moisture, con i percentili ORDINATI in modo crescente lungo l'ultimo
    asse per ciascuna riga/orizzonte (fix del "quantile crossing" descritto
    nel docstring del modulo: np.sort non altera quale valore è la mediana,
    corregge solo eventuali code previste fuori ordine).
    """
    model.eval()
    X_scaled = feature_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        quantile_predictions_scaled = model(X_tensor).cpu().numpy()

    y_mean, y_std = y_stats
    # y_mean/y_std hanno shape (n_horizons,): si espandono per moltiplicare
    # per l'asse dei percentili senza cambiarne l'ordine relativo.
    quantile_predictions = (
        quantile_predictions_scaled * y_std[np.newaxis, :, np.newaxis]
        + y_mean[np.newaxis, :, np.newaxis]
    )

    quantile_predictions_sorted = np.sort(quantile_predictions, axis=-1)
    return quantile_predictions_sorted


# ═══════════════════════════════════════════════════════════════════════════
# Tuning deterministico via validation (punteggio: pinball loss di
# validation, l'equivalente della NLL per un modello di quantili)
# ═══════════════════════════════════════════════════════════════════════════

def tune_mlp_quantile_multi_val(X_tr, Y_tr, X_val, Y_val, grid, sample_weight=None, nn_fixed=None):
    nn_fixed = nn_fixed or {}

    best_val_pinball = np.inf
    best_params = None
    best_fit_result = None

    for params in ParameterGrid(grid):
        model, feature_scaler, y_stats, val_pinball = fit_mlp_quantile_multi(
            X_tr, Y_tr, X_val, Y_val, sample_weight=sample_weight, **params, **nn_fixed,
        )
        if val_pinball < best_val_pinball:
            best_val_pinball = val_pinball
            best_params = params
            best_fit_result = (model, feature_scaler, y_stats)

    return best_fit_result, best_params, best_val_pinball
