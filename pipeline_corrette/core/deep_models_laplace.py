"""
deep_models_laplace.py
─────────────────────────────────────────────────────────────────────────────
APPROSSIMAZIONE DI LAPLACE (last-layer) su una rete multi-output che predice
tutti e 7 gli orizzonti insieme, t+1..t+7, con le stesse DUE incertezze delle
altre pipeline di questa cartella.

L'IDEA IN UNA FRASE
Si allena una rete perfettamente normale (stessa architettura e stesso
training loop di MC-Dropout: tronco + testa mu + testa log_var, loss NLL
gaussiana). Finito il training, i pesi trovati sono il MASSIMO della
posteriore — il MAP. L'approssimazione di Laplace dice: invece di tenere solo
quel punto, guardiamo quanto la posteriore è CURVA attorno ad esso e
sostituiamola con la gaussiana che ha lì il suo massimo e quella curvatura.
Curvatura alta (la loss cresce ripidamente appena ci si muove) significa che i
dati vincolano bene quei pesi → poca incertezza epistemica; curvatura piatta
significa che molti valori diversi dei pesi spiegherebbero i dati altrettanto
bene → molta incertezza epistemica.

È un metodo POST-HOC: non cambia nulla del training, si applica dopo. Questa è
la differenza pratica più grande rispetto a Bayes by Backprop
(deep_models_bayesian_vi.py), dove invece la distribuzione sui pesi va appresa
durante il training.

PERCHÉ SOLO L'ULTIMO STRATO
L'Hessiana su TUTTI i pesi sarebbe una matrice (n_pesi x n_pesi) — decine di
migliaia di righe: impossibile da invertire e da conservare. Si congela quindi
il tronco e si tratta come bayesiano il solo ultimo strato lineare, quello che
va dalle feature nascoste phi(x) alla media mu. È la scelta standard
("last-layer Laplace", Kristiadi et al. 2020) e ha un vantaggio che le
approssimazioni diagonali non hanno: a tronco fissato e verosimiglianza
gaussiana il modello è lineare nei pesi dell'ultimo strato, quindi la
gaussiana NON è un'approssimazione — è la posteriore ESATTA di quello strato.
La matrice da invertire è (hidden+1) x (hidden+1), cioè al massimo qualche
decina di righe per orizzonte: costo trascurabile.

LA MATEMATICA CHE SERVE
Con phi(x) le feature del tronco (aumentate di una colonna di 1 per includere
il bias) e sigma^2_j(x) la varianza aleatoria che la testa logvar predice in
quel punto, per ogni orizzonte j:

    H_j = SOMMA_i [ phi(x_i) phi(x_i)^T / sigma^2_j(x_i) ]  +  tau * I

dove tau è la precisione del prior (vedi sotto). H_j è la precisione della
posteriore sui pesi dell'ultimo strato; la sua inversa è la covarianza. Nota
il peso 1/sigma^2: i punti su cui la rete si dichiara sicura contribuiscono
di più a vincolare i pesi, quelli rumorosi di meno — esattamente come deve
essere.

In previsione, per un nuovo punto x*:

    epistemic_var_j(x*) = phi(x*)^T H_j^-1 phi(x*)
    aleatoric_var_j(x*) = exp(logvar_j(x*))
    total_var_j(x*)     = epistemic_var_j + aleatoric_var_j

che è la stessa legge di combinazione di MC-Dropout, Deep Ensemble e Bayesian
VI — solo che qui l'incertezza epistemica si ottiene in FORMA CHIUSA, senza
campionare: nessun rumore Monte Carlo, risultato bit-identico a ogni
esecuzione.

LA PRECISIONE DEL PRIOR (tau)
tau corrisponde a un prior N(0, 1/tau) sui pesi dell'ultimo strato e regola
direttamente quanto è larga l'incertezza epistemica: tau grande = prior
stretto = poca incertezza; tau piccolo = molta. Non entra in Optuna perché
non serve riaddestrare per cambiarla: si sceglie DOPO, provando una griglia
logaritmica e tenendo il valore che minimizza la NLL di validation (vedi
calibra_prior_precision). È la calibrazione standard per Laplace e costa
secondi invece di ore.

Modulo autosufficiente: l'architettura MAP e il suo training loop sono
duplicati qui invece di essere importati da deep_models_uncertainty.py, come
già fanno fra loro gli altri moduli di core/ — un po' di duplicazione accettata
per tenere le cartelle indipendenti.
"""

from __future__ import annotations

import os
# Deve essere impostata PRIMA di "import torch": viene letta da cuBLAS alla
# prima inizializzazione del contesto CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
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
print(f"[deep_models_laplace] device = {DEVICE}")

LOG_VAR_MIN = -8.0
LOG_VAR_MAX = 8.0

# Griglia di precisioni del prior scandita da calibra_prior_precision:
# 21 punti logaritmicamente equispaziati fra 1e-3 (prior larghissimo, molta
# incertezza epistemica) e 1e3 (prior strettissimo, epistemica quasi nulla).
PRIOR_PRECISION_GRID = np.logspace(-3, 3, 21)

LOG_2PI = float(np.log(2.0 * np.pi))


def _set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ═══════════════════════════════════════════════════════════════════════════
# Architettura MAP: identica a MLPGaussianMulti, più l'accesso alle feature
# ═══════════════════════════════════════════════════════════════════════════

class MLPMapMulti(nn.Module):
    """Tronco Linear -> ReLU -> Dropout, poi le due teste mu e logvar.

    Unica aggiunta rispetto alla versione usata da MC-Dropout: il metodo
    `features()`, che espone l'uscita del tronco phi(x). Serve perché
    l'approssimazione di Laplace last-layer lavora proprio su quel vettore:
    congelato il tronco, mu = W phi(x) + b è lineare nei pesi W, ed è questa
    linearità a rendere la posteriore esattamente gaussiana.
    """

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

        self.trunk_output_dim = input_dim
        self.mu_head = nn.Linear(input_dim, n_outputs)
        self.logvar_head = nn.Linear(input_dim, n_outputs)

    def features(self, x):
        return self.trunk(x)

    def forward(self, x):
        shared_features = self.trunk(x)
        mu = self.mu_head(shared_features)
        log_var = torch.clamp(self.logvar_head(shared_features),
                              min=LOG_VAR_MIN, max=LOG_VAR_MAX)
        return mu, log_var


# ═══════════════════════════════════════════════════════════════════════════
# Loss e training loop del MAP (identici alle altre pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def gaussian_nll_loss(mu, log_var, y_true, sample_weight=None):
    """nll = 0.5 * log_var + 0.5 * exp(-log_var) * (y_true - mu)^2

    Stessa formula esatta delle altre pipeline (senza la costante
    0.5*log(2*pi), che non sposta il minimo).
    """
    inverse_variance = torch.exp(-log_var)
    squared_error = (y_true - mu) ** 2
    per_output_loss = 0.5 * log_var + 0.5 * inverse_variance * squared_error

    per_row_loss = per_output_loss.mean(dim=1)

    if sample_weight is not None:
        return (per_row_loss * sample_weight).mean()
    return per_row_loss.mean()


def _train_loop_map(model, X_tr, Y_tr, X_val, Y_val, sample_weight=None,
                    epochs=300, lr=1e-3, batch_size=32, patience=30):
    """Adam + early stopping sulla NLL di validation, con best_model_state
    clonato e ricaricato: identico a _train_loop_gaussian. Il dropout qui si
    comporta in modo standard (attivo in train, spento in eval): non viene
    riacceso in predizione, perché l'incertezza epistemica in questo metodo
    viene dall'Hessiana, non dal campionamento.
    """
    _set_seed()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
    Y_tr_tensor = torch.tensor(Y_tr, dtype=torch.float32)
    if sample_weight is None:
        weight_tensor = torch.ones(len(Y_tr_tensor), dtype=torch.float32)
    else:
        weight_tensor = torch.tensor(np.asarray(sample_weight), dtype=torch.float32)

    generator = torch.Generator()
    generator.manual_seed(SEED)
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


# ═══════════════════════════════════════════════════════════════════════════
# Approssimazione di Laplace sull'ultimo strato
# ═══════════════════════════════════════════════════════════════════════════

class LaplaceLastLayer:
    """Contiene la rete MAP allenata e la covarianza posteriore dei pesi
    dell'ultimo strato, una matrice per orizzonte.

    `cov` ha shape (n_outputs, D, D) con D = hidden_finale + 1 (il +1 è la
    colonna costante che rappresenta il bias). È l'inversa di H_j.
    """

    def __init__(self, model, cov, prior_precision):
        self.model = model
        self.cov = cov
        self.prior_precision = float(prior_precision)


def _phi_e_logvar(model, X_scaled):
    """Feature del tronco (con la colonna di 1 per il bias) e log-varianze
    aleatorie, calcolate in eval() — dropout SPENTO: il tronco dev'essere una
    funzione deterministica, altrimenti phi(x) non sarebbe ben definita e
    l'Hessiana verrebbe calcolata su un modello diverso da quello che poi
    predice.
    """
    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        phi = model.features(X_tensor).cpu().numpy()
        _, log_var = model(X_tensor)
        log_var = log_var.cpu().numpy()

    ones = np.ones((phi.shape[0], 1), dtype=phi.dtype)
    phi_aug = np.concatenate([phi, ones], axis=1).astype(np.float64)
    return phi_aug, log_var.astype(np.float64)


def fit_laplace_last_layer(model, X_train_scaled, prior_precision=1.0):
    """Costruisce la posteriore gaussiana dei pesi dell'ultimo strato.

    Per ogni orizzonte j:
        H_j   = phi^T diag(1/sigma^2_j) phi + prior_precision * I
        cov_j = H_j^-1

    Il calcolo è in float64 (non float32) perché H_j va invertita: in singola
    precisione, con feature molto correlate fra loro, l'inversione perde
    cifre significative e l'incertezza epistemica può uscire negativa.

    Non richiede i target: la curvatura della NLL gaussiana rispetto ai pesi
    dell'ultimo strato non dipende da y — dipende solo da dove cadono i punti
    (phi) e da quanto la rete li dichiara rumorosi (sigma^2).
    """
    phi, log_var = _phi_e_logvar(model, X_train_scaled)
    n_outputs = log_var.shape[1]
    dim = phi.shape[1]

    identity = np.eye(dim)
    cov = np.empty((n_outputs, dim, dim), dtype=np.float64)
    for j in range(n_outputs):
        inverse_variance = np.exp(-log_var[:, j])          # (n_righe,)
        hessian = (phi * inverse_variance[:, None]).T @ phi + prior_precision * identity
        cov[j] = np.linalg.inv(hessian)

    return LaplaceLastLayer(model, cov, prior_precision)


def laplace_predict(laplace, feature_scaler, y_stats, X):
    """Predizione con incertezza, in forma chiusa.

        mu_pred        = mu della rete MAP (dropout spento)
        epistemic_var  = phi(x*)^T cov_j phi(x*)
        aleatoric_var  = exp(logvar_j(x*))
        total_var      = epistemic_var + aleatoric_var

    Nessun campionamento: a differenza di MC-Dropout, Deep Ensemble e
    Bayesian VI il risultato è esatto e perfettamente riproducibile.

    Ritorna (mu_pred, epistemic_std, aleatoric_std, total_std), array
    (n_righe, n_orizzonti) in unità originali di moisture — stessa quadrupla
    delle altre pipeline, quindi stesso schema di CSV.
    """
    model = laplace.model
    model.eval()

    X_scaled = feature_scaler.transform(X)
    phi, log_var = _phi_e_logvar(model, X_scaled)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        mu_scaled = model(X_tensor)[0].cpu().numpy().astype(np.float64)

    # forma quadratica phi^T cov_j phi calcolata per tutte le righe insieme:
    # (phi @ cov_j) * phi, sommato sull'ultima dimensione
    n_outputs = log_var.shape[1]
    epistemic_variance_scaled = np.empty_like(mu_scaled)
    for j in range(n_outputs):
        epistemic_variance_scaled[:, j] = np.einsum("nd,dk,nk->n", phi, laplace.cov[j], phi)

    # la forma quadratica è teoricamente >= 0 (cov è definita positiva); il
    # clip difende solo da un residuo di errore numerico attorno allo zero
    epistemic_variance_scaled = np.clip(epistemic_variance_scaled, 0.0, None)
    aleatoric_variance_scaled = np.exp(log_var)
    total_variance_scaled = epistemic_variance_scaled + aleatoric_variance_scaled

    y_mean, y_std = y_stats
    mu_pred = mu_scaled * y_std + y_mean
    epistemic_std = np.sqrt(epistemic_variance_scaled) * y_std
    aleatoric_std = np.sqrt(aleatoric_variance_scaled) * y_std
    total_std = np.sqrt(total_variance_scaled) * y_std

    return mu_pred, epistemic_std, aleatoric_std, total_std


def _nll_gaussiana_media(y_true, mu, sigma):
    """NLL gaussiana media per riga/orizzonte, costante 2*pi INCLUSA.

    Qui la costante si tiene (a differenza della loss di training): questo
    numero viene confrontato fra diverse precisioni del prior e riportato come
    punteggio, quindi conviene che sia la verosimiglianza vera e non una
    versione traslata.
    """
    variance = np.maximum(sigma ** 2, 1e-12)
    return float(np.mean(0.5 * (LOG_2PI + np.log(variance)) + (y_true - mu) ** 2 / (2.0 * variance)))


def calibra_prior_precision(model, feature_scaler, y_stats, X_tr, X_val, Y_val,
                            grid=None):
    """Sceglie la precisione del prior che minimizza la NLL di validation.

    Perché fuori da Optuna: cambiare tau NON richiede di riaddestrare la rete
    — si ricostruisce solo l'Hessiana (una manciata di matrici piccole), quindi
    l'intera griglia costa secondi. Metterla fra gli iperparametri di Optuna
    significherebbe spendere un training completo per ogni valore, senza
    guadagnare nulla.

    L'Hessiana si costruisce sui dati di TRAIN (è lì che la posteriore si
    forma) e la si giudica sulla validation, mai vista: è ciò che impedisce di
    scegliere un tau che semplicemente stringe l'incertezza fin quasi a zero.

    Ritorna (laplace, best_prior_precision, best_val_nll).
    """
    grid = PRIOR_PRECISION_GRID if grid is None else np.asarray(grid, dtype=float)
    Y_val = np.asarray(Y_val, dtype=float)

    best_laplace = None
    best_prior_precision = None
    best_val_nll = np.inf

    for prior_precision in grid:
        laplace = fit_laplace_last_layer(model, X_tr, prior_precision=prior_precision)
        mu_pred, _, _, total_std = laplace_predict(laplace, feature_scaler, y_stats, X_val)
        val_nll = _nll_gaussiana_media(Y_val, mu_pred, total_std)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_prior_precision = float(prior_precision)
            best_laplace = laplace

    return best_laplace, best_prior_precision, best_val_nll


# ═══════════════════════════════════════════════════════════════════════════
# Interfaccia di alto livello: MAP + Laplace in un colpo solo
# ═══════════════════════════════════════════════════════════════════════════

def fit_laplace_multi(X_tr, Y_tr, X_val, Y_val, sample_weight=None,
                      hidden=(64, 32), dropout=0.2, lr=1e-3,
                      batch_size=32, patience=30, epochs=300,
                      prior_precision_grid=None):
    """Allena la rete MAP, ci costruisce sopra l'approssimazione di Laplace e
    ne calibra la precisione del prior sulla validation.

    Ritorna (laplace, feature_scaler, y_stats, val_nll) — stessa forma di
    ritorno di fit_mlp_gaussian_multi, con `laplace` al posto del modello nudo
    (contiene comunque la rete in `laplace.model`).

    Attenzione a quale NLL viene restituita: NON è quella dell'early stopping
    della rete MAP, ma quella DOPO Laplace, cioè con l'incertezza epistemica
    inclusa e con la precisione del prior già scelta. È il punteggio giusto da
    dare a Optuna, perché è il metodo completo a essere confrontato, non la
    sola rete puntuale.

    Il sample_weight pesa il training (come nelle altre pipeline) ma NON entra
    nell'Hessiana: i pesi degli eventi pioggia/irrigazione sono una scelta di
    ottimizzazione, non una vera replicazione delle osservazioni, e usarli
    nella posteriore stringerebbe artificialmente l'incertezza sui giorni
    ripesati.
    """
    _set_seed()   # prima della costruzione del modello: i pesi si inizializzano lì

    feature_scaler = StandardScaler().fit(X_tr)
    X_tr_scaled = feature_scaler.transform(X_tr)
    X_val_scaled = feature_scaler.transform(X_val)

    Y_tr = np.asarray(Y_tr, dtype=float)
    Y_val = np.asarray(Y_val, dtype=float)
    y_mean = Y_tr.mean(axis=0)
    y_std = Y_tr.std(axis=0) + 1e-8
    Y_tr_scaled = (Y_tr - y_mean) / y_std
    Y_val_scaled = (Y_val - y_mean) / y_std
    y_stats = (y_mean, y_std)

    model = MLPMapMulti(
        n_features=X_tr_scaled.shape[1], n_outputs=Y_tr_scaled.shape[1],
        hidden=hidden, dropout=dropout,
    )
    model, _ = _train_loop_map(
        model, X_tr_scaled, Y_tr_scaled, X_val_scaled, Y_val_scaled,
        sample_weight=sample_weight, epochs=epochs, lr=lr,
        batch_size=batch_size, patience=patience,
    )

    laplace, _, val_nll = calibra_prior_precision(
        model, feature_scaler, y_stats, X_tr_scaled, X_val, Y_val,
        grid=prior_precision_grid,
    )
    return laplace, feature_scaler, y_stats, val_nll


def tune_mlp_laplace_multi_val(X_tr, Y_tr, X_val, Y_val, grid, sample_weight=None, nn_fixed=None):
    """Grid search deterministico sulla NLL di validation post-Laplace.
    Ritorna ((laplace, feature_scaler, y_stats), best_params, best_val_nll).
    """
    nn_fixed = nn_fixed or {}
    best_fit_result = None
    best_params = None
    best_val_nll = np.inf

    for params in ParameterGrid(grid):
        laplace, feature_scaler, y_stats, val_nll = fit_laplace_multi(
            X_tr, Y_tr, X_val, Y_val, sample_weight=sample_weight, **params, **nn_fixed,
        )
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_params = params
            best_fit_result = (laplace, feature_scaler, y_stats)

    return best_fit_result, best_params, best_val_nll
