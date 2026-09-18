"""
deep_models_bayesian_vi.py
─────────────────────────────────────────────────────────────────────────────
Rete neurale BAYESIANA allenata con VARIATIONAL INFERENCE (Bayes by Backprop,
Blundell et al. 2015), multi-output: predice tutti e 7 gli orizzonti insieme,
t+1..t+7, e quantifica le stesse DUE incertezze delle altre pipeline di
questa cartella.

DIFFERENZA RISPETTO A MC-DROPOUT E DEEP ENSEMBLE
In una rete normale ogni peso è UN numero. Qui ogni peso è una
DISTRIBUZIONE: la rete impara, per ciascun peso, una media e una deviazione
standard. In predizione si campiona ripetutamente dai pesi e si guarda quanto
le previsioni oscillano.

  - Incertezza EPISTEMICA: quanto oscillano le mu ottenute campionando pesi
    diversi dalla posteriore variazionale. Se i dati vincolano bene i pesi, le
    sigma dei pesi sono piccole e le previsioni si somigliano; dove i dati
    dicono poco, le sigma restano larghe e le previsioni divergono.
  - Incertezza ALEATORIA: come nelle altre pipeline, la rete predice anche
    log(sigma^2) punto per punto con una testa dedicata (eteroscedastica).

Le due si combinano ESATTAMENTE come in deep_models_uncertainty.py e
deep_models_ensemble.py (varianza totale = epistemica + aleatoria), quindi i
CSV prodotti hanno lo stesso schema e sono confrontabili riga per riga con
MC-Dropout, Deep Ensemble e Quantile Regression.

COME SI ALLENA — ELBO
Non si minimizza solo l'errore: si minimizza

    ELBO = NLL gaussiana  +  kl_weight * KL(q(w) || p(w)) / n_batch

dove q(w) è la distribuzione appresa sui pesi e p(w) = N(0, prior_sigma^2) è
il prior. Il secondo termine tira i pesi verso il prior e impedisce alla rete
di azzerare le sigma (cioè di tornare deterministica). La divisione per il
numero di batch distribuisce la KL totale sull'epoca: è la pratica standard
di Bayes by Backprop, senza la quale la KL — che è una somma su TUTTI i pesi —
schiaccia il termine di verosimiglianza e la rete collassa sul prior.

REPARAMETERIZATION TRICK
Campionare w ~ N(mu, sigma^2) direttamente non è derivabile rispetto a mu e
sigma. Si campiona invece eps ~ N(0,1) e si scrive w = mu + sigma * eps: il
caso è confinato in eps, che non dipende dai parametri, e il gradiente
attraversa mu e sigma senza problemi.

Nessuna libreria bayesiana esterna (pyro, torchbnn, blitz): solo PyTorch, in
linea con la scelta di tenere i moduli di core/ autosufficienti.

Determinismo: stessa configurazione delle altre pipeline
(CUBLAS_WORKSPACE_CONFIG prima di importare torch, cudnn deterministico,
generator fisso per il DataLoader).
"""

from __future__ import annotations

import os
# Deve essere impostata PRIMA di "import torch": viene letta da cuBLAS alla
# prima inizializzazione del contesto CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
print(f"[deep_models_bayesian_vi] device = {DEVICE}")

# Stesso limite di sicurezza numerica per log(sigma^2) usato dalle altre
# pipeline: impedisce alla testa aleatoria di dichiarare varianza ~0 (loss
# artificialmente piccola, gradiente esplosivo) o di arrendersi verso +inf.
LOG_VAR_MIN = -8.0
LOG_VAR_MAX = 8.0

# Inizializzazione di rho: softplus(-5) ~ 0.0067, cioè sigma iniziale molto
# piccola. La rete parte quasi deterministica e "apre" il rumore sui pesi solo
# dove i dati non la vincolano. Partendo con sigma grandi le prime epoche sono
# rumorosissime e il training spesso non decolla.
RHO_INIT = -5.0


def _set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ═══════════════════════════════════════════════════════════════════════════
# Strato lineare bayesiano
# ═══════════════════════════════════════════════════════════════════════════

class BayesianLinear(nn.Module):
    """Strato lineare in cui pesi e bias sono distribuzioni gaussiane
    indipendenti (approssimazione "mean-field") invece che numeri fissi.

    Parametri appresi: weight_mu, weight_rho, bias_mu, bias_rho.
    La deviazione standard si ottiene come sigma = softplus(rho) invece di
    parametrizzarla direttamente: softplus è sempre positiva e liscia, quindi
    l'ottimizzatore può muovere rho su tutto R senza vincoli e senza che sigma
    diventi mai negativa.
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = float(prior_sigma)

        # media inizializzata come uno strato lineare normale (scala 1/sqrt(fan_in)),
        # così la rete parte dallo stesso regime di una rete deterministica
        bound = 1.0 / math.sqrt(in_features)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.bias_mu = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))

        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), RHO_INIT))
        self.bias_rho = nn.Parameter(torch.full((out_features,), RHO_INIT))

    def forward(self, x, sample: bool = True):
        """sample=True: campiona i pesi (reparameterization trick).
        sample=False: usa le medie, cioè il passaggio deterministico — serve
        per una validation riproducibile (vedi _train_loop_bayesian)."""
        if not sample:
            return F.linear(x, self.weight_mu, self.bias_mu)

        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """KL(q || p) in forma chiusa fra q = N(mu, sigma^2) appresa e il
        prior p = N(0, prior_sigma^2), sommata su tutti i pesi e i bias:

            KL = log(sigma_p / sigma_q) + (sigma_q^2 + mu^2) / (2 sigma_p^2) - 1/2

        Essendo entrambe gaussiane non serve nessun campionamento: è esatta.
        """
        total = 0.0
        for mu, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            sigma = F.softplus(rho)
            total = total + (
                math.log(self.prior_sigma) - torch.log(sigma)
                + (sigma ** 2 + mu ** 2) / (2.0 * self.prior_sigma ** 2)
                - 0.5
            ).sum()
        return total


# ═══════════════════════════════════════════════════════════════════════════
# Architettura: MLP bayesiano multi-output con testa eteroscedastica
# ═══════════════════════════════════════════════════════════════════════════

class MLPBayesianMulti(nn.Module):
    """Tronco condiviso di BayesianLinear + ReLU, poi DUE teste bayesiane:

      - mu_head:     una previsione puntuale per ciascun orizzonte
      - logvar_head: log(sigma^2) per ciascun orizzonte (incertezza aleatoria)

    Nessun Dropout: qui il rumore sui pesi svolge già il ruolo di
    regolarizzatore stocastico, e sovrapporre i due meccanismi renderebbe
    impossibile attribuire l'incertezza epistemica all'uno o all'altro.
    """

    def __init__(self, n_features: int, n_outputs: int, hidden=(64, 32),
                 prior_sigma: float = 0.1):
        super().__init__()

        trunk_layers = []
        input_dim = n_features
        for hidden_dim in hidden:
            trunk_layers.append(BayesianLinear(input_dim, hidden_dim, prior_sigma=prior_sigma))
            trunk_layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.trunk = nn.ModuleList(trunk_layers)

        self.mu_head = BayesianLinear(input_dim, n_outputs, prior_sigma=prior_sigma)
        self.logvar_head = BayesianLinear(input_dim, n_outputs, prior_sigma=prior_sigma)

    def forward(self, x, sample: bool = True):
        # ModuleList e non Sequential: il flag `sample` va inoltrato solo agli
        # strati bayesiani, mentre le ReLU si chiamano normalmente
        for layer in self.trunk:
            x = layer(x, sample=sample) if isinstance(layer, BayesianLinear) else layer(x)

        mu = self.mu_head(x, sample=sample)
        log_var = self.logvar_head(x, sample=sample)
        log_var = torch.clamp(log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX)
        return mu, log_var

    def kl_divergence(self):
        """KL totale della rete: somma dei contributi di ogni strato bayesiano."""
        total = 0.0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                total = total + module.kl_divergence()
        return total


# ═══════════════════════════════════════════════════════════════════════════
# Loss: NLL gaussiana eteroscedastica (identica alle altre pipeline) + KL
# ═══════════════════════════════════════════════════════════════════════════

def gaussian_nll_loss(mu, log_var, y_true, sample_weight=None):
    """Identica, formula per formula, a quella di deep_models_uncertainty.py e
    deep_models_ensemble.py — la costante 0.5*log(2*pi) è omessa perché non
    dipende dai parametri e non sposta il minimo:

        nll = 0.5 * log_var + 0.5 * exp(-log_var) * (y_true - mu)^2
    """
    inverse_variance = torch.exp(-log_var)
    squared_error = (y_true - mu) ** 2
    per_output_loss = 0.5 * log_var + 0.5 * inverse_variance * squared_error

    per_row_loss = per_output_loss.mean(dim=1)

    if sample_weight is not None:
        return (per_row_loss * sample_weight).mean()
    return per_row_loss.mean()


def elbo_loss(mu, log_var, y_true, kl, kl_weight, n_train, sample_weight=None):
    """ELBO negativa da minimizzare: verosimiglianza + KL riscalata.

    L'ELBO vera è `SOMMA_i NLL_i + KL`, con la somma su tutto il dataset. Qui
    il primo termine è la NLL MEDIA per riga (come in tutte le altre pipeline,
    per avere loss confrontabili), quindi per restare proporzionali va diviso
    tutto per n_train — ed è per questo che la KL si divide per il numero di
    RIGHE di training, non per il numero di batch.

    La differenza non è cosmetica: dividendo per i batch la KL risulta
    `batch_size` volte troppo grande (con batch 32, 32 volte), domina la
    verosimiglianza e la rete collassa sul prior — predice ovunque lo stesso
    valore dichiarando un'incertezza enorme. Sintomo tipico: MAE altissima
    anche sul TRAIN e aleatoric_std dell'ordine della deviazione standard del
    target.

    `kl_weight` resta come manopola esplicita: a 1.0 è l'ELBO esatta, sotto
    avvicina la rete a una rete deterministica.
    """
    nll = gaussian_nll_loss(mu, log_var, y_true, sample_weight)
    return nll + kl_weight * kl / max(n_train, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def _train_loop_bayesian(model, X_tr, Y_tr, X_val, Y_val, sample_weight=None,
                         kl_weight=1.0, epochs=300, lr=1e-3, batch_size=32, patience=30):
    """Stesso schema delle altre pipeline (Adam, early stopping su una
    validation esplicita, best_model_state clonato e ricaricato), con due
    differenze:

      - la loss di training è l'ELBO, non la sola NLL;
      - la NLL di validation si calcola con `sample=False`, cioè coi pesi
        MEDI. Campionare anche in validation renderebbe il criterio di early
        stopping rumoroso da un'epoca all'altra (si fermerebbe su un campione
        fortunato invece che su un minimo vero). La KL non entra nel punteggio
        di validation: si seleziona sulla qualità predittiva, coerentemente
        con le altre pipeline che early-stoppano sulla NLL.
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
    loader = DataLoader(
        TensorDataset(X_tr_tensor, Y_tr_tensor, weight_tensor),
        batch_size=batch_size, shuffle=True, generator=generator,
    )
    n_train = len(X_tr_tensor)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(DEVICE)

    best_val_nll = np.inf
    best_model_state = None
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        for X_batch, Y_batch, weight_batch in loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            weight_batch = weight_batch.to(DEVICE)

            optimizer.zero_grad()
            mu_batch, log_var_batch = model(X_batch, sample=True)
            loss = elbo_loss(mu_batch, log_var_batch, Y_batch,
                             model.kl_divergence(), kl_weight, n_train, weight_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            mu_val, log_var_val = model(X_val_tensor, sample=False)
            val_nll = gaussian_nll_loss(mu_val, log_var_val, Y_val_tensor).item()

        if val_nll < best_val_nll - 1e-5:
            best_val_nll = val_nll
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_val_nll


def fit_mlp_bayesian_multi(X_tr, Y_tr, X_val, Y_val, sample_weight=None,
                           hidden=(64, 32), prior_sigma=0.1, kl_weight=1.0,
                           lr=1e-3, batch_size=32, patience=30, epochs=300):
    """Allena la rete bayesiana e ritorna
    (model, feature_scaler, y_stats, best_val_nll) — stessa firma di ritorno
    di fit_mlp_gaussian_multi nelle altre pipeline.

    Scaling identico alle altre: StandardScaler solo sulle feature, target
    normalizzato a mano colonna per colonna (una media e una deviazione per
    orizzonte).
    """
    _set_seed()   # prima di costruire il modello: è qui che si inizializzano i pesi

    feature_scaler = StandardScaler()
    X_tr_scaled = feature_scaler.fit_transform(X_tr)
    X_val_scaled = feature_scaler.transform(X_val)

    Y_tr = np.asarray(Y_tr, dtype=float)
    Y_val = np.asarray(Y_val, dtype=float)
    y_mean = Y_tr.mean(axis=0)
    y_std = Y_tr.std(axis=0) + 1e-8
    Y_tr_scaled = (Y_tr - y_mean) / y_std
    Y_val_scaled = (Y_val - y_mean) / y_std

    model = MLPBayesianMulti(
        n_features=X_tr_scaled.shape[1], n_outputs=Y_tr_scaled.shape[1],
        hidden=hidden, prior_sigma=prior_sigma,
    )
    model, best_val_nll = _train_loop_bayesian(
        model, X_tr_scaled, Y_tr_scaled, X_val_scaled, Y_val_scaled,
        sample_weight=sample_weight, kl_weight=kl_weight,
        epochs=epochs, lr=lr, batch_size=batch_size, patience=patience,
    )

    y_stats = (y_mean, y_std)
    return model, feature_scaler, y_stats, best_val_nll


# ═══════════════════════════════════════════════════════════════════════════
# Predizione: campionamento dalla posteriore variazionale sui pesi
# ═══════════════════════════════════════════════════════════════════════════

def vi_predict_mlp_bayesian_multi(model, feature_scaler, y_stats, X, n_samples=100):
    """Predizione con incertezza.

    Si ripete il forward n_samples volte campionando OGNI VOLTA un set di pesi
    diverso dalla posteriore variazionale. È l'analogo dei T forward pass di
    MC-Dropout e degli N membri del Deep Ensemble — cambia la sorgente della
    variabilità (la distribuzione appresa sui pesi invece che il dropout
    casuale o l'inizializzazione), ma la combinazione finale è la STESSA:

        mu_pred        = media delle mu campionate
        epistemic_var  = varianza delle mu campionate
        aleatoric_var  = media di exp(log_var) campionate
        total_var      = epistemic_var + aleatoric_var

    I calcoli avvengono in scala normalizzata; alla fine si torna in punti di
    umidità moltiplicando per y_std (trasformazione affine: per le deviazioni
    standard basta la moltiplicazione, senza il +y_mean).

    Ritorna (mu_pred, epistemic_std, aleatoric_std, total_std), array
    (n_righe, n_orizzonti) in unità originali.
    """
    model.eval()   # qui eval() non spegne nulla: il campionamento dei pesi è
                   # controllato dal flag `sample`, non dalla modalità del modulo

    X_scaled = feature_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    mu_samples = []
    log_var_samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            mu_sample, log_var_sample = model(X_tensor, sample=True)
            mu_samples.append(mu_sample.cpu().numpy())
            log_var_samples.append(log_var_sample.cpu().numpy())

    mu_samples = np.stack(mu_samples, axis=0)             # (n_samples, n_rows, n_outputs)
    log_var_samples = np.stack(log_var_samples, axis=0)

    mu_mean_scaled = mu_samples.mean(axis=0)
    epistemic_variance_scaled = mu_samples.var(axis=0)
    aleatoric_variance_scaled = np.exp(log_var_samples).mean(axis=0)
    total_variance_scaled = epistemic_variance_scaled + aleatoric_variance_scaled

    y_mean, y_std = y_stats
    mu_pred = mu_mean_scaled * y_std + y_mean
    epistemic_std = np.sqrt(epistemic_variance_scaled) * y_std
    aleatoric_std = np.sqrt(aleatoric_variance_scaled) * y_std
    total_std = np.sqrt(total_variance_scaled) * y_std

    return mu_pred, epistemic_std, aleatoric_std, total_std


# ═══════════════════════════════════════════════════════════════════════════
# Tuning deterministico via validation NLL (stesso schema delle altre pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def tune_mlp_bayesian_multi_val(X_tr, Y_tr, X_val, Y_val, grid, sample_weight=None, nn_fixed=None):
    """Per ogni combinazione della griglia allena e tiene quella con la NLL di
    validation più bassa. Ritorna
    ((model, feature_scaler, y_stats), best_params, best_val_nll).
    """
    nn_fixed = nn_fixed or {}
    best_fit_result = None
    best_params = None
    best_val_nll = np.inf

    for params in ParameterGrid(grid):
        model, feature_scaler, y_stats, val_nll = fit_mlp_bayesian_multi(
            X_tr, Y_tr, X_val, Y_val, sample_weight=sample_weight, **params, **nn_fixed,
        )
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_params = params
            best_fit_result = (model, feature_scaler, y_stats)

    return best_fit_result, best_params, best_val_nll
