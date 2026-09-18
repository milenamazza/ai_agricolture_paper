"""
deep_models_uncertainty.py
─────────────────────────────────────────────────────────────────────────────
Un solo modello MLP multi-output (predice tutti e 7 gli orizzonti insieme,
t+1..t+7) che quantifica DUE tipi di incertezza allo stesso tempo:

  - Incertezza EPISTEMICA (quanto il modello stesso è incerto, per via di
    dati/parametri limitati): stimata con MC-Dropout. Il dropout normale di
    PyTorch viene tenuto ATTIVO anche in fase di predizione (non solo in
    training) e si ripete lo stesso forward pass T volte: ogni volta il
    dropout "spegne" neuroni diversi a caso, quindi si ottengono T previsioni
    leggermente diverse. Quanto queste T previsioni oscillano tra loro è la
    misura dell'incertezza epistemica.

  - Incertezza ALEATORIA (rumore intrinseco nei dati, che non sparirebbe
    nemmeno con infiniti dati): il modello non predice solo un valore (mu)
    ma anche una varianza (sigma^2) associata a quel valore, diversa punto
    per punto ("eteroscedastica"). Si allena con una loss di verosimiglianza
    gaussiana (Negative Log-Likelihood) invece della solita MSE/MAE, così il
    modello impara da solo dove è più "rumoroso" il fenomeno.

Le due incertezze si combinano in un'unica varianza totale in fase di
predizione (vedi mc_predict_mlp_gaussian_multi): varianza_totale =
varianza_epistemica + varianza_aleatoria (legge della varianza totale).
Questo è lo schema standard descritto in Kendall & Gal, "What Uncertainties
Do We Need in Bayesian Deep Learning for Computer Vision?" (2017).

Modulo autosufficiente (nessun import da data_driven_ml_multioutput/ o da
modelli singoli/data_driven_ml/deep_models.py): stessa scelta di
compartimentazione già usata da deep_models_multi.py nelle altre cartelle,
un po' di duplicazione accettata per tenere le cartelle indipendenti.

Determinismo: stessa configurazione (CUBLAS_WORKSPACE_CONFIG prima di
importare torch, cudnn deterministico, generator fisso per il DataLoader)
già verificata bit-esatta nelle altre pipeline di questo progetto.
"""

from __future__ import annotations

import os
# Deve essere impostata PRIMA di "import torch": viene letta da cuBLAS alla
# prima inizializzazione del contesto CUDA.
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
LOOKBACK_DEFAULT = 14

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
print(f"[deep_models_uncertainty] device = {DEVICE}")

# Limite di sicurezza numerica per log(sigma^2), spiegato in dettaglio più
# sotto vicino a dove viene applicato (MLPGaussianMulti.forward).
LOG_VAR_MIN = -8.0
LOG_VAR_MAX = 8.0


def _set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ═══════════════════════════════════════════════════════════════════════════
# Architettura: MLP multi-output con testa eteroscedastica (mu, log_var)
# ═══════════════════════════════════════════════════════════════════════════

class MLPGaussianMulti(nn.Module):
    """
    Tronco condiviso Linear -> ReLU -> Dropout (stessa struttura di
    MLPRegressorMulti nelle altre cartelle), seguito da DUE teste separate:

      - mu_head:     una previsione puntuale per ciascun orizzonte.
      - logvar_head: log(sigma^2) per ciascun orizzonte, cioè il logaritmo
                     della varianza prevista per quella previsione.

    Perché log(sigma^2) e non sigma^2 direttamente: una rete neurale può
    restituire qualunque numero reale (anche negativo), ma una varianza deve
    sempre essere positiva. Prevedendo il LOGARITMO della varianza non serve
    nessun vincolo esplicito sull'output della rete: per riottenere la
    varianza vera basta fare exp(log_var), che è sempre >= 0 qualunque sia
    log_var.

    Il dropout NON viene disattivato manualmente da questa classe: il
    training normale lo usa in modalità standard (attivo in model.train(),
    disattivo in model.eval()). Il trucco del MC-Dropout è tutto nella
    funzione mc_predict_mlp_gaussian_multi più sotto, che richiama
    esplicitamente model.train() anche in fase di predizione.
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

        trunk_output_dim = input_dim
        self.mu_head = nn.Linear(trunk_output_dim, n_outputs)
        self.logvar_head = nn.Linear(trunk_output_dim, n_outputs)

    def forward(self, x):
        shared_features = self.trunk(x)
        mu = self.mu_head(shared_features)
        log_var = self.logvar_head(shared_features)

        # Clamp di sicurezza numerica. Senza questo limite, durante il
        # training la rete potrebbe imparare a spingere log_var verso valori
        # molto negativi su alcuni punti (dichiarando varianza quasi zero,
        # per far sembrare la loss piccolissima lì): la loss contiene un
        # termine exp(-log_var) che in quel caso esploderebbe, rendendo il
        # gradiente instabile. Simmetricamente, log_var potrebbe esplodere
        # verso valori molto positivi come scorciatoia per "arrendersi" su
        # punti difficili. [-8, 8] corrisponde a sigma^2 tra circa 0.0003 e
        # circa 2981 (in scala normalizzata, dove il target ha varianza 1 per
        # costruzione) — un intervallo ampio che non limita mai un output
        # ragionevole, ma impedisce questi due comportamenti degeneri.
        log_var = torch.clamp(log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX)

        return mu, log_var


# ═══════════════════════════════════════════════════════════════════════════
# Loss: Negative Log-Likelihood gaussiana eteroscedastica
# ═══════════════════════════════════════════════════════════════════════════

def gaussian_nll_loss(mu, log_var, y_true, sample_weight=None):
    """
    NLL di una gaussiana con media mu e varianza exp(log_var), calcolata
    elemento per elemento e poi mediata. Formula (per un singolo output):

        nll = 0.5 * log_var + 0.5 * exp(-log_var) * (y_true - mu)^2

    (si è tolta la costante 0.5*log(2*pi): non dipende dai parametri del
    modello, quindi non cambia dove si trova il minimo della loss).

    Interpretazione dei due termini:
      - 0.5 * log_var: cresce se il modello dichiara una varianza grande.
        Impedisce alla rete di "vincere facile" dichiarando sempre
        un'incertezza enorme dappertutto.
      - 0.5 * exp(-log_var) * (y_true - mu)^2: è l'errore quadratico, ma
        ridimensionato dalla varianza dichiarata. Se il modello ha previsto
        (onestamente) un sigma grande in quel punto, un errore grande lì
        costa relativamente poco; se aveva dichiarato un sigma piccolo, lo
        stesso errore costa molto di più.

    mu, log_var, y_true hanno tutti shape (batch, n_outputs) — un valore per
    ciascuno dei 7 orizzonti. Si media prima sugli orizzonti (così ogni riga
    pesa 1, indipendentemente da quanti orizzonti ha), poi sul batch,
    applicando il sample_weight per riga se fornito (stesso schema di peso
    usato per i giorni pioggia/irrigazione nelle altre pipeline).
    """
    inverse_variance = torch.exp(-log_var)
    squared_error = (y_true - mu) ** 2
    per_output_loss = 0.5 * log_var + 0.5 * inverse_variance * squared_error

    per_row_loss = per_output_loss.mean(dim=1)

    if sample_weight is not None:
        weighted_loss = per_row_loss * sample_weight
        return weighted_loss.mean()

    return per_row_loss.mean()


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def _train_loop_gaussian(model, X_tr, Y_tr, X_val, Y_val, sample_weight=None,
                         epochs=300, lr=1e-3, batch_size=32, patience=30):
    """
    Stesso schema delle altre pipeline (stesso seed, stesso DataLoader con
    generator fisso per lo shuffling, stesso early stopping sulla
    validation), ma:
      - la loss è gaussian_nll_loss invece di MAE/SmoothL1;
      - il criterio di early stopping è la NLL di validation (non la MAE):
        è la quantità che il modello sta davvero minimizzando, quindi è
        coerente fermarsi quando quella smette di migliorare.

    Durante training e validation il dropout si comporta in modo standard
    (attivo in model.train(), disattivo in model.eval()): qui NON si fa
    ancora MC-Dropout. Il campionamento MC-Dropout avviene solo più tardi,
    in fase di predizione finale (mc_predict_mlp_gaussian_multi).
    """
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


def fit_mlp_gaussian_multi(X_tr, Y_tr, X_val, Y_val, sample_weight=None,
                           hidden=(64, 32), dropout=0.2, lr=1e-3,
                           batch_size=32, patience=30, epochs=300):
    """
    Normalizza feature e target (media/std calcolate SOLO sul train, come
    nelle altre pipeline), allena il modello, e ritorna tutto ciò che serve
    per predire più tardi in unità originali: il modello allenato, lo scaler
    delle feature, e (y_mean, y_std) per riportare mu/sigma alla scala di
    moisture originale.
    """
    _set_seed()   # deve precedere la costruzione del modello: i pesi si inizializzano lì

    feature_scaler = StandardScaler().fit(X_tr)
    X_tr_scaled = feature_scaler.transform(X_tr)
    X_val_scaled = feature_scaler.transform(X_val)

    # Normalizzazione del target PER COLONNA (un orizzonte diverso può avere
    # scala/varianza leggermente diversa). Senza normalizzare, la rete
    # convergerebbe troppo lentamente rispetto alla scala di moisture
    # (~0-100) e l'early stopping la fermerebbe su un output quasi costante.
    y_mean = Y_tr.mean(axis=0)
    y_std = Y_tr.std(axis=0) + 1e-8
    Y_tr_scaled = (Y_tr - y_mean) / y_std
    Y_val_scaled = (Y_val - y_mean) / y_std

    model = MLPGaussianMulti(
        n_features=X_tr.shape[1], n_outputs=Y_tr.shape[1],
        hidden=hidden, dropout=dropout,
    )
    model, best_val_nll = _train_loop_gaussian(
        model, X_tr_scaled, Y_tr_scaled, X_val_scaled, Y_val_scaled,
        sample_weight=sample_weight, epochs=epochs, lr=lr,
        batch_size=batch_size, patience=patience,
    )

    y_stats = (y_mean, y_std)
    return model, feature_scaler, y_stats, best_val_nll


# ═══════════════════════════════════════════════════════════════════════════
# MC-Dropout: predizione con decomposizione epistemica / aleatoria / totale
# ═══════════════════════════════════════════════════════════════════════════

def mc_predict_mlp_gaussian_multi(model, feature_scaler, y_stats, X, n_mc_samples=100):
    """
    Predizione con incertezza via MC-Dropout.

    Idea: si ripete lo STESSO forward pass n_mc_samples volte, tenendo il
    dropout ATTIVO (model.train(), non model.eval()) anche se non stiamo più
    allenando. Ogni ripetizione "spegne" a caso neuroni diversi del tronco,
    quindi ogni volta il modello dà una mu e un log_var leggermente diversi.
    Da queste n_mc_samples coppie (mu, log_var) si ricavano:

      - mu_pred:        la previsione puntuale finale, media delle mu.
      - epistemic_std:  quanto oscillano tra loro le mu ottenute nei diversi
                        pass. Se il modello "non sa bene" cosa rispondere in
                        quella zona (poca informazione/pochi dati simili
                        visti in training), il dropout casuale sposta la
                        previsione in modo più marcato da un pass all'altro
                        → oscillazione grande → epistemic_std grande.
      - aleatoric_std:  la media delle sigma dichiarate dal modello nei
                        diversi pass (cioè il rumore che il modello stesso
                        dichiara di aspettarsi in quel punto, a prescindere
                        dal dropout).
      - total_std:      la deviazione standard predittiva complessiva,
                        combinando le due varianze (varianza totale =
                        varianza epistemica + varianza aleatoria — legge
                        della varianza totale, valida perché si sta
                        mediando su un insieme di distribuzioni gaussiane
                        con medie diverse).

    Tutti i calcoli intermedi avvengono in scala NORMALIZZATA (quella in cui
    il modello è stato allenato); alla fine si riportano mu e le tre std in
    unità originali di moisture moltiplicando per y_std. Questo è corretto
    perché normalizzare il target è una trasformazione lineare (affine):
    per una trasformazione lineare, una deviazione standard in scala
    normalizzata si converte in quella in scala originale moltiplicando
    semplicemente per y_std (non serve nessuna correzione più complicata).
    """
    # model.train() invece di model.eval(): è la riga che rende attivo il
    # dropout anche adesso, in predizione. Nessun'altra parte del modello
    # (qui non ci sono batch-norm o simili) risente di questa scelta.
    model.train()

    X_scaled = feature_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    mu_samples = []
    log_var_samples = []
    with torch.no_grad():
        for _ in range(n_mc_samples):
            mu_sample, log_var_sample = model(X_tensor)
            mu_samples.append(mu_sample.cpu().numpy())
            log_var_samples.append(log_var_sample.cpu().numpy())

    model.eval()   # si rimette il modello in modalità normale per un uso successivo

    # shape di ciascuno: (n_mc_samples, n_rows, n_outputs)
    mu_samples = np.stack(mu_samples, axis=0)
    log_var_samples = np.stack(log_var_samples, axis=0)

    mu_mean_scaled = mu_samples.mean(axis=0)
    epistemic_variance_scaled = mu_samples.var(axis=0)

    aleatoric_variance_per_sample = np.exp(log_var_samples)
    aleatoric_variance_scaled = aleatoric_variance_per_sample.mean(axis=0)

    total_variance_scaled = epistemic_variance_scaled + aleatoric_variance_scaled

    y_mean, y_std = y_stats
    mu_pred = mu_mean_scaled * y_std + y_mean
    epistemic_std = np.sqrt(epistemic_variance_scaled) * y_std
    aleatoric_std = np.sqrt(aleatoric_variance_scaled) * y_std
    total_std = np.sqrt(total_variance_scaled) * y_std

    return mu_pred, epistemic_std, aleatoric_std, total_std


# ═══════════════════════════════════════════════════════════════════════════
# Tuning deterministico via validation (stesso schema delle altre pipeline,
# ma il punteggio è la NLL di validation invece della MAE)
# ═══════════════════════════════════════════════════════════════════════════

def tune_mlp_gaussian_multi_val(X_tr, Y_tr, X_val, Y_val, grid, sample_weight=None, nn_fixed=None):
    """
    Grid search deterministico: per ogni combinazione della griglia (ordine
    fisso di ParameterGrid) allena su train e guarda la NLL di validation
    già calcolata da fit_mlp_gaussian_multi durante l'early stopping — si
    tiene la combinazione con la NLL di validation più bassa.

    Nota sulla scelta della metrica: si usa la NLL e non la MAE perché qui il
    modello non è giudicato solo sulla qualità del valore centrale (mu), ma
    anche sulla qualità della sigma dichiarata. Un modello con una MAE bassa
    ma una sigma inventata a caso avrebbe comunque una NLL di validation
    alta, e verrebbe correttamente scartato a favore di uno con mu magari
    leggermente peggiore ma sigma più onesta.
    """
    nn_fixed = nn_fixed or {}

    best_val_nll = np.inf
    best_params = None
    best_fit_result = None

    for params in ParameterGrid(grid):
        model, feature_scaler, y_stats, val_nll = fit_mlp_gaussian_multi(
            X_tr, Y_tr, X_val, Y_val, sample_weight=sample_weight, **params, **nn_fixed,
        )
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_params = params
            best_fit_result = (model, feature_scaler, y_stats)

    return best_fit_result, best_params, best_val_nll
