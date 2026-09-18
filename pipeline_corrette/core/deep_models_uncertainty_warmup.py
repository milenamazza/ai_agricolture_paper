"""
deep_models_uncertainty_warmup.py
─────────────────────────────────────────────────────────────────────────────
MC-Dropout con WARM-UP della media: stessa rete, stessa predizione e stessa
scomposizione dell'incertezza di deep_models_uncertainty.py — cambia SOLO come
la rete viene allenata.

IL PROBLEMA CHE AFFRONTA
La rete ha due teste, mu e log(sigma^2), allenate insieme con la NLL gaussiana

    0.5 * log_var + 0.5 * exp(-log_var) * (y - mu)^2

fin dalla prima epoca. Sui punti difficili la rete ha due modi di abbassare il
secondo termine: migliorare mu, oppure alzare log_var. Il secondo è spesso più
rapido, e all'inizio del training la rete tende a prenderlo: sigma "assorbe"
errori che la media avrebbe potuto ridurre, e mu resta sottoallenata proprio
dove servirebbe (Skafte et al. 2019, "Reliable training and estimation of
variance networks"; Seitzer et al. 2022, "On the pitfalls of heteroscedastic
uncertainty estimation with probabilistic neural networks").

COSA FA IL WARM-UP
  Fase 1 — si allena SOLO mu, con l'errore quadratico medio. La testa log_var
           non riceve gradiente, quindi non c'è modo di "arrendersi" gonfiando
           sigma. Si ferma quando l'MSE di validation smette di migliorare
           (stessa patience dell'early stopping di sempre) e si ricarica il
           miglior stato.
  Passaggio — la testa log_var viene inizializzata a un sigma COSTANTE pari
           all'errore effettivo della media appena allenata: pesi a zero, bias
           = log(MSE di train) per orizzonte. Senza questo passo log_var
           partirebbe da valori casuali, la NLL avrebbe gradienti enormi nelle
           prime epoche e rovinerebbe la media appena costruita.
  Fase 2 — si allena TUTTA la rete con la NLL, come nella versione senza
           warm-up, con un ottimizzatore nuovo (i momenti di Adam accumulati in
           fase 1 riguardano un'altra loss). Early stopping sulla NLL di
           validation.

Il valore ritornato come punteggio è la NLL di validation della fase 2: la
stessa metrica delle pipeline senza warm-up, quindi i valori di Optuna sono
direttamente confrontabili.

COSA ASPETTARSI, E COSA NO
Tipicamente mu migliora e, di conseguenza, sigma si restringe. Ma bande più
strette non sono di per sé un risultato: se la coverage scende sotto il
nominale l'incertezza è solo diventata ottimista. Vanno guardati insieme MAE,
NLL, coverage e sharpness.

PERCHÉ IMPORTA DA deep_models_uncertainty.py
Architettura (MLPGaussianMulti), loss (gaussian_nll_loss), predizione
(mc_predict_mlp_gaussian_multi) e costanti sono importate da lì, non copiate:
per confrontare con e senza warm-up deve cambiare solo il training, e una
copia divergente per errore falserebbe il confronto. Importare un modulo non
lo modifica.

Diagnostica: dopo il fit, model.warmup_info contiene epoche e punteggi di
entrambe le fasi (vedi _train_loop_warmup).
"""

from __future__ import annotations

import os
# Deve essere impostata PRIMA di "import torch": viene letta da cuBLAS alla
# prima inizializzazione del contesto CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from deep_models_uncertainty import (  # noqa: F401  (import in sola lettura; alcuni nomi sono riesportati)
    SEED, DEVICE, LOG_VAR_MIN, LOG_VAR_MAX, _set_seed,
    MLPGaussianMulti, gaussian_nll_loss, mc_predict_mlp_gaussian_multi,
)


# ═══════════════════════════════════════════════════════════════════════════
# Loss della fase 1: solo la media
# ═══════════════════════════════════════════════════════════════════════════

def mse_mu_loss(mu, y_true, sample_weight=None):
    """Errore quadratico medio sulla sola mu, con la stessa struttura di media
    di gaussian_nll_loss (prima sugli orizzonti, così ogni riga pesa 1, poi sul
    batch applicando il peso per riga). È la NLL con log_var fissato a zero, a
    meno del fattore 0.5 e di una costante: stesso minimo per mu."""
    per_row_loss = ((y_true - mu) ** 2).mean(dim=1)
    if sample_weight is not None:
        return (per_row_loss * sample_weight).mean()
    return per_row_loss.mean()


def _copia_stato(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ═══════════════════════════════════════════════════════════════════════════
# Training a due fasi
# ═══════════════════════════════════════════════════════════════════════════

def _train_loop_warmup(model, X_tr, Y_tr, X_val, Y_val, seed, sample_weight=None,
                       epochs=300, lr=1e-3, batch_size=32, patience=30):
    """Fase 1 (solo mu, MSE) -> inizializzazione di log_var -> fase 2 (NLL).

    Il seed è un parametro esplicito perché la stessa procedura serve anche a
    deep_models_ensemble_warmup.py, dove ogni membro ha il suo. Ciascuna fase
    fa al più `epochs` epoche con la stessa `patience`, quindi il costo
    massimo è il doppio della versione senza warm-up.

    Ritorna (model, best_val_nll) e lascia in model.warmup_info:
      epoche_fase1, epoca_migliore_fase1, val_mse_passaggio,
      epoche_fase2, epoca_migliore_fase2, val_nll_inizio_fase2, val_nll_finale
    """
    _set_seed(seed)
    model = model.to(DEVICE)

    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
    Y_tr_tensor = torch.tensor(Y_tr, dtype=torch.float32)
    if sample_weight is not None:
        weight_tensor = torch.tensor(np.asarray(sample_weight), dtype=torch.float32)
    else:
        weight_tensor = torch.ones(len(Y_tr_tensor))

    # un solo generator per entrambe le fasi: l'ordine dei batch della fase 2
    # prosegue la sequenza della fase 1, ed è comunque deterministico
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_tr_tensor, Y_tr_tensor, weight_tensor),
        batch_size=batch_size, shuffle=True, generator=generator,
    )

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(DEVICE)

    # ── Fase 1: solo mu ──────────────────────────────────────────────────────
    # L'ottimizzatore contiene solo tronco e testa mu: la testa log_var resta
    # esattamente com'è (nemmeno il weight decay la tocca).
    parametri_mu = list(model.trunk.parameters()) + list(model.mu_head.parameters())
    optimizer = torch.optim.Adam(parametri_mu, lr=lr, weight_decay=1e-5)

    best_val_mse = np.inf
    best_state = None
    epoca_migliore_fase1 = 0
    epoche_fase1 = 0
    epochs_without_improvement = 0

    for epoca in range(1, epochs + 1):
        epoche_fase1 = epoca
        model.train()
        for X_batch, Y_batch, weight_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            weight_batch = weight_batch.to(DEVICE)

            optimizer.zero_grad()
            mu_batch, _ = model(X_batch)
            loss = mse_mu_loss(mu_batch, Y_batch, weight_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            mu_val, _ = model(X_val_tensor)
            val_mse = mse_mu_loss(mu_val, Y_val_tensor).item()

        if val_mse < best_val_mse - 1e-5:
            best_val_mse = val_mse
            best_state = _copia_stato(model)
            epoca_migliore_fase1 = epoca
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Passaggio: sigma costante pari all'errore reale della media ─────────
    # Dropout spento: si vuole l'errore della rete che poi predice, non di una
    # versione rumorosa. MSE per orizzonte, in scala normalizzata.
    model.eval()
    with torch.no_grad():
        mu_tr, _ = model(X_tr_tensor.to(DEVICE))
        mse_per_orizzonte = ((Y_tr_tensor.to(DEVICE) - mu_tr) ** 2).mean(dim=0)
        log_var_iniziale = torch.log(mse_per_orizzonte.clamp_min(1e-12))
        log_var_iniziale = log_var_iniziale.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        model.logvar_head.weight.zero_()
        model.logvar_head.bias.copy_(log_var_iniziale)

        mu_val, log_var_val = model(X_val_tensor)
        val_nll_inizio_fase2 = gaussian_nll_loss(mu_val, log_var_val, Y_val_tensor).item()

    # ── Fase 2: tutta la rete con la NLL ─────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # si parte dallo stato del passaggio: se la fase 2 non migliorasse mai la
    # NLL, il modello restituito è la media della fase 1 con sigma costante
    best_val_nll = val_nll_inizio_fase2
    best_state = _copia_stato(model)
    epoca_migliore_fase2 = 0
    epoche_fase2 = 0
    epochs_without_improvement = 0

    for epoca in range(1, epochs + 1):
        epoche_fase2 = epoca
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
            best_state = _copia_stato(model)
            epoca_migliore_fase2 = epoca
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)

    model.warmup_info = {
        "epoche_fase1": int(epoche_fase1),
        "epoca_migliore_fase1": int(epoca_migliore_fase1),
        "val_mse_passaggio": round(float(best_val_mse), 5),
        "epoche_fase2": int(epoche_fase2),
        "epoca_migliore_fase2": int(epoca_migliore_fase2),
        "val_nll_inizio_fase2": round(float(val_nll_inizio_fase2), 5),
        "val_nll_finale": round(float(best_val_nll), 5),
    }
    return model, best_val_nll


def fit_mlp_gaussian_multi(X_tr, Y_tr, X_val, Y_val, sample_weight=None,
                           hidden=(64, 32), dropout=0.2, lr=1e-3,
                           batch_size=32, patience=30, epochs=300):
    """Stessa firma e stesso ritorno di
    deep_models_uncertainty.fit_mlp_gaussian_multi — (model, feature_scaler,
    y_stats, best_val_nll) — e stessa normalizzazione di feature e target.
    Cambia solo il training (_train_loop_warmup)."""
    _set_seed()   # deve precedere la costruzione del modello: i pesi si inizializzano lì

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
        model, X_tr_scaled, Y_tr_scaled, X_val_scaled, Y_val_scaled, seed=SEED,
        sample_weight=sample_weight, epochs=epochs, lr=lr,
        batch_size=batch_size, patience=patience,
    )

    y_stats = (y_mean, y_std)
    return model, feature_scaler, y_stats, best_val_nll
