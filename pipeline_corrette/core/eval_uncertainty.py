"""
eval_uncertainty.py
─────────────────────────────────────────────────────────────────────────────
Funzioni per rispondere alla domanda: "l'incertezza che il modello dichiara è
calcolata bene?". Una previsione puntuale si giudica con MAE/RMSE/R² (quanto
è vicina alla realtà); una previsione PROBABILISTICA (mu, sigma) si deve
giudicare in modo diverso, su due assi separati che vanno guardati insieme:

  1. CALIBRAZIONE (coverage): se il modello dice "sono sicuro al 90% che il
     valore vero sia in questo intervallo", nella pratica il valore vero
     dovrebbe caderci dentro circa il 90% delle volte — né di più né di
     meno. Si misura con la "coverage empirica".

  2. NITIDEZZA (sharpness): a parità di calibrazione, un modello che dà
     intervalli più STRETTI è più utile di uno che dà intervalli larghissimi
     (dire sempre "potrebbe essere qualunque cosa" è calibrato ma inutile).

Un modello di incertezza va bene solo se è BUONO su entrambi gli assi insieme
— la NLL (Negative Log-Likelihood) è la metrica riassuntiva che li combina
in un solo numero (vedi gaussian_nll_score più sotto).

Tutte le funzioni qui assumono che la distribuzione predittiva sia una
gaussiana N(mu, sigma^2) — la stessa assunzione usata per allenare il
modello in deep_models_uncertainty.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, kstest


# ═══════════════════════════════════════════════════════════════════════════
# Negative Log-Likelihood — metrica riassuntiva principale
# ═══════════════════════════════════════════════════════════════════════════

def gaussian_nll_score(y_true, mu, sigma):
    """
    NLL media di una gaussiana N(mu, sigma^2) valutata nei punti y_true
    (qui SENZA scartare la costante 0.5*log(2*pi), a differenza della loss
    di training: qui vogliamo un numero interpretabile e confrontabile tra
    modelli diversi, non solo un gradiente da minimizzare).

    Un numero più basso è meglio. Penalizza automaticamente sia un mu
    sbagliato sia un sigma scelto male (troppo piccolo => grande penalità se
    l'errore è alto; troppo grande => piccola penalità sull'errore ma il
    termine log(sigma) cresce comunque).
    """
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    variance = sigma ** 2
    squared_error = (y_true - mu) ** 2

    per_point_nll = 0.5 * np.log(2 * np.pi * variance) + 0.5 * squared_error / variance
    return float(np.mean(per_point_nll))


# ═══════════════════════════════════════════════════════════════════════════
# Coverage e sharpness a un dato livello di confidenza nominale
# ═══════════════════════════════════════════════════════════════════════════

def _z_score_for_level(nominal_level):
    """
    Quantile normale standard per un intervallo simmetrico al livello
    nominale dato. Esempio: per il 90% di copertura nominale, l'intervallo è
    [mu - z*sigma, mu + z*sigma] con z tale che il 90% dell'area di una
    gaussiana standard sta tra -z e +z. Per il 90%, z vale circa 1.645.
    """
    tail_probability = (1.0 - nominal_level) / 2.0
    return norm.ppf(1.0 - tail_probability)


def coverage_at_level(y_true, mu, sigma, nominal_level):
    """
    Percentuale di punti in cui y_true cade dentro l'intervallo
    [mu - z*sigma, mu + z*sigma], dove z è scelto in modo che, SE il modello
    fosse perfettamente calibrato, questa percentuale dovrebbe essere circa
    uguale a nominal_level (es. 0.90 per un intervallo al 90%).

    coverage_empirica ≈ nominal_level         → ben calibrato
    coverage_empirica <  nominal_level        → intervalli troppo STRETTI
                                                (il modello è overconfident:
                                                 il caso più pericoloso)
    coverage_empirica >  nominal_level        → intervalli troppo LARGHI
                                                (il modello è prudente/pigro)
    """
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    z = _z_score_for_level(nominal_level)
    lower_bound = mu - z * sigma
    upper_bound = mu + z * sigma

    inside_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    return float(np.mean(inside_interval))


def sharpness_at_level(sigma, nominal_level):
    """
    Larghezza media dell'intervallo allo stesso livello nominale usato in
    coverage_at_level: 2 * z * sigma, mediata su tutti i punti. Va guardata
    SEMPRE insieme alla coverage: un modello con intervalli larghissimi può
    avere coverage perfetta ma essere inutile in pratica.
    """
    sigma = np.asarray(sigma, dtype=float)
    z = _z_score_for_level(nominal_level)
    interval_width = 2.0 * z * sigma
    return float(np.mean(interval_width))


# ═══════════════════════════════════════════════════════════════════════════
# PIT (Probability Integral Transform) — diagnostica sulla FORMA della
# calibrazione, non solo un numero riassuntivo
# ═══════════════════════════════════════════════════════════════════════════

def pit_values(y_true, mu, sigma):
    """
    Per ciascun punto di test calcola:

        u_i = Phi( (y_true_i - mu_i) / sigma_i )

    dove Phi è la funzione di ripartizione (CDF) della gaussiana standard.
    Interpretazione: u_i è "in che percentile della distribuzione predetta
    cade il valore vero". Se il modello è ben calibrato, questi percentili
    dovrebbero essere sparsi UNIFORMEMENTE tra 0 e 1 (nessuna zona più
    probabile di un'altra) — esattamente come succederebbe se estraessimo
    y_true a caso dalla distribuzione che il modello ha previsto.

    Uso tipico: costruire un istogramma dei valori restituiti da questa
    funzione. Forme tipiche e cosa significano:
      - istogramma piatto (uniforme)     → ben calibrato
      - a forma di "U" (troppi vicino a 0 e 1) → intervalli troppo STRETTI
      - a "gobba" (troppi vicino a 0.5)  → intervalli troppo LARGHI
      - asimmetrico (sbilanciato a sinistra o destra) → mu sistematicamente
        troppo alto o troppo basso (bias)
    """
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    standardized_residual = (y_true - mu) / sigma
    return norm.cdf(standardized_residual)


def pit_uniformity_test(y_true, mu, sigma):
    """
    Test di Kolmogorov-Smirnov: confronta i valori PIT osservati con una
    distribuzione Uniforme(0,1) teorica, e restituisce (statistica, p-value).

    Una statistica KS vicina a 0 (p-value alto) è compatibile con una buona
    calibrazione. Una statistica alta (p-value basso, es. < 0.05) è
    un'evidenza formale di miscalibrazione — da leggere insieme alla forma
    dell'istogramma PIT per capire IN CHE MODO il modello è miscalibrato.
    """
    pit = pit_values(y_true, mu, sigma)
    ks_statistic, p_value = kstest(pit, "uniform")
    return float(ks_statistic), float(p_value)


# ═══════════════════════════════════════════════════════════════════════════
# Tabella di calibrazione: nominal vs coverage empirica vs sharpness, per più
# livelli insieme — il modo più diretto di "vedere" se l'incertezza è buona
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CALIBRATION_LEVELS = (0.50, 0.68, 0.80, 0.90, 0.95, 0.99)


def calibration_table(y_true, mu, sigma, levels=DEFAULT_CALIBRATION_LEVELS):
    """
    Costruisce una tabella con una riga per ciascun livello nominale
    richiesto (default: 50/68/80/90/95/99%), con colonne:
      nominal_level    — il livello di confidenza richiesto
      empirical_coverage — quanto viene osservato davvero nei dati
      mean_interval_width — sharpness a quel livello

    Un modello ben calibrato ha empirical_coverage vicina a nominal_level per
    OGNI riga (non solo per una). Questa tabella è pensata per essere anche
    tracciata come grafico (nominal sull'asse x, empirical sull'asse y): i
    punti dovrebbero stare vicino alla diagonale.
    """
    rows = []
    for level in levels:
        rows.append({
            "nominal_level": level,
            "empirical_coverage": coverage_at_level(y_true, mu, sigma, level),
            "mean_interval_width": sharpness_at_level(sigma, level),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Riepilogo completo per un gruppo di righe (una combinazione di
# modello/feature-set/orizzonte/split) — pensato per popolare una riga del
# CSV finale di metriche
# ═══════════════════════════════════════════════════════════════════════════

def uncertainty_metric_dict(y_true, mu, epistemic_std, aleatoric_std, total_std):
    """
    Riassume in un dizionario piatto (adatto a diventare una riga di CSV)
    tutte le metriche di incertezza per un gruppo di previsioni: NLL, PIT
    (statistica KS + p-value), coverage e sharpness ai livelli standard, più
    la grandezza media delle due componenti di incertezza (utile per capire
    quale delle due domina, e se il loro comportamento è sensato: es. più
    incertezza epistemica su orizzonti lunghi, più aleatoria nei giorni di
    pioggia/irrigazione).
    """
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    total_std = np.asarray(total_std, dtype=float)

    result = {
        "NLL": round(gaussian_nll_score(y_true, mu, total_std), 4),
        "mean_epistemic_std": round(float(np.mean(epistemic_std)), 4),
        "mean_aleatoric_std": round(float(np.mean(aleatoric_std)), 4),
        "mean_total_std": round(float(np.mean(total_std)), 4),
    }

    ks_statistic, ks_p_value = pit_uniformity_test(y_true, mu, total_std)
    result["pit_ks_stat"] = round(ks_statistic, 4)
    result["pit_ks_pvalue"] = round(ks_p_value, 4)

    for level in DEFAULT_CALIBRATION_LEVELS:
        level_label = f"{int(round(level * 100))}"
        result[f"coverage_{level_label}"] = round(coverage_at_level(y_true, mu, total_std, level), 4)
        result[f"sharpness_{level_label}"] = round(sharpness_at_level(total_std, level), 4)

    return result
