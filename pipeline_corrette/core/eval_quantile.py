"""
eval_quantile.py
─────────────────────────────────────────────────────────────────────────────
Metriche per giudicare un modello di Quantile Regression. A differenza di
eval_uncertainty.py (in uncertainty_ml/, per MC-Dropout/Deep Ensemble), qui
NON si assume che l'errore sia gaussiano: gli intervalli di previsione si
leggono direttamente dai percentili previsti, senza bisogno di uno z-score
o di una sigma.

Non è incluso un equivalente del PIT (Probability Integral Transform) usato
per i modelli gaussiani: il PIT richiede di poter valutare la funzione di
ripartizione (CDF) in modo continuo, mentre qui si hanno solo alcuni
percentili discreti (5°, 25°, 50°, 75°, 95°) — un'approssimazione sarebbe
possibile (interpolando linearmente tra i percentili noti) ma poco robusta
con così pochi punti, quindi si omette: la pinball loss e le coppie
coverage/sharpness già dicono se il modello è ben calibrato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pinball_loss_score(y_true, quantile_predictions, quantile_levels):
    """
    Media della pinball loss su tutti i percentili richiesti — l'equivalente,
    per la quantile regression, della NLL gaussiana: un solo numero che
    scende quanto più i percentili previsti sono vicini a quelli veri della
    distribuzione condizionata (più basso è meglio).

    y_true:               (n_rows,)
    quantile_predictions: (n_rows, n_quantiles)
    quantile_levels:      (n_quantiles,) — es. [0.05, 0.25, 0.50, 0.75, 0.95]
    """
    y_true = np.asarray(y_true, dtype=float)
    quantile_predictions = np.asarray(quantile_predictions, dtype=float)
    quantile_levels = np.asarray(quantile_levels, dtype=float)

    error = y_true[:, np.newaxis] - quantile_predictions   # (n_rows, n_quantiles)
    under_estimate_cost = quantile_levels * error
    over_estimate_cost = (quantile_levels - 1.0) * error
    per_element_loss = np.maximum(under_estimate_cost, over_estimate_cost)

    return float(np.mean(per_element_loss))


def _index_of_quantile_level(quantile_levels, target_level):
    """Trova la posizione del percentile richiesto (es. 0.90) nell'elenco
    dei percentili effettivamente previsti dal modello. Solleva un errore
    chiaro se quel percentile non è stato previsto — meglio di un indice
    silenziosamente sbagliato."""
    quantile_levels = np.asarray(quantile_levels, dtype=float)
    closest_index = int(np.argmin(np.abs(quantile_levels - target_level)))
    if not np.isclose(quantile_levels[closest_index], target_level):
        raise ValueError(
            f"Il percentile {target_level} non è tra quelli previsti dal modello: "
            f"{quantile_levels.tolist()}"
        )
    return closest_index


def coverage_from_quantiles(y_true, quantile_predictions, quantile_levels, lower_level, upper_level):
    """
    Percentuale di punti in cui y_true cade dentro [Q(lower_level), Q(upper_level)].
    Esempio: lower_level=0.05, upper_level=0.95 → intervallo al 90%, e se il
    modello è ben calibrato la coverage empirica dovrebbe essere vicina a 0.90
    — stessa idea di eval_uncertainty.coverage_at_level, ma qui l'intervallo è
    preso DIRETTAMENTE dai percentili previsti, senza nessuna formula/z-score.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower_index = _index_of_quantile_level(quantile_levels, lower_level)
    upper_index = _index_of_quantile_level(quantile_levels, upper_level)

    lower_bound = quantile_predictions[:, lower_index]
    upper_bound = quantile_predictions[:, upper_index]

    inside_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    return float(np.mean(inside_interval))


def sharpness_from_quantiles(quantile_predictions, quantile_levels, lower_level, upper_level):
    """Larghezza media dell'intervallo [Q(lower_level), Q(upper_level)] — da
    guardare sempre insieme alla coverage corrispondente (stesso principio
    di eval_uncertainty.sharpness_at_level)."""
    lower_index = _index_of_quantile_level(quantile_levels, lower_level)
    upper_index = _index_of_quantile_level(quantile_levels, upper_level)

    interval_width = quantile_predictions[:, upper_index] - quantile_predictions[:, lower_index]
    return float(np.mean(interval_width))


# Coppie di percentili per cui costruire intervalli standard, coerenti con
# QUANTILE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95) in deep_models_quantile.py.
STANDARD_INTERVALS = (
    (0.25, 0.75, "50"),   # intervallo al 50%
    (0.05, 0.95, "90"),   # intervallo al 90%
)


def quantile_metric_dict(y_true, quantile_predictions, quantile_levels):
    """
    Riassume in un dizionario piatto (adatto a diventare una riga di CSV) le
    metriche di un gruppo di previsioni a quantili: pinball loss complessiva,
    più coverage/sharpness per gli intervalli standard al 50% e al 90%.
    """
    result = {
        "pinball_loss": round(pinball_loss_score(y_true, quantile_predictions, quantile_levels), 4),
    }

    for lower_level, upper_level, label in STANDARD_INTERVALS:
        result[f"coverage_{label}"] = round(
            coverage_from_quantiles(y_true, quantile_predictions, quantile_levels, lower_level, upper_level), 4,
        )
        result[f"sharpness_{label}"] = round(
            sharpness_from_quantiles(quantile_predictions, quantile_levels, lower_level, upper_level), 4,
        )

    return result
