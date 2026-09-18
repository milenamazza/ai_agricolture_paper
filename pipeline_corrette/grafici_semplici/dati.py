"""
dati.py — caricamento dati condiviso dai 5 script di grafici
─────────────────────────────────────────────────────────────────────────────
Stessa logica di crea_grafici.py, spostata qui perché usata da più script:
nessuna riscrittura, solo un posto solo da cui caricare.

carica_filtrato()  i tre approcci del confronto filtrato (Fisica / Data
                   driven / Residual error), da confronto_filtrato_optuna.xlsx.
                   Ridge escluso: RF per il data driven, MLP per il residuo
                   (senza esclusione il miglior data driven sarebbe Ridge,
                   MAE 3.86).

carica_gruppi()    Deep Ensemble e Quantile Regression, ciascuno nella
                   variante diretta e residuo, con Penman ricalcolato sulle
                   STESSE righe (join su data+sensore+orizzonte) per un
                   confronto alla pari. Il feature-set di ciascuna famiglia è
                   quello migliore per il SUO punteggio primario (NLL per i
                   modelli gaussiani, pinball loss per la Quantile
                   Regression) — lo stesso criterio del foglio "Riepilogo"
                   dell'Excel:

                     Deep Ensemble        diretta TERRA_ACQUA   residuo ACQUA
                     Quantile Regression  diretta TERRA         residuo TERRA
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, "..")

REPORT_FILTRATO = os.path.join(PIPE, "confronto", "confronto_filtrato_optuna.xlsx")
PENMAN_PREDS = os.path.join(PIPE, "penman_baseline", "results", "penman_preds.csv")

ORIZZONTI = [f"t+{h}" for h in range(1, 8)]
Z90 = 1.645

FISICA = "Fisica (Penman-Monteith)"
DATA_DRIVEN = "Data driven"
RESIDUAL = "Residual error"

FILTRATO = {
    DATA_DRIVEN: ("ML puro (componibile, filtrato, Optuna)", "RF", "TERRA_METEO"),
    RESIDUAL: ("Penman + ML (residuo, solo feature, filtrato, Optuna)", "MLP", "ACQUA"),
    FISICA: ("Penman puro", "Penman-Monteith", "--"),
}

GRUPPI = {
    "deep_ensemble": {
        "nome": "Deep Ensemble",
        "scompone": True,
        "famiglie": [
            ("diretta", "deep_ensemble/results_optuna/ensemble_preds.csv",
             "TERRA_ACQUA", "gaussiana"),
            ("residuo", "deep_ensemble/results_residual_optuna/ensemble_residual_preds.csv",
             "ACQUA", "gaussiana"),
        ],
    },
    "quantile": {
        "nome": "Quantile Regression",
        "scompone": False,
        "famiglie": [
            ("diretta", "quantile_regression/results_optuna/quantile_preds.csv",
             "TERRA", "quantili"),
            ("residuo", "quantile_regression/results_residual_optuna/quantile_residual_preds.csv",
             "TERRA", "quantili"),
        ],
    },
}


def carica_filtrato():
    d = pd.read_excel(REPORT_FILTRATO, sheet_name="Dati completi")
    righe = []
    for approccio, (tipo, modello, fset) in FILTRATO.items():
        g = d[(d["tipo"] == tipo) & (d["model"] == modello)
              & (d["feature_set"].astype(str) == fset)]
        for _, r in g.iterrows():
            righe.append({"approccio": approccio, "modello": modello, "feature_set": fset,
                          "orizzonte": r["horizon"], "errore": r["MAE"], "n": r["n"]})
    return pd.DataFrame(righe)


def carica_gruppi():
    penman = pd.read_csv(PENMAN_PREDS)
    penman = penman[penman["split"] == "test"][["date", "sensor", "horizon", "y_pred"]]

    gruppi = {}
    for chiave, cfg in GRUPPI.items():
        famiglie = []
        for variante, percorso, fset, tipo in cfg["famiglie"]:
            d = pd.read_csv(os.path.join(PIPE, percorso))
            d = d[(d["split"] == "test") & (d["feature_set"] == fset)].copy()
            if tipo == "quantili":
                prev = d["q50"].to_numpy()
                basso, alto = d["q05"].to_numpy(), d["q95"].to_numpy()
                d["epistemica"] = np.nan
                d["aleatoria"] = np.nan
            else:
                prev = d["mu_pred"].to_numpy()
                sd = d["total_std"].to_numpy()
                basso, alto = prev - Z90 * sd, prev + Z90 * sd
                d["epistemica"] = d["epistemic_std"]
                d["aleatoria"] = d["aleatoric_std"]
            d["previsione"] = prev
            d["basso"], d["alto"] = basso, alto
            d["dentro"] = (d["y_true"] >= basso) & (d["y_true"] <= alto)
            d["ampiezza"] = alto - basso
            d["errore"] = (d["y_true"] - prev).abs()
            d = d.merge(penman, on=["date", "sensor", "horizon"], how="inner")
            famiglie.append({"variante": variante, "feature_set": fset, "dati": d})
        gruppi[chiave] = {**cfg, "famiglie": famiglie}
    return gruppi


def serie(d, colonna):
    return [d[d["horizon"] == h][colonna].mean() for h in ORIZZONTI]


# ═══════════════════════════════════════════════════════════════════════════
# Per il grafico 06 (storico + previsione da una singola data di emissione)
# ═══════════════════════════════════════════════════════════════════════════
#
# Nota sulla semantica delle date, verificata numericamente sui dati:
# `date` è il giorno di EMISSIONE, non quello previsto. La riga
# (date=D, horizon=t+h) ha `y_true` = umidità misurata a D+h.
# Controllo: y_true(D, t+2) == y_true(D+1, t+1) con scarto 0.0 su 147 righe.

def carica_previsioni(chiave, variante, splits=("train", "val", "test")):
    """Predizioni di UNA famiglia su tutti gli split richiesti, SENZA il join
    con Penman.

    Perché non riusare carica_gruppi(): quella filtra a split=="test" e fa un
    merge interno con le predizioni Penman di test, quindi scarterebbe le
    righe val/train che qui servono per lo storico.

    Aggiunge `bersaglio` = date + h, cioè il giorno a cui la previsione si
    riferisce davvero.
    """
    cfg = next(f for f in GRUPPI[chiave]["famiglie"] if f[0] == variante)
    _, percorso, fset, tipo = cfg

    d = pd.read_csv(os.path.join(PIPE, percorso))
    d = d[(d["split"].isin(splits)) & (d["feature_set"] == fset)].copy()

    if tipo == "quantili":
        prev = d["q50"].to_numpy()
        basso, alto = d["q05"].to_numpy(), d["q95"].to_numpy()
    else:
        prev = d["mu_pred"].to_numpy()
        sd = d["total_std"].to_numpy()
        basso, alto = prev - Z90 * sd, prev + Z90 * sd

    d["previsione"] = prev
    d["basso"], d["alto"] = basso, alto
    d["dentro"] = (d["y_true"] >= basso) & (d["y_true"] <= alto)
    d["date"] = pd.to_datetime(d["date"])
    d["h"] = d["horizon"].str.replace("t+", "", regex=False).astype(int)
    d["bersaglio"] = d["date"] + pd.to_timedelta(d["h"], unit="D")
    return d


def serie_misurata(d):
    """Umidità misurata giorno per giorno, ricostruita da TUTTI gli orizzonti.

    Ogni riga (date, h) porta con sé la misura del giorno date+h: usandole
    tutte invece del solo t+1 la serie risulta molto più fitta — per
    EM-500-12 nel 2026 sono 70 giorni consecutivi senza buchi, contro i 43
    con buchi che si otterrebbero dal solo t+1.

    Ritorna un DataFrame ordinato con le colonne `giorno` e `umidita`.
    """
    s = (d.drop_duplicates("bersaglio")[["bersaglio", "y_true"]]
         .rename(columns={"bersaglio": "giorno", "y_true": "umidita"})
         .sort_values("giorno")
         .reset_index(drop=True))
    return s
