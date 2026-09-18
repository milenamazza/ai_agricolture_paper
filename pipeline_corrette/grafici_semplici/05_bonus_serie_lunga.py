"""
05_bonus_serie_lunga.py
─────────────────────────────────────────────────────────────────────────────
BONUS del grafico 05: stesso schema (storico solo linea, poi riga verticale,
poi previsione + fascia, con una copia con e una senza i puntini veri), ma
sul tratto continuo più lungo disponibile fra tutti i 14 sensori.

Perché EM-500-9 e non EM-500-11 (come chiesto): EM-500-11 non ha righe di
split "val" né "test" — solo "train" (luglio-novembre 2025), quindi non
esiste nessuna previsione da mostrare per quel sensore. Cercando su tutti i
sensori il tratto senza buchi più lungo si trova su EM-500-9: da fine marzo
2026 in poi, train + val + test sono 100 giorni CONSECUTIVI senza un solo
buco (verificato: zero salti >1 giorno in quel tratto) — contro i 37-58
giorni disponibili per EM-500-12 nel grafico 05 originale.

Non tocca dati.py, stile.py, 05_serie_temporale.py né 06_previsione_storico.py:
riusa solo le funzioni già esistenti in dati.py (carica_previsioni,
già scritta per lo script 06).

Output (2 gruppi x 2 versioni, 4 file):
  05_bonus_serie_lunga_{gruppo}.png
  05_bonus_serie_lunga_{gruppo}_con_misurato.png

Utilizzo:
    python -u 05_bonus_serie_lunga.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import dati
import stile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "risultati")

SENSORE = "EM-500-9"
# inizio del tratto continuo (prima di questa data c'è un buco di 227 giorni
# nel train di questo sensore, verificato sui dati)
INIZIO_CONTINUO = "2026-03-31"
INIZIO_PREVISIONI = "2026-05-28"   # dove comincia val: da qui la previsione ha senso mostrarla

COLORE_STORICO = "#17222B"
COLORE_ANDAMENTO = "#2E6F8E"
COLORE_FUORI = "#C1483C"


def disegna(chiave, gruppo, con_misurato):
    variante = gruppo["variante_migliore"]
    d = dati.carica_previsioni(chiave, variante, splits=("train", "val", "test"))
    d = d[(d["sensor"] == SENSORE) & (d["horizon"] == "t+1")].copy()
    d = d[d["date"] >= pd.Timestamp(INIZIO_CONTINUO)].sort_values("date").reset_index(drop=True)

    cutoff = pd.Timestamp(INIZIO_PREVISIONI)
    storico = d[d["date"] < cutoff]
    previsioni = d[d["date"] >= cutoff]
    if storico.empty or previsioni.empty:
        print(f"  saltato {chiave}: storico={len(storico)}, previsioni={len(previsioni)}")
        return

    colore = stile.COLORE_VARIANTE[variante]
    fig, ax = plt.subplots(figsize=(13, 5.5))   # più largo: 100 giorni invece di 37-58

    ax.plot(storico["date"], storico["y_true"], "-", color=COLORE_ANDAMENTO,
            linewidth=1.8, label="Umidità misurata")

    ultimo_giorno = storico["date"].iloc[-1]
    ultimo_valore = storico["y_true"].iloc[-1]
    ponte_x = [ultimo_giorno] + list(previsioni["date"])
    ponte_y = [ultimo_valore] + list(previsioni["previsione"])

    ax.fill_between(previsioni["date"], previsioni["basso"], previsioni["alto"],
                    color=colore, alpha=0.18, label="Intervallo al 90%")
    ax.plot(ponte_x, ponte_y, "-", color=colore, linewidth=1.8, label="Previsione")

    if con_misurato:
        dentro = previsioni[previsioni["dentro"]]
        fuori = previsioni[~previsioni["dentro"]]
        if len(dentro):
            ax.plot(dentro["date"], dentro["y_true"], "o", color=COLORE_STORICO,
                    markersize=6, label="Misurato (dentro la fascia)")
        if len(fuori):
            ax.plot(fuori["date"], fuori["y_true"], "o", color=COLORE_FUORI, markersize=7,
                    label="Misurato (fuori dalla fascia)")

    ax.axvline(cutoff, color="#555555", linestyle="--", linewidth=1.4)
    ax.text(cutoff, 0.97, " Previsioni", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=11, fontweight="bold", color="#555555")

    ax.set_ylabel("Umidità del suolo (%)")
    ax.set_ylim(25, 85)
    #stile.titolo(ax, f"{gruppo['nome']} — {stile.nome_variante(variante)}, "
    #                 f"t+1, sensore {SENSORE}")
    ax.legend(loc="lower right", fontsize=10)
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    suffisso = "_con_misurato" if con_misurato else ""
    percorso = os.path.join(OUT, f"05_bonus_serie_lunga_{chiave}{suffisso}.png")
    fig.savefig(percorso, dpi=300)
    plt.close(fig)
    print(f"Salvato: {percorso}  (storico: {len(storico)} giorni, "
          f"previsioni: {len(previsioni)} giorni, totale: {len(d)})")


def main():
    os.makedirs(OUT, exist_ok=True)
    stile.applica()
    gruppi = dati.carica_gruppi()   # solo per scegliere la variante migliore, come in 05/06
    for chiave, gruppo in gruppi.items():
        migliore = min(gruppo["famiglie"], key=lambda f: f["dati"]["errore"].mean())
        gruppo["variante_migliore"] = migliore["variante"]
        for con_misurato in (False, True):
            disegna(chiave, gruppo, con_misurato)


if __name__ == "__main__":
    main()
