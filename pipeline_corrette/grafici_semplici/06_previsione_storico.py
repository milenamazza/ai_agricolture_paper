"""
06_previsione_storico.py
─────────────────────────────────────────────────────────────────────────────
Storico misurato + riga verticale del giorno di emissione + previsione a 7
giorni con fascia di incertezza. Due versioni, pensate per essere
SOVRAPPOSTE nelle slide:

  06_previsione_storico.png          solo previsione e fascia dopo la riga
  06_previsione_storico_misure.png   + i valori realmente misurati, come
                                     puntini neri se dentro la fascia al 90%,
                                     rossi se fuori (convenzione del graf. 05)

PERCHÉ NIENTE tight_layout()
Le due immagini devono combaciare al pixel. `tight_layout()` ricalcola i
margini in base al contenuto, e in versione B la legenda più alta sposterebbe
l'area di disegno: in sovrapposizione il grafico "salterebbe". Qui l'area è
fissata a mano con subplots_adjust() a valori identici, e xlim/ylim sono
calcolati UNA volta sull'unione di tutto ciò che compare in una qualsiasi
delle due versioni.

NOTA SULLE DATE
`date` nei CSV è il giorno di EMISSIONE: la riga (date=D, t+h) porta la
misura del giorno D+h. Lo storico va quindi ricostruito da tutti gli
orizzonti (vedi dati.serie_misurata), non dal solo t+1.

Con ORIGINE = 2026-06-14 tutti e 7 i valori reali cadono dentro la fascia,
quindi i puntini saranno tutti neri. Le origini che producono un punto rosso
sono 2026-06-25, 06-27, 06-29 e 06-30 (in tutte il punto fuori fascia è il
picco di pioggia del 2 luglio). Il colore è automatico: basta cambiare
ORIGINE qui sotto.

Utilizzo:
    python -u 06_previsione_storico.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

import dati
import stile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "risultati")

# ─── tutto modificabile in una riga ─────────────────────────────────────────
GRUPPO = "deep_ensemble"
VARIANTE = "diretta"          # la stessa che il grafico 05 seleziona per questo gruppo
SENSORE = "EM-500-12"
ORIGINE = "2026-06-2"        # giorno di emissione della previsione
GIORNI_STORICO = 30           # quanti giorni di misure mostrare prima della riga

# ─── geometria fissa: garantisce la sovrapponibilità delle due versioni ─────
FIGSIZE = (11, 5.5)
MARGINI = dict(left=0.085, right=0.985, top=0.90, bottom=0.17)

COLORE_STORICO = "#17222B"          # puntini "misurato" nella zona di previsione
COLORE_ANDAMENTO = "#2E6F8E"        # linea dell'andamento vero, prima della riga
COLORE_PREVISIONE = stile.COLORE_VARIANTE[VARIANTE]
COLORE_FUORI = "#C1483C"

# matplotlib formatta le date in inglese (locale C): qui si passano i nomi
# italiani a mano, sia per il titolo sia per le etichette dell'asse x
MESI = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]
MESI_BREVI = ["gen", "feb", "mar", "apr", "mag", "giu",
              "lug", "ago", "set", "ott", "nov", "dic"]


def data_estesa(giorno):
    return f"{giorno.day} {MESI[giorno.month - 1]} {giorno.year}"


def _etichetta_asse(valore, _pos):
    g = mdates.num2date(valore)
    return f"{g.day} {MESI_BREVI[g.month - 1]}"


def prepara():
    """Storico, previsione e misure future per l'origine scelta."""
    d = dati.carica_previsioni(GRUPPO, VARIANTE)
    d = d[d["sensor"] == SENSORE]
    origine = pd.Timestamp(ORIGINE)

    prev = d[d["date"] == origine].sort_values("h")
    if prev.empty:
        disponibili = sorted(d["date"].dt.date.unique())
        raise SystemExit(
            f"Nessuna previsione emessa il {ORIGINE} per {SENSORE}.\n"
            f"Date di emissione disponibili vicine: "
            f"{[str(x) for x in disponibili if abs((pd.Timestamp(x) - origine).days) <= 4]}")

    misurata = dati.serie_misurata(d)
    storico = misurata[(misurata["giorno"] <= origine)
                       & (misurata["giorno"] >= origine - pd.Timedelta(days=GIORNI_STORICO))]
    return storico, prev, origine


def disegna(storico, prev, origine, con_misure, limiti):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**MARGINI)   # niente tight_layout: vedi docstring

    # andamento vero, fino alla riga: solo linea, nessun puntino
    ax.plot(storico["giorno"], storico["umidita"], "-", color=COLORE_ANDAMENTO,
            linewidth=1.8, label="Umidità misurata")

    # riga verticale: giorno di emissione
    ax.axvline(origine, color="#555555", linestyle="--", linewidth=1.4)
    # etichetta in basso: in alto finirebbe sotto la legenda
    ax.annotate("previsione emessa qui", xy=(origine, limiti["ylim"][0]),
                xytext=(-8, 10), textcoords="offset points",
                ha="right", va="bottom", fontsize=10, color="#555555")

    # previsione: solo puntini, nessuna linea
    ax.fill_between(prev["bersaglio"], prev["basso"], prev["alto"],
                    color=COLORE_PREVISIONE, alpha=0.18, label="Intervallo al 90%")
    ax.plot(prev["bersaglio"], prev["previsione"], "o", color=COLORE_PREVISIONE,
            markersize=6, label="Previsione")

    if con_misure:
        dentro = prev[prev["dentro"]]
        fuori = prev[~prev["dentro"]]
        if len(dentro):
            ax.plot(dentro["bersaglio"], dentro["y_true"], "o", color=COLORE_STORICO,
                    markersize=6, label="Misurato (dentro la fascia)")
        if len(fuori):
            ax.plot(fuori["bersaglio"], fuori["y_true"], "o", color=COLORE_FUORI,
                    markersize=7, label="Misurato (fuori dalla fascia)")

    ax.set_xlim(*limiti["xlim"])
    ax.set_ylim(*limiti["ylim"])
    ax.set_ylabel("Umidità del suolo (%)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_etichetta_asse))
    for etichetta in ax.get_xticklabels():
        etichetta.set_rotation(30)
        etichetta.set_horizontalalignment("right")

    stile.titolo(ax, f"Previsione a 7 giorni emessa il {data_estesa(origine)} "
                     f"— sensore {SENSORE}")
    ax.legend(loc="upper right", fontsize=10)

    nome = "06_previsione_storico_misure.png" if con_misure else "06_previsione_storico.png"
    percorso = os.path.join(OUT, nome)
    fig.savefig(percorso)
    plt.close(fig)
    print(f"Salvato: {percorso}")


def main():
    os.makedirs(OUT, exist_ok=True)
    stile.applica()
    storico, prev, origine = prepara()

    # limiti calcolati UNA volta sull'unione di tutto ciò che può comparire in
    # una qualsiasi delle due versioni: è ciò che rende le immagini sovrapponibili
    valori = list(storico["umidita"]) + list(prev["basso"]) + list(prev["alto"]) \
        + list(prev["previsione"]) + list(prev["y_true"])
    margine = (max(valori) - min(valori)) * 0.08
    limiti = {
        "xlim": (storico["giorno"].min() - pd.Timedelta(days=1),
                 prev["bersaglio"].max() + pd.Timedelta(days=1)),
        "ylim": (min(valori) - margine, max(valori) + margine),
    }

    n_fuori = int((~prev["dentro"]).sum())
    print(f"  origine {origine.date()}, {len(storico)} giorni di storico, "
          f"{len(prev)} orizzonti previsti, {n_fuori} fuori fascia")

    for con_misure in (False, True):
        disegna(storico, prev, origine, con_misure, limiti)


if __name__ == "__main__":
    main()
