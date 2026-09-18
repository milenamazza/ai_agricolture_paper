"""
05_serie_temporale.py
─────────────────────────────────────────────────────────────────────────────
Andamento nel tempo su un sensore reale: i primi GIORNI_STORICO giorni del
periodo di test mostrano il solo andamento VERO dell'umidità (nessuna
previsione disegnata, anche se esiste — è contesto, non il punto del
grafico). Una riga verticale segna dove comincia la parte di previsione; da
lì in poi si vede previsione e intervallo al 90%.

La linea dello storico e quella della previsione sono UNITE in un unico
tratto continuo: l'ultimo punto vero fa anche da primo punto della linea di
previsione, così le due si toccano senza salto — anche se sono due
grandezze diverse (misurato prima, previsto dopo), non due serie continue
della stessa cosa.

Perché i 30 giorni sono presi DENTRO il periodo di test e non nei giorni
subito precedenti: train e test sono separati da un buco di ~7 mesi (il
train finisce a ottobre, il test comincia a maggio dell'anno dopo), quindi
non esiste un mese continuo di dati veri appena prima del test. Il periodo
di test invece è continuo (37 giorni su ~54 di calendario per EM-500-12), e
tagliarlo a metà dà 22 giorni di "storico" e 15 di "previsioni" — nessun
salto temporale nel grafico.

Due versioni per gruppo (Deep Ensemble, Quantile Regression), come nello
script 06:
  05_serie_temporale_{gruppo}.png                solo storico + previsione
  05_serie_temporale_{gruppo}_con_misurato.png    + i valori VERI nella zona
                                                  di previsione (puntini neri
                                                  se dentro la fascia al 90%,
                                                  rossi se fuori)

Utilizzo:
    python -u 05_serie_temporale.py
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

SENSORE_ESEMPIO = "EM-500-12"
GIORNI_STORICO = 18   # quanti giorni dall'inizio del test mostrare come solo storico

COLORE_STORICO = "#17222B"      # puntini "misurato" nella zona di previsione
COLORE_ANDAMENTO = "#2E6F8E"    # linea dell'andamento vero, prima della riga
COLORE_FUORI = "#C1483C"


def disegna(chiave, gruppo, con_misurato):
    f = min(gruppo["famiglie"], key=lambda f: f["dati"]["errore"].mean())
    d = f["dati"]
    d = d[(d["sensor"] == SENSORE_ESEMPIO) & (d["horizon"] == "t+1")].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    if len(d) < 10:
        print(f"  saltato {chiave}: solo {len(d)} giorni per {SENSORE_ESEMPIO}")
        return

    cutoff = d["date"].min() + pd.Timedelta(days=GIORNI_STORICO)
    storico = d[d["date"] < cutoff]
    previsioni = d[d["date"] >= cutoff]
    if previsioni.empty:
        print(f"  saltato {chiave}: nessun giorno dopo il cutoff del {cutoff.date()}")
        return

    colore = stile.COLORE_VARIANTE[f["variante"]]
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # andamento vero, fino alla riga: solo linea, nessun puntino
    ax.plot(storico["date"], storico["y_true"], "-", color=COLORE_ANDAMENTO,
            linewidth=1.8, label="Umidità misurata")

    # previsione: si aggancia all'ultimo punto vero, così le due linee si
    # toccano senza salto — anche se sono due grandezze diverse (misurato
    # prima, previsto dopo), non un'unica serie continua
    ultimo_giorno = storico["date"].iloc[-1]
    ultimo_valore = storico["y_true"].iloc[-1]
    ponte_x = [ultimo_giorno] + list(previsioni["date"])
    ponte_y = [ultimo_valore] + list(previsioni["previsione"])

    ax.fill_between(previsioni["date"], previsioni["basso"], previsioni["alto"],
                    color=colore, alpha=0.18, label="Intervallo al 90%")
    ax.plot(ponte_x, ponte_y, "-", color=colore, linewidth=1.8, label="Previsione")

    if con_misurato:
        # valori VERI nella zona di previsione: solo puntini, dentro/fuori la
        # fascia, nessuna linea a collegarli
        dentro = previsioni[(previsioni["y_true"] >= previsioni["basso"])
                            & (previsioni["y_true"] <= previsioni["alto"])]
        fuori = previsioni[(previsioni["y_true"] < previsioni["basso"])
                           | (previsioni["y_true"] > previsioni["alto"])]
        if len(dentro):
            ax.plot(dentro["date"], dentro["y_true"], "o", color=COLORE_STORICO,
                    markersize=6, label="Misurato (dentro la fascia)")
        if len(fuori):
            ax.plot(fuori["date"], fuori["y_true"], "o", color=COLORE_FUORI, markersize=7,
                    label="Misurato (fuori dalla fascia)")

    ax.axvline(cutoff, color="#555555", linestyle="--", linewidth=1.4)
    # etichetta ancorata all'asse (non ai dati): resta in alto qualunque sia l'ylim
    ax.text(cutoff, 0.97, " Previsioni", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=11, fontweight="bold", color="#555555")

    ax.set_ylabel("Umidità del suolo (%)")
    stile.titolo(ax, f"{gruppo['nome']} — {stile.nome_variante(f['variante'])}, "
                     f"t+1, sensore {SENSORE_ESEMPIO}")
    ax.legend(loc="upper right", fontsize=10)
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    suffisso = "_con_misurato" if con_misurato else ""
    percorso = os.path.join(OUT, f"05_serie_temporale_{chiave}{suffisso}.png")
    fig.savefig(percorso)
    plt.close(fig)
    print(f"Salvato: {percorso}  (storico: {len(d) - len(previsioni)} giorni, "
          f"previsioni: {len(previsioni)} giorni)")


def main():
    os.makedirs(OUT, exist_ok=True)
    stile.applica()
    gruppi = dati.carica_gruppi()
    for chiave, gruppo in gruppi.items():
        for con_misurato in (False, True):
            disegna(chiave, gruppo, con_misurato)


if __name__ == "__main__":
    main()
