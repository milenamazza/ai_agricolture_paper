"""
03_errore_incertezza.py
─────────────────────────────────────────────────────────────────────────────
Errore medio per orizzonte, Deep Ensemble e Quantile Regression (un PNG per
gruppo), ciascuno nelle due varianti — rinominate "Data driven" e "Residual
error" (stessi nomi del grafico 01), SENZA il feature-set in etichetta.
Penman-Monteith come riferimento, ricalcolato sulle stesse righe.

Utilizzo:
    python -u 03_errore_incertezza.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dati
import stile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "risultati")


def disegna(chiave, gruppo):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    d0 = gruppo["famiglie"][0]["dati"]
    yp = [(d0[d0["horizon"] == h]["y_pred"] - d0[d0["horizon"] == h]["y_true"]).abs().mean()
          for h in dati.ORIZZONTI]
    ax.plot(stile.X, yp, "o:", color=stile.COLORE_PENMAN, linewidth=1.6, markersize=5,
            label=stile.FISICA)

    for f in gruppo["famiglie"]:
        y = dati.serie(f["dati"], "errore")
        ax.plot(stile.X, y, "o-", color=stile.COLORE_VARIANTE[f["variante"]],
                linewidth=1.8, markersize=6, label=stile.nome_variante(f["variante"]))

    stile.assi_orizzonte(ax, "MAE (punti di umidità)")
    ax.set_ylim(0, None)
    stile.titolo(ax, f"{gruppo['nome']} — errore medio per orizzonte")
    ax.legend(loc="upper left", fontsize=10.5)

    fig.tight_layout()
    percorso = os.path.join(OUT, f"03_errore_{chiave}.png")
    fig.savefig(percorso)
    plt.close(fig)
    print(f"Salvato: {percorso}")


def main():
    os.makedirs(OUT, exist_ok=True)
    stile.applica()
    gruppi = dati.carica_gruppi()
    for chiave, gruppo in gruppi.items():
        disegna(chiave, gruppo)


if __name__ == "__main__":
    main()
