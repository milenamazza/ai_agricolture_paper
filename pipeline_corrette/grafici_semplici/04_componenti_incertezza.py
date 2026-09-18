"""
04_componenti_incertezza.py
─────────────────────────────────────────────────────────────────────────────
Deep Ensemble: componenti epistemica e aleatoria dell'incertezza, per
orizzonte. Solo Deep Ensemble, perché la Quantile Regression non separa le
due componenti (predice direttamente i percentili).

La legenda è UNICA e condivisa sotto i due pannelli (fig.legend), non dentro
il primo asse: dentro copriva le linee nel pannello "Epistemica", dove le due
varianti partono quasi sovrapposte a t+1.

Utilizzo:
    python -u 04_componenti_incertezza.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dati
import stile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "risultati")


def main():
    os.makedirs(OUT, exist_ok=True)
    stile.applica()
    gruppo = dati.carica_gruppi()["deep_ensemble"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.5))
    for f in gruppo["famiglie"]:
        colore = stile.COLORE_VARIANTE[f["variante"]]
        etichetta = stile.nome_variante(f["variante"])
        a1.plot(stile.X, dati.serie(f["dati"], "epistemica"), "o-", color=colore,
                linewidth=1.8, markersize=6, label=etichetta)
        a2.plot(stile.X, dati.serie(f["dati"], "aleatoria"), "o-", color=colore,
                linewidth=1.8, markersize=6, label=etichetta)

    for ax, tit, yl in ((a1, "Epistemica (incertezza del modello)", "σ epistemica"),
                        (a2, "Aleatoria (rumore nei dati)", "σ aleatoria")):
        stile.assi_orizzonte(ax, yl)
        ax.set_ylim(0, None)
        stile.titolo(ax, tit)

    # legenda unica, sotto i due pannelli: dentro l'asse coprirebbe le linee
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{gruppo['nome']} — componenti dell'incertezza per orizzonte",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])

    percorso = os.path.join(OUT, "04_componenti_deep_ensemble.png")
    fig.savefig(percorso)
    plt.close(fig)
    print(f"Salvato: {percorso}")


if __name__ == "__main__":
    main()
