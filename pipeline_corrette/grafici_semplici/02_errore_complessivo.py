"""
02_errore_complessivo.py
─────────────────────────────────────────────────────────────────────────────
Errore medio complessivo (Complessivo 7gg) dei tre approcci, a BARRE
VERTICALI (a differenza della versione precedente, che era orizzontale).

Utilizzo:
    python -u 02_errore_complessivo.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dati
import stile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "risultati")

ORDINE = [stile.FISICA, stile.DATA_DRIVEN, stile.RESIDUAL]


def main():
    os.makedirs(OUT, exist_ok=True)
    stile.applica()
    filtrato = dati.carica_filtrato()

    c = filtrato[filtrato["orizzonte"] == "Complessivo (7gg)"].copy()
    c = c.set_index("approccio").reindex(ORDINE).reset_index()

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(c))
    ax.bar(x, c["errore"], width=0.55, color=[stile.COLORE_APPROCCIO[a] for a in c["approccio"]],
           edgecolor="#222222", linewidth=0.9)
    for i, r in enumerate(c.itertuples()):
        ax.text(i, r.errore + 0.06, f"{r.errore:.2f}", ha="center",
                fontweight="bold", color=stile.COLORE_APPROCCIO[r.approccio])

    ax.set_xticks(x)
    ax.set_xticklabels(c["approccio"], fontsize=10)
    ax.tick_params(axis="x", pad=10)
    ax.set_ylabel("MAE su 7 orizzonti (punti di umidità)", fontsize=11)
    ax.set_ylim(0, c["errore"].max() * 1.18)
    ax.grid(axis="x", visible=False)
    stile.titolo(ax, "Errore medio complessivo")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_errore_complessivo.png"))
    plt.close(fig)
    print(f"Salvato: {os.path.join(OUT, '02_errore_complessivo.png')}")


if __name__ == "__main__":
    main()
