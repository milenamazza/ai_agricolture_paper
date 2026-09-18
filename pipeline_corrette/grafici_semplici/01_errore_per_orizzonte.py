"""
01_errore_per_orizzonte.py
─────────────────────────────────────────────────────────────────────────────
Errore medio di previsione per orizzonte, tre approcci: Fisica
(Penman-Monteith), Data driven (RF, TERRA_METEO), Residual error
(MLP, ACQUA) — dal confronto filtrato (confronto_filtrato_optuna.xlsx),
Ridge escluso.

Utilizzo:
    python -u 01_errore_per_orizzonte.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dati
import stile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "risultati")

ORDINE = [stile.FISICA, stile.DATA_DRIVEN, stile.RESIDUAL]


def main():
    os.makedirs(OUT, exist_ok=True)
    stile.applica()
    filtrato = dati.carica_filtrato()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for approccio in ORDINE:
        g = filtrato[(filtrato["approccio"] == approccio)
                     & (filtrato["orizzonte"].isin(dati.ORIZZONTI))]
        g = g.set_index("orizzonte").reindex(dati.ORIZZONTI)
        ax.plot(stile.X, g["errore"].to_numpy(dtype=float), "o-",
                color=stile.COLORE_APPROCCIO[approccio], linewidth=1.8, markersize=6,
                label=approccio)

    stile.assi_orizzonte(ax, "Errore medio umidità (%)")
    ax.set_ylim(0, None)
    stile.titolo(ax, "Errore medio di previsione per orizzonte")
    ax.legend(loc="upper left", fontsize=10.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_errore_per_orizzonte.png"))
    plt.close(fig)
    print(f"Salvato: {os.path.join(OUT, '01_errore_per_orizzonte.png')}")


if __name__ == "__main__":
    main()
