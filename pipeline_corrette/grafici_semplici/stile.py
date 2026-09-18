"""
stile.py — stile accademico condiviso dai 5 script di grafici
─────────────────────────────────────────────────────────────────────────────
Un solo posto per: nomi coerenti degli approcci (usati identici in tutti gli
script), i colori (invariati rispetto alla versione precedente — la richiesta
riguardava tipografia/legenda/griglia, non la palette), e i parametri
matplotlib che danno l'aspetto da figura di paper invece che da slide
divulgativa: riquadro completo su tutti e 4 i lati, griglia tratteggiata
sottile, titolo in grassetto centrato, legenda con cornice.

Importato da ogni script (`import stile`), MAI modificato da uno script per
gli altri: se un grafico ha bisogno di uno scostamento dallo stile comune,
lo fa in loco dopo aver chiamato applica().
"""

import matplotlib.pyplot as plt

# ─── nomi coerenti in tutti i grafici ────────────────────────────────────────
FISICA = "Fisica (Penman-Monteith)"
DATA_DRIVEN = "Data driven"
RESIDUAL = "Residual error"

COLORE_APPROCCIO = {FISICA: "#0F766E", DATA_DRIVEN: "#B45309", RESIDUAL: "#1D4ED8"}
COLORE_VARIANTE = {"diretta": "#B45309", "residuo": "#1D4ED8"}
COLORE_PENMAN = "#6B7280"

ORIZZONTI = [f"t+{h}" for h in range(1, 8)]
X = list(range(1, 8))
Z90 = 1.645


def nome_variante(variante):
    """'diretta' -> Data driven, 'residuo' -> Residual error — stessi nomi
    del confronto filtrato, senza feature-set in etichetta."""
    return DATA_DRIVEN if variante == "diretta" else RESIDUAL


def applica():
    """Da chiamare una volta, all'inizio di ogni script."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150,
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.linestyle": ":", "grid.linewidth": 0.7, "grid.alpha": 0.55,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.edgecolor": "#222222", "axes.linewidth": 0.9,
        "legend.frameon": True, "legend.edgecolor": "#222222",
        "legend.framealpha": 1.0, "legend.fancybox": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "xtick.direction": "in", "ytick.direction": "in",
    })


def titolo(ax, testo):
    """Titolo centrato, in grassetto — stile figura di paper."""
    ax.set_title(testo, loc="center", pad=12)


def assi_orizzonte(ax, ylabel):
    ax.set_xticks(X)
    ax.set_xticklabels(ORIZZONTI)
    ax.set_xlabel("Orizzonte di previsione")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.8, 7.2)
