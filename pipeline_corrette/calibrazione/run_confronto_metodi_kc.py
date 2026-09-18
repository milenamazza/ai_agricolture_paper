# ─────────────────────────────────────────────────────────────────────────────
# COPIA di AI_agricolture/pipeline_corrette/calibrazione/run_confronto_metodi_kc.py.
# Script identico all'originale. Le differenze stanno tutte in mh_core.py (vedi
# il blocco "COPIA IN AI_agricolture_paper" nella sua docstring): registro
# irrigazioni nuovo, target del rollout cercato per data, curva Kc che parte
# dalla DATA SEMINA dove indicata. Dataset: quello di default di core/paths.py.
# ─────────────────────────────────────────────────────────────────────────────

"""
run_confronto_metodi_kc.py
─────────────────────────────────────────────────────────────────────────────
Stima le sole scale Kc (kc_ini, kc_mid, kc_end) con metodi indipendenti,
tenendo FISSI tutti gli altri parametri, e confronta i risultati: se metodi
con meccanismi diversi convergono sullo stesso valore, non e' un artefatto
del campionatore.

Tre metodi:
  1. Metropolis-Hastings — bayesiano, con prior N(1, 0.3) sulle scale.
  2. Regressione NNLS (minimi quadrati NON NEGATIVI). Con gli altri parametri
     fissi il bilancio idrico e' LINEARE nelle 3 scale, perche' Kc_eff e' una
     combinazione dei 3 nodi con pesi noti (interpolazione FAO-56):
         ETc_implicita = m_prev + pioggia + irrigazione - percolazione - m_curr
                       = X1*scala_ini + X2*scala_mid + X3*scala_end
     Nella versione precedente questa era una OLS libera, e dava scale
     NEGATIVE (-1.8, -1.25): non e' un risultato, e' un metodo mal posto,
     perche' Kc >= 0 e' un vincolo fisico. Qui si usa scipy.optimize.nnls,
     che lo impone.
  3. Bootstrap sulla NNLS — ricampiona le righe di train con reinserimento e
     rifa' la stima: distribuzione frequentista, meccanismo completamente
     diverso da MCMC, utile per controllare lo spread della posteriori.

In tabella ci sono sempre il VALORE DI LIBRO e la PERSISTENZA come
riferimenti: senza, una stima puo' sembrare buona ed essere inutile.

Output in risultati_confronto_kc/:
  confronto_kc.csv        coltura x fase x metodo: SCALE (scala_*) e valori ASSOLUTI (kc_*)
  kc_assoluti.csv         tabella compatta dei soli Kc assoluti, un metodo per colonna
  metriche_metodi.csv     MAE a 1 giorno e sul rollout per ciascun metodo (+ persistenza)
  confronto_kc.png        istogrammi MH e Bootstrap, punto NNLS, valore di libro

Utilizzo:
    python -u run_confronto_metodi_kc.py
    python -u run_confronto_metodi_kc.py --crops melissa --n-burnin 500 --n-sample 2000
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

import mh_core as mh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "risultati_confronto_kc")
METHOD_MH = "Metropolis-Hastings"
METHOD_NNLS = "NNLS (regressione vincolata)"
METHOD_BOOT = "Bootstrap (NNLS)"


def build_design(df, domain):
    """Matrice di disegno lineare nelle 3 scale Kc, con gli altri parametri
    fissi. Nessuna divisione: X e y restano sulla scala dell'umidita'."""
    fc, wp, mm = domain["fc"], domain["wp"], domain["mm_to_pct"]
    ks = mh._water_stress_vec(df["m_prev"].values, fc, wp,
                              mh._p_efficace(domain, df["p_coltura"].values))
    base = ks * df["et0"].values * mm
    X = np.column_stack([
        base * df["w_ini"].values * df["kc_ini"].values,
        base * df["w_mid"].values * df["kc_mid"].values,
        base * df["w_end"].values * df["kc_end"].values,
    ])
    rain_pct = domain["alpha_rain"] * df["rain_mm"].values * mm
    irr_pct = domain["alpha_irr"] * df["irr_mm"].values * mm
    perc_pct = mh._percolation_vec(df["m_prev"].values, fc, domain["drain_coeff"])
    y = df["m_prev"].values + rain_pct + irr_pct - perc_pct - df["m_curr"].values
    return X, y


def fit_nnls(X, y):
    coef, _ = nnls(X, y)
    return coef


def run_bootstrap(X, y, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.zeros((n_boot, 3))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = fit_nnls(X[idx], y[idx])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensors-filter", nargs="+", default=None)
    parser.add_argument("--crops", nargs="+", default=None,
                        help="colture da stimare (default: tutte quelle con dati)")
    parser.add_argument("--rain-col", default=mh.OBSERVED_RAIN_COL)
    parser.add_argument("--n-burnin", type=int, default=5000)
    parser.add_argument("--n-sample", type=int, default=20000)
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Costruisco i dataset (pioggia: {args.rain_col})...")
    df_trans, df_roll = mh.build_datasets(rain_col=args.rain_col, sensors_filter=args.sensors_filter)
    all_crops = mh.crops_in(df_trans)
    crops = [c for c in (args.crops or all_crops) if c in all_crops]
    if not crops:
        raise RuntimeError(f"Nessuna delle colture richieste ha dati. Disponibili: {all_crops}")
    print(f"  colture da stimare: {crops}   (disponibili: {all_crops})")

    mm_hat, n_ev = mh.estimate_mm_to_pct(df_trans)
    domain = dict(mh.INIT_GLOBAL)
    if mm_hat is not None:
        domain["mm_to_pct"] = mm_hat
        print(f"  mm_to_pct fissato alla stima di regressione: {mm_hat:.4f} ({n_ev} eventi)")
    else:
        print(f"  mm_to_pct fissato al default {domain['mm_to_pct']} (solo {n_ev} eventi di pioggia)")
    print("  altri parametri fissi: " +
          ", ".join(f"{k}={domain[k]}" for k in ("alpha_rain", "alpha_irr", "drain_coeff",
                                               "fc", "wp", "p_scale")))

    rows, metric_rows, plot_data = [], [], {}
    book = df_trans.drop_duplicates("coltura").set_index("coltura")

    for crop in crops:
        d = df_trans[df_trans["coltura"] == crop]
        tr = d[d["split"] == "train"].reset_index(drop=True)
        te = d[d["split"] == "test"].reset_index(drop=True)
        print(f"\n--- {crop} --- train={len(tr)}  test={len(te)}")
        if len(tr) < 20 or len(te) < 10:
            print("  SALTATA: dati insufficienti.")
            continue

        # 1. Metropolis-Hastings (solo il blocco Kc libero)
        init = mh.initial_params([crop], mm_init=domain["mm_to_pct"])
        init.update(domain)
        # Qui si stima SOLO il Kc: tutto il dominio resta fissato, p_scale
        # compreso. Calibrare anche p renderebbe il confronto fra i tre metodi
        # senza senso — MH avrebbe un parametro in piu' di NNLS e del bootstrap,
        # che lavorano su un disegno lineare a p fisso. p_scale resta a 1.0,
        # cioe' ai valori di libro di colture.csv, gli stessi che usa NNLS.
        fixed = set(n for n in mh.GLOBAL_NAMES if n != "sigma")
        assert "p_scale" in fixed, "p_scale deve restare fisso negli script che stimano solo Kc"
        chain, acc = mh.run_mh(tr, [crop], domain["mm_to_pct"], max(0.5 * domain["mm_to_pct"], 0.02),
                               n_burnin=args.n_burnin, n_sample=args.n_sample, thin=args.thin,
                               seed=args.seed, fixed=fixed, init=init)
        print("  MH accettazione: " + "  ".join(f"{k}={v:.3f}" for k, v in acc.items()))

        # 2. NNLS  3. Bootstrap
        X, y = build_design(tr, domain)
        coef = fit_nnls(X, y)
        boot = run_bootstrap(X, y, args.n_bootstrap, args.seed)
        print(f"  NNLS: ini={coef[0]:.3f} mid={coef[1]:.3f} end={coef[2]:.3f}")

        for i, (suffix, col) in enumerate(zip(mh.KC_SUFFIXES, ["kc_ini", "kc_mid", "kc_end"])):
            libro = float(book.at[crop, col])
            v = chain[mh.kc_param(suffix, crop)].values
            common = {"coltura": crop, "fase": col, "valore_libro": libro}
            # colonne "scala_*" = moltiplicatore sul libro; "kc_*" = valore ASSOLUTO
            rows.append({**common, "metodo": METHOD_MH, "scala_puntuale": np.nan,
                         "scala_media": float(v.mean()), "scala_std": float(v.std()),
                         "scala_p5": float(np.percentile(v, 5)), "scala_p95": float(np.percentile(v, 95)),
                         "kc_assoluto": float(libro * v.mean()),
                         "kc_p5": float(libro * np.percentile(v, 5)),
                         "kc_p95": float(libro * np.percentile(v, 95))})
            rows.append({**common, "metodo": METHOD_NNLS, "scala_puntuale": float(coef[i]),
                         "scala_media": np.nan, "scala_std": np.nan,
                         "scala_p5": np.nan, "scala_p95": np.nan,
                         "kc_assoluto": float(libro * coef[i]), "kc_p5": np.nan, "kc_p95": np.nan})
            b = boot[:, i]
            rows.append({**common, "metodo": METHOD_BOOT, "scala_puntuale": np.nan,
                         "scala_media": float(b.mean()), "scala_std": float(b.std()),
                         "scala_p5": float(np.percentile(b, 5)), "scala_p95": float(np.percentile(b, 95)),
                         "kc_assoluto": float(libro * b.mean()),
                         "kc_p5": float(libro * np.percentile(b, 5)),
                         "kc_p95": float(libro * np.percentile(b, 95))})

        # metriche a 1 giorno per ciascun metodo, piu' libro e persistenza
        def point_chain(scales):
            p = dict(domain)
            p["sigma"] = float(chain["sigma"].mean())
            for suffix, s in zip(mh.KC_SUFFIXES, scales):
                p[mh.kc_param(suffix, crop)] = float(s)
            return pd.DataFrame([p])

        variants = {
            METHOD_MH: [chain[mh.kc_param(s, crop)].mean() for s in mh.KC_SUFFIXES],
            METHOD_NNLS: list(coef),
            METHOD_BOOT: list(boot.mean(axis=0)),
            "valore di libro (scala 1.0)": [1.0, 1.0, 1.0],
        }
        for label, scales in variants.items():
            mu = mh.predict_mu(point_chain(scales).iloc[0].to_dict(), te, [crop])
            metric_rows.append({"coltura": crop, "metodo": label, "n": len(te),
                                "MAE_1giorno": float(np.mean(np.abs(te["m_curr"].values - mu)))})
        metric_rows.append({"coltura": crop, "metodo": "persistenza", "n": len(te),
                            "MAE_1giorno": float(np.mean(np.abs(te["m_curr"].values - te["m_prev"].values)))})

        plot_data[crop] = {"chain": chain, "coef": coef, "boot": boot,
                           "book": {c: float(book.at[crop, c]) for c in ["kc_ini", "kc_mid", "kc_end"]}}

    if not rows:
        print("\nNessuna coltura con dati sufficienti.")
        return

    df_out = pd.DataFrame(rows)
    df_met = pd.DataFrame(metric_rows)
    df_out.to_csv(os.path.join(args.out, "confronto_kc.csv"), index=False)
    df_met.to_csv(os.path.join(args.out, "metriche_metodi.csv"), index=False)

    # tabella compatta dei VALORI ASSOLUTI, un metodo per colonna
    pivot = df_out.pivot_table(index=["coltura", "fase", "valore_libro"], columns="metodo",
                               values="kc_assoluto", aggfunc="first").reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"valore_libro": "libro", METHOD_MH: "MH",
                                  METHOD_NNLS: "NNLS", METHOD_BOOT: "Bootstrap"})
    pivot.to_csv(os.path.join(args.out, "kc_assoluti.csv"), index=False)

    print("\n=== Scale Kc (moltiplicatore sul valore di libro) ===")
    print(df_out[["coltura", "fase", "valore_libro", "metodo", "scala_puntuale", "scala_media", "scala_std"]]
          .to_string(index=False))
    print("\n=== Kc in valore ASSOLUTO (libro x scala), per metodo ===")
    print(pivot.round(3).to_string(index=False))
    print("\n=== MAE a 1 giorno per metodo (test) ===")
    print(df_met.to_string(index=False))

    ncrops = len(plot_data)
    fig, axes = plt.subplots(ncrops, 3, figsize=(15, 4 * ncrops), squeeze=False)
    for r, (crop, d) in enumerate(plot_data.items()):
        for c, (suffix, col) in enumerate(zip(mh.KC_SUFFIXES, ["kc_ini", "kc_mid", "kc_end"])):
            ax = axes[r, c]
            ax.hist(d["chain"][mh.kc_param(suffix, crop)].values, bins=30, density=True,
                    color="#1D4ED8", alpha=0.5, label="MH")
            ax.hist(d["boot"][:, c], bins=30, density=True, color="#059669", alpha=0.5, label="Bootstrap")
            ax.axvline(d["coef"][c], color="#B91C1C", linewidth=2, label="NNLS")
            ax.axvline(1.0, color="black", linestyle="--", linewidth=1.4, label="libro (scala 1.0)")
            if r == 0:
                ax.set_title(col, fontsize=11)
            if c == 0:
                ax.set_ylabel(crop, fontsize=11)
            if r == 0 and c == 2:
                ax.legend(fontsize=7)
    fig.suptitle("Scale Kc: confronto tra metodi (altri parametri fissi)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "confronto_kc.png"))
    plt.close(fig)
    print(f"\nOutput in {args.out}")


if __name__ == "__main__":
    main()
