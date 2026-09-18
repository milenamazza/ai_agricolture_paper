# ─────────────────────────────────────────────────────────────────────────────
# COPIA di AI_agricolture/pipeline_corrette/calibrazione/run_calibrazione_da_mais.py.
# Script identico all'originale. Le differenze stanno tutte in mh_core.py (vedi
# il blocco "COPIA IN AI_agricolture_paper" nella sua docstring): registro
# irrigazioni nuovo, target del rollout cercato per data, curva Kc che parte
# dalla DATA SEMINA dove indicata. Dataset: quello di default di core/paths.py.
# ─────────────────────────────────────────────────────────────────────────────

"""
run_calibrazione_da_mais.py
─────────────────────────────────────────────────────────────────────────────
Calibrazione ANCORATA sul mais: si prende per buono il Kc di libro del mais
(FAO-56 Tab.11/12, la coltura meglio documentata delle sette) e lo si usa
come ancora per stimare tutto il resto.

Motivo: stimando una coltura sola, "Kc basso" e "gli altri parametri sono
starati e Kc li compensa" sono indistinguibili. Fissando il Kc di una coltura
di cui ci si fida, gli altri parametri non hanno piu' un colpevole con cui
confondersi.

Rispetto alla versione precedente in metropoli-hastings/, la calibrazione e'
in TRE stadi invece che due, e ogni stadio usa informazione che gli altri non
usano — cosi' non si stima tutto insieme lasciando che i parametri si
scambino di ruolo:

  1. mm_to_pct globale, dagli EVENTI DI PIOGGIA (regressione sul solo train).
     E' una proprieta' del suolo e della sonda, non della coltura: si stima
     su tutti i sensori insieme. E' anche l'unico punto in cui mm_to_pct e'
     identificabile senza Kc (nella pioggia compare da solo).
  2. Parametri globali di dominio (alpha_rain, alpha_irr, drain_coeff, fc,
     wp, sigma) sui SOLI dati di mais, con il Kc di mais fissato al libro e
     mm_to_pct fissato allo stadio 1.
  3. Kc assoluto delle altre colture, con gli stadi 1 e 2 fissati.

Output in risultati_da_mais/:
  stadio1_mm_to_pct.csv       stima di regressione + numero di eventi usati
  stadio2_dominio_mais.csv    posteriori dei parametri globali
  stadio3_kc_altre_colture.csv  Kc assoluto e scala per coltura
  metriche_1giorno.csv        (+ persistenza)
  metriche_rollout.csv        (+ persistenza)
  dominio_mais.png            istogrammi dello stadio 2

Utilizzo:
    python -u run_calibrazione_da_mais.py
    python -u run_calibrazione_da_mais.py --sensors-filter EM-500-9 EM-500-15 --n-burnin 500 --n-sample 2000
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mh_core as mh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "risultati_da_mais")
REFERENCE_CROP = "mais"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensors-filter", nargs="+", default=None)
    parser.add_argument("--rain-col", default=mh.OBSERVED_RAIN_COL)
    parser.add_argument("--n-burnin", type=int, default=5000)
    parser.add_argument("--n-sample", type=int, default=20000)
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=OUT_DIR)
    # Larghezze dei prior. I default riproducono mh_core.PRIOR_STD, quindi
    # senza passarle lo script si comporta esattamente come prima. Servono a
    # ripetere la calibrazione con prior quasi non informativi e misurare
    # quanto della posteriori sia informazione dei dati: sul Kc la riduzione
    # rispetto al prior qui e' 0.017, cioe' praticamente nulla.
    parser.add_argument("--kc-prior-std", type=float, default=mh.PRIOR_STD["kc"])
    parser.add_argument("--fc-prior-std", type=float, default=mh.PRIOR_STD["fc"])
    parser.add_argument("--p-prior-std", type=float, default=mh.PRIOR_STD["p_scale"])
    parser.add_argument("--mm-prior-std", type=float, default=None,
                        help="se assente usa 0.5 * la stima di regressione dello stadio 1")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    prior_std = {"kc": args.kc_prior_std, "fc": args.fc_prior_std,
                 "p_scale": args.p_prior_std}

    print(f"Costruisco i dataset (pioggia: {args.rain_col})...")
    df_trans, df_roll = mh.build_datasets(rain_col=args.rain_col, sensors_filter=args.sensors_filter)
    crops = mh.crops_in(df_trans)
    print(f"  transizioni: {len(df_trans)}   colture con dati: {crops}")
    if REFERENCE_CROP not in crops:
        raise RuntimeError(f"Nessun dato per la coltura di riferimento '{REFERENCE_CROP}'.")
    other_crops = [c for c in crops if c != REFERENCE_CROP]

    # ── STADIO 1 ────────────────────────────────────────────────────────────
    print("\n=== STADIO 1: mm_to_pct globale dagli eventi di pioggia ===")
    mm_hat, n_ev = mh.estimate_mm_to_pct(df_trans)
    if mm_hat is None:
        raise RuntimeError(f"Solo {n_ev} eventi di pioggia sul train: non bastano per lo stadio 1.")
    mm_std = args.mm_prior_std if args.mm_prior_std is not None else max(0.5 * abs(mm_hat), 0.02)
    print(f"  stima = {mm_hat:.4f} %/mm su {n_ev} eventi (train)   [valore di libro: 0.1]")
    print(f"  larghezze dei prior:  mm_to_pct {mm_std:.4f}   Kc scala {args.kc_prior_std}   "
          f"fc {args.fc_prior_std}   (default: {max(0.5*abs(mm_hat), 0.02):.4f} / "
          f"{mh.PRIOR_STD['kc']} / {mh.PRIOR_STD['fc']})")
    pd.DataFrame([{"parametro": "mm_to_pct", "stima_regressione": mm_hat,
                   "n_eventi_pioggia": n_ev, "prior_std_usato": mm_std}]
                 ).to_csv(os.path.join(args.out, "stadio1_mm_to_pct.csv"), index=False)

    # ── STADIO 2 ────────────────────────────────────────────────────────────
    print(f"\n=== STADIO 2: dominio globale sui dati di {REFERENCE_CROP} (Kc di libro fissato) ===")
    tr_mais = df_trans[(df_trans["coltura"] == REFERENCE_CROP) & (df_trans["split"] == "train")].reset_index(drop=True)
    print(f"  transizioni di {REFERENCE_CROP} nel train: {len(tr_mais)}")
    if len(tr_mais) < 20:
        raise RuntimeError(f"Troppe poche transizioni di {REFERENCE_CROP} per lo stadio 2.")

    init2 = mh.initial_params([REFERENCE_CROP], mm_init=mm_hat)
    fixed2 = {"mm_to_pct"} | {mh.kc_param(s, REFERENCE_CROP) for s in mh.KC_SUFFIXES}
    chain2, acc2 = mh.run_mh(tr_mais, [REFERENCE_CROP], mm_hat, mm_std,
                             n_burnin=args.n_burnin, n_sample=args.n_sample, thin=args.thin,
                             seed=args.seed, fixed=fixed2, init=init2, prior_std=prior_std)
    print("  accettazione: " + "  ".join(f"{k}={v:.3f}" for k, v in acc2.items()))
    sum2 = mh.summarize_posterior(chain2, [REFERENCE_CROP], mm_hat)
    print(sum2[sum2["parametro"].isin(mh.GLOBAL_NAMES)].to_string(index=False))
    sum2.to_csv(os.path.join(args.out, "stadio2_dominio_mais.csv"), index=False)

    domain = {n: float(chain2[n].mean()) for n in mh.GLOBAL_NAMES}
    domain["mm_to_pct"] = mm_hat          # fissato allo stadio 1, non ricalibrato qui

    # ── STADIO 3 ────────────────────────────────────────────────────────────
    print(f"\n=== STADIO 3: Kc delle altre colture ({other_crops}) con stadi 1-2 fissati ===")
    rows3 = []
    chain3 = None
    if other_crops:
        tr_other = df_trans[(df_trans["coltura"].isin(other_crops)) &
                            (df_trans["split"] == "train")].reset_index(drop=True)
        print(f"  transizioni delle altre colture nel train: {len(tr_other)}")
        if len(tr_other) >= 20:
            init3 = mh.initial_params(other_crops, mm_init=mm_hat)
            init3.update(domain)
            # Stadio 3: si stima SOLO il Kc delle altre colture. Tutto il
            # dominio resta congelato ai valori degli stadi 1-2, p_scale
            # compreso: e' il senso dell'ancoraggio sul mais, e lasciarlo
            # libero qui rimetterebbe in gioco proprio il parametro che il
            # mais doveva inchiodare. Solo sigma resta libero.
            fixed3 = set(n for n in mh.GLOBAL_NAMES if n != "sigma")
            assert "p_scale" in fixed3, "p_scale deve restare fisso nello stadio 3"
            print(f"  dominio congelato dagli stadi 1-2 (p_scale={domain['p_scale']:.3f}); "
                  f"liberi: Kc delle altre colture e sigma")
            chain3, acc3 = mh.run_mh(tr_other, other_crops, mm_hat, mm_std,
                                     n_burnin=args.n_burnin, n_sample=args.n_sample,
                                     thin=args.thin, seed=args.seed, fixed=fixed3, init=init3,
                                     prior_std=prior_std)
            print("  accettazione: " + "  ".join(f"{k}={v:.3f}" for k, v in acc3.items()))

            book = df_trans.drop_duplicates("coltura").set_index("coltura")
            for c in other_crops:
                for suffix, col in zip(mh.KC_SUFFIXES, ["kc_ini", "kc_mid", "kc_end"]):
                    v = chain3[mh.kc_param(suffix, c)].values
                    libro = float(book.at[c, col])
                    rows3.append({"coltura": c, "fase": col, "valore_libro": libro,
                                  "scala_media": float(v.mean()), "scala_std": float(v.std()),
                                  "scala_p5": float(np.percentile(v, 5)),
                                  "scala_p95": float(np.percentile(v, 95)),
                                  "kc_calibrato": float(libro * v.mean()),
                                  "prior_std_kc": args.kc_prior_std,
                                  # 1 - std_post/std_prior: vicino a 0 vuol dire che i dati
                                  # non hanno spostato il Kc dal prior, quindi kc_calibrato
                                  # e' il valore di libro che si rilegge, non una stima
                                  "riduzione_vs_prior": 1.0 - float(v.std()) / args.kc_prior_std})
            df3 = pd.DataFrame(rows3)
            print(df3.round(4).to_string(index=False))
            print(f"\n  riduzione media rispetto al prior: {df3['riduzione_vs_prior'].mean():.3f}"
                  "\n  (vicino a 0 = i dati non informano i Kc: il mais, l'unica coltura con"
                  "\n   un po' di segnale, e' escluso da questo stadio perche' fa da ancora)")
        else:
            print("  SALTATO: troppe poche transizioni.")
    df3 = pd.DataFrame(rows3)
    df3.to_csv(os.path.join(args.out, "stadio3_kc_altre_colture.csv"), index=False)

    # ── validazione con il set di parametri completo ────────────────────────
    print("\n=== Validazione del set completo (stadi 1+2+3) ===")
    final = dict(domain)
    for s in mh.KC_SUFFIXES:
        final[mh.kc_param(s, REFERENCE_CROP)] = 1.0        # mais: libro, per costruzione
    for c in other_crops:
        for s in mh.KC_SUFFIXES:
            name = mh.kc_param(s, c)
            final[name] = float(chain3[name].mean()) if chain3 is not None and name in chain3 else 1.0

    # catena "puntuale": un solo campione, i valori finali — l'incertezza dei
    # singoli stadi non si propaga, e va detto invece che finto propagata
    chain_final = pd.DataFrame([final])
    te = df_trans[df_trans["split"] == "test"].reset_index(drop=True)
    roll_te = df_roll[df_roll["split"] == "test"].reset_index(drop=True)

    m1 = mh.metrics_1day(chain_final, te, crops)
    print("\nMetriche a 1 giorno (test):")
    print(m1[["modello", "n", "MAE", "RMSE"]].to_string(index=False))
    m1.to_csv(os.path.join(args.out, "metriche_1giorno.csv"), index=False)

    if len(roll_te) >= 10:
        mr = mh.metrics_rollout(chain_final, roll_te, crops)
        print("\nMetriche sul rollout t+1..t+7 (test):")
        print(mr[["modello", "horizon", "n", "MAE", "RMSE"]].to_string(index=False))
    else:
        mr = pd.DataFrame()
        print("\nRollout: troppe poche righe complete.")
    mr.to_csv(os.path.join(args.out, "metriche_rollout.csv"), index=False)

    print("\nNota: con un solo set di valori puntuali l'incertezza NON e' propagata,"
          "\nquindi la parte epistemica delle metriche e' nulla per costruzione."
          "\nPer l'incertezza completa usare run_calibrazione.py (calibrazione congiunta).")

    # mm_to_pct e' fissato allo stadio 1, quindi non si grafica: restano i
    # parametri davvero campionati qui (p_scale incluso da quando e' calibrato)
    names2 = [n for n in mh.GLOBAL_NAMES if n != "mm_to_pct"]
    ncols = 4
    nrows = int(np.ceil(len(names2) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.ravel()[len(names2):]:
        ax.axis("off")
    for ax, name in zip(axes.ravel(), names2):
        ax.hist(chain2[name].values, bins=40, color="#0F766E", alpha=0.85)
        ax.axvline(mh.INIT_GLOBAL[name], color="black", linestyle="--", linewidth=1.4,
                   label="valore di partenza")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"Stadio 2 — dominio calibrato su {REFERENCE_CROP} (Kc di libro fissato)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "dominio_mais.png"))
    plt.close(fig)

    print(f"\nOutput in {args.out}")


if __name__ == "__main__":
    main()
