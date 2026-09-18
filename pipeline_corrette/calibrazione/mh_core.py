"""
mh_core.py
─────────────────────────────────────────────────────────────────────────────
Motore condiviso dei 3 script di calibrazione bayesiana di questa cartella.
Riscrittura di metropoli-hastings/mh_utils.py alla luce di quello che le
prove successive hanno mostrato:

  1. mm_to_pct era tenuto FISSO a 0.3 in tutte le vecchie calibrazioni, ed e'
     invece il parametro con lo scarto piu' grande dai dati (risposta misurata
     0.03-0.21 %/mm): qui e' un parametro LIBERO, globale, con prior centrato
     sulla stima di regressione dagli eventi di pioggia del solo train.
     Dentro ETc, mm_to_pct e Kc sono collineari (compaiono solo come
     prodotto); si separano perche' mm_to_pct moltiplica ANCHE la pioggia e
     l'irrigazione, dove Kc non c'e'. E' li' che il prior di regressione
     prende l'informazione.
  2. Nessuna vecchia calibrazione si confrontava con la PERSISTENZA ("l'umidita'
     resta com'e'"), e su questi dati e' un baseline forte: con i valori di
     libro Penman era 2x peggio del non fare nulla. Qui la persistenza e' una
     riga fissa in ogni tabella di metriche.

Verosimiglianza sulle transizioni a 1 giorno (t -> t+1). Il rollout t+1..t+7
si calcola SOLO in validazione, per vedere se parametri buoni a 1 giorno
reggono anche sull'uso reale a 7 giorni.

Parametri calibrati:
  globali (7)      mm_to_pct, alpha_rain, alpha_irr, drain_coeff, fc, wp, sigma
  per coltura (3)  kc_ini_scale, kc_mid_scale, kc_end_scale — solo per le
                   colture che hanno davvero dati (sorgo/malva/escolzia non
                   hanno nessun sensore utilizzabile). Sono SCALE sopra i
                   valori di libro, cosi' le differenze relative tra colture
                   di colture.csv restano.

Campionatore: Metropolis-within-Gibbs a blocchi (blocco globale + blocco
per-coltura, aggiornati in sequenza a ogni iterazione, ognuno con la propria
scala adattiva durante il burn-in). Una proposta congiunta su ~19 parametri
farebbe crollare il tasso di accettazione.

Fisica identica a pipeline_corrette/core (stesse formule di water_stress,
percolation e della curva Kc FAO-56), qui riscritta in forma vettorizzata
perche' deve essere parametrica sui valori proposti a ogni iterazione.

COPIA IN AI_agricolture_paper — differenze dall'originale
─────────────────────────────────────────────────────────────────────────────
  1. Registro irrigazioni: il percorso arriva da
     paths.DEFAULT_IRRIGAZIONE_RICOSTRUITA (nome in uso in questa cartella), che
     punta al registro "effettive" (71 eventi, tutti da log; nessuno inferito
     dai salti di umidita'). _load_context chiama anche
     paths.require_data_files(): con un percorso sbagliato l'irrigazione
     verrebbe letta vuota SENZA errore, e alpha_irr sarebbe stimato su
     irrigazioni tutte nulle.
     L'allineamento temporale dell'irrigazione era gia' corretto e non e'
     cambiato: la transizione t->t+1 usa l'irrigazione del giorno di arrivo e
     esiste solo fra giorni consecutivi; il rollout la cerca per data.

  2. Target del rollout PER DATA: y_h e' l'umidita' misurata il giorno D+h,
     cercata per data nel periodo di installazione. L'originale leggeva
     target_t{h}, uno shift posizionale: dove la serie ha buchi confrontava la
     previsione per D+h con la misura di un altro giorno. Se il giorno D+h non
     c'e', la riga di rollout non e' completa e si scarta. Tocca solo le
     metriche di validazione a 7 giorni, non la calibrazione.

  3. Origine della curva Kc: day_index si misura da
     data_utils.origine_curva_kc(srow), cioe' dalla DATA SEMINA dove indicata
     in sensori-corretti.csv e dall'installazione altrimenti. I giorni prima
     della semina sono esclusi (senza coltura in campo il modello colturale
     non si applica: stessa regola di penman.py). L'installazione continua a
     delimitare il periodo in cui il sensore misura.

  Non cambiato, da sapere: dopo la fine del ciclo colturale _kc_weights_vec
  tiene Kc fermo a kc_end, mentre penman.py smette di prevedere. E' una
  differenza preesistente fra calibrazione e Penman; allinearla toglierebbe
  molti giorni alle colture con periodi lunghi (melissa).
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "..", "core")
sys.path.insert(0, CORE)

import paths  # noqa: E402
from data_utils import (  # noqa: E402
    HORIZONS, SAT_OVERRIDE,
    load_colture, load_sensori, apply_soil_overrides, get_coltura_params,
    origine_curva_kc,
)
from dataset_builder import (  # noqa: E402
    build_all_datasets_with_reconstructed_irrigation,
    build_canonical_split,
)
import eval_uncertainty as eu  # noqa: E402

ET0_COL = "fc1_et0_fao_evapotranspiration"      # corretta in core/data_utils.py
OBSERVED_RAIN_COL = "ws_rainfall_total"
TARGET_COLS = [f"target_t{h}" for h in HORIZONS]

GLOBAL_NAMES = ["mm_to_pct", "alpha_rain", "alpha_irr", "drain_coeff", "fc", "wp",
                "p_scale", "sigma"]
KC_SUFFIXES = ["kc_ini_scale", "kc_mid_scale", "kc_end_scale"]

# p_scale: scala globale sulla frazione di deplezione p (FAO-56), che governa
# a quale umidita' Ks comincia a scendere, cioe' dove il modello passa da
# "evapotraspira liberamente" a "evapotraspira meno perche' e' in stress".
# Era l'unico parametro della fisica mai calibrato: veniva letto da
# colture.csv e tenuto fisso.
#
# Si stima come SCALA e non come valore assoluto perche' p in colture.csv
# varia per coltura (mais 0.55, carciofo e passiflora 0.45, melissa 0.40):
# una scala globale preserva quelle differenze FAO-56 e stima UN numero ben
# identificato invece di quattro deboli — stessa logica gia' usata per i Kc.
# Il p efficace viene comunque limitato a [0.01, 0.99]: e' una frazione.

INIT_GLOBAL = {"mm_to_pct": 0.10, "alpha_rain": 0.50, "alpha_irr": 0.70,
               "drain_coeff": 0.005, "fc": 45.0, "wp": 18.0, "p_scale": 1.0,
               "sigma": 3.0}

# Larghezze dei prior. Sono DEFAULT sovrascrivibili per singola voce dal
# parametro prior_std di log_prior/log_posterior/run_mh: servono agli script
# che vogliono misurare quanto della posteriori e' informazione dei dati e
# quanto e' il prior che si rilegge da solo. La diagnostica sulle finestre di
# asciutta ha mostrato che con "kc": 0.3 la regione verso cui puntano i dati
# (scala ~2) sta a 3.3 sigma dal centro, cioe' e' esclusa a priori.
PRIOR_STD = {"drain_coeff": 0.005, "fc": 15.0, "wp": 8.0, "kc": 0.8, "p_scale": 0.3}
STEP_GLOBAL = {"mm_to_pct": 0.10, "alpha_rain": 0.05, "alpha_irr": 0.05,
               "drain_coeff": 0.0008, "fc": 1.5, "wp": 1.5, "p_scale": 0.05,
               "sigma": 0.25}
STEP_KC = 0.05


def kc_param(suffix, crop):
    return f"{suffix}__{crop}"


# ═══════════════════════════════════════════════════════════════════════════
# Fisica vettorizzata (stesse formule di core/data_utils.py)
# ═══════════════════════════════════════════════════════════════════════════

def _kc_weights_vec(day, lini, ldev, lmid, llate):
    """Pesi dell'interpolazione FAO-56 a 5 nodi, tali che
    Kc_eff = w_ini*kc_ini + w_mid*kc_mid + w_end*kc_end. Si salvano i PESI e
    non il Kc perche' cosi' Kc_eff resta lineare nelle scale calibrate."""
    day = np.asarray(day, dtype=float)
    t1 = lini
    t2 = lini + ldev
    t3 = lini + ldev + lmid
    t4 = lini + ldev + lmid + llate
    day_c = np.clip(day, 0.0, t4)

    frac_dev = np.where(ldev > 0, (day_c - t1) / np.maximum(ldev, 1e-9), 0.0)
    frac_late = np.where(llate > 0, (day_c - t3) / np.maximum(llate, 1e-9), 0.0)

    w_ini = np.where(day_c <= t1, 1.0, np.where(day_c <= t2, 1.0 - frac_dev, 0.0))
    w_mid = np.where(day_c <= t1, 0.0,
                     np.where(day_c <= t2, frac_dev,
                              np.where(day_c <= t3, 1.0, 1.0 - frac_late)))
    w_end = np.where(day_c <= t3, 0.0, frac_late)
    return w_ini, w_mid, w_end


def _p_efficace(params, p_libro):
    """Frazione di deplezione calibrata: valore di libro per coltura x scala
    globale, limitata a [0.01, 0.99] perche' e' una frazione di TAW."""
    return np.clip(np.asarray(p_libro, dtype=float) * params.get("p_scale", 1.0), 0.01, 0.99)


def _water_stress_vec(m, fc, wp, p):
    """Stessa formula di data_utils.water_stress (FAO-56 eq. 84)."""
    TAW = fc - wp
    RAW = p * TAW
    threshold = fc - RAW
    ks = (TAW - (fc - m)) / np.maximum(TAW - RAW, 1e-9)
    ks = np.where(m >= threshold, 1.0, np.where(m <= wp, 0.0, ks))
    return np.clip(ks, 0.0, 1.0)


def _percolation_vec(m, fc, drain_coeff):
    """Stessa formula di data_utils.percolation."""
    excess = np.maximum(m - fc, 0.0)
    return np.minimum(excess * drain_coeff * 10.0, excess)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset di transizioni (t -> t+1): e' su questo che si calibra
# ═══════════════════════════════════════════════════════════════════════════

def _crop_rows(datasets, sensori_df, colture_df, crop_filter, sat_override):
    """Itera su (sensore, periodo di installazione) restituendo il pezzo di
    dataset e i parametri di coltura. Un device puo' avere piu' periodi con
    colture diverse nel tempo, quindi il filtro coltura va applicato qui."""
    for sensor, ds in datasets.items():
        srows = sensori_df[sensori_df["device"] == sensor]
        if srows.empty:
            continue
        ds = ds.copy()
        ds["date"] = pd.to_datetime(ds["date"])
        ds = ds.sort_values("date").reset_index(drop=True)

        for _, srow in srows.iterrows():
            coltura = str(srow.get("coltura", "")).strip().lower()
            if crop_filter is not None and coltura != str(crop_filter).strip().lower():
                continue
            try:
                cp = get_coltura_params(colture_df, coltura)
            except ValueError:
                continue
            install = pd.to_datetime(srow["installazione"], errors="coerce")
            removal = pd.to_datetime(srow.get("rimozione", pd.NaT), errors="coerce")
            if pd.isna(install):
                continue
            sub = ds[ds["date"] >= install]
            if pd.notna(removal):
                sub = sub[sub["date"] <= removal]
            sub = sub.reset_index(drop=True)
            if len(sub) < 3:
                continue
            sat_raw = cp.get("sat", np.nan)
            sat = float(sat_raw) if pd.notna(sat_raw) else float(sat_override or 100.0)
            # origine della fenologia: semina se indicata, installazione altrimenti
            # (install resta quello che delimita il periodo di misura qui sopra)
            origine = origine_curva_kc(srow)
            yield sensor, coltura, cp, install, origine, sub, sat


def _load_context(sensors_filter, sat_override):
    # fallisce subito e in modo leggibile se manca un file: un percorso di
    # irrigazione sbagliato altrimenti darebbe irrigazioni tutte nulle in silenzio
    paths.require_data_files()
    colture_df = apply_soil_overrides(load_colture(paths.DEFAULT_COLTURE), sat=sat_override)
    sensori_df = load_sensori(paths.DEFAULT_SENSORS)
    datasets = build_all_datasets_with_reconstructed_irrigation(
        paths.DEFAULT_DATASET, paths.DEFAULT_SENSORS, paths.DEFAULT_IRRIGAZIONE_RICOSTRUITA,
        sensors_filter=sensors_filter)
    canonical = build_canonical_split(datasets, TARGET_COLS, paths.TRAIN_FRAC, paths.VAL_FRAC)
    split_lookup = {
        (r["device"], pd.Timestamp(r["date"]).strftime("%Y-%m-%d")): r["_split"]
        for _, r in canonical[["device", "date", "_split"]].iterrows()
    }
    return datasets, sensori_df, colture_df, split_lookup


def build_datasets(rain_col=OBSERVED_RAIN_COL, sensors_filter=None, crop_filter=None,
                   sat_override=SAT_OVERRIDE):
    """Costruisce in UNA sola passata sui dati (il caricamento grezzo costa
    1-2 minuti) sia il dataset di transizioni t->t+1, su cui si calibra, sia
    quello di rollout t+1..t+7, usato solo in validazione.

    rain_col: colonna di pioggia per la transizione. Default: quella
    OSSERVATA — questa e' una calibrazione su dati storici, non una pipeline
    deployabile, quindi si usa il dato piu' accurato disponibile.
    Il rollout usa invece sempre le PREVISIONI fc{h}_rain / fc{h}_et0_...
    (gia' corrette in core/data_utils.py), perche' li' si vuole misurare
    l'uso reale.
    """
    datasets, sensori_df, colture_df, split_lookup = _load_context(sensors_filter, sat_override)

    trans_rows, roll_rows = [], []
    for sensor, coltura, cp, install, origine, sub, sat in _crop_rows(
            datasets, sensori_df, colture_df, crop_filter, sat_override):

        kc_ini, kc_mid, kc_end = float(cp["kc_ini"]), float(cp["kc_mid"]), float(cp["kc_end"])
        lini, ldev = float(cp["lini"]), float(cp["ldev"])
        lmid, llate = float(cp["lmid"]), float(cp["llate"])
        p_coltura = float(cp["p"])

        # umidita' misurata per DATA, dentro il periodo di installazione: serve
        # al rollout per leggere il valore vero del giorno D+h (vedi sotto)
        umidita_per_data = (sub.dropna(subset=["date"]).drop_duplicates("date")
                            .set_index("date")["moisture"]
                            if "moisture" in sub.columns else pd.Series(dtype=float))

        for i in range(len(sub)):
            row_date = sub.at[i, "date"]
            m_now = sub.at[i, "moisture"] if "moisture" in sub.columns else np.nan
            if pd.isna(m_now):
                continue
            # prima della semina non c'e' coltura in campo: il modello colturale
            # non si applica (stessa regola di penman.py). Con la semina uguale
            # o precedente all'installazione non esclude nulla.
            if row_date < origine:
                continue
            day_index = float((row_date - origine).days)
            split_label = split_lookup.get((sensor, row_date.strftime("%Y-%m-%d")))
            if split_label is None:
                continue

            common = dict(sensor=sensor, coltura=coltura, split=split_label,
                          p_coltura=p_coltura, sat=sat,
                          kc_ini=kc_ini, kc_mid=kc_mid, kc_end=kc_end,
                          lini=lini, ldev=ldev, lmid=lmid, llate=llate)

            # ── transizione t -> t+1 (giorno precedente -> questo giorno) ──
            # il giorno di partenza deve anch'esso essere dopo la semina
            if (i >= 1 and sub.at[i - 1, "date"] >= origine
                    and (row_date - sub.at[i - 1, "date"]) == pd.Timedelta(days=1)):
                m_prev = sub.at[i - 1, "moisture"]
                et0 = sub.at[i - 1, ET0_COL] if ET0_COL in sub.columns else np.nan
                rain = sub.at[i - 1, rain_col] if rain_col in sub.columns else np.nan
                if pd.notna(m_prev) and pd.notna(et0) and pd.notna(rain):
                    irr_raw = sub.at[i, "irrigation_mm"] if "irrigation_mm" in sub.columns else 0.0
                    w_ini, w_mid, w_end = _kc_weights_vec(day_index, lini, ldev, lmid, llate)
                    trans_rows.append({
                        **common, "date": row_date,
                        "m_prev": float(m_prev), "m_curr": float(m_now),
                        "et0": float(et0), "rain_mm": float(rain),
                        "irr_mm": 0.0 if pd.isna(irr_raw) else float(irr_raw),
                        "day_index": day_index,
                        "w_ini": float(w_ini), "w_mid": float(w_mid), "w_end": float(w_end),
                    })

            # ── rollout t+1..t+7 (solo validazione, con le PREVISIONI) ──
            roll = {**common, "date": row_date, "m0": float(m_now), "day_index": day_index}
            complete = True
            for h in HORIZONS:
                ecol = f"fc{h}_et0_fao_evapotranspiration"
                rcol = f"fc{h}_rain"
                target_date = row_date + pd.Timedelta(days=h)
                # valore vero cercato PER DATA. target_t{h} e' uno shift
                # posizionale: dove la serie ha buchi punterebbe a un altro
                # giorno, e si confronterebbe la previsione per D+h con la misura
                # di una data diversa. Se D+h manca, la riga non e' completa.
                y = umidita_per_data.get(target_date, np.nan)
                e = sub.at[i, ecol] if ecol in sub.columns else np.nan
                r = sub.at[i, rcol] if rcol in sub.columns else np.nan
                if pd.isna(y) or pd.isna(e) or pd.isna(r):
                    complete = False
                    break
                irr_h = 0.0
                if "irrigation_mm" in sub.columns:
                    match = sub.loc[sub["date"] == target_date, "irrigation_mm"]
                    if len(match) and pd.notna(match.iloc[0]):
                        irr_h = float(match.iloc[0])
                roll[f"y_{h}"] = float(y)
                roll[f"et0_{h}"] = float(e)
                roll[f"rain_{h}"] = float(r)
                roll[f"irr_{h}"] = irr_h
            if complete:
                roll_rows.append(roll)

    df_trans = pd.DataFrame(trans_rows)
    df_roll = pd.DataFrame(roll_rows)
    return df_trans, df_roll


def crops_in(df):
    return sorted(df["coltura"].unique().tolist())


# ═══════════════════════════════════════════════════════════════════════════
# Previsione
# ═══════════════════════════════════════════════════════════════════════════

def _kc_scales_per_row(params, df, crops):
    """Espande le scale Kc per-coltura su ogni riga."""
    idx = pd.Categorical(df["coltura"], categories=crops).codes
    out = []
    for suffix in KC_SUFFIXES:
        vals = np.array([params[kc_param(suffix, c)] for c in crops], dtype=float)
        out.append(vals[idx])
    return out


def predict_mu(params, df, crops):
    """Un passo di bilancio idrico: mu(D+1) per ogni riga di df."""
    fc, wp = params["fc"], params["wp"]
    mm = params["mm_to_pct"]
    m_prev = df["m_prev"].values

    s_ini, s_mid, s_end = _kc_scales_per_row(params, df, crops)
    kc_eff = (df["w_ini"].values * df["kc_ini"].values * s_ini
              + df["w_mid"].values * df["kc_mid"].values * s_mid
              + df["w_end"].values * df["kc_end"].values * s_end)

    ks = _water_stress_vec(m_prev, fc, wp, _p_efficace(params, df["p_coltura"].values))
    etc_pct = ks * kc_eff * df["et0"].values * mm
    rain_pct = params["alpha_rain"] * df["rain_mm"].values * mm
    irr_pct = params["alpha_irr"] * df["irr_mm"].values * mm
    perc_pct = _percolation_vec(m_prev, fc, params["drain_coeff"])

    mu = m_prev + rain_pct + irr_pct - etc_pct - perc_pct
    return np.clip(mu, wp, df["sat"].values)


def predict_rollout(params, df_roll, crops):
    """Simulazione iterativa t+1..t+7 (solo validazione). Restituisce una
    matrice (n_righe x 7) con la previsione a ciascun orizzonte."""
    fc, wp = params["fc"], params["wp"]
    mm = params["mm_to_pct"]
    s_ini, s_mid, s_end = _kc_scales_per_row(params, df_roll, crops)

    m = df_roll["m0"].values.astype(float).copy()
    sat = df_roll["sat"].values
    preds = np.zeros((len(df_roll), len(HORIZONS)))

    for j, h in enumerate(HORIZONS):
        w_ini, w_mid, w_end = _kc_weights_vec(
            df_roll["day_index"].values + h, df_roll["lini"].values, df_roll["ldev"].values,
            df_roll["lmid"].values, df_roll["llate"].values)
        kc_eff = (w_ini * df_roll["kc_ini"].values * s_ini
                  + w_mid * df_roll["kc_mid"].values * s_mid
                  + w_end * df_roll["kc_end"].values * s_end)

        ks = _water_stress_vec(m, fc, wp, _p_efficace(params, df_roll["p_coltura"].values))
        etc_pct = ks * kc_eff * df_roll[f"et0_{h}"].values * mm
        rain_pct = params["alpha_rain"] * df_roll[f"rain_{h}"].values * mm
        irr_pct = params["alpha_irr"] * df_roll[f"irr_{h}"].values * mm
        perc_pct = _percolation_vec(m, fc, params["drain_coeff"])

        m = np.clip(m + rain_pct + irr_pct - etc_pct - perc_pct, wp, sat)
        preds[:, j] = m
    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Stima preliminare di mm_to_pct dagli eventi di pioggia (prior)
# ═══════════════════════════════════════════════════════════════════════════

def estimate_mm_to_pct(df_trans, min_rain=2.0, split="train"):
    """Regressione di (m_curr - m_prev) sulla pioggia, sui soli giorni di
    pioggia significativa del TRAIN. E' l'unico punto in cui mm_to_pct e'
    identificabile SENZA Kc: nella pioggia il coefficiente compare da solo,
    mentre dentro ETc compare solo moltiplicato per Kc.

    Ritorna (stima, n_eventi). Se gli eventi sono troppo pochi ritorna
    (None, n) e il chiamante deve ricadere su un prior debole.
    """
    d = df_trans[(df_trans["split"] == split) & (df_trans["rain_mm"] > min_rain)]
    if len(d) < 6:
        return None, len(d)
    x = d["rain_mm"].values[:, None]
    y = (d["m_curr"] - d["m_prev"]).values
    beta = float(np.linalg.lstsq(x, y, rcond=None)[0][0])
    return beta, len(d)


# ═══════════════════════════════════════════════════════════════════════════
# Prior / verosimiglianza / posteriori
# ═══════════════════════════════════════════════════════════════════════════

def log_prior(params, crops, mm_prior_mean, mm_prior_std, prior_std=None):
    p = params
    ps = dict(PRIOR_STD)
    if prior_std:
        ps.update(prior_std)
    if not (0.005 <= p["mm_to_pct"] <= 0.60):
        return -np.inf
    # alpha_rain e alpha_irr sono FRAZIONI di acqua assorbita: di 1 mm caduto
    # al piu' 1 mm entra nel suolo. Il bound a 2.0 della vecchia calibrazione
    # non era fisico e lasciava che alpha_rain salisse a 1.49 per compensare
    # un mm_to_pct troppo basso, cioe' "assorbo il 149% della pioggia caduta".
    if not (0.0 <= p["alpha_rain"] <= 1.0):
        return -np.inf
    if not (0.0 <= p["alpha_irr"] <= 1.0):
        return -np.inf
    if p["drain_coeff"] <= 0:
        return -np.inf
    if not (25.0 <= p["fc"] <= 70.0):
        return -np.inf
    if not (5.0 <= p["wp"] <= 35.0):
        return -np.inf
    # p_scale: il p efficace e' comunque limitato a [0.01, 0.99] da _p_efficace;
    # questi sono i bound sulla scala, larghi ma non assurdi (2.0 porterebbe
    # gia' il mais oltre il tetto fisico)
    if not (0.10 <= p.get("p_scale", 1.0) <= 2.0):
        return -np.inf
    if p["fc"] - p["wp"] <= 5.0:          # TAW fisicamente positivo
        return -np.inf
    if not (0.05 <= p["sigma"] <= 20.0):
        return -np.inf
    for c in crops:
        for suffix in KC_SUFFIXES:
            if not (0.05 <= p[kc_param(suffix, c)] <= 3.0):
                return -np.inf

    lp = norm.logpdf(p["mm_to_pct"], mm_prior_mean, mm_prior_std)
    lp += norm.logpdf(p["drain_coeff"], 0.005, ps["drain_coeff"])
    lp += norm.logpdf(p["fc"], 45.0, ps["fc"])
    lp += norm.logpdf(p["wp"], 18.0, ps["wp"])
    lp += norm.logpdf(p.get("p_scale", 1.0), 1.0, ps["p_scale"])
    for c in crops:
        for suffix in KC_SUFFIXES:
            lp += norm.logpdf(p[kc_param(suffix, c)], 1.0, ps["kc"])
    # alpha_rain, alpha_irr, sigma: uniformi sui bound gia' verificati sopra
    return lp


def log_likelihood(params, df, crops):
    mu = predict_mu(params, df, crops)
    resid = df["m_curr"].values - mu
    sigma = params["sigma"]
    n = len(resid)
    return -0.5 * np.sum((resid / sigma) ** 2) - n * np.log(sigma) - 0.5 * n * np.log(2.0 * np.pi)


def log_posterior(params, df, crops, mm_prior_mean, mm_prior_std, prior_std=None):
    lp = log_prior(params, crops, mm_prior_mean, mm_prior_std, prior_std)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, df, crops)


# ═══════════════════════════════════════════════════════════════════════════
# Metropolis-within-Gibbs a blocchi
# ═══════════════════════════════════════════════════════════════════════════

def initial_params(crops, mm_init=None):
    p = dict(INIT_GLOBAL)
    if mm_init is not None:
        p["mm_to_pct"] = float(mm_init)
    for c in crops:
        for suffix in KC_SUFFIXES:
            p[kc_param(suffix, c)] = 1.0
    return p


def run_mh(df_train, crops, mm_prior_mean, mm_prior_std,
           n_burnin=5000, n_sample=20000, thin=20, seed=42,
           adapt_every=200, target_accept=0.30, fixed=None, init=None,
           prior_std=None):
    """Metropolis-within-Gibbs: a ogni iterazione si propone prima il blocco
    globale, poi quello per-coltura, ognuno con la propria scala adattiva.
    Con ~19 parametri una proposta congiunta unica farebbe crollare
    l'accettazione.

    fixed: insieme di nomi da NON campionare (restano al valore iniziale) —
    serve agli script che calibrano per stadi.
    """
    rng = np.random.default_rng(seed)
    fixed = set(fixed or ())

    current = init if init is not None else initial_params(crops, mm_prior_mean)
    current = dict(current)

    block_global = [n for n in GLOBAL_NAMES if n not in fixed]
    block_kc = [kc_param(s, c) for c in crops for s in KC_SUFFIXES
                if kc_param(s, c) not in fixed]
    blocks = [("globale", block_global), ("per-coltura", block_kc)]
    blocks = [(name, names) for name, names in blocks if names]

    step = {n: STEP_GLOBAL[n] for n in GLOBAL_NAMES}
    for c in crops:
        for s in KC_SUFFIXES:
            step[kc_param(s, c)] = STEP_KC
    scale = {name: 1.0 for name, _ in blocks}

    current_lp = log_posterior(current, df_train, crops, mm_prior_mean, mm_prior_std, prior_std)
    if not np.isfinite(current_lp):
        raise RuntimeError("Posteriori iniziale non finita: controlla i valori di partenza.")

    n_total = n_burnin + n_sample
    chain_rows = []
    acc = {name: 0 for name, _ in blocks}
    acc_win = {name: 0 for name, _ in blocks}

    for it in range(n_total):
        for name, names in blocks:
            proposal = dict(current)
            for p_name in names:
                proposal[p_name] = current[p_name] + rng.normal(0.0, step[p_name] * scale[name])
            prop_lp = log_posterior(proposal, df_train, crops, mm_prior_mean, mm_prior_std, prior_std)
            if np.log(rng.uniform()) < (prop_lp - current_lp):
                current, current_lp = proposal, prop_lp
                acc[name] += 1
                acc_win[name] += 1

        if it < n_burnin and (it + 1) % adapt_every == 0:
            for name, _ in blocks:
                rate = acc_win[name] / adapt_every
                scale[name] *= 1.15 if rate > target_accept else 1.0 / 1.15
                acc_win[name] = 0

        if it >= n_burnin and (it - n_burnin) % thin == 0:
            row = dict(current)
            row["log_posterior"] = current_lp
            chain_rows.append(row)

    accept_rates = {name: acc[name] / n_total for name, _ in blocks}
    return pd.DataFrame(chain_rows), accept_rates


def summarize_posterior(chain_df, crops, mm_prior_mean=None):
    names = GLOBAL_NAMES + [kc_param(s, c) for c in crops for s in KC_SUFFIXES]
    rows = []
    for name in names:
        if name not in chain_df.columns:
            continue
        v = chain_df[name].values
        ref = INIT_GLOBAL.get(name, 1.0 if name not in GLOBAL_NAMES else np.nan)
        if name == "mm_to_pct" and mm_prior_mean is not None:
            ref = mm_prior_mean
        rows.append({"parametro": name, "media": float(v.mean()), "std": float(v.std()),
                     "mediana": float(np.median(v)),
                     "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                     "riferimento": ref})
    return pd.DataFrame(rows)


def kc_assoluti(chain_df, df_trans, crops):
    """Converte le scale Kc a posteriori nei VALORI ASSOLUTI di Kc
    (valore di libro x scala), con l'intervallo al 90%. Le scale da sole
    dicono "quanto ci si scosta dal libro"; questa tabella dice quanto vale
    davvero il coefficiente colturale."""
    book = df_trans.drop_duplicates("coltura").set_index("coltura")
    rows = []
    for c in crops:
        for suffix, col in zip(KC_SUFFIXES, ["kc_ini", "kc_mid", "kc_end"]):
            name = kc_param(suffix, c)
            if name not in chain_df.columns:
                continue
            v = chain_df[name].values
            libro = float(book.at[c, col])
            rows.append({
                "coltura": c, "fase": col, "valore_libro": libro,
                "scala_media": float(v.mean()), "scala_std": float(v.std()),
                "kc_assoluto": float(libro * v.mean()),
                "kc_p5": float(libro * np.percentile(v, 5)),
                "kc_p95": float(libro * np.percentile(v, 95)),
            })
    return pd.DataFrame(rows)


def p_assoluti(chain_df, df_trans, crops):
    """La frazione di deplezione p in valore ASSOLUTO per coltura (valore di
    libro x p_scale stimata), con l'intervallo al 90%. La scala da sola dice
    quanto ci si scosta dal libro; questa tabella dice a quale umidita' il
    modello fa cominciare lo stress idrico, che e' la grandezza interpretabile.

    Il p efficace e' limitato a [0.01, 0.99] come nella fisica, quindi i
    percentili qui riflettono lo stesso limite applicato in previsione."""
    if "p_scale" not in chain_df.columns:
        return pd.DataFrame()
    book = df_trans.drop_duplicates("coltura").set_index("coltura")
    v = chain_df["p_scale"].values
    rows = []
    for c in crops:
        libro = float(book.at[c, "p_coltura"])
        eff = np.clip(libro * v, 0.01, 0.99)
        rows.append({
            "coltura": c, "p_libro": libro,
            "scala_media": float(v.mean()), "scala_std": float(v.std()),
            "p_assoluto": float(eff.mean()),
            "p_p5": float(np.percentile(eff, 5)), "p_p95": float(np.percentile(eff, 95)),
        })
    return pd.DataFrame(rows)


def posterior_correlations(chain_df, crops):
    """Correlazione a posteriori tra mm_to_pct e ogni scala Kc. Dentro ETc i
    due sono collineari: se la correlazione resta vicina a -1 vuol dire che i
    dati non riescono a separarli, ed e' un limite da dichiarare."""
    rows = []
    if "mm_to_pct" not in chain_df.columns:
        return pd.DataFrame(rows)
    mm = chain_df["mm_to_pct"].values
    for c in crops:
        for s in KC_SUFFIXES:
            name = kc_param(s, c)
            if name in chain_df.columns:
                rows.append({"coltura": c, "parametro": s,
                             "corr_con_mm_to_pct": float(np.corrcoef(mm, chain_df[name].values)[0, 1])})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Metriche predittive (1 giorno e rollout), sempre con la persistenza
# ═══════════════════════════════════════════════════════════════════════════

def _uncertainty_row(label, y_true, mu_pred, epistemic, aleatoric, levels=(0.50, 0.90), extra=None):
    total = np.sqrt(epistemic ** 2 + aleatoric ** 2)
    row = {"modello": label, "n": int(len(y_true)),
           "MAE": float(np.mean(np.abs(y_true - mu_pred))),
           "RMSE": float(np.sqrt(np.mean((y_true - mu_pred) ** 2))),
           "mean_epistemic_std": float(np.mean(epistemic)),
           "mean_aleatoric_std": float(np.mean(aleatoric)),
           "mean_total_std": float(np.mean(total))}
    for level in levels:
        pct = int(round(level * 100))
        row[f"coverage_{pct}"] = float(eu.coverage_at_level(y_true, mu_pred, total, level))
        row[f"sharpness_{pct}"] = float(eu.sharpness_at_level(total, level))
    if extra:
        row.update(extra)
    return row


def metrics_1day(chain_df, df_test, crops, levels=(0.50, 0.90)):
    """Metriche a 1 giorno: ensemble di previsioni, una per campione a
    posteriori (stessa scomposizione epistemica/aleatoria gia' usata dalle
    pipeline Deep Ensemble). Include sempre la riga PERSISTENZA."""
    param_names = [c for c in chain_df.columns if c != "log_posterior"]
    mu_samples = np.stack([
        predict_mu({k: row[k] for k in param_names}, df_test, crops)
        for _, row in chain_df.iterrows()
    ])
    y_true = df_test["m_curr"].values
    mu = mu_samples.mean(axis=0)
    epi = mu_samples.std(axis=0)
    ale = np.full_like(mu, np.sqrt(np.mean(chain_df["sigma"].values ** 2)))

    rows = [_uncertainty_row("Penman calibrato (MH)", y_true, mu, epi, ale, levels)]
    pers = df_test["m_prev"].values
    rows.append({"modello": "persistenza", "n": int(len(y_true)),
                 "MAE": float(np.mean(np.abs(y_true - pers))),
                 "RMSE": float(np.sqrt(np.mean((y_true - pers) ** 2)))})
    return pd.DataFrame(rows)


def metrics_rollout(chain_df, df_roll_test, crops, levels=(0.50, 0.90)):
    """Metriche sul rollout t+1..t+7 con le PREVISIONI meteo: e' l'uso reale.
    Si calibra a 1 giorno, quindi questa tabella serve a vedere se i
    parametri reggono anche a 7. Persistenza inclusa a ogni orizzonte."""
    param_names = [c for c in chain_df.columns if c != "log_posterior"]
    samples = np.stack([
        predict_rollout({k: row[k] for k in param_names}, df_roll_test, crops)
        for _, row in chain_df.iterrows()
    ])                                        # (n_campioni, n_righe, 7)
    sigma_mean = np.sqrt(np.mean(chain_df["sigma"].values ** 2))

    rows = []
    for j, h in enumerate(HORIZONS):
        y_true = df_roll_test[f"y_{h}"].values
        mu = samples[:, :, j].mean(axis=0)
        epi = samples[:, :, j].std(axis=0)
        ale = np.full_like(mu, sigma_mean * np.sqrt(h))   # l'errore si accumula
        rows.append(_uncertainty_row("Penman calibrato (MH)", y_true, mu, epi, ale, levels,
                                     extra={"horizon": f"t+{h}"}))
        pers = df_roll_test["m0"].values
        rows.append({"modello": "persistenza", "horizon": f"t+{h}", "n": int(len(y_true)),
                     "MAE": float(np.mean(np.abs(y_true - pers))),
                     "RMSE": float(np.sqrt(np.mean((y_true - pers) ** 2)))})

    df = pd.DataFrame(rows)
    cols = ["modello", "horizon"] + [c for c in df.columns if c not in ("modello", "horizon")]
    return df[cols]
