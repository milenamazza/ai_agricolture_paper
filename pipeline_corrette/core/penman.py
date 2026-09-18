"""
penman.py
─────────────────────────────────────────────────────────────────────────────
Simulazione Penman-Monteith (FAO-56) iterativa t+1..t+7, copia locale di
penman_baseline/run_penman.py limitata alla sola fisica (la parte di
valutazione/CLI di quel file non serve qui: ogni pipeline ha la sua).

Fisica INVARIATA rispetto all'originale. Quello che cambia è l'INPUT: le
colonne fc{h}_rain e fc{h}_et0_fao_evapotranspiration arrivano da
data_utils.pivot_forecast, che qui somma correttamente le letture orarie
invece di mediarle (vedi il FIX in data_utils.py) — nell'originale erano
~23-24x troppo piccole, e finivano dritte nel bilancio idrico.

Per ogni giorno D si simula in avanti giorno per giorno: al passo k si usano
le previsioni fatte il giorno D per il giorno D+k (fc{k}_rain,
fc{k}_et0_fao_evapotranspiration), la Kc al giorno D+k, e l'irrigazione
registrata alla data D+k. Se manca anche solo un input al passo k, la catena
si interrompe (t_k..t_7 restano NaN per quel giorno) — nessun dato inventato.
"""

import numpy as np
import pandas as pd

from data_utils import (
    HORIZONS, SAT_OVERRIDE,
    get_coltura_params, build_kc_curve, water_stress, percolation,
    origine_curva_kc,
)


def run_penman_multi_horizon_for_sensor(ds, sensori_df, colture_df,
                                        alpha_rain, alpha_irr, horizons=HORIZONS,
                                        observed_rain_col=None):
    """Aggiunge penman_pred_t1..t{max(horizons)} a ds, simulando in avanti
    giorno per giorno per ciascuna installazione del sensore in sensori_df.

    observed_rain_col: se dato (es. "ws_rainfall_total"), al passo k si usa la
    pioggia OSSERVATA alla data D+k al posto della previsione fc{k}_rain —
    lookup per data, come già si fa per l'irrigazione. È un ORACOLO: guarda il
    futuro, non è utilizzabile in produzione, serve solo a misurare il limite
    superiore della fisica se la previsione di pioggia fosse perfetta."""
    device = str(ds["device"].iloc[0]) if "device" in ds.columns else "unknown"
    rows = sensori_df[sensori_df["device"] == device]
    if rows.empty:
        return ds

    result_pieces = []
    for _, srow in rows.iterrows():
        coltura = str(srow.get("coltura", "")).strip().lower()
        try:
            cp = get_coltura_params(colture_df, coltura)
        except ValueError:
            continue
        install = pd.to_datetime(srow["installazione"], errors="coerce")
        removal = pd.to_datetime(srow.get("rimozione", pd.NaT), errors="coerce")
        if pd.isna(install):
            continue

        sub = ds[ds["date"] >= install].copy()
        if pd.notna(removal):
            sub = sub[sub["date"] <= removal]
        if len(sub) < 5:
            continue

        fc_val = float(cp["fc"]); wp = float(cp["wp"])
        p_val = float(cp["p"]); mm_to_pct = float(cp["mm_to_pct"])
        sat_raw = cp.get("sat", np.nan)
        sat = float(sat_raw) if pd.notna(sat_raw) else float(SAT_OVERRIDE or 100.0)

        sub = sub.sort_values("date").reset_index(drop=True)
        sub["date"] = pd.to_datetime(sub["date"])
        # La curva Kc descrive lo sviluppo della PIANTA, quindi parte dalla
        # semina quando è nota (colonna DATA SEMINA in sensori-corretti.csv) e
        # dall'installazione altrimenti. `install` qui sopra continua invece a
        # delimitare il periodo in cui il sensore misura: sono due date diverse
        # e non vanno confuse.
        # Fuori dall'intervallo coperto dalla curva (prima della semina, o dopo
        # la fine del ciclo colturale) kc_curve.get() restituisce NaN e la
        # catena si interrompe più sotto: nessuna previsione inventata dove non
        # c'è una coltura in campo.
        kc_curve = build_kc_curve(origine_curva_kc(srow), cp)

        if "irrigation_mm" in sub.columns:
            irr_series = sub.set_index("date")["irrigation_mm"].fillna(0)
        else:
            irr_series = pd.Series(dtype=float)

        if observed_rain_col and observed_rain_col in sub.columns:
            rain_series = sub.set_index("date")[observed_rain_col]
        else:
            rain_series = None

        for h in horizons:
            sub[f"penman_pred_t{h}"] = np.nan

        moisture_col = sub["moisture"] if "moisture" in sub.columns else pd.Series(np.nan, index=sub.index)

        for i in range(len(sub)):
            m_now = moisture_col.iat[i]
            if pd.isna(m_now):
                continue
            row_date = sub.at[i, "date"]
            m = float(m_now)

            for h in horizons:
                target_date = row_date + pd.Timedelta(days=h)
                kc_h = kc_curve.get(target_date, np.nan)
                et0_h = sub.at[i, f"fc{h}_et0_fao_evapotranspiration"] \
                    if f"fc{h}_et0_fao_evapotranspiration" in sub.columns else np.nan
                if rain_series is not None:
                    rain_h = rain_series.get(target_date, np.nan)
                else:
                    rain_h = sub.at[i, f"fc{h}_rain"] if f"fc{h}_rain" in sub.columns else np.nan

                if pd.isna(kc_h) or pd.isna(et0_h) or pd.isna(rain_h):
                    break   # catena interrotta: t_h..t_max restano NaN per questo giorno

                irr_h = float(irr_series.get(target_date, 0.0))

                ks = water_stress(m, fc_val, wp, p_val)
                etc_pct = ks * float(kc_h) * float(et0_h) * mm_to_pct
                rain_pct = alpha_rain * float(rain_h) * mm_to_pct
                irr_pct = alpha_irr * irr_h * mm_to_pct
                perc_pct = percolation(m, fc_val)
                m = float(np.clip(m + rain_pct + irr_pct - etc_pct - perc_pct, wp, sat))
                sub.at[i, f"penman_pred_t{h}"] = m

        result_pieces.append(sub)

    if not result_pieces:
        return ds

    wb = pd.concat(result_pieces, ignore_index=True).sort_values("date").drop_duplicates("date")
    pred_cols = [f"penman_pred_t{h}" for h in horizons if f"penman_pred_t{h}" in wb.columns]
    ds = ds.merge(wb[["date"] + pred_cols], on="date", how="left")
    return ds


def enrich_all_with_penman(datasets, sensori_df, colture_df, alpha_rain, alpha_irr,
                           observed_rain_col=None):
    enriched = {}
    for sensor, ds in datasets.items():
        ds2 = ds.copy()
        if "device" not in ds2.columns:
            ds2["device"] = sensor
        enriched[sensor] = run_penman_multi_horizon_for_sensor(
            ds2, sensori_df, colture_df, alpha_rain, alpha_irr,
            observed_rain_col=observed_rain_col)
    return enriched
