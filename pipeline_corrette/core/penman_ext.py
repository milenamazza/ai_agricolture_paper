"""
penman_ext.py
─────────────────────────────────────────────────────────────────────────────
Come penman.py, ma salva anche gli INTERMEDI FISICI per orizzonte (Kc, Ks,
ETc, pioggia/irrigazione efficaci, percolazione, irrigazione pianificata) —
copia locale della sola parte fisica di
modelli singoli/penman_residual/run_penman_residual.py.

Serve alle pipeline che usano le colonne di dominio Penman come feature
(variante "_dominio" del multioutput). Fisica INVARIATA rispetto
all'originale; cambia solo l'input, ora corretto (vedi data_utils.py).
"""

import numpy as np
import pandas as pd

from data_utils import (
    HORIZONS, SAT_OVERRIDE,
    get_coltura_params, build_kc_curve, water_stress, percolation,
    origine_curva_kc,
)

PENMAN_PHYS_SUFFIXES = [
    "penman_pred", "penman_kc", "penman_etc_pct", "penman_ks",
    "penman_rain_eff_pct", "penman_irr_eff_pct", "penman_perc_pct",
    "penman_irrigation_planned",
]


def penman_phys_cols(h):
    return [f"{s}_t{h}" for s in PENMAN_PHYS_SUFFIXES]


def run_penman_multi_horizon_ext_for_sensor(ds, sensori_df, colture_df,
                                            alpha_rain, alpha_irr, horizons=HORIZONS):
    device = str(ds["device"].iloc[0]) if "device" in ds.columns else "unknown"
    rows = sensori_df[sensori_df["device"] == device]
    if rows.empty:
        return ds

    out_cols = ["date"] + [c for h in horizons for c in penman_phys_cols(h)]

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
        # dalla semina se nota, dall'installazione altrimenti — vedi penman.py
        kc_curve = build_kc_curve(origine_curva_kc(srow), cp)

        if "irrigation_mm" in sub.columns:
            irr_series = sub.set_index("date")["irrigation_mm"].fillna(0)
        else:
            irr_series = pd.Series(dtype=float)

        for h in horizons:
            for c in penman_phys_cols(h):
                sub[c] = np.nan

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
                sub.at[i, f"penman_kc_t{h}"] = float(kc_h)
                sub.at[i, f"penman_etc_pct_t{h}"] = etc_pct
                sub.at[i, f"penman_ks_t{h}"] = ks
                sub.at[i, f"penman_rain_eff_pct_t{h}"] = rain_pct
                sub.at[i, f"penman_irr_eff_pct_t{h}"] = irr_pct
                sub.at[i, f"penman_perc_pct_t{h}"] = perc_pct
                sub.at[i, f"penman_irrigation_planned_t{h}"] = irr_h

        result_pieces.append(sub)

    if not result_pieces:
        return ds

    wb = pd.concat(result_pieces, ignore_index=True).sort_values("date").drop_duplicates("date")
    avail_out = [c for c in out_cols if c in wb.columns]
    ds = ds.merge(wb[avail_out], on="date", how="left")

    if "penman_pred_t1" in ds.columns and "moisture" in ds.columns:
        ds["penman_residual_lag1"] = ds["moisture"] - ds["penman_pred_t1"].shift(1)
    else:
        ds["penman_residual_lag1"] = np.nan
    return ds


def enrich_all_with_penman_ext(datasets, sensori_df, colture_df, alpha_rain, alpha_irr):
    enriched = {}
    for sensor, ds in datasets.items():
        ds2 = ds.copy()
        if "device" not in ds2.columns:
            ds2["device"] = sensor
        enriched[sensor] = run_penman_multi_horizon_ext_for_sensor(
            ds2, sensori_df, colture_df, alpha_rain, alpha_irr)
    return enriched
