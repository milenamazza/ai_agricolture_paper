"""
utils.py — Funzioni condivise self-contained per l'analisi umidità del suolo
=============================================================================
Modulo autonomo: nessuna dipendenza da altri file del progetto padre.
Importato da penman_ml.py, baseline_ml.py e pm_overview.py.

Sezioni:
  0  Costanti & percorsi
  1  Colori per i plot
  2  Feature set (FSET_MAP)
  3  Caricamento & pivot dati
  4  Statistiche WS e feature derivate
  5  Fisica Penman-Monteith (FAO-56)
  6  Configurazione colture & sensori
  7  Bilancio idrico con correzione SAT
  8  Codifica coltura
  9  Flag eventi & piogge
 10  Arricchimento Penman per sensore
 11  Split train/test
 12  Metriche
 13  ARIMA & naive helpers
 14  Tuning ML & imputation
"""

from __future__ import annotations

import csv as _csv
import os
import warnings
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, ParameterGrid

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# 0. COSTANTI & PERCORSI
# ═══════════════════════════════════════════════════════════════════════════════

LOCAL_TZ    = "Europe/Rome"
FRMPAYLOAD_PREFIX     = "device_frmpayload_data_"
MIN_SAMPLES           = 20
SEED                  = 42
COMMON_PM_SPLIT_COL   = "common_pm_split"
SAT_OVERRIDE          = 100.0     # limite superiore predizione se non in colture.csv

IGNORE_MEASUREMENTS   = {"device_uplink", "battery", "sensor_in_soil"}
SUM_MEASUREMENTS      = {"rainfall_total", "precipitation_sum",
                         "precipitation_hours", "precipitation", "rain"}
FORECAST_MEASUREMENTS = {"openmeteo_previous_runs_daily",
                         "openmeteo_previous_runs_hourly"}
FORECAST_DROP_FIELDS  = {
    "elevation", "latitude", "longitude",
    "sunrise_unix", "sunset_unix", "utc_offset_seconds",
    "timezone", "timezone_abbreviation", "generationtime_ms",
    "reference_time_unix", "target_time_unix",
}

# Campi previsionali CUMULATIVI: vanno SOMMATI sulle ~24 letture orarie del
# giorno, non mediati (vedi il FIX in pivot_forecast). Stessa classificazione
# di SUM_FIELDS in pivot_forecast_hourly, più "precipitation" per coerenza.
FORECAST_CUMULATIVE_FIELDS = {
    "rain", "showers", "snowfall", "precipitation",
    "et0_fao_evapotranspiration", "evapotranspiration",
}

ET0_ALIASES  = ["et0_fc1", "fc1_et0_fao_evapotranspiration",
                "fc1h_et0_fao_evapotranspiration_sum"]
RAIN_ALIASES = ["rain_fc1", "fc1_rain", "fc1h_rain_sum"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. COLORI PER I PLOT
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_COLORS = {
    "Ridge":            "#1565C0",
    "Ridge-MO":         "#1E88E5",
    "RF":               "#2E7D32",
    "RF-MO":            "#43A047",
    "LGBM":             "#E65100",
    "LGBM-MO":          "#FB8C00",
    "Penman-Monteith":  "#6A1B9A",
    "Naive":            "#546E7A",
    "MA-3":             "#78909C",
    "MA-7":             "#90A4AE",
    "ARIMA(1,1,1)":     "#6D4C41",
}

COLTURA_COLORS = {
    "mais":       "#E53935",
    "sorgo":      "#FB8C00",
    "carciofo":   "#43A047",
    "melissa":    "#00ACC1",
    "passiflora": "#8E24AA",
    "malva":      "#F06292",
    "escolzia":   "#FFB300",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE SET
# ═══════════════════════════════════════════════════════════════════════════════

_BASE = [
    "moisture", "ec", "temperature",
    "moisture_lag1", "moisture_lag2", "moisture_roll3", "moisture_roll7",
    "ec_lag1",
    "crop_encoded",
    "ws_temperature", "ws_humidity", "ws_rainfall_total",
    "ws_temperature_lag1", "ws_temperature_roll3",
    "ws_rainfall_total_lag1", "ws_rainfall_total_roll7",
    "irrigation_mm", "irrigation_lag1",
]

_WS_EXTRA = [
    "ws_temp_min", "ws_temp_max", "ws_temp_median",
    "ws_temp_amplitude", "ws_temp_std",
    "ws_hum_min", "ws_hum_max", "ws_hum_median", "ws_hum_std",
    "ws_wind_max", "ws_wind_std", "ws_wind_median",
]

_HFC_T1 = [
    "fc1h_temperature_2m_mean", "fc1h_temperature_2m_min",
    "fc1h_temperature_2m_max", "fc1h_temperature_2m_amp",
    "fc1h_apparent_temperature_min", "fc1h_apparent_temperature_max",
    "fc1h_relative_humidity_2m_mean", "fc1h_relative_humidity_2m_min",
    "fc1h_wind_speed_10m_max", "fc1h_wind_speed_10m_mean",
    "fc1h_rain_sum", "fc1h_rain_hours",
    "fc1h_et0_fao_evapotranspiration_sum", "fc1h_evapotranspiration_sum",
    "fc1h_showers_sum", "fc1h_snowfall_sum",
]
_HFC_T2 = [c.replace("fc1h_", "fc2h_") for c in _HFC_T1]

_INTERACTIONS_T1 = ["fc1_net_balance", "fc1_moist_x_et", "fc1_temp_delta"]
_INTERACTIONS_T2 = ["fc2_net_balance", "fc2_moist_x_et", "fc2_temp_delta"]
_EVENT_FLAGS      = ["has_rain", "has_irr", "has_water_event"]

FEATURES_T1 = _BASE + [
    "fc1_temperature_2m", "fc1_apparent_temperature",
    "fc1_rain", "fc1_precipitation_hours",
    "fc1_relative_humidity_2m",
    "fc1_et0_fao_evapotranspiration",
    "fc1_evapotranspiration",
]
FEATURES_T2 = _BASE + [
    "fc2_temperature_2m", "fc2_apparent_temperature",
    "fc2_rain", "fc2_precipitation_hours",
    "fc2_relative_humidity_2m",
    "fc2_et0_fao_evapotranspiration",
    "fc2_evapotranspiration",
]

MINIMAL_T1 = [
    "moisture", "moisture_lag1", "moisture_roll3",
    "fc1h_rain_sum", "fc1h_et0_fao_evapotranspiration_sum",
    "irrigation_mm", "temperature", "crop_encoded",
]
MINIMAL_T2 = [
    "moisture", "moisture_lag1", "moisture_roll3",
    "fc2h_rain_sum", "fc2h_et0_fao_evapotranspiration_sum",
    "irrigation_mm", "temperature", "crop_encoded",
]

_WS_OBSERVED = [
    "ws_temperature", "ws_humidity", "ws_rainfall_total",
    "ws_temperature_lag1", "ws_temperature_roll3",
    "ws_rainfall_total_lag1", "ws_rainfall_total_roll7",
]

NO_FORECAST_T1 = [
    "moisture", "moisture_lag1", "moisture_lag2", "moisture_roll3", "moisture_roll7",
    "ec", "ec_lag1", "temperature", "crop_encoded",
] + _WS_OBSERVED + ["irrigation_mm", "irrigation_lag1"]
NO_FORECAST_T2 = NO_FORECAST_T1

FORECAST_ONLY_T1 = [
    "moisture", "crop_encoded",
    "fc1_temperature_2m", "fc1_apparent_temperature",
    "fc1_rain", "fc1_precipitation_hours",
    "fc1_relative_humidity_2m",
    "fc1_et0_fao_evapotranspiration", "fc1_evapotranspiration",
] + _HFC_T1 + _INTERACTIONS_T1
FORECAST_ONLY_T2 = [
    c.replace("fc1_", "fc2_").replace("fc1h_", "fc2h_")
    for c in FORECAST_ONLY_T1
]

FORECAST_METEO_MOISTURE_T1 = [
    "moisture", "moisture_lag1", "moisture_roll3",
    "temperature", "irrigation_mm", "irrigation_lag1",
] + _WS_OBSERVED + _WS_EXTRA + [
    "fc1_temperature_2m", "fc1_rain",
    "fc1_et0_fao_evapotranspiration", "fc1_relative_humidity_2m",
] + _HFC_T1 + _INTERACTIONS_T1 + _EVENT_FLAGS

FORECAST_METEO_MOISTURE_T2 = [
    "moisture", "moisture_lag1", "moisture_roll3",
    "temperature", "irrigation_mm", "irrigation_lag1",
] + _WS_OBSERVED + _WS_EXTRA + [
    "fc2_temperature_2m", "fc2_rain",
    "fc2_et0_fao_evapotranspiration", "fc2_relative_humidity_2m",
] + _HFC_T2 + _INTERACTIONS_T2 + _EVENT_FLAGS

LAG_RAIN_IRR_T1 = [
    "moisture", "moisture_lag1", "moisture_lag2", "moisture_roll3", "moisture_roll7",
    "crop_encoded",
    "ws_rainfall_total", "ws_rainfall_total_lag1", "ws_rainfall_total_roll7",
    "irrigation_mm", "irrigation_lag1",
    "fc1_rain", "fc1h_rain_sum",
    "has_rain", "has_irr",
]
LAG_RAIN_IRR_T2 = [
    "moisture", "moisture_lag1", "moisture_lag2", "moisture_roll3", "moisture_roll7",
    "crop_encoded",
    "ws_rainfall_total", "ws_rainfall_total_lag1", "ws_rainfall_total_roll7",
    "irrigation_mm", "irrigation_lag1",
    "fc2_rain", "fc2h_rain_sum",
    "has_rain", "has_irr",
]

DERIVED_ALL_T1 = FEATURES_T1 + _WS_EXTRA + _HFC_T1 + _INTERACTIONS_T1 + _EVENT_FLAGS
DERIVED_ALL_T2 = FEATURES_T2 + _WS_EXTRA + _HFC_T2 + _INTERACTIONS_T2 + _EVENT_FLAGS

FSET_MAP: dict[str, tuple[list, list]] = {
    "MINIMAL":                 (MINIMAL_T1,                MINIMAL_T2),
    "LAG_RAIN_IRR":            (LAG_RAIN_IRR_T1,           LAG_RAIN_IRR_T2),
    "NO_FORECAST":             (NO_FORECAST_T1,            NO_FORECAST_T2),
    "FORECAST_ONLY":           (FORECAST_ONLY_T1,          FORECAST_ONLY_T2),
    "FORECAST_METEO_MOISTURE": (FORECAST_METEO_MOISTURE_T1, FORECAST_METEO_MOISTURE_T2),
    "DERIVED_ALL":             (DERIVED_ALL_T1,            DERIVED_ALL_T2),
}


# ── Orizzonte generico h=1..7 ────────────────────────────────────────────────
# Non tocca FSET_MAP (per non rompere baseline_ml.py, che si aspetta tuple
# (T1, T2)): la versione T1 di ogni feature-set è l'unica sorgente di verità,
# le colonne "fc1_"/"fc1h_" vengono rimappate a "fc{h}_"/"fc{h}h_" per
# qualsiasi orizzonte richiesto — stessa trasformazione già usata per T2/T7,
# generalizzata per evitare 7 blocchi duplicati (uno per orizzonte).

_FSET_T1_BASE: dict[str, list] = {
    "MINIMAL":                 MINIMAL_T1,
    "LAG_RAIN_IRR":            LAG_RAIN_IRR_T1,
    "NO_FORECAST":             NO_FORECAST_T1,
    "FORECAST_ONLY":           FORECAST_ONLY_T1,
    "FORECAST_METEO_MOISTURE": FORECAST_METEO_MOISTURE_T1,
    "DERIVED_ALL":             DERIVED_ALL_T1,
}

HORIZONS = list(range(1, 8))   # t+1 .. t+7


def get_fset_features(fset_name: str, horizon: int) -> list:
    """Feature list per un feature-set e un orizzonte h in 1..7, derivata
    deterministicamente dalla versione t+1 (fc1_->fc{h}_, fc1h_->fc{h}h_)."""
    if fset_name not in _FSET_T1_BASE:
        raise ValueError(f"Feature-set '{fset_name}' non trovato. Disponibili: {list(_FSET_T1_BASE)}")
    base = _FSET_T1_BASE[fset_name]
    if horizon == 1:
        return list(base)
    return [c.replace("fc1h_", f"fc{horizon}h_").replace("fc1_", f"fc{horizon}_") for c in base]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CARICAMENTO & PIVOT DATI
# ═══════════════════════════════════════════════════════════════════════════════

def load_dump(path: str) -> pd.DataFrame:
    """Carica il dump CSV grezzo e normalizza colonne chiave."""
    print(f"[load] {path} ...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df["date"] = df["time"].dt.normalize().dt.tz_localize(None)

    df["value"]      = pd.to_numeric(df["value"],      errors="coerce")
    df["lead_days"]  = pd.to_numeric(df.get("lead_days",  np.nan), errors="coerce")
    df["lead_hours"] = pd.to_numeric(df.get("lead_hours", np.nan), errors="coerce")

    df["device_name"] = df["device_name"].fillna("").astype(str).str.strip()
    df["measurement"] = df["measurement"].fillna("").str.strip()
    df["field"]       = df["field"].fillna("").str.strip()
    df["timezone"]    = (df["timezone"].fillna(LOCAL_TZ).str.strip()
                         if "timezone" in df.columns else LOCAL_TZ)
    df["measurement"] = df["measurement"].str.replace(FRMPAYLOAD_PREFIX, "", regex=False)
    print(f"       {len(df):,} righe")
    return df


def pivot_em500(df: pd.DataFrame) -> pd.DataFrame:
    """Media giornaliera di ogni measurement dei sensori EM-500."""
    mask = (df["device_name"].str.startswith("EM-500", na=False) &
            ~df["measurement"].isin(IGNORE_MEASUREMENTS))
    agg = (df[mask]
           .groupby(["device_name", "date", "measurement"])["value"]
           .mean().reset_index())
    pivot = agg.pivot_table(index=["device_name", "date"],
                            columns="measurement", values="value", aggfunc="mean")
    pivot.columns.name = None
    return pivot.reset_index().rename(columns={"device_name": "device"})


def pivot_ws(df: pd.DataFrame) -> pd.DataFrame:
    """
    Media giornaliera per i measurement WS normali; per quelli in
    SUM_MEASUREMENTS (es. rainfall_total) l'aggregazione giornaliera è
    diversa: quelle misure arrivano da un CONTATORE CUMULATIVO che si
    azzera periodicamente (non necessariamente a mezzanotte) quando il
    dispositivo esegue un reset — verificato sui dati grezzi: il contatore
    resta piatto per decine di letture consecutive allo stesso valore, poi
    cala di colpo a 0 quando si azzera.

    Il totale giornaliero NON è la somma di tutte le letture del giorno
    (letture ripetute identiche verrebbero contate una volta per lettura,
    gonfiando il totale di un fattore pari al numero di letture — è quello
    che succedeva prima di questo fix, con letture aggregate anche a
    centinaia di mm in un giorno). È invece la somma dei soli INCREMENTI tra
    una lettura e la successiva, calcolati sull'INTERA serie ordinata nel
    tempo per dispositivo (non spezzata per giorno, altrimenti un reset a
    cavallo di mezzanotte spezzerebbe il conteggio a metà), con gli
    incrementi negativi (i reset stessi, dove il contatore torna a un
    valore più basso) azzerati a 0 anziché sottratti.
    """
    mask = (df["device_name"].str.startswith("WS-", na=False) &
            ~df["measurement"].isin(IGNORE_MEASUREMENTS))
    sub = df[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    records = []
    for (dev, meas), grp in sub.groupby(["device_name", "measurement"]):
        grp = grp.sort_values("time")
        if meas in SUM_MEASUREMENTS:
            positive_increments = grp["value"].diff().clip(lower=0).fillna(0)
            daily_totals = positive_increments.groupby(grp["date"]).sum()
        else:
            daily_totals = grp.groupby("date")["value"].mean()
        for date, val in daily_totals.items():
            records.append({"device_name": dev, "date": date, "measurement": meas, "value": val})

    agg   = pd.DataFrame(records)
    pivot = agg.pivot_table(index=["device_name", "date"],
                            columns="measurement", values="value", aggfunc="mean")
    pivot.columns = [f"ws_{c}" for c in pivot.columns]
    pivot.columns.name = None
    return pivot.reset_index().rename(columns={"device_name": "ws"})


def _circular_mean(angles_deg: pd.Series) -> float:
    rad = np.radians(angles_deg.dropna())
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)


def pivot_forecast_hourly(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Aggrega forecast orario Open-Meteo a sommari giornalieri (lead 1 e 2)."""
    fc = df[df["measurement"] == "openmeteo_previous_runs_hourly"].copy()
    if fc.empty:
        return {}

    fc["lead_days_eff"] = (fc["lead_hours"] / 24).round().astype("Int64")
    fc["forecasted_date"] = (fc["time"].dt.tz_convert(LOCAL_TZ)
                              .dt.normalize().dt.tz_localize(None))
    fc["emission_date"] = fc["forecasted_date"] - pd.to_timedelta(
        fc["lead_days_eff"].astype(float), unit="D"
    )
    fc = fc[~fc["field"].isin(FORECAST_DROP_FIELDS)]

    SUM_FIELDS   = {"rain", "showers", "snowfall",
                    "et0_fao_evapotranspiration", "evapotranspiration"}
    MINMAX_FIELDS = {"temperature_2m", "apparent_temperature", "relative_humidity_2m"}
    CIRC_FIELDS   = {"wind_direction_10m"}
    TIME_FIELDS   = {"sunrise_unix", "sunset_unix"}

    results = {}
    for lead in HORIZONS:
        sub = fc[fc["lead_days_eff"] == lead].copy()
        if sub.empty:
            continue
        prefix  = f"fc{lead}h_"
        records = []
        for (emdate, field), grp in sub.groupby(["emission_date", "field"]):
            vals = grp["value"].dropna()
            if len(vals) == 0:
                continue
            rec = {"date": emdate}
            if field in SUM_FIELDS:
                rec[f"{prefix}{field}_sum"] = vals.sum()
                if "rain" in field or "shower" in field:
                    rec[f"{prefix}{field}_hours"] = (vals > 0.1).sum()
            elif field in MINMAX_FIELDS:
                rec[f"{prefix}{field}_mean"]   = vals.mean()
                rec[f"{prefix}{field}_min"]    = vals.min()
                rec[f"{prefix}{field}_max"]    = vals.max()
                rec[f"{prefix}{field}_median"] = vals.median()
                rec[f"{prefix}{field}_amp"]    = vals.max() - vals.min()
            elif field in CIRC_FIELDS:
                rec[f"{prefix}{field}_mean"] = _circular_mean(vals)
            elif field in TIME_FIELDS:
                rec[f"{prefix}{field}"] = vals.iloc[0]
            else:
                rec[f"{prefix}{field}_mean"] = vals.mean()
                rec[f"{prefix}{field}_max"]  = vals.max()
            records.append(rec)
        if not records:
            continue
        day_df = pd.DataFrame(records).groupby("date").first().reset_index()
        results[lead] = day_df
        print(f"  Hourly agg lead={lead}: {len(day_df)} giorni")
    return results


def pivot_forecast(df: pd.DataFrame) -> dict:
    """Pivot forecast daily Open-Meteo per lead=1..7.

    FIX rispetto alla versione originale (modelli singoli/data_driven_ml/
    utils.py): le grandezze CUMULATIVE (pioggia, ET0, evapotraspirazione,
    rovesci, neve) vengono SOMMATE sulle ~24 letture orarie del giorno, non
    mediate. L'originale usava .mean() per tutti i campi indistintamente,
    producendo fc{h}_rain ~24x troppo piccola (0.026 invece di 0.608 mm/gg) e
    fc{h}_et0_fao_evapotranspiration ~23x troppo piccola (0.22 invece di 5.12
    mm/gg) — valori che finivano direttamente nel bilancio idrico di Penman.
    La classificazione dei campi è la stessa già usata da
    pivot_forecast_hourly() qui sopra (SUM_FIELDS), qui estesa a
    "precipitation" per coerenza."""
    mask = df["measurement"].isin(FORECAST_MEASUREMENTS)
    if mask.sum() == 0:
        mask = df["field"].isin(FORECAST_MEASUREMENTS)
    sub = df[mask].copy()
    if sub.empty:
        return {}

    local_tz = LOCAL_TZ
    if "timezone" in sub.columns and sub["timezone"].notna().any():
        local_tz = sub["timezone"].dropna().iloc[0]

    time_local = sub["time"].dt.tz_convert(local_tz)
    sub["forecasted_date"] = time_local.dt.normalize().dt.tz_localize(None)

    if "lead_hours" in sub.columns:
        lead_h = pd.to_numeric(sub["lead_hours"], errors="coerce")
        sub["lead_days_eff"] = sub["lead_days"].combine_first(
            (lead_h / 24).round().astype("Int64")
        )
    else:
        sub["lead_days_eff"] = sub["lead_days"]

    sub = sub[~sub["field"].isin(FORECAST_DROP_FIELDS)]

    results = {}
    for lead in HORIZONS:
        sub_lead = sub[sub["lead_days_eff"] == lead].copy()
        if sub_lead.empty:
            continue
        sub_lead["emission_date"] = (
            sub_lead["forecasted_date"] - pd.Timedelta(days=int(lead))
        )
        is_cumulative = sub_lead["field"].isin(FORECAST_CUMULATIVE_FIELDS)
        agg_sum = (sub_lead[is_cumulative].groupby(["emission_date", "field"])["value"]
                   .sum().reset_index())
        agg_mean = (sub_lead[~is_cumulative].groupby(["emission_date", "field"])["value"]
                    .mean().reset_index())
        agg = pd.concat([agg_sum, agg_mean], ignore_index=True)
        pivot = agg.pivot_table(index="emission_date", columns="field",
                                values="value", aggfunc="first")
        pivot.columns = [f"fc{lead}_{c}" for c in pivot.columns]
        pivot.columns.name = None
        results[lead] = pivot.reset_index().rename(columns={"emission_date": "date"})
    return results


def load_irrigation(path: str) -> pd.DataFrame:
    """Carica irrigazioni.csv. Restituisce DataFrame vuoto se assente."""
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    irr = pd.read_csv(path)
    irr.columns = irr.columns.str.strip()
    irr["date"] = pd.to_datetime(irr["data"], errors="coerce").dt.normalize()
    irr = irr.dropna(subset=["date", "device_name"])
    irr["irrigation_mm"] = pd.to_numeric(irr["mm"], errors="coerce").fillna(0)
    return irr[["device_name", "date", "irrigation_mm"]].copy()


def build_sensor_ws_map(df: pd.DataFrame, sensor_csv: str | None) -> dict:
    """Costruisce {sensor -> [(d_from, d_to, ws_name), ...]} da sensori.csv."""
    if sensor_csv and os.path.exists(sensor_csv):
        def _parse(s):
            for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y"):
                try:
                    return pd.Timestamp(datetime.strptime(s.strip(), fmt)
                                        .replace(tzinfo=timezone.utc))
                except (ValueError, AttributeError):
                    pass
            return None

        mapping = {}
        with open(sensor_csv, newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                num    = row.get("SENSORE", "").strip()
                ws_raw = row.get("WS rispettiva", "").strip()
                if not num or not ws_raw:
                    continue
                try:
                    n = int(float(num))
                except ValueError:
                    continue
                dev    = f"EM-500-{n}"
                d_from = _parse(row.get("DATA INSTALLAZIONE", "")) or pd.Timestamp("2020-01-01")
                d_to   = _parse(row.get("DATA RIMOZIONE", ""))     or pd.Timestamp("2099-01-01")
                for ts in (d_from, d_to):
                    if getattr(ts, "tz", None) is not None:
                        ts = ts.tz_localize(None)
                d_from = d_from.tz_localize(None) if getattr(d_from, "tz", None) else d_from
                d_to   = d_to.tz_localize(None)   if getattr(d_to,   "tz", None) else d_to
                mapping.setdefault(dev, []).append((d_from, d_to, f"WS-{ws_raw}"))
        return mapping

    # Fallback automatico dai dati
    ws_list   = df[df["device_name"].str.startswith("WS-", na=False)]["device_name"].unique()
    em_ranges = (df[df["device_name"].str.startswith("EM-500", na=False)]
                 .groupby("device_name")["date"].agg(["min", "max"]))
    mapping = {}
    for dev, row in em_ranges.iterrows():
        best_ws, best_n = None, -1
        for ws in ws_list:
            n = df[(df["device_name"] == ws) &
                   (df["date"] >= row["min"]) &
                   (df["date"] <= row["max"])].shape[0]
            if n > best_n:
                best_n, best_ws = n, ws
        d_min = row["min"].tz_localize(None) if getattr(row["min"], "tz", None) else row["min"]
        d_max = row["max"].tz_localize(None) if getattr(row["max"], "tz", None) else row["max"]
        mapping[dev] = [(d_min, d_max, best_ws)]
    return mapping


def build_sensor_dataset(sensor, em_pivot, ws_pivot, ws_periods, fc_pivots, irrigation):
    """Assembla il dataset giornaliero per un singolo sensore."""
    base = em_pivot[em_pivot["device"] == sensor].copy()
    base = base.sort_values("date").set_index("date")

    if "moisture" in base.columns:
        base["moisture_lag1"]  = base["moisture"].shift(1)
        base["moisture_lag2"]  = base["moisture"].shift(2)
        base["moisture_roll3"] = base["moisture"].shift(1).rolling(3, min_periods=2).mean()
        base["moisture_roll7"] = base["moisture"].shift(1).rolling(7, min_periods=4).mean()
    if "ec" in base.columns:
        base["ec_lag1"] = base["ec"].shift(1)

    idx = base.index
    base["day_of_year"] = idx.dayofyear
    base["month"]       = idx.month
    base["sin_doy"]     = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    base["cos_doy"]     = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    if not ws_pivot.empty:
        ws_frames = []
        for (d_from, d_to, ws_name) in ws_periods:
            if not ws_name:
                continue
            ws_sub = ws_pivot[ws_pivot["ws"] == ws_name].set_index("date")
            ws_sub = ws_sub[(ws_sub.index >= d_from) & (ws_sub.index <= d_to)]
            if not ws_sub.empty:
                ws_frames.append(ws_sub.drop(columns=["ws"], errors="ignore"))
        if ws_frames:
            ws_all = pd.concat(ws_frames)
            ws_all = ws_all[~ws_all.index.duplicated(keep="last")]
            base   = base.join(ws_all, how="left")
            rain_col = next((c for c in base.columns if "rainfall" in c or "precipitation" in c), None)
            temp_col = next((c for c in base.columns if c.startswith("ws_temp")), None)
            hum_col  = next((c for c in base.columns if c.startswith("ws_hum")), None)
            if rain_col:
                base[f"{rain_col}_lag1"]  = base[rain_col].shift(1)
                base[f"{rain_col}_roll7"] = base[rain_col].shift(1).rolling(7, min_periods=4).sum()
            if temp_col:
                base[f"{temp_col}_lag1"]  = base[temp_col].shift(1)
                base[f"{temp_col}_roll3"] = base[temp_col].shift(1).rolling(3, min_periods=2).mean()
            if hum_col:
                base[f"{hum_col}_lag1"] = base[hum_col].shift(1)

    for lead, fc in fc_pivots.items():
        base = base.join(fc.set_index("date"), how="left")

    if not irrigation.empty:
        irr_sensor = (irrigation[irrigation["device_name"] == sensor]
                      .set_index("date")[["irrigation_mm"]])
        irr_sensor = irr_sensor[~irr_sensor.index.duplicated(keep="last")]
        base = base.join(irr_sensor, how="left")
        base["irrigation_mm"]   = base["irrigation_mm"].fillna(0)
        base["irrigation_lag1"] = base["irrigation_mm"].shift(1)

        # ── Irrigazione PIANIFICATA per i giorni futuri ─────────────────────
        # irrigation_mm e irrigation_lag1 guardano solo indietro (oggi e ieri):
        # con quelle sole colonne il modello, per prevedere t+7, vede esattamente
        # ciò che vede per t+1, e l'irrigazione futura resta per lui un'incognita.
        # Ma l'irrigazione non è un fenomeno da prevedere: è una decisione nota
        # in anticipo, cioè un input esogeno — ed è già così che la tratta la
        # fisica (penman.py fa irr_series.get(row_date + h giorni, 0.0) per ogni
        # orizzonte). Queste 7 colonne allineano il lato ML a quel trattamento.
        #
        # LOOKUP PER DATA, non shift(-h). La differenza non è una sfumatura: le
        # righe sono consecutive nella TABELLA, non nel CALENDARIO (quando un
        # sensore smette di misurare, i giorni mancanti semplicemente non hanno
        # una riga). Con shift(-1), una riga seguita nella tabella da un giorno
        # di sette mesi dopo si vedrebbe assegnare l'irrigazione di quel giorno
        # come se fosse "domani": un valore inventato, e per giunta invisibile,
        # perché la colonna è riempita di zeri e non produce il NaN che farebbe
        # scartare la riga dal pool. Il reindex sulle date vere scrive 0 dove
        # quel giorno non è stato irrigato, che è la verità.
        #
        # Il default 0.0 è lo stesso di penman.py: "nessuna irrigazione nota per
        # quel giorno" si traduce in zero millimetri, non in un dato mancante.
        irr_per_data = base["irrigation_mm"]
        for h in HORIZONS:
            base[f"irr_planned_t{h}"] = (
                irr_per_data.reindex(base.index + pd.Timedelta(days=h))
                .fillna(0.0)
                .to_numpy()
            )

    if "moisture" in base.columns:
        for h in HORIZONS:
            base[f"target_t{h}"] = base["moisture"].shift(-h)

    return base.reset_index()


def build_all_datasets(dump, sensors_csv, irr_csv, verbose=True):
    """Entry point: carica tutto e restituisce {sensor: DataFrame}."""
    df         = load_dump(dump)
    em_pivot   = pivot_em500(df)
    ws_pivot   = pivot_ws(df)
    fc_pivots  = pivot_forecast(df)
    irrigation = load_irrigation(irr_csv)
    ws_map     = build_sensor_ws_map(df, sensors_csv)

    datasets: dict[str, pd.DataFrame] = {}
    for sensor in sorted(em_pivot["device"].unique()):
        periods = ws_map.get(sensor, [])
        ds = build_sensor_dataset(sensor, em_pivot, ws_pivot, periods,
                                   fc_pivots, irrigation)
        if "target_t1" not in ds.columns:
            continue
        if ds["target_t1"].notna().sum() < MIN_SAMPLES:
            continue
        datasets[sensor] = ds

    if verbose:
        print(f"\nDataset pronti: {len(datasets)} sensori")
        for s, d in datasets.items():
            print(f"  {s}: {len(d)} giorni "
                  f"(t1={d['target_t1'].notna().sum()}, t2={d['target_t2'].notna().sum()})")
    return datasets


def pool_datasets(datasets, features, target):
    """Combina tutti i sensori in un DataFrame con le feature richieste."""
    _EVENT_COLS = ["has_rain", "has_irr", "has_water_event",
                   "rain_median_mm", "device"]
    frames = []
    for sensor, ds in datasets.items():
        avail = [f for f in features if f in ds.columns]
        extra = [c for c in _EVENT_COLS if c in ds.columns and c not in avail]
        sub   = ds[["date"] + avail + extra + [target]].copy().dropna(subset=[target])
        for f in features:
            if f not in sub.columns:
                sub[f] = np.nan
        if "device" not in sub.columns:
            sub["device"] = sensor
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return (pd.concat(frames, ignore_index=True)
              .sort_values("date")
              .reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STATISTICHE WS E FEATURE DERIVATE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ws_daily_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Statistiche sub-giornaliere delle WS (temp, humidity, wind)."""
    mask = raw_df["device_name"].str.startswith("WS-", na=False)
    sub  = raw_df[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    CONFIG = {
        "temperature": ("ws_temp", ["min", "max", "std", "median"]),
        "humidity":    ("ws_hum",  ["min", "max", "std", "median"]),
        "wind_speed":  ("ws_wind", ["max", "std", "median"]),
    }
    pieces = []
    for meas, (prefix, aggs) in CONFIG.items():
        s = sub[sub["measurement"] == meas].copy()
        if s.empty:
            continue
        grp  = s.groupby(["device_name", "date"])["value"]
        part = grp.agg(aggs).rename(columns={a: f"{prefix}_{a}" for a in aggs})
        part = part.reset_index().rename(columns={"device_name": "ws"})
        if meas == "temperature":
            part["ws_temp_amplitude"] = part["ws_temp_max"] - part["ws_temp_min"]
        pieces.append(part)

    if not pieces:
        return pd.DataFrame()
    result = pieces[0]
    for p in pieces[1:]:
        result = result.merge(p, on=["ws", "date"], how="outer")
    return result


def add_hourly_forecast(datasets: dict, raw_df: pd.DataFrame) -> dict:
    """Merge degli aggregati orari del forecast in ogni dataset sensore."""
    hfc = pivot_forecast_hourly(raw_df)
    enhanced = {}
    for sensor, ds in datasets.items():
        ds = ds.copy()
        for lead, hdf in hfc.items():
            ds = ds.merge(hdf, on="date", how="left")
        enhanced[sensor] = ds
    return enhanced


def add_derived_features(datasets: dict, ws_stats: pd.DataFrame,
                         ws_map: dict) -> dict:
    """Arricchisce ogni sensore con statistiche WS e interazioni fisiche."""
    enhanced = {}
    for sensor, ds in datasets.items():
        ds = ds.copy()

        if not ws_stats.empty and sensor in ws_map:
            rows = []
            for d_from, d_to, ws_name in ws_map[sensor]:
                chunk = ws_stats[
                    (ws_stats["ws"] == ws_name) &
                    (ws_stats["date"] >= d_from) &
                    (ws_stats["date"] <= d_to)
                ].drop(columns=["ws"], errors="ignore")
                rows.append(chunk)
            if rows:
                ws_chunk = (pd.concat(rows, ignore_index=True)
                            .drop_duplicates("date"))
                ds = ds.merge(ws_chunk, on="date", how="left")

        moist = ds.get("moisture",      pd.Series(dtype=float))
        irr   = ds.get("irrigation_mm", pd.Series(0.0, index=ds.index))
        ws_t  = ds.get("ws_temperature", pd.Series(dtype=float))

        for fc in (f"fc{h}" for h in HORIZONS):
            rain_h = ds.get(f"{fc}h_rain_sum", None)
            rain   = (rain_h if rain_h is not None
                      else ds.get(f"{fc}_rain", pd.Series(0.0, index=ds.index)))
            et0_h  = ds.get(f"{fc}h_et0_fao_evapotranspiration_sum", None)
            et0    = (et0_h if et0_h is not None
                      else ds.get(f"{fc}_et0_fao_evapotranspiration",
                                  pd.Series(0.0, index=ds.index)))
            fc_tmp = ds.get(f"{fc}h_temperature_2m_max",
                            ds.get(f"{fc}_temperature_2m", pd.Series(dtype=float)))
            ds[f"{fc}_net_balance"] = rain.fillna(0) + irr.fillna(0) - et0.fillna(0)
            ds[f"{fc}_moist_x_et"]  = moist * et0.fillna(0)
            if fc_tmp.notna().any() and ws_t.notna().any():
                ds[f"{fc}_temp_delta"] = fc_tmp - ws_t
            else:
                ds[f"{fc}_temp_delta"] = np.nan

        enhanced[sensor] = ds
    return enhanced


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FISICA PENMAN-MONTEITH (FAO-56)
# ═══════════════════════════════════════════════════════════════════════════════

def water_stress(moisture: float, fc: float, wp: float, p: float) -> float:
    """Coefficiente di stress idrico Ks (FAO-56 eq. 84)."""
    TAW = fc - wp
    RAW = p * TAW
    threshold = fc - RAW
    if moisture >= threshold:
        return 1.0
    if moisture <= wp:
        return 0.0
    return (TAW - (fc - moisture)) / (TAW - RAW)


def percolation(moisture: float, fc: float, drain_coeff: float = 0.005) -> float:
    """Percolazione profonda (attiva solo quando moisture > FC)."""
    if moisture > fc:
        excess = moisture - fc
        return min(excess * drain_coeff * 10, excess)
    return 0.0


def build_kc_curve(install_date: pd.Timestamp, p: pd.Series) -> pd.Series:
    """Curva Kc giornaliera interpolata linearmente tra le 4 fasi FAO-56."""
    total = int(p["lini"] + p["ldev"] + p["lmid"] + p["llate"])
    ctrl  = [
        (0,                                         p["kc_ini"]),
        (p["lini"],                                 p["kc_ini"]),
        (p["lini"] + p["ldev"],                     p["kc_mid"]),
        (p["lini"] + p["ldev"] + p["lmid"],         p["kc_mid"]),
        (total,                                     p["kc_end"]),
    ]
    days    = np.arange(0, total + 1)
    kc_vals = np.interp(days, [c[0] for c in ctrl], [c[1] for c in ctrl])
    dates   = pd.date_range(install_date, periods=total + 1, freq="D")
    return pd.Series(kc_vals, index=dates, name="Kc")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CONFIGURAZIONE COLTURE & SENSORI
# ═══════════════════════════════════════════════════════════════════════════════

def load_colture(path: str) -> pd.DataFrame:
    """Legge colture.csv (ignora righe con #). Restituisce df indicizzato per coltura."""
    rows = [l for l in open(path, encoding="utf-8")
            if not l.strip().startswith("#")]
    df = pd.read_csv(StringIO("".join(rows)))
    df.columns = df.columns.str.strip()
    df["coltura"] = df["coltura"].str.strip().str.lower()
    df = df.set_index("coltura")
    for col in ["kc_ini", "kc_mid", "kc_end",
                "lini", "ldev", "lmid", "llate",
                "fc", "wp", "p", "mm_to_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[colture] caricate: {list(df.index)}")
    return df


def load_sensori(path: str) -> pd.DataFrame:
    """Legge sensori.csv; aggiunge colonna 'device' (es. EM-500-8).

    Colonna opzionale DATA SEMINA
    ─────────────────────────────
    Se presente e valorizzata, indica quando la coltura di quella riga è stata
    seminata/trapiantata. Serve a far partire da lì la curva dei coefficienti
    colturali Kc, che descrive lo sviluppo della PIANTA e non ha niente a che
    vedere con il giorno in cui è stato attaccato il sensore (che può essere
    posato prima o dopo la semina).

    La colonna 'origine_kc' che si ricava qui è quella da usare per la
    fenologia: è la semina dove c'è, l'installazione dove manca — quindi un
    CSV senza la colonna si comporta esattamente come prima.

    Attenzione a non confondere le due date a valle: 'installazione' continua a
    delimitare il periodo in cui il sensore misura davvero (è lì che i dati
    esistono), 'origine_kc' riguarda solo il calcolo dei Kc.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "SENSORE":            "sensore_num",
        "CAMPO":              "campo",
        "WS rispettiva":      "ws",
        "COLTURA":            "coltura",
        "DATA INSTALLAZIONE": "installazione",
        "DATA RIMOZIONE":     "rimozione",
        "DATA SEMINA":        "semina",
    })
    df["sensore_num"]   = pd.to_numeric(df["sensore_num"], errors="coerce")
    df["device"]        = "EM-500-" + df["sensore_num"].astype(int).astype(str)
    df["ws"]            = "WS-" + df["ws"].astype(str).str.strip()
    df["coltura"]       = df["coltura"].str.strip().str.lower()
    df["installazione"] = pd.to_datetime(df["installazione"], dayfirst=True, errors="coerce")
    df["rimozione"]     = pd.to_datetime(df["rimozione"],     dayfirst=True, errors="coerce")
    if "semina" in df.columns:
        df["semina"] = pd.to_datetime(df["semina"], dayfirst=True, errors="coerce")
    else:
        df["semina"] = pd.NaT
    df = df.dropna(subset=["installazione"])

    # origine della curva Kc: la semina dove nota, altrimenti l'installazione
    df["origine_kc"] = df["semina"].fillna(df["installazione"])

    n_semina = int(df["semina"].notna().sum())
    print(f"[sensori] {len(df)} periodi caricati da {path}")
    if n_semina:
        anticipi = (df["installazione"] - df["semina"]).dt.days.dropna()
        print(f"[sensori] {n_semina} periodi con DATA SEMINA esplicita "
              f"(scarto dall'installazione: da {int(anticipi.min())} a "
              f"{int(anticipi.max())} giorni; positivo = seminato PRIMA di posare "
              f"il sensore). Sugli altri {len(df) - n_semina} la curva Kc parte "
              f"dall'installazione, come prima.")
    else:
        print("[sensori] nessuna DATA SEMINA indicata: la curva Kc parte "
              "dall'installazione per tutti i periodi")
    return df.reset_index(drop=True)


def origine_curva_kc(srow: pd.Series) -> pd.Timestamp:
    """Data da cui far partire la curva Kc per una riga di sensori-corretti.csv.

    È la semina quando è indicata, l'installazione altrimenti. Sta in una
    funzione sola perché la stessa scelta va fatta identica nei tre punti che
    costruiscono la curva (penman.py, penman_ext.py e il bilancio a un passo
    qui in data_utils.py): se divergessero, la fisica a 7 orizzonti e quella a
    un passo userebbero fenologie diverse sugli stessi dati.
    """
    for campo in ("origine_kc", "semina"):
        if campo in srow.index:
            valore = pd.to_datetime(srow.get(campo), errors="coerce")
            if pd.notna(valore):
                return valore
    return pd.to_datetime(srow.get("installazione"), errors="coerce")


def get_coltura_params(colture_df: pd.DataFrame, coltura: str) -> pd.Series:
    """Ritorna la riga dei parametri per una coltura; lancia ValueError se assente."""
    key = coltura.strip().lower()
    if key not in colture_df.index:
        raise ValueError(f"Coltura '{coltura}' non trovata. Disponibili: {list(colture_df.index)}")
    return colture_df.loc[key]


def _parse_crop_override_map(text) -> dict:
    """Converte 'mais=38,sorgo=36' in {'mais': 38.0, 'sorgo': 36.0}."""
    if text is None or str(text).strip() == "":
        return {}
    out = {}
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Override non valido: '{part}'. Usa formato coltura=valore")
        crop, value = part.split("=", 1)
        out[crop.strip().lower()] = float(value.strip())
    return out


def apply_soil_overrides(colture_df, fc=None, wp=None, sat=None,
                         fc_by_crop=None, wp_by_crop=None,
                         sat_by_crop=None) -> pd.DataFrame:
    """Applica override di FC, WP, SAT a colture_df senza modificare il CSV originale."""
    df = colture_df.copy()
    fc_by_crop  = dict(fc_by_crop  or {})
    wp_by_crop  = dict(wp_by_crop  or {})
    sat_by_crop = dict(sat_by_crop or {})

    if "sat" not in df.columns:
        df["sat"] = np.nan

    changed = []
    if fc  is not None:
        df["fc"]  = float(fc);  changed.append(f"FC={fc:.3g}")
    if wp  is not None:
        df["wp"]  = float(wp);  changed.append(f"WP={wp:.3g}")
    if sat is not None:
        df["sat"] = float(sat); changed.append(f"SAT={sat:.3g}")

    for crop, v in fc_by_crop.items():
        if crop in df.index:
            df.loc[crop, "fc"] = float(v); changed.append(f"FC[{crop}]={v:.3g}")
    for crop, v in wp_by_crop.items():
        if crop in df.index:
            df.loc[crop, "wp"] = float(v); changed.append(f"WP[{crop}]={v:.3g}")
    for crop, v in sat_by_crop.items():
        if crop in df.index:
            df.loc[crop, "sat"] = float(v); changed.append(f"SAT[{crop}]={v:.3g}")

    if changed:
        print("Override suolo:", ", ".join(changed))
    else:
        print("Nessun override FC/WP/SAT: uso i valori di colture.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BILANCIO IDRICO CON CORREZIONE SAT
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_aliases(sub: pd.DataFrame) -> pd.DataFrame:
    """Mappa gli alias di colonne ET0 e pioggia ai nomi canonici attesi."""
    sub = sub.copy()
    if "et0_fc1" not in sub.columns or sub["et0_fc1"].isna().all():
        for alias in ET0_ALIASES[1:]:
            if alias in sub.columns and not sub[alias].isna().all():
                sub["et0_fc1"] = sub[alias]
                break
    if "rain_fc1" not in sub.columns or sub["rain_fc1"].isna().all():
        for alias in RAIN_ALIASES[1:]:
            if alias in sub.columns and not sub[alias].isna().all():
                sub["rain_fc1"] = sub[alias]
                break
    return sub


def run_water_balance_fc_corrected(ds, kc_curve, cp,
                                   alpha_rain=0.50, alpha_irr=0.90) -> pd.DataFrame:
    """
    Bilancio idrico FAO-56 D+1.
    FC usata per stress idrico e percolazione; SAT usata come tetto della predizione.
    """
    fc        = float(cp["fc"])
    wp        = float(cp["wp"])
    p_val     = float(cp["p"])
    mm_to_pct = float(cp["mm_to_pct"])
    sat_val   = cp.get("sat", np.nan)
    sat       = float(sat_val) if pd.notna(sat_val) else float(SAT_OVERRIDE or 100.0)

    df = ds.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["Kc"]   = df["date"].map(kc_curve)
    df["irrigation_planned"] = (
        df["irrigation_mm"].shift(-1).fillna(0)
        if "irrigation_mm" in df.columns else 0.0
    )
    for col in ["ETc_pct", "rain_eff_pct", "irr_eff_pct",
                "perc_pct", "Ks", "moisture_pred_t1", "sat"]:
        df[col] = np.nan

    for i in range(len(df) - 1):
        row   = df.iloc[i]
        m_now = row.get("moisture", np.nan)
        kc_d1 = row["Kc"]
        et0   = row.get("et0_fc1", np.nan)
        if pd.isna(m_now) or pd.isna(kc_d1) or pd.isna(et0):
            continue
        rain     = float(row.get("rain_fc1", 0) or 0)
        irr      = float(row.get("irrigation_planned", 0) or 0)
        ks       = water_stress(m_now, fc, wp, p_val)
        etc_pct  = ks * kc_d1 * et0 * mm_to_pct
        rain_pct = alpha_rain * rain * mm_to_pct
        irr_pct  = alpha_irr  * irr  * mm_to_pct
        perc_pct = percolation(m_now, fc)
        m_t1     = np.clip(m_now + rain_pct + irr_pct - etc_pct - perc_pct, wp, sat)
        df.at[i, "ETc_pct"]          = etc_pct
        df.at[i, "rain_eff_pct"]     = rain_pct
        df.at[i, "irr_eff_pct"]      = irr_pct
        df.at[i, "perc_pct"]         = perc_pct
        df.at[i, "Ks"]               = ks
        df.at[i, "sat"]              = sat
        df.at[i, "moisture_pred_t1"] = m_t1
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CODIFICA COLTURA
# ═══════════════════════════════════════════════════════════════════════════════

def build_crop_map(sensors_csv: str) -> pd.DataFrame:
    """Costruisce mappa {device, crop_type, date_start, date_end} da sensori.csv."""
    df = pd.read_csv(sensors_csv)
    df.columns = df.columns.str.strip()
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("sensore", "sensor"):              col_map[c] = "device"
        elif "coltura" in cl or "crop" in cl:        col_map[c] = "crop_type"
        elif "installazione" in cl or "start" in cl: col_map[c] = "date_start"
        elif "rimozione" in cl or "end" in cl:       col_map[c] = "date_end"
    df = df.rename(columns=col_map)
    for col in ("device", "crop_type", "date_start", "date_end"):
        if col not in df.columns:
            df[col] = np.nan
    df["device"]     = df["device"].astype(str).str.strip()
    df["crop_type"]  = df["crop_type"].astype(str).str.strip().str.upper()
    df["date_start"] = pd.to_datetime(df["date_start"], dayfirst=True, errors="coerce")
    df["date_end"]   = pd.to_datetime(df["date_end"],   dayfirst=True, errors="coerce")
    return df[["device", "crop_type", "date_start", "date_end"]].dropna(subset=["device"])


def add_crop_feature(datasets: dict, crop_map: pd.DataFrame) -> dict:
    """Aggiunge crop_type e crop_encoded (int) a ogni dataset sensore."""
    all_crops = sorted(crop_map["crop_type"].dropna().unique())
    crop2idx  = {c: i for i, c in enumerate(all_crops)}
    enriched  = {}
    for sensor, df in datasets.items():
        df   = df.copy()
        rows = crop_map[crop_map["device"] == str(sensor).strip()]
        if rows.empty:
            df["crop_type"] = "UNKNOWN"
            df["crop_encoded"] = -1
            enriched[sensor] = df
            continue
        df["_date_dt"] = pd.to_datetime(df["date"], errors="coerce")

        def _get_crop(d, _rows=rows):
            if pd.isna(d):
                return "UNKNOWN"
            mask   = ((_rows["date_start"].isna() | (_rows["date_start"] <= d)) &
                      (_rows["date_end"].isna()   | (_rows["date_end"]   >= d)))
            active = _rows[mask]
            if not active.empty:
                return active.iloc[0]["crop_type"]
            past = _rows[_rows["date_start"] <= d]
            if not past.empty:
                return past.sort_values("date_start").iloc[-1]["crop_type"]
            return "UNKNOWN"

        df["crop_type"]    = df["_date_dt"].apply(_get_crop)
        df["crop_encoded"] = df["crop_type"].map(crop2idx).fillna(-1).astype(int)
        df = df.drop(columns=["_date_dt"], errors="ignore")
        enriched[sensor] = df
    return enriched


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FLAG EVENTI & PIOGGE
# ═══════════════════════════════════════════════════════════════════════════════

def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge has_rain, has_irr, has_water_event."""
    df = df.copy()
    has_rain = (df["rain_median_mm"] > 0.1) if "rain_median_mm" in df.columns \
               else pd.Series(False, index=df.index)
    has_irr  = (df["irrigation_mm"]  > 0.1) if "irrigation_mm"  in df.columns \
               else pd.Series(False, index=df.index)
    df["has_rain"]        = has_rain.fillna(False)
    df["has_irr"]         = has_irr.fillna(False)
    df["has_water_event"] = df["has_rain"] | df["has_irr"]
    return df


def compute_daily_median_rain(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mediana giornaliera della pioggia tra le stazioni WS, dal contatore
    cumulativo rainfall_total. Stessa logica (e stesso motivo) del fix in
    pivot_ws: il contatore si azzera periodicamente, quindi il totale
    giornaliero è la somma dei soli INCREMENTI POSITIVI tra letture
    consecutive sull'intera serie ordinata per stazione (i reset, cioè i
    cali del contatore, contano 0 — non vengono sottratti), non un sum()
    diretto delle letture del giorno (che prima di questo fix contava più
    volte letture ripetute identiche, gonfiando enormemente il totale nei
    giorni con pioggia forte — es. 261mm calcolati per un giorno con in
    realtà circa 1mm di pioggia reale).
    """
    mask = (raw_df["device_name"].str.startswith("WS-", na=False) &
            (raw_df["measurement"] == "rainfall_total"))
    sub = raw_df[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=["date", "rain_median_mm"])

    daily_totals_per_station = []
    for ws_name, grp in sub.groupby("device_name"):
        grp = grp.sort_values("time")
        positive_increments = grp["value"].diff().clip(lower=0).fillna(0)
        daily_total = positive_increments.groupby(grp["date"]).sum()
        daily_totals_per_station.append(daily_total.rename(ws_name))

    daily_by_station = pd.concat(daily_totals_per_station, axis=1)
    med = daily_by_station.median(axis=1).reset_index()
    med.columns = ["date", "rain_median_mm"]
    return med


# ═══════════════════════════════════════════════════════════════════════════════
# 10. ARRICCHIMENTO PENMAN PER SENSORE
# ═══════════════════════════════════════════════════════════════════════════════

def run_penman_for_sensor(ds: pd.DataFrame, sensori_df: pd.DataFrame,
                          colture_df: pd.DataFrame,
                          alpha_rain: float, alpha_irr: float) -> pd.DataFrame:
    """Bilancio idrico FAO-56 per l'intera serie di un sensore."""
    device = str(ds["device"].iloc[0]) if "device" in ds.columns else "unknown"
    rows   = sensori_df[sensori_df["device"] == device]
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
        sub = _resolve_aliases(sub)
        # dalla semina se nota, dall'installazione altrimenti — stessa scelta
        # dei tre punti che costruiscono la curva Kc, centralizzata in
        # origine_curva_kc() perché non possano divergere
        kc_curve  = build_kc_curve(origine_curva_kc(srow), cp)
        wb_result = run_water_balance_fc_corrected(sub, kc_curve, cp,
                                                   alpha_rain, alpha_irr)
        result_pieces.append(wb_result)

    if not result_pieces:
        return ds

    wb = pd.concat(result_pieces).sort_values("date").drop_duplicates("date")
    rename = {
        "moisture_pred_t1":   "penman_pred_t1",
        "Kc":                 "penman_kc",
        "ETc_pct":            "penman_etc_pct",
        "Ks":                 "penman_ks",
        "rain_eff_pct":       "penman_rain_eff_pct",
        "irr_eff_pct":        "penman_irr_eff_pct",
        "perc_pct":           "penman_perc_pct",
        "irrigation_planned": "penman_irrigation_planned",
        "sat":                "penman_sat",
    }
    wb = wb.rename(columns={k: v for k, v in rename.items() if k in wb.columns})
    penman_cols = [c for c in rename.values() if c in wb.columns]
    if not penman_cols:
        return ds

    ds = ds.merge(wb[["date"] + penman_cols], on="date", how="left")
    if "penman_pred_t1" in ds.columns and "penman_etc_pct" in ds.columns:
        ds["penman_pred_t2"] = (ds["penman_pred_t1"] - ds["penman_etc_pct"]).clip(lower=0)
    if "penman_pred_t1" in ds.columns and "moisture" in ds.columns:
        ds["penman_residual_lag1"] = ds["moisture"] - ds["penman_pred_t1"].shift(1)
    return ds


def enrich_with_penman(datasets: dict, sensori_df: pd.DataFrame,
                       colture_df: pd.DataFrame,
                       alpha_rain: float, alpha_irr: float) -> dict:
    """Arricchisce tutti i dataset con le feature Penman-Monteith."""
    enriched = {}
    for sensor, ds in datasets.items():
        ds2 = ds.copy()
        if "device" not in ds2.columns:
            ds2["device"] = sensor
        ds2 = run_penman_for_sensor(ds2, sensori_df, colture_df,
                                    alpha_rain, alpha_irr)
        enriched[sensor] = ds2
    return enriched


# ═══════════════════════════════════════════════════════════════════════════════
# 11. SPLIT TRAIN/TEST
# ═══════════════════════════════════════════════════════════════════════════════

def sensor_7030(df: pd.DataFrame, train_frac: float = 0.70):
    """Split cronologico 70/30 per sensore, poi pooled."""
    trains, tests = [], []
    for _, g in df.groupby("device", sort=False):
        g   = g.sort_values("date")
        cut = max(1, int(len(g) * train_frac))
        trains.append(g.iloc[:cut])
        tests.append(g.iloc[cut:])
    tr = (pd.concat(trains).sort_values("date").reset_index(drop=True)
          if trains else pd.DataFrame())
    te = (pd.concat(tests).sort_values("date").reset_index(drop=True)
          if tests else pd.DataFrame())
    return tr, te


def sensor_train_val_test(df: pd.DataFrame, train_frac: float = 0.70,
                          val_frac: float = 0.15):
    """
    Split cronologico train/validation/test per sensore, poi pooled.
    Completamente deterministico: nessun campionamento casuale, solo un
    taglio per data (crescente) entro ciascun sensore. test_frac = 1 -
    train_frac - val_frac.
    """
    trains, vals, tests = [], [], []
    for _, g in df.groupby("device", sort=False):
        g = g.sort_values("date")
        n = len(g)
        n_train = max(1, int(n * train_frac))
        n_val   = max(1, int(n * val_frac))
        n_train = min(n_train, max(1, n - 2))
        n_val   = min(n_val, max(1, n - n_train - 1))
        trains.append(g.iloc[:n_train])
        vals.append(g.iloc[n_train:n_train + n_val])
        tests.append(g.iloc[n_train + n_val:])

    def _cat(parts):
        return (pd.concat(parts).sort_values("date").reset_index(drop=True)
                if parts else pd.DataFrame())

    return _cat(trains), _cat(vals), _cat(tests)


def add_common_pm_split(datasets_penman: dict,
                        min_valid: int = MIN_SAMPLES,
                        split_col: str = COMMON_PM_SPLIT_COL) -> dict:
    """
    Split 70/30 basato sui giorni validi Penman + target_t1.
    Garantisce confrontabilità diretta tra Penman puro e ML.
    """
    out, skipped = {}, []
    for sensor, ds in sorted(datasets_penman.items()):
        df = ds.copy().sort_values("date").reset_index(drop=True)
        df[split_col] = np.nan
        df["device"]  = sensor

        if "target_t1" not in df.columns:
            if "moisture" in df.columns:
                df["target_t1"] = df["moisture"].shift(-1)
            else:
                skipped.append((sensor, "no moisture/target_t1"))
                out[sensor] = df
                continue

        if "penman_pred_t1" not in df.columns:
            skipped.append((sensor, "no penman_pred_t1"))
            out[sensor] = df
            continue

        valid   = df["target_t1"].notna() & df["penman_pred_t1"].notna()
        n_valid = int(valid.sum())
        if n_valid < min_valid:
            skipped.append((sensor, f"PM-valid days too few ({n_valid})"))
            out[sensor] = df
            continue

        valid_idx = df.index[valid].to_numpy()
        n_train   = max(1, min(int(np.floor(0.70 * len(valid_idx))), len(valid_idx) - 1))
        df.loc[valid_idx[:n_train], split_col] = "train"
        df.loc[valid_idx[n_train:], split_col] = "test"
        out[sensor] = df

    n_common = sum(split_col in ds.columns and ds[split_col].notna().any()
                   for ds in out.values())
    n_test   = sum(int((ds[split_col] == "test").sum())
                   for ds in out.values() if split_col in ds.columns)
    print(f"  Split PM-valid: {n_common}/{len(out)} sensori, {n_test} righe test")
    if skipped:
        print(f"  [split] {len(skipped)} sensori esclusi:")
        for s, r in skipped:
            print(f"    {s}: {r}")
    return out


def _split_common_7030(df: pd.DataFrame,
                       split_col: str = COMMON_PM_SPLIT_COL):
    """Usa lo split PM-valid se presente, altrimenti ricade su sensor_7030."""
    if split_col in df.columns and df[split_col].notna().any():
        return df[df[split_col] == "train"].copy(), df[df[split_col] == "test"].copy()
    return sensor_7030(df)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. METRICHE
# ═══════════════════════════════════════════════════════════════════════════════

def _mape(y_true, y_pred, min_abs_true=1.0):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(y_true) >= float(min_abs_true)
    if mask.sum() < 3:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _smape(y_true, y_pred, min_denom=1.0):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask  = denom >= float(min_denom)
    if mask.sum() < 3:
        return np.nan
    return float(np.mean(200.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def _wape(y_true, y_pred, min_sum=1.0):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = float(np.sum(np.abs(y_true)))
    if len(y_true) < 3 or denom < float(min_sum):
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def _nmae(y_true, y_pred):
    rng = float(np.nanmax(y_true) - np.nanmin(y_true))
    return float(mean_absolute_error(y_true, y_pred) / rng) if rng >= 0.1 else np.nan


def _nrmse(y_true, y_pred):
    rng = float(np.nanmax(y_true) - np.nanmin(y_true))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)) / rng) if rng >= 0.1 else np.nan


def _safe_pearson(y_true, y_pred):
    if len(y_true) < 5 or np.nanstd(y_true) < 1e-12 or np.nanstd(y_pred) < 1e-12:
        return np.nan
    try:
        return float(pearsonr(y_true, y_pred)[0])
    except Exception:
        return np.nan


def _metric_dict(y_true, y_pred, prefix="") -> dict:
    """Calcola MAE, RMSE, R², r, Bias, MAPE, sMAPE, WAPE, nMAE, nRMSE."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 5:
        return {
            f"{prefix}MAE": np.nan, f"{prefix}RMSE": np.nan,
            f"{prefix}R2": np.nan,  f"{prefix}r": np.nan,
            f"{prefix}Bias": np.nan, f"{prefix}MAPE": np.nan,
            f"{prefix}sMAPE": np.nan, f"{prefix}WAPE": np.nan,
            f"{prefix}nMAE": np.nan, f"{prefix}nRMSE": np.nan,
            f"{prefix}range_y": np.nan, f"{prefix}n": int(len(yt)),
        }
    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    return {
        f"{prefix}MAE":     round(float(mae), 4),
        f"{prefix}RMSE":    round(float(rmse), 4),
        f"{prefix}R2":      round(float(r2_score(yt, yp)), 4),
        f"{prefix}r":       round(_safe_pearson(yt, yp), 4),
        f"{prefix}Bias":    round(float((yp - yt).mean()), 4),
        f"{prefix}MAPE":    round(_mape(yt, yp), 2),
        f"{prefix}sMAPE":   round(_smape(yt, yp), 2),
        f"{prefix}WAPE":    round(_wape(yt, yp), 2),
        f"{prefix}nMAE":    round(_nmae(yt, yp), 4),
        f"{prefix}nRMSE":   round(_nrmse(yt, yp), 4),
        f"{prefix}range_y": round(float(np.nanmax(yt) - np.nanmin(yt)), 4),
        f"{prefix}n":       int(len(yt)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 13. ARIMA & NAIVE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_arima(train_y: np.ndarray, order=(1, 1, 1)):
    """Fit ARIMA; fallback a AR(2)-OLS se statsmodels non disponibile."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        return ARIMA(train_y, order=order).fit(), "statsmodels"
    except Exception:
        pass

    class _AR2:
        def __init__(self, coef, intercept, p, last_vals):
            self.coef, self.intercept, self.p = coef, intercept, p
            self.last_vals = list(last_vals[-p:])

        def forecast(self, steps=1):
            preds, hist = [], list(self.last_vals)
            for _ in range(steps):
                x     = np.array(hist[-self.p:][::-1])
                y_hat = self.intercept + x @ self.coef
                preds.append(float(y_hat))
                hist.append(float(y_hat))
            return np.array(preds)

    p = min(2, len(train_y) - 1)
    X = np.column_stack([train_y[p-i-1: len(train_y)-i-1] for i in range(p)])
    y = train_y[p:]
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return _AR2(coef[1:], coef[0], p, train_y), "AR(2)-OLS"


def _arima_walkforward(model, backend, test_y, train_y, order, steps=1):
    """Walk-forward ARIMA: refit ad ogni passo per valutazione onesta."""
    preds, history = [], list(train_y.copy())
    for obs in test_y:
        if backend == "statsmodels":
            from statsmodels.tsa.arima.model import ARIMA
            try:
                m  = ARIMA(history, order=order).fit()
                fc = m.forecast(steps=steps)[-1]
            except Exception:
                fc = history[-1]
        else:
            m, _ = _fit_arima(np.array(history), order)
            fc   = float(m.forecast(steps=steps)[-1])
        preds.append(fc)
        history.append(float(obs))
    return np.array(preds)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. TUNING ML & IMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

CV = TimeSeriesSplit(n_splits=3)

GRID_RIDGE = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
GRID_RF    = {"n_estimators": [100, 200], "max_depth": [4, 8, None],
              "min_samples_leaf": [2, 5]}
GRID_LGBM  = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7],
              "learning_rate": [0.05, 0.10], "num_leaves": [31, 63]}


def _impute(X_tr: np.ndarray, X_te: np.ndarray):
    imp  = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)
    return X_tr, X_te


def _tune(model_cls, grid, X_tr, y_tr, sample_weight=None, **kwargs):
    """GridSearchCV con sample_weight opzionale."""
    gs = GridSearchCV(model_cls(**kwargs), grid, cv=CV,
                      scoring="neg_mean_absolute_error", n_jobs=-1)
    if sample_weight is not None:
        gs.fit(X_tr, y_tr, sample_weight=sample_weight)
    else:
        gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_


def _tune_val(model_cls, grid, X_tr, y_tr, X_val, y_val,
              sample_weight=None, **kwargs):
    """
    Grid search deterministico via validation set esplicito (nessuna CV
    interna, nessuna divisione casuale): per ogni combinazione della griglia
    (ordine fisso di ParameterGrid) allena su train e valuta la MAE su
    validation, tiene la combinazione migliore. A parità di dati e seed dei
    modelli (random_state fisso) il risultato è sempre lo stesso.
    Ritorna (best_model, best_params, best_val_mae). best_model è allenato
    solo su X_tr/y_tr (validation usata solo per scegliere gli iperparametri).
    """
    best_score, best_params, best_model = np.inf, None, None
    for params in ParameterGrid(grid):
        model = model_cls(**params, **kwargs)
        if sample_weight is not None:
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
        else:
            model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        score = float(mean_absolute_error(y_val, pred))
        if score < best_score:
            best_score, best_params, best_model = score, params, model
    return best_model, best_params, best_score


def _make_sample_weights(df_train: pd.DataFrame, event_weight: float) -> np.ndarray:
    """Peso event_weight per i giorni pioggia/irrigazione, 1.0 per gli altri."""
    has_rain = (df_train["has_rain"].fillna(False).values.astype(bool)
                if "has_rain" in df_train.columns
                else np.zeros(len(df_train), dtype=bool))
    has_irr  = (df_train["has_irr"].fillna(False).values.astype(bool)
                if "has_irr" in df_train.columns
                else np.zeros(len(df_train), dtype=bool))
    return np.where(has_rain | has_irr, float(event_weight), 1.0)
