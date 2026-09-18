"""
load_clean_dataset.py
─────────────────────────────────────────────────────────────────────────────
Adatta data_cleaning/dataset_pulito.csv (formato long: _time,_value,_field,
_measurement,device_name,...) allo schema che utils.py (load_dump) si aspetta:
    time, value, device_name, measurement, field, lead_days, lead_hours, date

Applica in più le regole di pulizia sui valori:
  • rainfall_counter          → scartata interamente
  • rainfall_total >= 600     → sentinel hardware
  • moisture > 100            → fisicamente impossibile
  • humidity > 100            → fisicamente impossibile
  • temperatura WS == -0.1   → sentinel "nessuna lettura" delle WS

Ricostruisce inoltre il forecast Open-Meteo (misura
openmeteo_previous_runs_hourly) per lead time da 1 a 7 giorni: ogni blocco di
18 tabelle nel dump rappresenta un lead time fisso (target_time - reference_time
costante), il blocco viene individuato cercando la tabella che inizia con il
campo "apparent_temperature" (primo campo di ogni blocco completo).
"""

import numpy as np
import pandas as pd

FRMPAYLOAD_PREFIX = "device_frmpayload_data_"
LOCAL_TZ = "Europe/Rome"

MEAS_RAIN_COUNTER = "device_frmpayload_data_rainfall_counter"
MEAS_RAIN_TOTAL   = "device_frmpayload_data_rainfall_total"
MEAS_MOIST        = "device_frmpayload_data_moisture"
MEAS_HUM          = "device_frmpayload_data_humidity"
MEAS_TEMP         = "device_frmpayload_data_temperature"
RAIN_TOTAL_SENTINEL = 600.0

FORECAST_HOURLY_MEAS = "openmeteo_previous_runs_hourly"
BLOCK_SIZE = 18          # campi per blocco completo (con rain/temp/ecc.)
BLOCK_START_FIELD = "apparent_temperature"
LEADS_WANTED = (1, 2, 3, 4, 5, 6, 7)

_META_FIELDS = {
    "target_time_iso", "reference_time_iso",
    "target_time_unix", "reference_time_unix",
}


def _clean_em_ws_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Filtra un chunk grezzo (già con _time parsato e _valnum numerico) alle
    sole righe EM/WS valide, applicando le regole di pulizia richieste."""
    is_ws = chunk["device_name"].str.startswith("WS-", na=False)
    is_em = chunk["device_name"].str.match(r"^EM-500-\d+$", na=False)

    bad = (
        (chunk["_measurement"] == MEAS_RAIN_COUNTER) |
        ((chunk["_measurement"] == MEAS_RAIN_TOTAL) & (chunk["_valnum"] >= RAIN_TOTAL_SENTINEL)) |
        ((chunk["_measurement"] == MEAS_MOIST) & (chunk["_valnum"] > 100)) |
        ((chunk["_measurement"] == MEAS_HUM) & (chunk["_valnum"] > 100)) |
        ((chunk["_measurement"] == MEAS_TEMP) & is_ws & (chunk["_valnum"] == -0.1))
    )
    keep = (is_em | is_ws) & chunk["_valnum"].notna() & ~bad
    return chunk.loc[keep]


def _to_utils_schema(sub: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=sub.index)
    out["time"]        = sub["_time"]
    out["value"]       = sub["_valnum"]
    out["device_name"] = sub["device_name"]
    out["measurement"] = sub["_measurement"].str.replace(FRMPAYLOAD_PREFIX, "", regex=False)
    out["field"]       = sub["_field"]
    return out.reset_index(drop=True)


def _reconstruct_forecast(openmeteo_hourly_raw: pd.DataFrame,
                          leads=LEADS_WANTED) -> pd.DataFrame:
    """openmeteo_hourly_raw: colonne table,_time,_value,_field (solo measurement hourly).
    Ritorna un DataFrame long nello schema utils (time,value,device_name='',
    measurement,field,lead_days,lead_hours), uno per ciascun lead richiesto."""
    empty = pd.DataFrame(columns=["time", "value", "device_name", "measurement",
                                  "field", "lead_days", "lead_hours"])
    if openmeteo_hourly_raw.empty:
        return empty

    field_per_table = openmeteo_hourly_raw.groupby("table")["_field"].first()
    block_starts = sorted(field_per_table[field_per_table == BLOCK_START_FIELD].index.tolist())

    frames = []
    for b in block_starts:
        block_tables = list(range(b, b + BLOCK_SIZE))
        sub = openmeteo_hourly_raw[openmeteo_hourly_raw["table"].isin(block_tables)]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="_time", columns="_field", values="_value",
                              aggfunc="first").reset_index()
        if "target_time_iso" not in piv.columns or "reference_time_iso" not in piv.columns:
            continue
        target = pd.to_datetime(piv["target_time_iso"], errors="coerce", utc=True)
        reference = pd.to_datetime(piv["reference_time_iso"], errors="coerce", utc=True)
        valid = target.notna() & reference.notna()
        if valid.sum() == 0:
            continue
        delta_days = round(float((target[valid] - reference[valid]).dt.total_seconds().median() / 86400))
        if delta_days not in leads:
            continue

        value_fields = [c for c in piv.columns
                        if c not in _META_FIELDS and c != "_time"]
        piv["_target_time"] = target
        long = piv.loc[valid, ["_target_time"] + value_fields].copy()
        long = long.rename(columns={"_target_time": "time"})
        long = long.melt(id_vars=["time"], value_vars=value_fields,
                         var_name="field", value_name="value")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["value"])
        long["lead_days"] = float(delta_days)
        long["lead_hours"] = float(delta_days) * 24.0
        long["device_name"] = ""
        long["measurement"] = FORECAST_HOURLY_MEAS
        frames.append(long[["time", "value", "device_name", "measurement",
                            "field", "lead_days", "lead_hours"]])
        print(f"  [forecast] blocco tabelle {b}-{b+BLOCK_SIZE-1}: lead={delta_days}gg, "
              f"{long['time'].nunique()} timestamp")

    if not frames:
        return empty
    return pd.concat(frames, ignore_index=True)


def load_irrigation_wide(path: str) -> pd.DataFrame:
    """Adatta irrigazione.csv (formato largo: SENSORE,CAMPO,DATA,MM,DATA,MM,...
    con fino a 4 eventi per sensore in colonne ripetute) al formato lungo
    atteso da build_sensor_dataset: colonne device_name, date, irrigation_mm.
    Righe senza data valida (es. 'NON FATTE', 'NOT INSTALLED', vuote) vengono
    scartate."""
    raw = pd.read_csv(path, header=1, encoding="utf-8-sig")
    raw.columns = [str(c).strip() for c in raw.columns]

    date_cols = [c for c in raw.columns if c == "DATA"]
    # pandas rinomina le colonne duplicate "DATA" in "DATA", "DATA.1", ...
    data_cols = [c for c in raw.columns if c == "DATA" or c.startswith("DATA.")]
    mm_cols   = [c for c in raw.columns if c == "MM"   or c.startswith("MM.")]

    events = []
    for d_col, m_col in zip(data_cols, mm_cols):
        sub = raw[["SENSORE", d_col, m_col]].copy()
        sub.columns = ["SENSORE", "data", "mm"]
        events.append(sub)
    long = pd.concat(events, ignore_index=True)

    long["SENSORE"] = pd.to_numeric(long["SENSORE"], errors="coerce")
    long = long.dropna(subset=["SENSORE"])
    long["device_name"] = "EM-500-" + long["SENSORE"].astype(int).astype(str)
    long["date"] = pd.to_datetime(long["data"], dayfirst=True, format="mixed", errors="coerce").dt.normalize()
    long["irrigation_mm"] = pd.to_numeric(long["mm"], errors="coerce")
    long = long.dropna(subset=["date", "irrigation_mm"])
    return long[["device_name", "date", "irrigation_mm"]].reset_index(drop=True)


def load_and_clean(path: str, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Legge dataset_pulito.csv a chunk, pulisce EM/WS e ricostruisce il
    forecast, restituendo un DataFrame compatibile con utils.pivot_em500 /
    utils.pivot_ws / utils.pivot_forecast(_hourly)."""
    em_ws_chunks = []
    openmeteo_hourly_chunks = []

    print(f"[load_clean_dataset] leggo {path} ...")
    reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
    for i, chunk in enumerate(reader):
        chunk["_time"] = pd.to_datetime(chunk["_time"], utc=True, format="ISO8601", errors="coerce")
        chunk = chunk.dropna(subset=["_time"]).copy()
        if chunk.empty:
            continue
        chunk["_valnum"] = pd.to_numeric(chunk["_value"], errors="coerce")
        chunk["device_name"] = chunk["device_name"].fillna("").astype(str)

        om_mask = chunk["_measurement"] == FORECAST_HOURLY_MEAS
        if om_mask.any():
            openmeteo_hourly_chunks.append(chunk.loc[om_mask, ["table", "_time", "_value", "_field"]])

        cleaned = _clean_em_ws_chunk(chunk)
        if not cleaned.empty:
            em_ws_chunks.append(_to_utils_schema(cleaned))

        print(f"  chunk {i+1}: {len(chunk):,} righe processate", end="\r")
    print()

    em_ws_df = pd.concat(em_ws_chunks, ignore_index=True) if em_ws_chunks else pd.DataFrame(
        columns=["time", "value", "device_name", "measurement", "field"])
    print(f"[load_clean_dataset] righe EM/WS pulite: {len(em_ws_df):,}")

    om_raw = (pd.concat(openmeteo_hourly_chunks, ignore_index=True)
              if openmeteo_hourly_chunks else pd.DataFrame(columns=["table", "_time", "_value", "_field"]))
    fc_df = _reconstruct_forecast(om_raw, leads=LEADS_WANTED)
    print(f"[load_clean_dataset] righe forecast ricostruite: {len(fc_df):,}")

    df = pd.concat([em_ws_df, fc_df], ignore_index=True)
    df["lead_days"]  = pd.to_numeric(df.get("lead_days", np.nan),  errors="coerce")
    df["lead_hours"] = pd.to_numeric(df.get("lead_hours", np.nan), errors="coerce")
    df["field"]       = df["field"].fillna("").astype(str)
    df["measurement"] = df["measurement"].fillna("").astype(str)
    df["timezone"]    = LOCAL_TZ
    df["date"] = df["time"].dt.normalize().dt.tz_localize(None)
    print(f"[load_clean_dataset] totale righe: {len(df):,}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"..\data_cleaning\dataset_pulito.csv")
    args = parser.parse_args()
    out = load_and_clean(args.data)
    print(out.head())
    print(out["measurement"].value_counts())
