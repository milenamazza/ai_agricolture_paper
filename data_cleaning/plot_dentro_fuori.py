"""
plot_dentro_fuori.py
─────────────────────────────────────────────────────────────────────────────
Un'unica lettura (streaming a chunk) del dump grezzo produce:

  1. dataset_pulito.csv
     - EM (sensori di suolo): SOLO le righe nei giorni "dentro" (compresi in
       un periodo DATA INSTALLAZIONE->RIMOZIONE di sensori-corretti.csv E nel
       range GT_START/GT_END di inizio.csv)
     - WS (stazioni meteo): TUTTE le righe, tutte le misure, senza filtri
     - openmeteo (previous_runs_hourly + previous_runs_daily): TUTTE le righe,
       senza filtri

  2. plots_temperatura/<sensore>_temperatura.png
     - un solo grafico per sensore, su TUTTO il periodo reale di trasmissione
       del sensore (non limitato al range di inizio.csv)
     - due pannelli impilati: sopra scatter temperatura sensore, sotto
       scatter temperatura WS
     - banda verde continua sui periodi "dentro" (stessi intervalli usati per
       il filtro del dataset)

  3. plots_moisture_pioggia/<sensore>_moisture_pioggia.png
     - un solo grafico per sensore, sullo stesso tipo di periodo (basato sui
       timestamp di umidità)
     - due pannelli impilati: sopra scatter umidità suolo (con banda verde),
       sotto pioggia giornaliera osservata (WS) vs prevista da openmeteo il
       giorno prima (lead time 1 giorno)

COPIA IN AI_agricolture_paper — differenze dall'originale
─────────────────────────────────────────────────────────────────────────────
La logica di pulizia è identica a AI_agricolture/data_cleaning/. Cambiano solo:

  - il REGISTRO usato: quello aggiornato di questa cartella, che contiene i
    periodi 2026 dei sensori 3 e 4 (malva, escolzia), 8 (melissa) e 20
    (Boschi). La pulizia del 22 luglio 2026 era girata con un registro senza
    quei periodi, e siccome le righe EM "fuori" vengono buttate, i loro dati
    2026 non erano finiti nel dataset pur essendo presenti nel dump grezzo.
    La colonna DATA SEMINA del registro qui è ignorata: conta solo per i
    coefficienti Kc a valle, non per decidere quali misure tenere.

  - i DEFAULT dei percorsi: sono relativi alla cartella di QUESTO script, non
    a quella da cui si lancia il comando. Lanciandolo da un'altra cartella,
    con i default relativi alla cwd, leggerebbe un altro registro o
    scriverebbe il dataset altrove.

  - un PRESIDIO sull'output: lo script cancella --out-csv all'avvio, quindi se
    il percorso esce da AI_agricolture_paper/ si ferma con un errore. Serve a
    non poter cancellare per sbaglio il dataset_pulito.csv della cartella
    originale, che le pipeline possono star leggendo.

Il dump grezzo resta in db_dump/, fuori da questa cartella: è un input della
sola pulizia, non delle pipeline.

Utilizzo (da qualunque cartella, i default bastano):
    python -u plot_dentro_fuori.py
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))       # AI_agricolture_paper/data_cleaning
ROOT_PAPER = os.path.dirname(HERE)                      # AI_agricolture_paper

TODAY = pd.Timestamp.now(tz="UTC").floor("D")
TEMP = "device_frmpayload_data_temperature"
MOIST = "device_frmpayload_data_moisture"
RAIN_TOTAL = "device_frmpayload_data_rainfall_total"
OPENMETEO_MEASUREMENTS = ["openmeteo_previous_runs_hourly", "openmeteo_previous_runs_daily"]
OPENMETEO_HOURLY = "openmeteo_previous_runs_hourly"
OPENMETEO_PREVDAY1_TABLES = (70, 87)   # lead time = 1 giorno (target - reference = 1 day)

C_IN       = "#2A7A3B"   # verde  - banda periodo di attività
C_SOIL     = "#1A3F6F"   # blu scuro - sensore (temperatura / umidità)
C_AIR      = "#D4660A"   # arancio   - WS (temperatura)
C_RAIN_OBS = "#2E6DB4"   # blu   - pioggia osservata
C_RAIN_FC  = "#D4660A"   # arancio - pioggia prevista (giorno prima)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#EEEEEE",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
})


# ══════════════════════════════════════════════════════════════════════════
# REGISTRO + GROUND TRUTH -> INTERVALLI "DENTRO"
# ══════════════════════════════════════════════════════════════════════════

def load_registry(path):
    reg = pd.read_csv(path)
    reg["SENSORE"] = reg["SENSORE"].astype(int)
    reg["DATA INSTALLAZIONE"] = pd.to_datetime(
        reg["DATA INSTALLAZIONE"], dayfirst=True, errors="coerce"
    ).dt.tz_localize("UTC")
    reg["DATA RIMOZIONE"] = pd.to_datetime(
        reg["DATA RIMOZIONE"], dayfirst=True, errors="coerce"
    ).dt.tz_localize("UTC")
    return reg


def load_gt(path):
    gt = pd.read_csv(path)
    gt["SENSORE"] = gt["SENSORE"].astype(int)
    gt["GT_START"] = pd.to_datetime(gt["GT_START"], dayfirst=True, errors="coerce").dt.tz_localize("UTC")
    gt["GT_END"] = pd.to_datetime(gt["GT_END"], dayfirst=True, errors="coerce").dt.tz_localize("UTC")
    return gt.set_index("SENSORE")


def build_dentro_map(reg, gt):
    """
    Ritorna:
      dentro_map[sensore] = {"gt_start", "gt_end", "dentro": [(s,e), ...]}
      periods_map[sensore] = [(start, end, campo, coltura), ...]  (grezzi, non ritagliati)
    """
    dentro_map = {}
    periods_map = {}

    for sensore, group in reg.groupby("SENSORE"):
        if sensore not in gt.index or pd.isna(gt.loc[sensore, "GT_START"]):
            continue
        gt_s = gt.loc[sensore, "GT_START"]
        gt_e = gt.loc[sensore, "GT_END"]
        if pd.isna(gt_e):
            gt_e = TODAY

        ivs = []
        raw_periods = []
        for _, row in group.iterrows():
            ps = row["DATA INSTALLAZIONE"]
            if pd.isna(ps):
                continue
            pe = row["DATA RIMOZIONE"] if pd.notna(row["DATA RIMOZIONE"]) else TODAY
            raw_periods.append((ps, pe, row.get("CAMPO", ""), row.get("COLTURA", "")))
            s, e = max(ps, gt_s), min(pe, gt_e)
            if s <= e:
                ivs.append((s, e))

        dentro_map[sensore] = {"gt_start": gt_s, "gt_end": gt_e, "dentro": ivs}
        periods_map[sensore] = raw_periods

    return dentro_map, periods_map


# ══════════════════════════════════════════════════════════════════════════
# STREAMING SUL DUMP GREZZO (un solo passaggio)
# ══════════════════════════════════════════════════════════════════════════

def stream_process(data_path, dentro_map, out_csv, chunksize=1_000_000):
    dev_ids = list(dentro_map.keys())
    orig_cols = None

    first_write = True
    total_em = 0
    total_ws = 0
    total_om = 0
    total_rows = 0

    em_temp_chunks = []
    ws_temp_chunks = []
    em_moist_chunks = []
    ws_rain_chunks = []
    openmeteo_chunks = []   # solo tabelle 70-87 (lead time 1 giorno), per il grafico

    reader = pd.read_csv(data_path, chunksize=chunksize, low_memory=False)
    for chunk_i, chunk in enumerate(reader):
        if orig_cols is None:
            orig_cols = list(chunk.columns)
        total_rows += len(chunk)
        chunk["_time"] = pd.to_datetime(chunk["_time"], utc=True, format="ISO8601", errors="coerce")
        chunk = chunk.dropna(subset=["_time"]).copy()
        if chunk.empty:
            continue
        chunk["_valnum"] = pd.to_numeric(chunk["_value"], errors="coerce")

        # ── openmeteo: righe complete (device_name vuoto, va preso prima) ───
        om_all_mask = chunk["_measurement"].isin(OPENMETEO_MEASUREMENTS)
        if om_all_mask.any():
            om_all = chunk.loc[om_all_mask, orig_cols]
            om_all.to_csv(out_csv, mode="a", header=first_write, index=False)
            first_write = False
            total_om += len(om_all)

            # sottoinsieme tabelle 70-87 (lead time 1 giorno) per ricostruire la previsione
            om_mask = om_all_mask & chunk["table"].between(*OPENMETEO_PREVDAY1_TABLES)
            if om_mask.any():
                openmeteo_chunks.append(chunk.loc[om_mask, ["table", "_time", "_value", "_field"]])

        # ── da qui in poi solo righe con device_name valido (EM / WS) ───────
        dchunk = chunk.dropna(subset=["device_name"]).copy()
        if dchunk.empty:
            print(f"  chunk {chunk_i+1}: {total_rows:,} righe lette (EM dentro={total_em:,}, WS={total_ws:,}, openmeteo={total_om:,})", end="\r")
            continue
        dchunk["device_name"] = dchunk["device_name"].astype(str)
        dchunk["_day"] = dchunk["_time"].dt.floor("D")

        is_em = dchunk["device_name"].str.match(r"^EM-500-\d+$", na=False)
        is_ws = dchunk["device_name"].str.startswith("WS", na=False)
        is_temp = dchunk["_measurement"] == TEMP
        is_moist = dchunk["_measurement"] == MOIST
        is_rain = dchunk["_measurement"] == RAIN_TOTAL

        # ── WS: righe complete, senza filtri ────────────────────────────────
        ws_all = dchunk.loc[is_ws, orig_cols]
        if not ws_all.empty:
            ws_all.to_csv(out_csv, mode="a", header=first_write, index=False)
            first_write = False
            total_ws += len(ws_all)

        # ── punti grezzi temperatura sensore (EM), per il grafico ──────────
        sub = dchunk.loc[is_em & is_temp & dchunk["_valnum"].notna()]
        if not sub.empty:
            dev = sub["device_name"].str.extract(r"EM-500-(\d+)")[0].astype(int)
            em_temp_chunks.append(pd.DataFrame({"dev": dev, "_time": sub["_time"], "_value": sub["_valnum"]}).reset_index(drop=True))

        # ── punti grezzi temperatura WS, scarta sentinel -0.1, per il grafico
        sub = dchunk.loc[is_ws & is_temp & dchunk["_valnum"].notna() & (dchunk["_valnum"] != -0.1)]
        if not sub.empty:
            ws_temp_chunks.append(sub[["_time", "_valnum"]].rename(columns={"_valnum": "_value"}))

        # ── punti grezzi umidità suolo (EM), scarta >100 (sentinel) ────────
        sub = dchunk.loc[is_em & is_moist & dchunk["_valnum"].notna() & (dchunk["_valnum"] <= 100)]
        if not sub.empty:
            dev = sub["device_name"].str.extract(r"EM-500-(\d+)")[0].astype(int)
            em_moist_chunks.append(pd.DataFrame({"dev": dev, "_time": sub["_time"], "_value": sub["_valnum"]}).reset_index(drop=True))

        # ── pioggia cumulativa WS (rainfall_total), scarta >=600 (sentinel) ─
        sub = dchunk.loc[is_ws & is_rain & dchunk["_valnum"].notna() & (dchunk["_valnum"] < 600)]
        if not sub.empty:
            ws_rain_chunks.append(sub[["device_name", "_time", "_valnum"]].rename(columns={"_valnum": "_value"}))

        # ── EM: filtro "dentro" da salvare (tutte le misure dei sensori EM) ─
        em_rows = dchunk.loc[is_em].copy()
        if not em_rows.empty:
            em_rows["dev"] = em_rows["device_name"].str.extract(r"EM-500-(\d+)")[0].astype(int)
            keep_mask = pd.Series(False, index=em_rows.index)
            for dev in dev_ids:
                dmask = em_rows["dev"] == dev
                if not dmask.any():
                    continue
                for s, e in dentro_map[dev]["dentro"]:
                    keep_mask |= dmask & (em_rows["_day"] >= s) & (em_rows["_day"] <= e)

            kept = em_rows.loc[keep_mask, orig_cols]
            if not kept.empty:
                kept.to_csv(out_csv, mode="a", header=first_write, index=False)
                first_write = False
                total_em += len(kept)

        print(f"  chunk {chunk_i+1}: {total_rows:,} righe lette (EM dentro={total_em:,}, WS={total_ws:,}, openmeteo={total_om:,})", end="\r")

    print()

    raw = {
        "em_temp": pd.concat(em_temp_chunks, ignore_index=True) if em_temp_chunks else pd.DataFrame(columns=["dev", "_time", "_value"]),
        "ws_temp": pd.concat(ws_temp_chunks, ignore_index=True) if ws_temp_chunks else pd.DataFrame(columns=["_time", "_value"]),
        "em_moist": pd.concat(em_moist_chunks, ignore_index=True) if em_moist_chunks else pd.DataFrame(columns=["dev", "_time", "_value"]),
        "ws_rain": pd.concat(ws_rain_chunks, ignore_index=True) if ws_rain_chunks else pd.DataFrame(columns=["device_name", "_time", "_value"]),
        "openmeteo": pd.concat(openmeteo_chunks, ignore_index=True) if openmeteo_chunks else pd.DataFrame(columns=["table", "_time", "_value", "_field"]),
    }
    return raw, total_em, total_ws, total_om


# ══════════════════════════════════════════════════════════════════════════
# DERIVAZIONI: raw-by-device, pioggia osservata giornaliera, pioggia prevista
# ══════════════════════════════════════════════════════════════════════════

def raw_by_dev(df):
    """df con colonne dev,_time,_value -> dict dev -> DataFrame ordinato per tempo"""
    out = {}
    if df.empty:
        return out
    for dev, g in df.groupby("dev"):
        out[int(dev)] = g.sort_values("_time")[["_time", "_value"]].reset_index(drop=True)
    return out


def compute_daily_rain_observed(ws_rain_raw):
    """Contatore cumulativo che si riazzera più volte al giorno: somma degli
    incrementi positivi; quando il contatore scende (reset) l'incremento è
    il valore stesso (si riparte da zero)."""
    if ws_rain_raw.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))

    df = ws_rain_raw.sort_values(["device_name", "_time"]).copy()
    df["inc"] = df.groupby("device_name")["_value"].diff()
    reset_mask = df["inc"] < 0
    df.loc[reset_mask, "inc"] = df.loc[reset_mask, "_value"]
    df["inc"] = df["inc"].fillna(0).clip(lower=0)
    df["day"] = df["_time"].dt.floor("D")

    daily_per_station = df.groupby(["device_name", "day"])["inc"].sum()
    daily = daily_per_station.groupby("day").mean()
    return daily.sort_index()


def compute_openmeteo_forecast_daily(openmeteo_raw):
    """Pioggia oraria prevista con lead time di 1 giorno (tabelle 70-87),
    ripivottata per campo e sommata per giorno reale (target_time)."""
    if openmeteo_raw.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))

    piv = openmeteo_raw.pivot_table(index="_time", columns="_field", values="_value", aggfunc="first").reset_index()
    if "target_time_iso" not in piv.columns or "rain" not in piv.columns:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))

    piv["target"] = pd.to_datetime(piv["target_time_iso"], errors="coerce", utc=True)
    piv["rain"] = pd.to_numeric(piv["rain"], errors="coerce")
    piv = piv.dropna(subset=["target", "rain"])
    piv["day"] = piv["target"].dt.floor("D")
    daily = piv.groupby("day")["rain"].sum()
    return daily.sort_index()


# ══════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════

def draw_activity_band(ax, sensore, dentro_map):
    info = dentro_map.get(sensore)
    if not info:
        return
    for s, e in info["dentro"]:
        ax.axvspan(s, e, color=C_IN, alpha=0.20, zorder=0, linewidth=0)


def format_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)


def transmission_range(*dfs):
    mins, maxs = [], []
    for df in dfs:
        if df is not None and not df.empty:
            mins.append(df["_time"].min())
            maxs.append(df["_time"].max())
    if not mins:
        return None, None
    return min(mins), max(maxs)


def make_temperature_plots(dentro_map, em_temp_by_dev, ws_temp_pooled, outdir):
    os.makedirs(outdir, exist_ok=True)

    for sensore in sorted(dentro_map.keys()):
        soil = em_temp_by_dev.get(sensore)
        t0, t1 = transmission_range(soil)
        if t0 is None:
            print(f"  EM-500-{sensore}: nessun dato di temperatura, salto.")
            continue

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                              gridspec_kw={"hspace": 0.12})

        draw_activity_band(ax_top, sensore, dentro_map)
        draw_activity_band(ax_bot, sensore, dentro_map)

        ax_top.scatter(soil["_time"], soil["_value"], s=3, color=C_SOIL, alpha=0.35, zorder=2, rasterized=True)
        ax_top.set_ylabel("Sensore (°C)", fontsize=11)
        ax_top.set_title(f"EM-500-{sensore} — temperatura ({t0.date()} → {t1.date()})", fontsize=13, pad=8)

        ws_sel = ws_temp_pooled.loc[(ws_temp_pooled["_time"] >= t0) & (ws_temp_pooled["_time"] <= t1)] if not ws_temp_pooled.empty else ws_temp_pooled
        if not ws_sel.empty:
            ax_bot.scatter(ws_sel["_time"], ws_sel["_value"], s=3, color=C_AIR, alpha=0.35, zorder=2, rasterized=True)
        ax_bot.set_ylabel("WS (°C)", fontsize=11)

        ax_top.set_xlim(t0, t1)
        format_time_axis(ax_bot)

        handles = [
            mpatches.Patch(facecolor=C_IN, alpha=0.20, label="Periodo di attività (dentro)"),
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor=C_SOIL, markersize=6, alpha=0.7, label="Sensore (grezzo)"),
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor=C_AIR, markersize=6, alpha=0.7, label="WS (grezzo)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
                   frameon=True, framealpha=0.95, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.04))
        fig.savefig(os.path.join(outdir, f"EM-500-{sensore}_temperatura.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def make_moisture_rain_plots(dentro_map, em_moist_by_dev, rain_obs_daily, rain_fc_daily, outdir):
    os.makedirs(outdir, exist_ok=True)

    for sensore in sorted(dentro_map.keys()):
        moist = em_moist_by_dev.get(sensore)
        t0, t1 = transmission_range(moist)
        if t0 is None:
            print(f"  EM-500-{sensore}: nessun dato di umidità, salto.")
            continue

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                              gridspec_kw={"hspace": 0.12})

        draw_activity_band(ax_top, sensore, dentro_map)
        draw_activity_band(ax_bot, sensore, dentro_map)

        ax_top.scatter(moist["_time"], moist["_value"], s=3, color=C_SOIL, alpha=0.35, zorder=2, rasterized=True)
        ax_top.set_ylabel("Umidità suolo (%)", fontsize=11)
        ax_top.set_title(f"EM-500-{sensore} — umidità suolo e pioggia ({t0.date()} → {t1.date()})", fontsize=13, pad=8)

        obs = rain_obs_daily.loc[(rain_obs_daily.index >= t0) & (rain_obs_daily.index <= t1)] if not rain_obs_daily.empty else rain_obs_daily
        fc = rain_fc_daily.loc[(rain_fc_daily.index >= t0) & (rain_fc_daily.index <= t1)] if not rain_fc_daily.empty else rain_fc_daily

        if not obs.empty:
            ax_bot.bar(obs.index, obs.values, width=0.9, color=C_RAIN_OBS, alpha=0.6, zorder=2)
        if not fc.empty:
            ax_bot.bar(fc.index, fc.values, width=0.5, color=C_RAIN_FC, alpha=0.85, zorder=3)
        ax_bot.set_ylabel("Pioggia (mm/giorno)", fontsize=11)

        ax_top.set_xlim(t0, t1)
        format_time_axis(ax_bot)

        handles = [
            mpatches.Patch(facecolor=C_IN, alpha=0.20, label="Periodo di attività (dentro)"),
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor=C_SOIL, markersize=6, alpha=0.7, label="Umidità suolo (grezzo)"),
            mpatches.Patch(facecolor=C_RAIN_OBS, alpha=0.6, label="Pioggia osservata (WS)"),
            mpatches.Patch(facecolor=C_RAIN_FC, alpha=0.85, label="Pioggia prevista (openmeteo, giorno prima)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
                   frameon=True, framealpha=0.95, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.10))
        fig.savefig(os.path.join(outdir, f"EM-500-{sensore}_moisture_pioggia.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"C:\Users\milen\Desktop\trace\db_dump\dump-luglio.csv")
    parser.add_argument("--registry", default=os.path.join(HERE, "sensori-corretti.csv"))
    parser.add_argument("--gt", default=os.path.join(HERE, "inizio.csv"))
    parser.add_argument("--outdir-temp", default=os.path.join(HERE, "plots_temperatura"))
    parser.add_argument("--outdir-moisture", default=os.path.join(HERE, "plots_moisture_pioggia"))
    parser.add_argument("--out-csv", default=os.path.join(HERE, "dataset_pulito.csv"))
    args = parser.parse_args()

    # Presidio: l'output viene CANCELLATO qui sotto prima di essere riscritto.
    # Se il percorso uscisse da questa cartella (per esempio puntando al
    # dataset_pulito.csv di AI_agricolture/, che le pipeline possono star
    # leggendo), lo si eliminerebbe senza possibilità di recupero.
    out_assoluto = os.path.normcase(os.path.abspath(args.out_csv))
    radice = os.path.normcase(ROOT_PAPER) + os.sep
    if not out_assoluto.startswith(radice):
        raise SystemExit(
            f"--out-csv deve stare dentro {ROOT_PAPER}\n"
            f"ricevuto: {os.path.abspath(args.out_csv)}\n"
            "Il file di output viene cancellato all'avvio: fuori da questa cartella "
            "si rischia di eliminare un dataset in uso."
        )

    if os.path.exists(args.out_csv):
        os.remove(args.out_csv)

    print("Carico registro e ground truth...")
    reg = load_registry(args.registry)
    gt = load_gt(args.gt)
    dentro_map, periods_map = build_dentro_map(reg, gt)
    print(f"Sensori con periodi 'dentro' definiti: {sorted(dentro_map.keys())}")

    print("\nLeggo il dump grezzo (streaming a chunk, potrebbe richiedere qualche minuto)...")
    raw, total_em, total_ws, total_om = stream_process(args.data, dentro_map, args.out_csv)
    print(f"Righe salvate: EM (dentro)={total_em:,}  WS (complete)={total_ws:,}  openmeteo (complete)={total_om:,}")

    print("\nCostruisco le serie derivate...")
    em_temp_by_dev = raw_by_dev(raw["em_temp"])
    em_moist_by_dev = raw_by_dev(raw["em_moist"])
    ws_temp_pooled = raw["ws_temp"].sort_values("_time")[["_time", "_value"]] if not raw["ws_temp"].empty else pd.DataFrame(columns=["_time", "_value"])
    rain_obs_daily = compute_daily_rain_observed(raw["ws_rain"])
    rain_fc_daily = compute_openmeteo_forecast_daily(raw["openmeteo"])

    print("\nGenero i grafici di temperatura (sensore sopra, WS sotto)...")
    make_temperature_plots(dentro_map, em_temp_by_dev, ws_temp_pooled, args.outdir_temp)
    print(f"  Salvati in '{args.outdir_temp}/'")

    print("\nGenero i grafici di umidità suolo + pioggia...")
    make_moisture_rain_plots(dentro_map, em_moist_by_dev, rain_obs_daily, rain_fc_daily, args.outdir_moisture)
    print(f"  Salvati in '{args.outdir_moisture}/'")

    print(f"\nFatto. Dataset in '{args.out_csv}'.")


if __name__ == "__main__":
    main()
