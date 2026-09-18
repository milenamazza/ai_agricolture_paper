"""
feature_groups_util.py
─────────────────────────────────────────────────────────────────────────────
Sistema COMPONIBILE di feature-set, alternativo ai 6 feature-set fissi e
scritti a mano di FSET_MAP (modelli singoli/data_driven_ml/utils.py,
documentati in confronto/feature_sets.md). Nessuna feature nuova: le stesse
colonne di FSET_MAP, riorganizzate in blocchi più piccoli e componibili.

Due livelli:

  1. ATOMIC_FEATURE_GROUPS — blocchi piccoli, ciascuno un tema tecnico
     preciso (es. "solo la pioggia osservata", "solo la previsione di
     temperatura"). Non vanno mai usati da soli come feature-set di un
     modello — sono i "mattoncini".

  2. THEMATIC_FEATURE_SETS — combinazioni di 2+ blocchi atomici, con un
     senso concettuale (es. ACQUA = tutto ciò che riguarda l'acqua in
     ingresso). Sono quelli pensati per essere effettivamente usati come
     feature-set di un modello. Alcuni sono a loro volta combinazioni di
     ALTRI gruppi tematici (es. TERRA_ACQUA = TERRA + ACQUA) — la
     composizione funziona "a cascata": blocchi atomici -> tematiche pure ->
     combinazioni di tematiche.

La funzione compose_feature_set() è quella che genera davvero l'elenco di
colonne per un dato orizzonte, e IMPONE la regola "nessun blocco da solo":
solleva un errore se le vengono passati meno di 2 nomi di gruppo.

Le colonne nei blocchi atomici sono sempre scritte nella forma "orizzonte 1" e
vengono tradotte all'orizzonte richiesto da _generalize_column_to_horizon():
fc1_->fc{h}_ e fc1h_->fc{h}h_ per prefisso (le previsioni meteo),
irr_planned_t1->irr_planned_t{h} per nome intero (l'irrigazione pianificata).
Le colonne che non dipendono dall'orizzonte restano invariate.
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════
# Livello 1 — blocchi atomici (forma base t+1, generalizzata più sotto)
# ═══════════════════════════════════════════════════════════════════════════

ATOMIC_FEATURE_GROUPS: dict[str, list[str]] = {

    # ── Storia locale del sensore/suolo/coltura ─────────────────────────────
    "MOISTURE_HISTORY": [
        "moisture", "moisture_lag1", "moisture_lag2", "moisture_roll3", "moisture_roll7",
    ],
    "EC_HISTORY": ["ec", "ec_lag1"],
    "TEMPERATURE_LOCAL": ["temperature"],
    "CROP": ["crop_encoded"],

    # ── Acqua in ingresso: osservata, prevista, irrigazione, flag evento ───
    "RAIN_OBSERVED": [
        "ws_rainfall_total", "ws_rainfall_total_lag1", "ws_rainfall_total_roll7",
    ],
    # irrigation_mm e irrigation_lag1 guardano indietro (oggi, ieri).
    # irr_planned_t1 guarda AVANTI e si generalizza a irr_planned_t{h}: è
    # l'irrigazione pianificata per il giorno che si sta prevedendo, cioè un
    # input esogeno noto e non un'incognita — lo stesso trattamento che la
    # fisica in penman.py dà già all'irrigazione.
    "IRRIGATION": ["irrigation_mm", "irrigation_lag1", "irr_planned_t1"],
    "RAIN_FORECAST": [
        "fc1_rain", "fc1_precipitation_hours",
        "fc1h_rain_sum", "fc1h_rain_hours",
        "fc1h_showers_sum", "fc1h_snowfall_sum",
    ],
    "EVENT_FLAGS": ["has_rain", "has_irr", "has_water_event"],

    # ── Condizioni atmosferiche (non l'acqua in ingresso) ───────────────────
    "TEMPERATURE_OBSERVED": [
        "ws_temperature", "ws_temperature_lag1", "ws_temperature_roll3",
        "ws_temp_min", "ws_temp_max", "ws_temp_median", "ws_temp_amplitude", "ws_temp_std",
    ],
    "HUMIDITY_WIND_OBSERVED": [
        "ws_humidity",
        "ws_hum_min", "ws_hum_max", "ws_hum_median", "ws_hum_std",
        "ws_wind_max", "ws_wind_std", "ws_wind_median",
    ],
    "TEMPERATURE_FORECAST": [
        "fc1_temperature_2m", "fc1_apparent_temperature",
        "fc1h_temperature_2m_mean", "fc1h_temperature_2m_min",
        "fc1h_temperature_2m_max", "fc1h_temperature_2m_amp",
        "fc1h_apparent_temperature_min", "fc1h_apparent_temperature_max",
    ],
    "HUMIDITY_WIND_FORECAST": [
        "fc1_relative_humidity_2m",
        "fc1h_relative_humidity_2m_mean", "fc1h_relative_humidity_2m_min",
        "fc1h_wind_speed_10m_max", "fc1h_wind_speed_10m_mean",
    ],
    "EVAPOTRANSPIRATION_FORECAST": [
        "fc1_et0_fao_evapotranspiration", "fc1_evapotranspiration",
        "fc1h_et0_fao_evapotranspiration_sum", "fc1h_evapotranspiration_sum",
    ],

    # ── Feature derivate/di interazione (né puramente acqua né puramente meteo) ─
    "DERIVED_INTERACTIONS": ["fc1_net_balance", "fc1_moist_x_et", "fc1_temp_delta"],
}


# ═══════════════════════════════════════════════════════════════════════════
# Composizione: unisce blocchi atomici, con la regola "minimo 2 blocchi"
# ═══════════════════════════════════════════════════════════════════════════

# Le colonne scritte nei blocchi atomici sono sempre nella forma "orizzonte 1";
# per un orizzonte diverso vanno tradotte nel nome della colonna corrispondente,
# che nel dataset esiste davvero (fc3_rain, irr_planned_t3, ...). Due regole:
#
#   PREFISSO  — sostituzione di sottostringa sul prefisso iniziale, la regola
#               storica per le colonne di previsione meteo.
#   ESATTA    — si applica solo a un nome INTERO, mai a un pezzo.
#
# Perché irr_planned_t1 sta fra le esatte e non si usa un più breve
# .replace("_t1", ...): il progetto è pieno di nomi *_t{h} che non sono feature
# componibili (target_t1, penman_pred_t1, penman_irrigation_planned_t1). Una
# regola di sottostringa li riscriverebbe il giorno in cui uno di essi finisse
# in un blocco atomico — senza sollevare nulla, producendo un feature-set
# sbagliato in silenzio. Col match esatto la collisione è impossibile per
# costruzione: il dominio della regola è un singolo nome, non un pattern.
_PREFIX_HORIZON_RULES = (("fc1h_", "fc{horizon}h_"), ("fc1_", "fc{horizon}_"))
_EXACT_HORIZON_COLUMNS = {"irr_planned_t1": "irr_planned_t{horizon}"}


def _generalize_column_to_horizon(column_name: str, horizon: int) -> str:
    """Traduce il nome di una colonna dalla forma t+1 a quella dell'orizzonte
    richiesto. Prima le regole a match esatto, poi quelle a prefisso (nello
    stesso ordine di sempre); un nome che non corrisponde a nessuna regola
    resta invariato — è il caso della maggior parte delle colonne, che non
    dipendono dall'orizzonte (moisture, ec, crop_encoded, ...)."""
    if column_name in _EXACT_HORIZON_COLUMNS:
        return _EXACT_HORIZON_COLUMNS[column_name].format(horizon=horizon)

    for prefisso_t1, modello in _PREFIX_HORIZON_RULES:
        if prefisso_t1 in column_name:
            return column_name.replace(prefisso_t1, modello.format(horizon=horizon))

    return column_name


def _resolve_to_atomic_group_names(group_names: tuple[str, ...]) -> list[str]:
    """Un nome può essere sia un blocco atomico (es. 'RAIN_OBSERVED') sia un
    gruppo tematico già definito (es. 'TERRA', risolto ricorsivamente nei
    suoi blocchi atomici) — questo permette la composizione "a cascata"
    (tematiche costruite da altre tematiche, non solo da blocchi atomici)."""
    resolved = []
    for name in group_names:
        if name in ATOMIC_FEATURE_GROUPS:
            resolved.append(name)
        elif name in THEMATIC_FEATURE_SETS:
            resolved.extend(_resolve_to_atomic_group_names(THEMATIC_FEATURE_SETS[name]))
        else:
            known_names = list(ATOMIC_FEATURE_GROUPS) + list(THEMATIC_FEATURE_SETS)
            raise ValueError(f"Gruppo '{name}' non riconosciuto. Gruppi disponibili: {known_names}")
    return resolved


def compose_feature_set(*group_names: str, horizon: int = 1) -> list[str]:
    """
    Unisce le colonne di due o più blocchi/gruppi tematici in un'unica lista
    di feature, per l'orizzonte richiesto (stessa generalizzazione fc1_ ->
    fc{h}_ / fc1h_ -> fc{h}h_ usata da get_fset_features() in utils.py).

    Regola imposta esplicitamente: servono ALMENO 2 gruppi. Un blocco
    atomico da solo non è pensato per essere un feature-set — è un
    "mattoncino", non un edificio. Passare un solo nome solleva un errore
    chiaro invece di produrre silenziosamente un feature-set con una sola
    fonte di informazione.
    """
    if len(group_names) < 2:
        raise ValueError(
            f"compose_feature_set richiede almeno 2 gruppi, ricevuto {len(group_names)}: "
            f"{group_names}. Nessun blocco va usato da solo come feature-set."
        )

    atomic_group_names = _resolve_to_atomic_group_names(group_names)

    base_columns = []
    for atomic_name in atomic_group_names:
        base_columns.extend(ATOMIC_FEATURE_GROUPS[atomic_name])
    deduplicated_columns = list(dict.fromkeys(base_columns))

    if horizon == 1:
        return deduplicated_columns

    return [_generalize_column_to_horizon(column_name, horizon)
            for column_name in deduplicated_columns]


# ═══════════════════════════════════════════════════════════════════════════
# Livello 2 — gruppi tematici, costruiti componendo i blocchi atomici (e, per
# le combinazioni a coppie/il totale, componendo altri gruppi tematici)
# ═══════════════════════════════════════════════════════════════════════════

THEMATIC_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "ACQUA": ("IRRIGATION", "RAIN_OBSERVED", "RAIN_FORECAST", "EVENT_FLAGS"),
    "TERRA": ("MOISTURE_HISTORY", "EC_HISTORY", "TEMPERATURE_LOCAL", "CROP"),
    "METEO": ("TEMPERATURE_OBSERVED", "HUMIDITY_WIND_OBSERVED", "TEMPERATURE_FORECAST",
             "HUMIDITY_WIND_FORECAST", "EVAPOTRANSPIRATION_FORECAST"),

    # Combinazioni a coppie: unione dei gruppi tematici puri qui sopra.
    "TERRA_ACQUA": ("TERRA", "ACQUA"),
    "TERRA_METEO": ("TERRA", "METEO"),
    "ACQUA_METEO": ("ACQUA", "METEO"),

    # Tutto insieme.
    "TOTALE": ("ACQUA", "TERRA", "METEO", "DERIVED_INTERACTIONS"),
}


def get_thematic_feature_set(name: str, horizon: int = 1) -> list[str]:
    """Scorciatoia per i gruppi tematici già definiti sopra, es.
    get_thematic_feature_set("ACQUA", horizon=3)."""
    if name not in THEMATIC_FEATURE_SETS:
        raise ValueError(f"Gruppo tematico '{name}' non trovato. Disponibili: {list(THEMATIC_FEATURE_SETS)}")
    return compose_feature_set(*THEMATIC_FEATURE_SETS[name], horizon=horizon)


# ═══════════════════════════════════════════════════════════════════════════
# Self-test — verifica rapida che la composizione funzioni come previsto
# ═══════════════════════════════════════════════════════════════════════════

def _run_self_test():
    print("Blocchi atomici disponibili:", len(ATOMIC_FEATURE_GROUPS))
    for group_name, columns in ATOMIC_FEATURE_GROUPS.items():
        print(f"  {group_name:28s} {len(columns):2d} feature")

    print("\nGruppi tematici (t+1):")
    for thematic_name in THEMATIC_FEATURE_SETS:
        columns = get_thematic_feature_set(thematic_name, horizon=1)
        print(f"  {thematic_name:14s} {len(columns):3d} feature")

    # TERRA_ACQUA deve essere esattamente l'unione di TERRA e ACQUA, senza duplicati.
    terra_columns = set(get_thematic_feature_set("TERRA"))
    acqua_columns = set(get_thematic_feature_set("ACQUA"))
    terra_acqua_columns = set(get_thematic_feature_set("TERRA_ACQUA"))
    assert terra_acqua_columns == terra_columns | acqua_columns, (
        "TERRA_ACQUA dovrebbe essere l'unione esatta di TERRA e ACQUA"
    )
    print("\n[OK] TERRA_ACQUA == unione(TERRA, ACQUA)")

    # La generalizzazione all'orizzonte deve sostituire fc1_/fc1h_ correttamente.
    acqua_t1 = get_thematic_feature_set("ACQUA", horizon=1)
    acqua_t5 = get_thematic_feature_set("ACQUA", horizon=5)
    assert "fc1_rain" in acqua_t1 and "fc5_rain" in acqua_t5
    assert "fc1h_rain_sum" in acqua_t1 and "fc5h_rain_sum" in acqua_t5
    print("[OK] generalizzazione fc1_ -> fc{h}_ funziona (verificato su ACQUA, h=5)")

    # L'irrigazione pianificata si generalizza per NOME INTERO, non per prefisso.
    assert "irr_planned_t1" in acqua_t1 and "irr_planned_t5" in acqua_t5
    assert "irr_planned_t1" not in acqua_t5, "a h=5 deve restare solo irr_planned_t5"
    print("[OK] generalizzazione irr_planned_t1 -> irr_planned_t{h} funziona (ACQUA, h=5)")

    # Le colonne che guardano al passato NON devono essere toccate dall'orizzonte.
    for colonna in ("irrigation_mm", "irrigation_lag1"):
        assert colonna in acqua_t1 and colonna in acqua_t5, f"{colonna} non deve cambiare con h"
    print("[OK] irrigation_mm / irrigation_lag1 restano invarianti all'orizzonte")

    # La regola a match esatto non deve toccare nomi che la contengono come pezzo.
    assert _generalize_column_to_horizon("target_t1", 5) == "target_t1"
    assert _generalize_column_to_horizon("penman_pred_t1", 5) == "penman_pred_t1"
    assert _generalize_column_to_horizon("irr_planned_t1_lag1", 5) == "irr_planned_t1_lag1"
    print("[OK] il match esatto non intacca target_t1 / penman_pred_t1 / irr_planned_t1_lag1")

    # L'irrigazione pianificata entra in tutti i set che contengono ACQUA, e in
    # nessun altro: è la conseguenza attesa della scelta di metterla in IRRIGATION.
    con_acqua = {"ACQUA", "TERRA_ACQUA", "ACQUA_METEO", "TOTALE"}
    for nome in THEMATIC_FEATURE_SETS:
        presente = "irr_planned_t3" in get_thematic_feature_set(nome, horizon=3)
        assert presente == (nome in con_acqua), f"{nome}: irr_planned atteso={nome in con_acqua}"
    print(f"[OK] irr_planned presente esattamente in {sorted(con_acqua)}")

    # Un solo gruppo deve sollevare un errore, non produrre un feature-set silenzioso.
    try:
        compose_feature_set("RAIN_OBSERVED")
        raise AssertionError("Ci si aspettava un ValueError con un solo gruppo")
    except ValueError as error:
        print(f"[OK] un solo gruppo solleva un errore come previsto: {error}")

    print("\nSelf-test completato senza errori.")


if __name__ == "__main__":
    _run_self_test()
