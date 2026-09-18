"""
paths.py
─────────────────────────────────────────────────────────────────────────────
Percorsi ai dati e costanti di split condivise da tutte le pipeline di
pipeline_corrette/. È l'UNICO punto in cui è scritto dove stanno i dati: ogni
script li importa da qui e nessuno se li ricalcola per conto proprio.

Perché conta: dataset_builder.load_irrigation_reconstructed, se il percorso
dell'irrigazione non esiste, NON solleva un errore — ritorna un DataFrame
vuoto, e la pipeline gira per ore producendo risultati apparentemente sani ma
con irrigazione ovunque nulla. Con un solo punto di verità quel fallimento
silenzioso può accadere in un posto solo, ed è coperto da require_data_files().

DOVE STANNO I DATI IN QUESTA COPIA
Due dei tre CSV di input sono stati portati DENTRO questa cartella
(data_cleaning/sensori-corretti.csv e irrigazione_mancante/irrigazione_ricostruita.csv:
pesano pochi KB, e il secondo è il file al centro del lavoro sull'irrigazione,
quindi averne una copia locale e modificabile è necessario).

Il dataset grande (dataset_pulito.csv, ~1.7 GB) NON è stato duplicato: si punta
ancora alla copia dentro AI_agricolture/. Vedi il blocco marcato più sotto —
è l'unica dipendenza esterna rimasta, e l'unica riga da cambiare per rendere
questa cartella davvero autonoma.
"""

import os

CORE_DIR = os.path.dirname(os.path.abspath(__file__))          # pipeline_corrette/core
PIPELINE_DIR = os.path.dirname(CORE_DIR)                        # pipeline_corrette
ROOT = os.path.dirname(PIPELINE_DIR)                            # radice di questa cartella

DATA_CLEANING_DIR = os.path.join(ROOT, "data_cleaning")
DEFAULT_SENSORS = os.path.join(DATA_CLEANING_DIR, "sensori-corretti.csv")
# Registro delle irrigazioni. Dal settembre 2026 si usa il file "effettive":
# 71 eventi, tutti da log reale. Sostituisce irrigazione_ricostruita.csv (37
# eventi, di cui 8 INFERITI dai salti di umidità senza pioggia): di quegli 8,
# 5 non erano mai avvenuti e 3 erano sottostime forti (15->40, 25->45, 15->45 mm).
# Oltre a essere più completo, elimina un problema di metodo: un evento dedotto
# dal salto di umidità del giorno t+h, usato come input per prevedere proprio
# quell'umidità, è informazione che al momento della previsione non si avrebbe.
DEFAULT_IRRIGAZIONE_RICOSTRUITA = os.path.join(
    ROOT, "irrigazione_mancante", "irrigazione_aggiornata_con_effettive_2026.csv")
DEFAULT_COLTURE = os.path.join(CORE_DIR, "colture.csv")

# ═══════════════════════════════════════════════════════════════════════════
# UNICA DIPENDENZA ESTERNA DI QUESTA CARTELLA
# ───────────────────────────────────────────────────────────────────────────
# dataset_pulito.csv (~1.7 GB) non è stato duplicato: si punta all'originale
# dentro AI_agricolture/. Finché questa riga resta com'è, questa cartella NON
# è autonoma: se AI_agricolture/ viene cancellata o rinominata, nessuna
# pipeline parte.
#
# PER RENDERLA AUTONOMA (operazione di un minuto):
#   1. copiare dataset_pulito.csv in ROOT/data_cleaning/
#   2. commentare la riga DATASET_ESTERNO qui sotto e scommentare quella dopo
#
# In alternativa, senza toccare il codice: impostare la variabile d'ambiente
# AI_AGRI_DATASET con il percorso del file.
# ═══════════════════════════════════════════════════════════════════════════
DATASET_ESTERNO = r"C:\Users\milen\Desktop\trace\AI_agricolture\data_cleaning\dataset_pulito.csv"
# DATASET_ESTERNO = os.path.join(DATA_CLEANING_DIR, "dataset_pulito.csv")

DEFAULT_DATASET = os.environ.get("AI_AGRI_DATASET", DATASET_ESTERNO)

# Dataset RIGENERATO in questa cartella (data_cleaning/plot_dentro_fuori.py, con
# il registro aggiornato): contiene in più i periodi 2026 dei sensori 3, 4, 8 e
# 20, che la pulizia originale aveva scartato. Lo usano solo le copie in
# pipeline_corrette/dataset_aggiornato/; tutte le altre pipeline restano su
# DEFAULT_DATASET qui sopra, così i risultati dei due dataset non si mescolano.
DATASET_AGGIORNATO = os.path.join(DATA_CLEANING_DIR, "dataset_pulito.csv")

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15                 # test_frac = 1 - TRAIN_FRAC - VAL_FRAC = 0.15


def require_data_files(dataset=None, sensors=None, irrigazione=None, colture=None):
    """Controlla che i file di input esistano, PRIMA di iniziare un run che
    può durare ore. Solleva FileNotFoundError elencando tutto ciò che manca.

    Va chiamata come prima riga di main(), dopo il parsing degli argomenti, con
    i percorsi effettivamente in uso (quelli da riga di comando possono essere
    diversi dai default di questo modulo).

    Senza questo controllo un percorso sbagliato dell'irrigazione non darebbe
    nessun errore: load_irrigation_reconstructed ritorna un DataFrame vuoto e
    l'intero esperimento verrebbe eseguito senza dati di irrigazione, in
    silenzio.
    """
    attesi = {
        "dataset": dataset if dataset is not None else DEFAULT_DATASET,
        "sensori": sensors if sensors is not None else DEFAULT_SENSORS,
        "irrigazione": irrigazione if irrigazione is not None else DEFAULT_IRRIGAZIONE_RICOSTRUITA,
        "colture": colture if colture is not None else DEFAULT_COLTURE,
    }
    mancanti = [f"  {etichetta:12s} {percorso}"
                for etichetta, percorso in attesi.items()
                if not percorso or not os.path.exists(percorso)]
    if mancanti:
        raise FileNotFoundError(
            "File di input mancanti:\n" + "\n".join(mancanti)
            + "\n\nSe manca solo il dataset: è l'unico file non copiato in questa "
              "cartella (vedi il blocco 'UNICA DIPENDENZA ESTERNA' in core/paths.py). "
              "Puoi indicarne un altro percorso con la variabile d'ambiente "
              "AI_AGRI_DATASET."
        )
