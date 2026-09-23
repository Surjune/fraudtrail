"""DuckDB warehouse over the raw dataset files.

Converts the two large CSVs to Parquet once, then exposes the views every script uses:

``tx_raw``     every original column
``tx``         transactions with the derived ``txn_id``, ``card_id`` and ``ts_at``
``ident``      identity records keyed by ``txn_id``
``cc`` / ``cp``  closed cases and the exam case pack, all columns as text
``card_keys``  one row per card
``cc_txn``     closed cases exploded to one row per transaction
``cc_joined``  ``cc_txn`` joined to the transaction it names
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from fraudtrail.data.sql import card_join_condition, card_keys_sql

log = logging.getLogger(__name__)

# The development laptop has 7.7 GB RAM with little free; DuckDB spills beyond this.
DUCKDB_MEMORY_LIMIT = "2GB"
DUCKDB_THREADS = 2

LARGE_CSVS = (("transactions.csv", "transactions.parquet"), ("identity.csv", "identity.parquet"))
TEXT_CSVS = (("cc", "closed_cases_history.csv"), ("cp", "case_pack.csv"))


def _to_parquet(con: duckdb.DuckDBPyConnection, csv: Path, parquet: Path) -> None:
    if parquet.exists() and parquet.stat().st_mtime >= csv.stat().st_mtime:
        return
    log.info("converting %s to Parquet (one-time, full type inference)", csv.name)
    tmp = parquet.with_suffix(".tmp.parquet")
    con.execute(
        f"COPY (SELECT * FROM read_csv('{csv.as_posix()}', header=true, sample_size=-1)) "
        f"TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"
    )
    tmp.replace(parquet)


def open_warehouse(raw: Path, processed: Path) -> duckdb.DuckDBPyConnection:
    processed.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads = {DUCKDB_THREADS}")
    con.execute(f"SET temp_directory = '{(processed / 'duckdb_tmp').as_posix()}'")

    for csv_name, parquet_name in LARGE_CSVS:
        _to_parquet(con, raw / csv_name, processed / parquet_name)

    tx_parquet = (processed / "transactions.parquet").as_posix()
    id_parquet = (processed / "identity.parquet").as_posix()
    con.execute(f"CREATE VIEW tx_raw AS SELECT * FROM read_parquet('{tx_parquet}')")
    con.execute(f"CREATE TABLE card_keys AS {card_keys_sql('tx_raw')}")
    con.execute(
        "CREATE VIEW tx AS SELECT CAST(t.TransactionID AS VARCHAR) AS txn_id, k.card_id, "
        "CAST(t.ts AS TIMESTAMP) AS ts_at, t.* FROM tx_raw t "
        f"JOIN card_keys k ON {card_join_condition('t', 'k')}"
    )
    con.execute(
        f"CREATE VIEW ident AS SELECT CAST(TransactionID AS VARCHAR) AS txn_id, * "
        f"FROM read_parquet('{id_parquet}')"
    )
    for view, name in TEXT_CSVS:
        con.execute(
            f"CREATE VIEW {view} AS SELECT * FROM read_csv('{(raw / name).as_posix()}', "
            f"header=true, all_varchar=true)"
        )
    con.execute(
        "CREATE TABLE cc_txn AS SELECT case_id, customer_id, card_id, outcome, pattern, "
        "CAST(opened_at AS TIMESTAMP) AS opened_at, trim(t) AS txn_id "
        "FROM (SELECT *, unnest(string_split(txn_ids, '|')) AS t FROM cc) WHERE trim(t) <> ''"
    )
    con.execute(
        "CREATE TABLE cc_joined AS SELECT c.*, t.ts_at, t.TransactionAmt, t.ProductCD, "
        "t.channel, t.risk_score, t.addr1, t.card1, t.card2, t.card3, t.card4, t.card5, "
        "t.card6 FROM cc_txn c JOIN tx t ON t.txn_id = c.txn_id"
    )
    return con
