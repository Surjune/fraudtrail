"""Write the TigerGraph load files from the raw dataset.

One CSV per vertex and edge type, in the order graph/loading.gsql expects. Files built
from transactions are split by month so each upload stays small.

Usage::

    uv run python scripts/export_graph.py [--out data/processed/load]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb

from fraudtrail.data.sql import device_profile_sql
from fraudtrail.data.warehouse import open_warehouse

log = logging.getLogger("export_graph")

REPO = Path(__file__).resolve().parent.parent

MONTHS = ("2016-07", "2016-08", "2016-09", "2016-10", "2016-11", "2016-12")

PROFILE = device_profile_sql("i")
# A profile with no device, OS, browser or screen carries no information, so it links
# nothing. 18% of identity records have no DeviceInfo, but most still have a browser.
HAS_PROFILE = (
    "(i.DeviceInfo IS NOT NULL OR i.id_30 IS NOT NULL OR i.id_31 IS NOT NULL "
    "OR i.id_33 IS NOT NULL)"
)

# Region and country codes are stored as floats (444.0); the graph keys them as "444".
REGION = "CASE WHEN t.addr1 IS NULL THEN '' ELSE CAST(CAST(t.addr1 AS INTEGER) AS VARCHAR) END"
COUNTRY = "CASE WHEN t.addr2 IS NULL THEN '' ELSE CAST(CAST(t.addr2 AS INTEGER) AS VARCHAR) END"

# Match flags kept as one compact string, e.g. "M1:T|M4:M0". Account takeover leans on
# these, and storing them together avoids nine sparse columns.
M_FLAGS = (
    "concat_ws('|', "
    + ", ".join(f"CASE WHEN t.M{i} IS NOT NULL THEN 'M{i}:' || t.M{i} END" for i in range(1, 10))
    + ")"
)

TS = "strftime(t.ts_at, '%Y-%m-%d %H:%M:%S')"

# The five documented patterns, plus the two answer-format values that are not patterns
# the analysts named.
PATTERNS: tuple[tuple[str, str, bool], ...] = (
    ("card_testing", "Small online authorizations before a larger purchase.", True),
    ("card_not_present_fraud", "Online use of the number without the card.", True),
    ("card_not_present_new_device", "Card-not-present use from a device new to the account.", True),
    ("out_of_region_use", "Card-present use in a billing region with no history.", True),
    ("account_takeover", "Mixed-channel activity pointing to stolen credentials.", True),
    ("undocumented", "Abuse matching none of the documented typologies.", False),
    ("none", "No fraud pattern; the alert was legitimate activity.", False),
)

WHOLE_FILES: dict[str, str] = {
    "customer": "SELECT DISTINCT customer_id FROM tx ORDER BY 1",
    "card": (
        "SELECT t.card_id, t.customer_id, coalesce(t.card4, '') AS network, "
        "coalesce(t.card6, '') AS card_type, "
        "strftime(min(t.ts_at), '%Y-%m-%d %H:%M:%S') AS first_seen, "
        "strftime(max(t.ts_at), '%Y-%m-%d %H:%M:%S') AS last_seen, count(*) AS txn_count "
        "FROM tx t GROUP BY ALL ORDER BY 1"
    ),
    "owns": "SELECT DISTINCT customer_id, card_id FROM tx ORDER BY 1, 2",
    "device_profile": (
        f"SELECT DISTINCT {PROFILE} AS profile_id, coalesce(i.DeviceInfo, '') AS device_info, "
        f"coalesce(i.id_30, '') AS os, coalesce(i.id_31, '') AS browser, "
        f"coalesce(i.id_33, '') AS screen, coalesce(i.DeviceType, '') AS device_type "
        f"FROM ident i WHERE {HAS_PROFILE}"
    ),
    "email_domain": (
        "SELECT DISTINCT domain FROM (SELECT P_emaildomain AS domain FROM tx "
        "UNION SELECT R_emaildomain FROM tx) WHERE domain IS NOT NULL ORDER BY 1"
    ),
    "billing_region": (
        f"SELECT DISTINCT {REGION} AS region FROM tx t WHERE t.addr1 IS NOT NULL ORDER BY 1"
    ),
    "closed_case": (
        "SELECT case_id, outcome, pattern, opened_at, closed_at, "
        "CAST(exposure_usd AS DOUBLE) AS exposure_usd, CAST(n_txns AS INTEGER) AS n_txns, "
        "coalesce(actions_taken, '') AS actions_taken, "
        "CASE WHEN report_filed = 'Yes' THEN 'true' ELSE 'false' END AS report_filed, "
        "coalesce(analyst_notes, '') AS analyst_notes FROM cc ORDER BY case_id"
    ),
    "involves": "SELECT case_id, txn_id FROM cc_txn ORDER BY 1, 2",
    "on_card": "SELECT case_id, card_id FROM cc ORDER BY 1",
    "connected_to": (
        "SELECT case_id, trim(c) AS card_id FROM (SELECT case_id, "
        "unnest(string_split(connected_card_ids, '|')) AS c FROM cc "
        "WHERE coalesce(connected_card_ids, '') <> '') WHERE trim(c) <> '' ORDER BY 1, 2"
    ),
    "case_pattern": "SELECT case_id, pattern FROM cc WHERE pattern <> 'none' ORDER BY 1",
}

MONTHLY_FILES: dict[str, str] = {
    "transaction": (
        f"SELECT t.txn_id, {TS} AS ts, t.TransactionAmt AS amount, t.ProductCD AS product_cd, "
        f"t.channel, t.risk_score, {REGION} AS region, {COUNTRY} AS country, t.dist1, t.dist2, "
        f"coalesce(i.id_15, '') AS device_status, coalesce(i.id_23, '') AS proxy, "
        f"{M_FLAGS} AS m_flags FROM tx t LEFT JOIN ident i USING (txn_id) WHERE {{month}}"
    ),
    "made": "SELECT t.card_id, t.txn_id FROM tx t WHERE {month}",
    "from_device": (
        f"SELECT t.txn_id, {PROFILE} AS profile_id FROM tx t JOIN ident i USING (txn_id) "
        f"WHERE {HAS_PROFILE} AND {{month}}"
    ),
    "purchaser_email": (
        "SELECT t.txn_id, t.P_emaildomain AS domain FROM tx t "
        "WHERE t.P_emaildomain IS NOT NULL AND {month}"
    ),
    "recipient_email": (
        "SELECT t.txn_id, t.R_emaildomain AS domain FROM tx t "
        "WHERE t.R_emaildomain IS NOT NULL AND {month}"
    ),
    "billed_in": (
        f"SELECT t.txn_id, {REGION} AS region FROM tx t WHERE t.addr1 IS NOT NULL AND {{month}}"
    ),
    # NEXT links a card's transactions in time order, so a query can walk an episode.
    "next": (
        "SELECT txn_id, next_txn_id, gap_minutes FROM (SELECT t.txn_id, "
        "lead(t.txn_id) OVER (PARTITION BY t.card_id ORDER BY t.ts_at) AS next_txn_id, "
        "date_diff('second', t.ts_at, lead(t.ts_at) OVER (PARTITION BY t.card_id "
        "ORDER BY t.ts_at)) / 60.0 AS gap_minutes, t.ts_at FROM tx t) t "
        "WHERE next_txn_id IS NOT NULL AND {month}"
    ),
}


def _copy(con: duckdb.DuckDBPyConnection, sql: str, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO '{path.as_posix()}' (FORMAT csv, HEADER, DELIMITER ',')")
    rows = con.execute(f"SELECT count(*) FROM ({sql})").fetchone()
    return 0 if rows is None else int(rows[0])


def export(con: duckdb.DuckDBPyConnection, out: Path) -> dict[str, int]:
    written: dict[str, int] = {}
    con.execute("CREATE TABLE pattern_rows (name VARCHAR, description VARCHAR, documented VARCHAR)")
    for name, description, documented in PATTERNS:
        con.execute(
            "INSERT INTO pattern_rows VALUES (?, ?, ?)",
            [name, description, "true" if documented else "false"],
        )
    written["pattern"] = _copy(con, "SELECT * FROM pattern_rows", out / "pattern.csv")

    for name, sql in WHOLE_FILES.items():
        written[name] = _copy(con, sql, out / f"{name}.csv")
        log.info("%s: %d rows", name, written[name])

    for name, template in MONTHLY_FILES.items():
        total = 0
        for month in MONTHS:
            where = f"strftime(t.ts_at, '%Y-%m') = '{month}'"
            sql = template.format(month=where)
            total += _copy(con, sql, out / f"{name}_{month}.csv")
        written[name] = total
        log.info("%s: %d rows across %d months", name, total, len(MONTHS))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", type=Path, default=REPO / "data" / "raw")
    parser.add_argument("--processed", type=Path, default=REPO / "data" / "processed")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "processed" / "load")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    con = open_warehouse(args.raw, args.processed)
    written = export(con, args.out)
    log.info(
        "wrote %d files to %s (%d rows)",
        len(list(args.out.glob("*.csv"))),
        args.out,
        sum(written.values()),
    )


if __name__ == "__main__":
    main()
