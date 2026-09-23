"""Profile the HHGOA_IEEE dataset before any graph work.

Answers the questions the build depends on:

* how card IDs (``C01234-K1``) map to the raw ``card1``-``card6`` columns
* what transaction and case ID formats look like
* what the closed cases teach: patterns, actions, reports, episode length, notes
* how the bank's risk score relates to confirmed outcomes
* what the 20 exam cases look like before any agent touches them

Converts the two large CSVs to Parquet once (``data/processed/``), then writes
``reports/data_profile.md``.

Usage::

    uv run python scripts/profile_data.py [--raw data/raw] [--out reports/data_profile.md]
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import duckdb

from fraudtrail.data.sql import device_profile_sql
from fraudtrail.data.warehouse import open_warehouse

log = logging.getLogger("profile_data")

REPO = Path(__file__).resolve().parent.parent

# Tolerance for "same amount" when looking for recurring charges (R7). The dataset's
# amount offsets stay within about $0.10 (a $100 charge appears as $99.94 to $100.09).
RECURRING_TOLERANCE_USD = 0.15

# Pattern 1 ("often under $5") and R5 ("within an hour"), used only as a hint here.
CARD_TESTING_SMALL_AUTH_USD = 5.0
CARD_TESTING_WINDOW_MINUTES = 60

# Exam window: the case pack covers November and December 2016.
EXAM_START = "2016-11-01"

# Rows shown for long listings.
SAMPLE_ROWS = 5
LISTING_ROWS = 40
NOTE_ROWS = 60

# Candidate raw-column combinations that could distinguish a customer's cards.
CARD_KEY_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("card2",),
    ("card3",),
    ("card5",),
    ("card4", "card6"),
    ("card2", "card3"),
    ("card2", "card5"),
    ("card2", "card3", "card5"),
    ("card2", "card3", "card4", "card5", "card6"),
    ("card1", "card2", "card3", "card4", "card5", "card6"),
    ("addr1",),
    ("card2", "card3", "card4", "card5", "card6", "addr1"),
)

DEVICE_PROFILE_SQL = device_profile_sql("i")


class Report:
    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con
        self._parts: list[str] = []

    def h(self, title: str, level: int = 2) -> None:
        log.info("section: %s", title)
        self._parts.append(f"\n{'#' * level} {title}\n")

    def text(self, body: str) -> None:
        self._parts.append(body.strip() + "\n")

    def table(self, sql: str, params: Sequence[object] = ()) -> list[tuple[object, ...]]:
        cursor = self._con.execute(sql, list(params))
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        self._parts.append(_markdown_table(columns, rows))
        return rows

    def execute(self, sql: str) -> None:
        self._con.execute(sql)

    def scalar(self, sql: str, params: Sequence[object] = ()) -> object:
        row = self._con.execute(sql, list(params)).fetchone()
        return None if row is None else row[0]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Dataset profile\n" + "\n".join(self._parts), encoding="utf-8")


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) >= 1 else f"{value:.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(columns: Sequence[str], rows: Sequence[tuple[object, ...]]) -> str:
    if not rows:
        return "_no rows_\n"
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join([header, rule, *body]) + "\n"


def section_shape(r: Report) -> None:
    r.h("Files and shape")
    r.table(
        "SELECT 'transactions' AS file, count(*) AS rows FROM tx_raw "
        "UNION ALL SELECT 'identity', count(*) FROM ident "
        "UNION ALL SELECT 'closed_cases_history', count(*) FROM cc "
        "UNION ALL SELECT 'case_pack', count(*) FROM cp"
    )
    r.table(
        "SELECT min(ts_at) AS first_ts, max(ts_at) AS last_ts, "
        "count(DISTINCT customer_id) AS customers, "
        "sum(CASE WHEN channel = 'online' THEN 1 ELSE 0 END) AS online, "
        "sum(CASE WHEN channel = 'in_person' THEN 1 ELSE 0 END) AS in_person FROM tx"
    )
    r.table("SELECT ProductCD, channel, count(*) AS n FROM tx GROUP BY ALL ORDER BY n DESC")


def section_ids(r: Report) -> None:
    r.h("ID formats")
    r.text("Answer files must use IDs exactly as the dataset spells them.")
    r.table(
        f"SELECT TransactionID, typeof(TransactionID) AS type, customer_id, ts FROM tx_raw "
        f"LIMIT {SAMPLE_ROWS}"
    )
    r.table(
        f"SELECT case_id, customer_id, card_id, first_fraud_txn_id, txn_ids, "
        f"connected_card_ids, actions_taken, report_filed FROM cc LIMIT {SAMPLE_ROWS}"
    )
    matched = r.scalar("SELECT count(*) FROM cc_joined")
    total = r.scalar("SELECT count(*) FROM cc_txn")
    r.text(f"Closed-case transaction IDs found in transactions.csv: {matched} of {total}.")


def section_case_pack(r: Report) -> None:
    r.h("Case pack integrity")
    r.table(
        f"SELECT cp.case_id, cp.trigger_type, cp.flagged_txn_id, cp.card_id, "
        f"t.card_id = cp.card_id AS card_rule_ok, t.ts_at, t.TransactionAmt, t.ProductCD, "
        f"t.channel, t.addr1, t.addr2, t.risk_score, i.id_15 AS device_status, i.id_23 AS proxy, "
        f"{DEVICE_PROFILE_SQL} AS device_profile "
        f"FROM cp LEFT JOIN tx t ON t.txn_id = cp.flagged_txn_id "
        f"LEFT JOIN ident i ON i.txn_id = cp.flagged_txn_id ORDER BY cp.case_id"
    )


def section_card_ids(r: Report) -> None:
    r.h("Card ID derivation")
    r.text(
        "transactions.csv has customer_id but no card_id. Closed cases and the case pack "
        "name cards like C01234-K1. For each candidate column set: `card_consistent` is the "
        "share of card IDs whose transactions all carry one key value; `key_unique` is the "
        "share of (customer, key) groups that map to a single card ID, over customers with "
        "two or more cards. A correct key scores 1.0 on both."
    )
    r.table(
        "SELECT (SELECT count(*) FROM (SELECT card1 FROM tx GROUP BY card1 "
        "HAVING count(DISTINCT customer_id) > 1)) AS card1_with_many_customers, "
        "(SELECT count(*) FROM (SELECT customer_id FROM tx GROUP BY customer_id "
        "HAVING count(DISTINCT card1) > 1)) AS customers_with_many_card1"
    )
    results: list[tuple[str, float, float]] = []
    for combo in CARD_KEY_CANDIDATES:
        key = " || '|' || ".join(f"coalesce(CAST({col} AS VARCHAR), '')" for col in combo)
        consistent = r.scalar(
            f"SELECT avg(CASE WHEN n = 1 THEN 1.0 ELSE 0.0 END) FROM "
            f"(SELECT card_id, count(DISTINCT {key}) AS n FROM cc_joined GROUP BY card_id)"
        )
        unique = r.scalar(
            f"WITH multi AS (SELECT customer_id FROM cc_joined GROUP BY customer_id "
            f"HAVING count(DISTINCT card_id) > 1) "
            f"SELECT avg(CASE WHEN n = 1 THEN 1.0 ELSE 0.0 END) FROM "
            f"(SELECT customer_id, {key} AS k, count(DISTINCT card_id) AS n FROM cc_joined "
            f"WHERE customer_id IN (SELECT customer_id FROM multi) GROUP BY ALL)"
        )
        results.append((" + ".join(combo), _as_float(consistent), _as_float(unique)))
    results.sort(key=lambda row: min(row[1], row[2]), reverse=True)
    r.text(_markdown_table(["key", "card_consistent", "key_unique"], results))

    r.text(
        "Confirmed rule: a customer's cards are their distinct (card4, card6) pairs, "
        "numbered K1, K2, ... in ascending order with nulls first. Check against every "
        "labelled transaction in the closed cases and the case pack:"
    )
    r.table(
        "WITH labels AS (SELECT DISTINCT card_id, txn_id FROM cc_txn "
        "UNION SELECT DISTINCT card_id, flagged_txn_id FROM cp) "
        "SELECT count(*) AS labelled_txns, "
        "avg(CASE WHEN t.card_id = l.card_id THEN 1.0 ELSE 0.0 END) AS rule_matches "
        "FROM labels l JOIN tx t ON t.txn_id = l.txn_id"
    )
    r.table(
        "SELECT cards_per_customer, count(*) AS customers FROM (SELECT customer_id, "
        "count(*) AS cards_per_customer FROM card_keys GROUP BY 1) GROUP BY 1 ORDER BY 1"
    )


def _as_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def section_closed_cases(r: Report) -> None:
    r.h("Closed cases")
    r.table("SELECT outcome, pattern, count(*) AS n FROM cc GROUP BY ALL ORDER BY outcome, n DESC")
    r.h("Reports filed", 3)
    r.table(
        "SELECT outcome, pattern, report_filed, count(*) AS n FROM cc GROUP BY ALL "
        "ORDER BY outcome, pattern, report_filed"
    )
    r.h("Actions taken", 3)
    r.table(
        "SELECT outcome, pattern, trim(a) AS action, count(*) AS n FROM "
        "(SELECT outcome, pattern, unnest(regexp_split_to_array(actions_taken, '[|;,]')) AS a "
        "FROM cc) WHERE trim(a) <> '' GROUP BY ALL ORDER BY outcome, pattern, n DESC"
    )
    r.h("Most common action lists (order kept)", 3)
    r.table(
        f"SELECT outcome, pattern, actions_taken, report_filed, count(*) AS n FROM cc "
        f"GROUP BY ALL ORDER BY n DESC LIMIT {LISTING_ROWS}"
    )
    r.h("Exposure, size and connected cards by pattern", 3)
    r.table(
        "SELECT outcome, pattern, count(*) AS n, "
        "median(CAST(exposure_usd AS DOUBLE)) AS median_exposure, "
        "quantile_cont(CAST(exposure_usd AS DOUBLE), 0.9) AS p90_exposure, "
        "max(CAST(exposure_usd AS DOUBLE)) AS max_exposure, "
        "median(CAST(n_txns AS INTEGER)) AS median_txns, "
        "max(CAST(n_txns AS INTEGER)) AS max_txns, "
        "avg(CASE WHEN coalesce(connected_card_ids, '') <> '' THEN 1.0 ELSE 0.0 END) "
        "AS share_with_connected_cards "
        "FROM cc GROUP BY ALL ORDER BY outcome, n DESC"
    )
    r.h("Episode span and timing relative to opened_at", 3)
    r.table(
        "WITH per_case AS (SELECT case_id, outcome, pattern, opened_at, min(ts_at) AS first_ts, "
        "max(ts_at) AS last_ts, avg(CASE WHEN ts_at > opened_at THEN 1.0 ELSE 0.0 END) AS after "
        "FROM cc_joined GROUP BY ALL) "
        "SELECT outcome, pattern, count(*) AS n, "
        "median(date_diff('minute', first_ts, last_ts) / 60.0) AS median_span_h, "
        "quantile_cont(date_diff('minute', first_ts, last_ts) / 60.0, 0.9) AS p90_span_h, "
        "median(date_diff('minute', last_ts, opened_at) / 60.0) AS median_open_lag_h, "
        "avg(after) AS share_txns_after_open "
        "FROM per_case GROUP BY ALL ORDER BY outcome, n DESC"
    )
    r.table(
        "SELECT c.pattern, avg(CASE WHEN c.first_fraud_txn_id = f.txn_id THEN 1.0 ELSE 0.0 END) "
        "AS first_txn_is_earliest FROM cc c JOIN (SELECT case_id, arg_min(txn_id, ts_at) AS "
        "txn_id FROM cc_joined GROUP BY case_id) f USING (case_id) "
        "WHERE c.outcome = 'confirmed_fraud' GROUP BY ALL"
    )
    r.h("Channels and products inside confirmed episodes", 3)
    r.table(
        "SELECT pattern, channel, ProductCD, count(*) AS n FROM cc_joined "
        "WHERE outcome = 'confirmed_fraud' GROUP BY ALL ORDER BY pattern, n DESC"
    )


def section_notes(r: Report) -> None:
    r.h("Analyst notes")
    r.h("Undocumented patterns (all)", 3)
    r.table(
        f"SELECT case_id, customer_id, card_id, opened_at, n_txns, exposure_usd, "
        f"connected_card_ids, actions_taken, analyst_notes FROM cc "
        f"WHERE pattern = 'undocumented' ORDER BY opened_at LIMIT {NOTE_ROWS}"
    )
    r.h("Why alerts were cleared (most common notes)", 3)
    r.table(
        f"SELECT analyst_notes, count(*) AS n FROM cc WHERE outcome = 'cleared' "
        f"GROUP BY ALL ORDER BY n DESC LIMIT {LISTING_ROWS}"
    )
    r.h("Confirmed fraud notes, three per pattern", 3)
    r.table(
        "SELECT pattern, case_id, analyst_notes FROM (SELECT *, row_number() OVER "
        "(PARTITION BY pattern ORDER BY case_id) AS rn FROM cc WHERE outcome = "
        "'confirmed_fraud') WHERE rn <= 3 ORDER BY pattern, case_id"
    )


def section_risk_score(r: Report) -> None:
    r.h("Risk score versus outcome")
    r.text("Case-level maximum risk score across each closed case's transactions.")
    r.table(
        "WITH per_case AS (SELECT case_id, outcome, max(risk_score) AS max_risk FROM cc_joined "
        "GROUP BY ALL) SELECT floor(max_risk * 10) / 10 AS risk_bin, count(*) AS cases, "
        "avg(CASE WHEN outcome = 'confirmed_fraud' THEN 1.0 ELSE 0.0 END) AS share_confirmed "
        "FROM per_case GROUP BY ALL ORDER BY risk_bin"
    )
    r.table(
        "SELECT floor(risk_score * 10) / 10 AS risk_bin, count(*) AS txns, "
        "avg(CASE WHEN ts_at >= DATE '2016-11-01' THEN 1.0 ELSE 0.0 END) AS share_exam_window "
        "FROM tx GROUP BY ALL ORDER BY risk_bin"
    )


def section_identity(r: Report) -> None:
    r.h("Device and identity signals")
    r.table("SELECT id_15 AS device_status, count(*) AS n FROM ident GROUP BY ALL ORDER BY n DESC")
    r.table("SELECT id_23 AS proxy, count(*) AS n FROM ident GROUP BY ALL ORDER BY n DESC")
    r.table(
        f"SELECT count(DISTINCT {DEVICE_PROFILE_SQL}) AS device_profiles, "
        f"avg(CASE WHEN i.DeviceInfo IS NULL THEN 1.0 ELSE 0.0 END) AS share_no_device_info "
        f"FROM ident i"
    )
    r.h("Device profiles shared by the most customers in the exam window", 3)
    r.table(
        f"SELECT {DEVICE_PROFILE_SQL} AS device_profile, count(DISTINCT t.customer_id) AS "
        f"customers, count(*) AS txns, min(t.ts_at) AS first_ts, max(t.ts_at) AS last_ts, "
        f"avg(t.risk_score) AS mean_risk FROM ident i JOIN tx t USING (txn_id) "
        f"WHERE t.ts_at >= DATE '{EXAM_START}' GROUP BY ALL ORDER BY customers DESC "
        f"LIMIT {LISTING_ROWS}"
    )


def section_exam_cases(r: Report) -> None:
    r.h("Exam cases: first look")
    r.text(
        "Per card, using the confirmed card ID rule. History counts only transactions "
        "before the flagged one."
    )
    r.execute(
        "CREATE TEMP TABLE exam AS SELECT cp.case_id, cp.trigger_type, "
        "CAST(cp.opened_at AS TIMESTAMP) AS opened_at, t.* FROM cp "
        "JOIN tx t ON t.txn_id = cp.flagged_txn_id"
    )
    r.h("History and region novelty", 3)
    r.table(
        "SELECT e.case_id, e.trigger_type, e.TransactionAmt AS amount, e.channel, e.addr1, "
        "count(h.txn_id) AS prior_txns, count(DISTINCT h.addr1) AS prior_regions, "
        "sum(CASE WHEN h.addr1 = e.addr1 THEN 1 ELSE 0 END) AS prior_in_flagged_region, "
        "mode(h.addr1) AS home_region, median(h.TransactionAmt) AS median_prior_amount, "
        "sum(CASE WHEN h.ProductCD = e.ProductCD THEN 1 ELSE 0 END) AS prior_same_product "
        "FROM exam e LEFT JOIN tx h ON h.card_id = e.card_id AND h.ts_at < e.ts_at "
        "GROUP BY ALL ORDER BY e.case_id"
    )
    r.h("Recurring-charge candidates (R7)", 3)
    r.table(
        f"SELECT e.case_id, e.trigger_type, e.TransactionAmt AS amount, count(h.txn_id) AS "
        f"similar_txns, count(DISTINCT date_trunc('month', h.ts_at)) AS distinct_months, "
        f"string_agg(strftime(h.ts_at, '%Y-%m-%d') || ' $' || CAST(h.TransactionAmt AS "
        f"VARCHAR), ', ' ORDER BY h.ts_at) AS matches "
        f"FROM exam e LEFT JOIN tx h ON h.card_id = e.card_id "
        f"AND h.txn_id <> e.txn_id AND h.ProductCD = e.ProductCD "
        f"AND abs(h.TransactionAmt - e.TransactionAmt) <= {RECURRING_TOLERANCE_USD} "
        f"GROUP BY ALL ORDER BY e.case_id"
    )
    r.h("Card-testing hint (R5)", 3)
    r.table(
        f"SELECT e.case_id, count(h.txn_id) AS small_online_auths_in_prior_hour "
        f"FROM exam e LEFT JOIN tx h ON h.card_id = e.card_id "
        f"AND h.channel = 'online' AND h.TransactionAmt < {CARD_TESTING_SMALL_AUTH_USD} "
        f"AND h.ts_at BETWEEN e.ts_at - INTERVAL {CARD_TESTING_WINDOW_MINUTES} MINUTE AND e.ts_at "
        f"AND h.txn_id <> e.txn_id GROUP BY ALL ORDER BY e.case_id"
    )
    r.h("Flagged device profile reach in the exam window", 3)
    r.table(
        f"WITH flagged AS (SELECT e.case_id, {DEVICE_PROFILE_SQL} AS profile, i.id_15, i.id_23 "
        f"FROM exam e JOIN ident i ON i.txn_id = e.txn_id), "
        f"usage AS (SELECT {DEVICE_PROFILE_SQL} AS profile, t.customer_id FROM ident i "
        f"JOIN tx t USING (txn_id) WHERE t.ts_at >= DATE '{EXAM_START}') "
        f"SELECT f.case_id, f.profile, f.id_15 AS device_status, f.id_23 AS proxy, "
        f"count(DISTINCT u.customer_id) AS customers_on_profile "
        f"FROM flagged f LEFT JOIN usage u USING (profile) GROUP BY ALL ORDER BY f.case_id"
    )
    r.h("Prior closed cases for the same customer", 3)
    r.table(
        "SELECT cp.case_id, cc.case_id AS prior_case, cc.card_id, cc.outcome, cc.pattern, "
        "cc.opened_at FROM cp JOIN cc ON cc.customer_id = cp.customer_id "
        "ORDER BY cp.case_id, cc.opened_at"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", type=Path, default=REPO / "data" / "raw")
    parser.add_argument("--processed", type=Path, default=REPO / "data" / "processed")
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "data_profile.md")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    con = open_warehouse(args.raw, args.processed)
    report = Report(con)
    for section in (
        section_shape,
        section_ids,
        section_case_pack,
        section_card_ids,
        section_closed_cases,
        section_notes,
        section_risk_score,
        section_identity,
        section_exam_cases,
    ):
        section(report)
    report.write(args.out)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
