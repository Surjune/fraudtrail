"""Evidence from the local DuckDB warehouse.

Answers the same questions as the TigerGraph provider, so detectors and the agent can run
and be tested without a live workspace. The graph remains the runtime path.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from fraudtrail.data.sql import device_profile_sql
from fraudtrail.data.warehouse import open_warehouse
from fraudtrail.evidence.models import (
    DEVICE_NEW,
    PROXY_PREFIX,
    CardBaseline,
    DeviceReach,
    PriorCase,
    RegionSpan,
    SharedOrigin,
    Txn,
)
from fraudtrail.evidence.provider import EvidenceError

log = logging.getLogger(__name__)

_PROFILE = device_profile_sql("i")
_HAS_PROFILE = (
    "(i.DeviceInfo IS NOT NULL OR i.id_30 IS NOT NULL OR i.id_31 IS NOT NULL "
    "OR i.id_33 IS NOT NULL)"
)
_M_FLAGS = (
    "concat_ws('|', "
    + ", ".join(f"CASE WHEN t.M{i} IS NOT NULL THEN 'M{i}:' || t.M{i} END" for i in range(1, 10))
    + ")"
)

# One flat table of everything a transaction-level question needs.
_TXV_SQL = f"""
CREATE TABLE txv AS SELECT
  t.txn_id,
  t.card_id,
  t.customer_id,
  t.ts_at AS ts,
  t.TransactionAmt AS amount,
  t.ProductCD AS product_cd,
  t.channel,
  t.risk_score,
  CASE WHEN t.addr1 IS NULL THEN '' ELSE CAST(CAST(t.addr1 AS INTEGER) AS VARCHAR) END AS region,
  CASE WHEN t.addr2 IS NULL THEN '' ELSE CAST(CAST(t.addr2 AS INTEGER) AS VARCHAR) END AS country,
  coalesce(i.id_15, '') AS device_status,
  coalesce(i.id_23, '') AS proxy,
  CASE WHEN {_HAS_PROFILE} THEN {_PROFILE} ELSE '' END AS profile_id,
  coalesce(t.P_emaildomain, '') AS purchaser_email,
  coalesce(t.R_emaildomain, '') AS recipient_email,
  {_M_FLAGS} AS m_flags
FROM tx t LEFT JOIN ident i USING (txn_id)
"""

_TXN_COLUMNS = (
    "txn_id, card_id, ts, amount, product_cd, channel, risk_score, region, country, "
    "device_status, proxy, profile_id, purchaser_email, recipient_email, m_flags"
)

# Words that say nothing about which case is similar.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "from",
        "this",
        "that",
        "they",
        "them",
        "their",
        "there",
        "here",
        "is",
        "was",
        "were",
        "are",
        "be",
        "been",
        "it",
        "its",
        "as",
        "at",
        "by",
        "not",
        "no",
    ]
)


def _keywords(text: str) -> frozenset[str]:
    """Distinctive words, lowercased: the units both sides of a similarity search share."""
    words = {token.strip(".,;:'\"()$").lower() for token in text.split()}
    return frozenset(w for w in words if len(w) > 3 and w not in _STOPWORDS and not w.isdigit())


def _txn(row: tuple[Any, ...]) -> Txn:
    return Txn(
        txn_id=str(row[0]),
        card_id=str(row[1]),
        ts=row[2],
        amount=float(row[3]),
        product_cd=str(row[4] or ""),
        channel=str(row[5] or ""),
        risk_score=float(row[6] or 0.0),
        region=str(row[7] or ""),
        country=str(row[8] or ""),
        device_status=str(row[9] or ""),
        proxy=str(row[10] or ""),
        profile_id=str(row[11] or ""),
        purchaser_email=str(row[12] or ""),
        recipient_email=str(row[13] or ""),
        m_flags=str(row[14] or ""),
    )


def _prior_case(row: tuple[Any, ...], relation: str) -> PriorCase:
    return PriorCase(
        case_id=str(row[0]),
        card_id=str(row[1]),
        outcome=str(row[2]),
        pattern=str(row[3]),
        opened_at=row[4],
        exposure_usd=float(row[5] or 0.0),
        n_txns=int(row[6] or 0),
        report_filed=str(row[7]).strip().lower() in {"yes", "true"},
        actions_taken=str(row[8] or ""),
        notes=str(row[9] or ""),
        relation=relation,
    )


_CASE_COLUMNS = (
    "case_id, card_id, outcome, pattern, CAST(opened_at AS TIMESTAMP), "
    "CAST(exposure_usd AS DOUBLE), CAST(n_txns AS INTEGER), report_filed, "
    "coalesce(actions_taken, ''), coalesce(analyst_notes, '')"
)


class DuckDbProvider:
    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con
        con.execute(_TXV_SQL)
        con.execute("CREATE INDEX txv_txn ON txv (txn_id)")
        con.execute("CREATE INDEX txv_card ON txv (card_id)")
        con.execute("CREATE INDEX txv_profile ON txv (profile_id)")
        # The closed cases are small and every investigation searches them, so hold their
        # notes in memory as word sets.
        self._closed: list[tuple[tuple[Any, ...], frozenset[str]]] = [
            (row, frozenset(_keywords(str(row[9]))))
            for row in self._rows(f"SELECT {_CASE_COLUMNS} FROM cc", [])
        ]

    @classmethod
    def open(cls, raw: Path, processed: Path) -> DuckDbProvider:
        return cls(open_warehouse(raw, processed))

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._con

    def _rows(self, sql: str, params: list[Any]) -> list[tuple[Any, ...]]:
        return self._con.execute(sql, params).fetchall()

    def flagged_txn(self, txn_id: str) -> Txn:
        rows = self._rows(f"SELECT {_TXN_COLUMNS} FROM txv WHERE txn_id = ?", [txn_id])
        if not rows:
            raise EvidenceError(f"transaction {txn_id} is not in the dataset")
        return _txn(rows[0])

    def card_window(
        self, card_id: str, start: datetime, end: datetime, limit: int = 500
    ) -> tuple[Txn, ...]:
        rows = self._rows(
            f"SELECT {_TXN_COLUMNS} FROM txv WHERE card_id = ? AND ts >= ? AND ts <= ? "
            f"ORDER BY ts DESC LIMIT {int(limit)}",
            [card_id, start, end],
        )
        return tuple(_txn(r) for r in reversed(rows))

    def card_baseline(self, card_id: str, before: datetime) -> CardBaseline:
        agg = self._rows(
            "SELECT count(*), sum(CASE WHEN channel = 'online' THEN 1 ELSE 0 END), min(ts), "
            "max(ts), avg(amount), max(amount), coalesce(stddev_samp(amount), 0) "
            "FROM txv WHERE card_id = ? AND ts < ?",
            [card_id, before],
        )
        n, n_online, first_ts, last_ts, mean, largest, stdev = agg[0]
        if not n:
            return CardBaseline(card_id=card_id)
        counts: dict[str, dict[str, int]] = {}
        for column, key in (
            ("region", "regions"),
            ("product_cd", "products"),
            ("profile_id", "profiles"),
            ("purchaser_email", "emails"),
        ):
            rows = self._rows(
                f"SELECT {column}, count(*) FROM txv WHERE card_id = ? AND ts < ? "
                f"AND {column} <> '' GROUP BY 1",
                [card_id, before],
            )
            counts[key] = {str(r[0]): int(r[1]) for r in rows}
        return CardBaseline(
            card_id=card_id,
            n_txns=int(n),
            n_online=int(n_online or 0),
            first_ts=first_ts,
            last_ts=last_ts,
            mean_amount=float(mean or 0.0),
            max_amount=float(largest or 0.0),
            stdev_amount=float(stdev or 0.0),
            regions=counts["regions"],
            products=counts["products"],
            profiles=counts["profiles"],
            emails=counts["emails"],
        )

    def device_reach(self, profile_id: str, start: datetime, end: datetime) -> DeviceReach:
        if not profile_id:
            return DeviceReach(profile_id="")
        rows = self._rows(
            "SELECT card_id, customer_id, txn_id, ts, device_status, proxy FROM txv "
            "WHERE profile_id = ? AND ts >= ? AND ts <= ? ORDER BY ts",
            [profile_id, start, end],
        )
        if not rows:
            return DeviceReach(profile_id=profile_id)
        txn_ids = [str(r[2]) for r in rows]
        cases = self._rows(
            "SELECT DISTINCT c.case_id FROM cc_txn c JOIN cc USING (case_id) "
            "WHERE c.txn_id IN ? AND cc.outcome = 'confirmed_fraud'",
            [txn_ids],
        )
        new_devices = sum(1 for r in rows if str(r[4] or "") == DEVICE_NEW)
        proxies = sum(1 for r in rows if str(r[5] or "").startswith(PROXY_PREFIX))
        return DeviceReach(
            profile_id=profile_id,
            cards=tuple(dict.fromkeys(str(r[0]) for r in rows)),
            customers=tuple(dict.fromkeys(str(r[1]) for r in rows)),
            txn_ids=tuple(txn_ids),
            prior_fraud_cases=tuple(sorted(str(r[0]) for r in cases)),
            first_ts=rows[0][3],
            last_ts=rows[-1][3],
            new_device_share=new_devices / len(rows),
            proxy_share=proxies / len(rows),
        )

    def region_timeline(self, card_id: str, before: datetime) -> tuple[RegionSpan, ...]:
        rows = self._rows(
            "SELECT region, count(*), min(ts), max(ts) FROM txv WHERE card_id = ? AND ts < ? "
            "AND region <> '' GROUP BY 1 ORDER BY 2 DESC",
            [card_id, before],
        )
        return tuple(RegionSpan(str(r[0]), int(r[1]), r[2], r[3]) for r in rows)

    def prior_cases(self, card_id: str, before: datetime, limit: int = 50) -> tuple[PriorCase, ...]:
        found: dict[str, PriorCase] = {}
        same = self._rows(
            f"SELECT {_CASE_COLUMNS} FROM cc WHERE card_id = ? "
            f"AND CAST(opened_at AS TIMESTAMP) < ? ORDER BY opened_at DESC LIMIT {int(limit)}",
            [card_id, before],
        )
        for row in same:
            found[str(row[0])] = _prior_case(row, "same_card")

        customer_id = card_id.split("-")[0]
        sibling = self._rows(
            f"SELECT {_CASE_COLUMNS} FROM cc WHERE customer_id = ? AND card_id <> ? "
            f"AND CAST(opened_at AS TIMESTAMP) < ? ORDER BY opened_at DESC LIMIT {int(limit)}",
            [customer_id, card_id, before],
        )
        for row in sibling:
            found.setdefault(str(row[0]), _prior_case(row, "sibling_card"))

        connected = self._rows(
            f"SELECT {_CASE_COLUMNS} FROM cc "
            f"WHERE contains('|' || coalesce(connected_card_ids, '') || '|', ?) "
            f"AND CAST(opened_at AS TIMESTAMP) < ? ORDER BY opened_at DESC "
            f"LIMIT {int(limit)}",
            [f"|{card_id}|", before],
        )
        for row in connected:
            found.setdefault(str(row[0]), _prior_case(row, "connected"))

        return tuple(sorted(found.values(), key=lambda c: c.opened_at, reverse=True))

    def shared_origins(
        self, card_id: str, start: datetime, end: datetime, min_cards: int
    ) -> tuple[SharedOrigin, ...]:
        """Device profiles this card shares with several others inside the window.

        One grouped query narrows thousands of profiles to the few worth expanding.
        """
        summary = self._rows(
            "WITH mine AS (SELECT DISTINCT profile_id FROM txv WHERE card_id = ? "
            "AND ts >= ? AND ts <= ? AND profile_id <> '') "
            "SELECT t.profile_id, count(DISTINCT t.card_id) AS cards "
            "FROM txv t JOIN mine USING (profile_id) WHERE t.ts >= ? AND t.ts <= ? "
            "GROUP BY 1 HAVING count(DISTINCT t.card_id) >= ? ORDER BY 2 DESC",
            [card_id, start, end, start, end, min_cards],
        )
        origins: list[SharedOrigin] = []
        for profile_id, _cards in summary:
            reach = self.device_reach(str(profile_id), start, end)
            origins.append(
                SharedOrigin(
                    kind="device_profile",
                    value=str(profile_id),
                    cards=reach.cards,
                    txn_ids=reach.txn_ids,
                    prior_fraud_cases=reach.prior_fraud_cases,
                    new_device_share=reach.new_device_share,
                    proxy_share=reach.proxy_share,
                    fully_specified=all(part.strip() for part in str(profile_id).split("|")),
                )
            )
        origins.sort(key=lambda o: o.card_count, reverse=True)
        return tuple(origins)

    def customer_cards(self, card_id: str) -> tuple[str, ...]:
        rows = self._rows(
            "SELECT DISTINCT card_id FROM txv WHERE customer_id = ? ORDER BY 1",
            [card_id.split("-")[0]],
        )
        return tuple(str(r[0]) for r in rows)

    def recurring_matches(
        self, card_id: str, amount: float, product_cd: str, tolerance: float, before: datetime
    ) -> tuple[Txn, ...]:
        rows = self._rows(
            f"SELECT {_TXN_COLUMNS} FROM txv WHERE card_id = ? AND ts < ? AND product_cd = ? "
            f"AND abs(amount - ?) <= ? ORDER BY ts",
            [card_id, before, product_cd, amount, tolerance],
        )
        return tuple(_txn(r) for r in rows)

    def similar_cases(self, query: str, before: datetime, k: int = 5) -> tuple[PriorCase, ...]:
        """Closed cases whose notes share the most distinctive words with this situation.

        The graph provider does this with vector search over the same notes; offline,
        overlap on the analysts' own wording retrieves the same kind of case.
        """
        words = _keywords(query)
        if not words:
            return ()
        scored: list[tuple[int, tuple[Any, ...]]] = []
        for row, note_words in self._closed:
            if row[4] >= before:
                continue
            overlap = len(words & note_words)
            if overlap:
                scored.append((overlap, row))
        scored.sort(key=lambda item: (item[0], item[1][4]), reverse=True)
        return tuple(_prior_case(row, "similar") for _, row in scored[:k])
