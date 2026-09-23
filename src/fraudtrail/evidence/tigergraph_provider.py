"""Evidence from TigerGraph: the runtime path.

Every question the investigation asks is one installed GSQL query, so the traversal runs
where the data is and the agent receives connected evidence rather than rows to join.
`DuckDbProvider` answers the same questions from the local warehouse and returns the same
types, which is what makes the two comparable: the same 20 cases run through either, and
a difference in an answer is a difference in the evidence, not in the logic.

Nothing here guesses. A query that fails raises `EvidenceError` rather than returning a
partial result, because a missing device profile silently read as "no ring" is worse than
an error.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fraudtrail.config import Settings
from fraudtrail.evidence.models import (
    CardBaseline,
    DeviceReach,
    PriorCase,
    RegionSpan,
    SharedOrigin,
    Txn,
)
from fraudtrail.evidence.provider import EvidenceError
from fraudtrail.evidence.similarity import keywords, most_similar
from fraudtrail.graph.client import GraphClient, GraphError

log = logging.getLogger(__name__)

TS_FORMAT = "%Y-%m-%d %H:%M:%S"

# An accumulator that never saw a value comes back as the epoch rather than as null.
EPOCH = "1970-01-01 00:00:00"

# The whole closed-case corpus is read once per run and ranked in memory. The cut-off is
# past the last date in the dataset, so the read takes everything and each retrieval
# applies the date its own investigation is allowed to see.
MAX_CLOSED_CASES = 8000
CORPUS_UNTIL = "2030-01-01 00:00:00"

# "device:" prefixes every device element in shared_origin's map.
DEVICE_PREFIX = "device:"

JsonDict = dict[str, Any]


def _ts(value: Any) -> datetime | None:
    if not value or str(value) == EPOCH:
        return None
    try:
        return datetime.strptime(str(value), TS_FORMAT)
    except ValueError as exc:
        raise EvidenceError(f"unreadable timestamp from the graph: {value!r}") from exc


def _required_ts(value: Any) -> datetime:
    parsed = _ts(value)
    if parsed is None:
        raise EvidenceError(f"a timestamp the investigation needs is missing: {value!r}")
    return parsed


def _param(ts: datetime) -> str:
    return ts.strftime(TS_FORMAT)


def _merge(blocks: list[Any]) -> JsonDict:
    """One dict of every printed name, whichever PRINT produced it."""
    merged: JsonDict = {}
    for block in blocks:
        if isinstance(block, dict):
            merged.update(block)
    return merged


def _vertices(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, dict)]


def _attrs(vertex: JsonDict) -> JsonDict:
    attributes = vertex.get("attributes")
    return attributes if isinstance(attributes, dict) else {}


def _counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(k): int(v) for k, v in value.items() if str(k)}


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(v) for v in value)


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()}


class TigerGraphProvider:
    """One installed query per question, and no query run twice for the same answer."""

    def __init__(self, client: GraphClient) -> None:
        self._client = client
        self._closed: list[tuple[PriorCase, frozenset[str]]] = []
        self._closed_loaded = False

    @classmethod
    def from_env(cls, settings: Settings | None = None) -> TigerGraphProvider:
        return cls(GraphClient.from_env(settings))

    @property
    def client(self) -> GraphClient:
        """The authenticated connection, shared with whatever writes cases back."""
        return self._client

    def _run(self, name: str, params: JsonDict) -> JsonDict:
        try:
            return _merge(self._client.run_query(name, params))
        except GraphError as exc:
            raise EvidenceError(f"{name} failed: {exc}") from exc

    # A vertex parameter is passed as a 1-tuple; a plain value is deprecated and fails
    # outright on ids containing spaces or pipes, which every device profile does.
    @staticmethod
    def _vertex(vertex_id: str) -> tuple[str]:
        return (vertex_id,)

    def _txn(self, vertex: JsonDict, card_id: str, profile_id: str = "") -> Txn:
        a = _attrs(vertex)
        return Txn(
            txn_id=str(vertex.get("v_id", a.get("txn_id", ""))),
            card_id=card_id,
            ts=_required_ts(a.get("ts")),
            amount=float(a.get("amount") or 0.0),
            product_cd=str(a.get("product_cd") or ""),
            channel=str(a.get("channel") or ""),
            risk_score=float(a.get("risk_score") or 0.0),
            region=str(a.get("region") or ""),
            country=str(a.get("country") or ""),
            device_status=str(a.get("device_status") or ""),
            proxy=str(a.get("proxy_flag") or ""),
            profile_id=profile_id,
            m_flags=str(a.get("m_flags") or ""),
        )

    def _prior_case(self, vertex: JsonDict, card_of: dict[str, str], relation: str) -> PriorCase:
        a = _attrs(vertex)
        case_id = str(vertex.get("v_id", a.get("case_id", "")))
        return PriorCase(
            case_id=case_id,
            card_id=card_of.get(case_id, ""),
            outcome=str(a.get("outcome") or ""),
            pattern=str(a.get("pattern") or ""),
            opened_at=_required_ts(a.get("opened_at")),
            exposure_usd=float(a.get("exposure_usd") or 0.0),
            n_txns=int(a.get("n_txns") or 0),
            report_filed=bool(a.get("report_filed")),
            actions_taken=str(a.get("actions_taken") or ""),
            notes=str(a.get("analyst_notes") or ""),
            relation=relation,
        )

    def flagged_txn(self, txn_id: str) -> Txn:
        result = self._run("case_context", {"flagged_txn": self._vertex(txn_id)})
        flagged = _vertices(result.get("flagged"))
        if not flagged:
            raise EvidenceError(f"transaction {txn_id} is not in the graph")
        cards = _vertices(result.get("card"))
        devices = _vertices(result.get("device"))
        card_id = str(cards[0].get("v_id", "")) if cards else ""
        profile_id = str(devices[0].get("v_id", "")) if devices else ""
        return self._txn(flagged[0], card_id, profile_id)

    def card_window(
        self, card_id: str, start: datetime, end: datetime, limit: int = 500
    ) -> tuple[Txn, ...]:
        result = self._run(
            "card_window",
            {
                "card": self._vertex(card_id),
                "from_ts": _param(start),
                "to_ts": _param(end),
                "max_txns": limit,
            },
        )
        device_of = result.get("device_of")
        devices = device_of if isinstance(device_of, dict) else {}
        txns = [
            self._txn(v, card_id, str(devices.get(str(v.get("v_id", "")), "")))
            for v in _vertices(result.get("txns"))
        ]
        # The query takes the newest inside the window so a busy card cannot push the
        # flagged transaction past the limit; the investigation reads them oldest first.
        txns.sort(key=lambda t: t.ts)
        return tuple(txns)

    def card_baseline(self, card_id: str, before: datetime) -> CardBaseline:
        result = self._run(
            "card_baseline", {"card": self._vertex(card_id), "before_ts": _param(before)}
        )
        n = int(result.get("n_txns") or 0)
        if not n:
            return CardBaseline(card_id=card_id)
        total = float(result.get("sum_amount") or 0.0)
        total_sq = float(result.get("sum_amount_sq") or 0.0)
        mean = total / n
        # Sample variance from the running sums, which is what the query can accumulate
        # in one pass. Floating point can push a near-zero variance below zero.
        variance = max((total_sq - n * mean * mean) / (n - 1), 0.0) if n > 1 else 0.0
        return CardBaseline(
            card_id=card_id,
            n_txns=n,
            n_online=int(result.get("n_online") or 0),
            first_ts=_ts(result.get("first_ts")),
            last_ts=_ts(result.get("last_ts")),
            mean_amount=mean,
            max_amount=float(result.get("max_amount") or 0.0),
            stdev_amount=variance**0.5,
            regions=_counts(result.get("regions")),
            products=_counts(result.get("products")),
            profiles=_counts(result.get("profiles")),
            emails=_counts(result.get("emails")),
        )

    def device_reach(self, profile_id: str, start: datetime, end: datetime) -> DeviceReach:
        if not profile_id:
            return DeviceReach(profile_id="")
        result = self._run(
            "device_reach",
            {
                "profile": self._vertex(profile_id),
                "from_ts": _param(start),
                "to_ts": _param(end),
            },
        )
        n_txns = int(result.get("n_txns") or 0)
        if not n_txns:
            return DeviceReach(profile_id=profile_id)
        return DeviceReach(
            profile_id=profile_id,
            cards=_strings(result.get("cards")),
            customers=_strings(result.get("customers")),
            txn_ids=_strings(result.get("txns")),
            prior_fraud_cases=tuple(sorted(_strings(result.get("prior_fraud_cases")))),
            first_ts=_ts(result.get("first_ts")),
            last_ts=_ts(result.get("last_ts")),
            new_device_share=int(result.get("n_new_device") or 0) / n_txns,
            proxy_share=int(result.get("n_proxy") or 0) / n_txns,
        )

    def region_timeline(self, card_id: str, before: datetime) -> tuple[RegionSpan, ...]:
        result = self._run(
            "region_timeline", {"card": self._vertex(card_id), "before_ts": _param(before)}
        )
        counts = _counts(result.get("counts"))
        first = _string_map(result.get("first_seen"))
        last = _string_map(result.get("last_seen"))
        spans = [
            RegionSpan(
                region=region,
                n_txns=n,
                first_ts=_required_ts(first.get(region)),
                last_ts=_required_ts(last.get(region)),
            )
            for region, n in counts.items()
        ]
        spans.sort(key=lambda s: s.n_txns, reverse=True)
        return tuple(spans)

    def prior_cases(self, card_id: str, before: datetime, limit: int = 50) -> tuple[PriorCase, ...]:
        result = self._run(
            "prior_cases",
            {"card": self._vertex(card_id), "before_ts": _param(before), "max_cases": limit},
        )
        card_of = _string_map(result.get("card_of"))
        found: dict[str, PriorCase] = {}
        # The nearest relation wins when a case reaches this card more than one way.
        for key, relation in (
            ("same_card", "same_card"),
            ("sibling_card", "sibling_card"),
            ("connected", "connected"),
        ):
            for vertex in _vertices(result.get(key)):
                case = self._prior_case(vertex, card_of, relation)
                found.setdefault(case.case_id, case)
        return tuple(sorted(found.values(), key=lambda c: c.opened_at, reverse=True))

    def shared_origins(
        self, card_id: str, start: datetime, end: datetime, min_cards: int
    ) -> tuple[SharedOrigin, ...]:
        result = self._run(
            "shared_origin",
            {
                "card": self._vertex(card_id),
                "from_ts": _param(start),
                "to_ts": _param(end),
                "min_cards": min_cards,
            },
        )
        by_element = result.get("cards_by_element")
        candidates = by_element if isinstance(by_element, dict) else {}
        origins: list[SharedOrigin] = []
        for element, cards in candidates.items():
            name = str(element)
            if not name.startswith(DEVICE_PREFIX) or len(_strings(cards)) < min_cards:
                continue
            profile_id = name[len(DEVICE_PREFIX) :]
            reach = self.device_reach(profile_id, start, end)
            if len(reach.cards) < min_cards:
                continue
            origins.append(
                SharedOrigin(
                    kind="device_profile",
                    value=profile_id,
                    cards=reach.cards,
                    txn_ids=reach.txn_ids,
                    prior_fraud_cases=reach.prior_fraud_cases,
                    new_device_share=reach.new_device_share,
                    proxy_share=reach.proxy_share,
                    fully_specified=all(part.strip() for part in profile_id.split("|")),
                )
            )
        origins.sort(key=lambda o: o.card_count, reverse=True)
        return tuple(origins)

    def customer_cards(self, card_id: str) -> tuple[str, ...]:
        result = self._run("customer_cards", {"card": self._vertex(card_id)})
        cards = {str(v.get("v_id", "")) for v in _vertices(result.get("cards"))}
        return tuple(sorted(c for c in cards if c))

    def recurring_matches(
        self, card_id: str, amount: float, product_cd: str, tolerance: float, before: datetime
    ) -> tuple[Txn, ...]:
        result = self._run(
            "recurring_charges",
            {
                "card": self._vertex(card_id),
                "amount": amount,
                "product_cd": product_cd,
                "tolerance": tolerance,
                "before_ts": _param(before),
            },
        )
        matches = [self._txn(v, card_id) for v in _vertices(result.get("matches"))]
        matches.sort(key=lambda t: t.ts)
        return tuple(matches)

    def _load_closed_cases(self) -> None:
        """Read the closed-case corpus once, then rank it per investigation in memory.

        The whole corpus is read, not the part before the first alert: each call then
        applies its own cut-off, so a later case can still retrieve a case the bank closed
        after the earlier ones and no investigation ever sees a case from its future.
        """
        result = self._run(
            "closed_case_notes", {"before_ts": CORPUS_UNTIL, "max_cases": MAX_CLOSED_CASES}
        )
        card_of = _string_map(result.get("card_of"))
        self._closed = [
            (case, keywords(case.notes))
            for case in (
                self._prior_case(v, card_of, "similar") for v in _vertices(result.get("cases"))
            )
        ]
        self._closed_loaded = True
        log.info("closed-case memory: %d cases", len(self._closed))

    def similar_cases(self, query: str, before: datetime, k: int = 5) -> tuple[PriorCase, ...]:
        if not self._closed_loaded:
            self._load_closed_cases()
        return most_similar(query, self._closed, before, k)
