"""Turning a closed case back into the alert it started as.

The bank's history is the only place the truth is written down: 5,565 cases the analysts
already decided, with the pattern they named, the transactions they held responsible, the
exposure they recorded and what they did about it. Replaying one is the only way to ask
whether this agent would have reached the same conclusion.

Two things make a replay honest. The agent sees the case as an alert, with nothing the
bank learned later. And the case itself is hidden: its own record, and every mention of it
in what other queries return, is withheld, so the agent cannot retrieve the answer it is
being asked for.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from fraudtrail.casepack import ExamCase
from fraudtrail.domain import Trigger
from fraudtrail.evidence.models import (
    CardBaseline,
    CaseEvidence,
    DeviceReach,
    PriorCase,
    RegionSpan,
    SharedOrigin,
    Txn,
)
from fraudtrail.evidence.provider import EvidenceProvider

TIMESTAMP = "%Y-%m-%d %H:%M:%S"

CONFIRMED_FRAUD = "confirmed_fraud"

# The answer schema only accepts exam case ids, and a replayed case must not be told
# which case it is. Scoring keeps the real id alongside.
REPLAY_CASE_ID = "HHG-000"

# The notes record how the case reached the bank. These are the words the analysts used.
CUSTOMER_WORDS = ("cardholder reported", "customer reported", "disputed", "reported")
ANALYST_WORDS = ("analyst", "review requested", "flagged for review")


@dataclass(frozen=True)
class ClosedCaseTruth:
    """What the analysts concluded, as written in closed_cases_history.csv."""

    case_id: str
    customer_id: str
    card_id: str
    opened_at: datetime
    outcome: str
    pattern: str
    flagged_txn_id: str
    txn_ids: tuple[str, ...]
    exposure_usd: float
    connected_card_ids: tuple[str, ...]
    actions: tuple[str, ...]
    report_filed: bool
    notes: str

    @property
    def is_fraud(self) -> bool:
        return self.outcome == CONFIRMED_FRAUD

    @property
    def trigger(self) -> Trigger:
        """How the case reached the bank, read from the analyst's own first sentence."""
        lowered = self.notes.lower()
        if any(word in lowered for word in CUSTOMER_WORDS):
            return Trigger.CUSTOMER_REPORT
        if any(word in lowered for word in ANALYST_WORDS):
            return Trigger.ANALYST_REQUEST
        return Trigger.RISK_SCORE

    def as_exam_case(self) -> ExamCase:
        """The same case as the agent would first see it, with no outcome attached.

        It carries the exam's own case id, because the answer schema only accepts that
        shape, and because a replay must not hand the agent the identifier of the case it
        is being scored against. The real id stays here, for the hold-out and the score.
        """
        return ExamCase(
            case_id=REPLAY_CASE_ID,
            opened_at=self.opened_at,
            trigger=self.trigger,
            trigger_text=(
                f"Alert on card {self.card_id} for transaction {self.flagged_txn_id}. "
                "Review and decide."
            ),
            flagged_txn_id=self.flagged_txn_id,
            card_id=self.card_id,
            customer_id=self.customer_id,
            # The history does not record the score the model gave, and inventing one
            # would hand the agent a signal the replay is meant to test.
            risk_score=None,
        )


def _split(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split("|") if part.strip())


def load_closed_cases(path: Path) -> tuple[ClosedCaseTruth, ...]:
    """Every closed case that can be replayed, oldest first."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    cases: list[ClosedCaseTruth] = []
    for row in rows:
        txn_ids = _split(row["txn_ids"])
        # A cleared case records no first fraudulent transaction, so the alert is the
        # transaction the analyst reviewed.
        flagged = row["first_fraud_txn_id"].strip() or (txn_ids[0] if txn_ids else "")
        if not flagged:
            continue
        cases.append(
            ClosedCaseTruth(
                case_id=row["case_id"],
                customer_id=row["customer_id"],
                card_id=row["card_id"],
                opened_at=datetime.strptime(row["opened_at"], TIMESTAMP),
                outcome=row["outcome"],
                pattern=row["pattern"],
                flagged_txn_id=flagged,
                txn_ids=txn_ids,
                exposure_usd=float(row["exposure_usd"] or 0.0),
                connected_card_ids=_split(row["connected_card_ids"]),
                actions=_split(row["actions_taken"]),
                report_filed=row["report_filed"].strip().lower() in {"yes", "true"},
                notes=row["analyst_notes"],
            )
        )
    cases.sort(key=lambda c: c.opened_at)
    return tuple(cases)


class HoldOutProvider:
    """Every question the agent asks, with the case under test withheld from the answers.

    The date filters already keep out anything the bank learned after the alert. This
    removes the remaining route to the answer: the case's own record reached through a
    device, a shared origin or a text search.
    """

    def __init__(self, inner: EvidenceProvider, hidden_case_id: str) -> None:
        self._inner = inner
        self._hidden = hidden_case_id

    def _visible(self, cases: tuple[PriorCase, ...]) -> tuple[PriorCase, ...]:
        return tuple(case for case in cases if case.case_id != self._hidden)

    def _clean_reach(self, reach: DeviceReach) -> DeviceReach:
        kept = tuple(c for c in reach.prior_fraud_cases if c != self._hidden)
        return replace(reach, prior_fraud_cases=kept)

    def flagged_txn(self, txn_id: str) -> Txn:
        return self._inner.flagged_txn(txn_id)

    def card_window(
        self, card_id: str, start: datetime, end: datetime, limit: int = 500
    ) -> tuple[Txn, ...]:
        return self._inner.card_window(card_id, start, end, limit)

    def card_baseline(self, card_id: str, before: datetime) -> CardBaseline:
        return self._inner.card_baseline(card_id, before)

    def device_reach(self, profile_id: str, start: datetime, end: datetime) -> DeviceReach:
        return self._clean_reach(self._inner.device_reach(profile_id, start, end))

    def region_timeline(self, card_id: str, before: datetime) -> tuple[RegionSpan, ...]:
        return self._inner.region_timeline(card_id, before)

    def prior_cases(self, card_id: str, before: datetime, limit: int = 50) -> tuple[PriorCase, ...]:
        return self._visible(self._inner.prior_cases(card_id, before, limit))

    def shared_origins(
        self, card_id: str, start: datetime, end: datetime, min_cards: int
    ) -> tuple[SharedOrigin, ...]:
        origins = self._inner.shared_origins(card_id, start, end, min_cards)
        return tuple(
            replace(o, prior_fraud_cases=tuple(c for c in o.prior_fraud_cases if c != self._hidden))
            for o in origins
        )

    def customer_cards(self, card_id: str) -> tuple[str, ...]:
        return self._inner.customer_cards(card_id)

    def recurring_matches(
        self, card_id: str, amount: float, product_cd: str, tolerance: float, before: datetime
    ) -> tuple[Txn, ...]:
        return self._inner.recurring_matches(card_id, amount, product_cd, tolerance, before)

    def similar_cases(self, query: str, before: datetime, k: int = 5) -> tuple[PriorCase, ...]:
        return self._visible(self._inner.similar_cases(query, before, k))


def evidence_is_usable(evidence: CaseEvidence) -> bool:
    """Whether the replay found anything to investigate at all."""
    return bool(evidence.window)
