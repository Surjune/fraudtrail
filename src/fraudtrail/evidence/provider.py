"""The evidence interface the investigation depends on.

`TigerGraphProvider` is the runtime path; `DuckDbProvider` answers the same questions from
the local warehouse so detectors, the scorer and the agent can be tested without a live
workspace. Both return the types in `models.py`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from fraudtrail.evidence.models import (
    CardBaseline,
    DeviceReach,
    PriorCase,
    RegionSpan,
    SharedOrigin,
    Txn,
)


class EvidenceError(RuntimeError):
    """Evidence could not be retrieved. Never answered with a guess."""


class EvidenceProvider(Protocol):
    """One call per question the investigation asks the graph."""

    def flagged_txn(self, txn_id: str) -> Txn:
        """The transaction an alert fired on."""
        ...

    def card_window(
        self, card_id: str, start: datetime, end: datetime, limit: int = 500
    ) -> tuple[Txn, ...]:
        """Every transaction on one card inside a window, oldest first."""
        ...

    def card_baseline(self, card_id: str, before: datetime) -> CardBaseline:
        """What the card did before the alert."""
        ...

    def device_reach(self, profile_id: str, start: datetime, end: datetime) -> DeviceReach:
        """Which cards and customers share a device profile in a window."""
        ...

    def region_timeline(self, card_id: str, before: datetime) -> tuple[RegionSpan, ...]:
        """One row per billing region the card has used."""
        ...

    def prior_cases(self, card_id: str, before: datetime, limit: int = 50) -> tuple[PriorCase, ...]:
        """Closed cases on this card, the customer's other cards, or naming this card."""
        ...

    def shared_origins(
        self, card_id: str, start: datetime, end: datetime, min_cards: int
    ) -> tuple[SharedOrigin, ...]:
        """Elements this card shares with several others in a window (R6)."""
        ...

    def customer_cards(self, card_id: str) -> tuple[str, ...]:
        """Every card the cardholder holds, including this one."""
        ...

    def recurring_matches(
        self, card_id: str, amount: float, product_cd: str, tolerance: float, before: datetime
    ) -> tuple[Txn, ...]:
        """Earlier charges on the card within a tolerance of one amount (R7)."""
        ...

    def similar_cases(self, query: str, before: datetime, k: int = 5) -> tuple[PriorCase, ...]:
        """Closed cases that read like this situation, newest first among equals."""
        ...
