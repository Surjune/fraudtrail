"""The shapes the evidence layer returns, whatever supplies them.

A provider fills these from TigerGraph (the runtime path) or from the DuckDB warehouse
(tests and offline runs). Detectors and the scorer only ever see these types, so the same
investigation logic runs against either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

# id_15 marks whether the device is new to the account.
DEVICE_NEW = "New"

# Values of id_23 all start with this prefix, e.g. IP_PROXY:ANONYMOUS.
PROXY_PREFIX = "IP_PROXY"


@dataclass(frozen=True)
class Txn:
    txn_id: str
    card_id: str
    ts: datetime
    amount: float
    product_cd: str
    channel: str
    risk_score: float
    region: str = ""
    country: str = ""
    device_status: str = ""
    proxy: str = ""
    profile_id: str = ""
    purchaser_email: str = ""
    recipient_email: str = ""
    m_flags: str = ""

    @property
    def is_online(self) -> bool:
        return self.channel == "online"

    @property
    def from_new_device(self) -> bool:
        return self.device_status == DEVICE_NEW

    @property
    def behind_proxy(self) -> bool:
        return self.proxy.startswith(PROXY_PREFIX)


@dataclass(frozen=True)
class CardBaseline:
    """What normal looks like on a card before the alert."""

    card_id: str
    n_txns: int = 0
    n_online: int = 0
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    mean_amount: float = 0.0
    max_amount: float = 0.0
    stdev_amount: float = 0.0
    regions: dict[str, int] = field(default_factory=dict)
    products: dict[str, int] = field(default_factory=dict)
    profiles: dict[str, int] = field(default_factory=dict)
    emails: dict[str, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.n_txns == 0

    @property
    def home_region(self) -> str:
        if not self.regions:
            return ""
        return max(self.regions.items(), key=lambda item: item[1])[0]

    @property
    def online_share(self) -> float:
        return self.n_online / self.n_txns if self.n_txns else 0.0

    def amount_z(self, amount: float) -> float:
        """How many standard deviations above this card's usual spend."""
        if self.n_txns < 2 or self.stdev_amount <= 0:
            return 0.0
        return (amount - self.mean_amount) / self.stdev_amount

    def region_seen(self, region: str) -> bool:
        return bool(region) and region in self.regions

    def product_seen(self, product_cd: str) -> bool:
        return product_cd in self.products

    def profile_seen(self, profile_id: str) -> bool:
        return bool(profile_id) and profile_id in self.profiles


@dataclass(frozen=True)
class DeviceReach:
    """How far one device profile spreads across the book in a window."""

    profile_id: str
    cards: tuple[str, ...] = ()
    customers: tuple[str, ...] = ()
    txn_ids: tuple[str, ...] = ()
    prior_fraud_cases: tuple[str, ...] = ()
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    new_device_share: float = 0.0
    """Share of the device's transactions marked New for their own account. A ring device
    is new everywhere it appears; a common browser fingerprint is not."""

    proxy_share: float = 0.0

    @property
    def customer_count(self) -> int:
        return len(self.customers)


@dataclass(frozen=True)
class RegionSpan:
    region: str
    n_txns: int
    first_ts: datetime
    last_ts: datetime


@dataclass(frozen=True)
class PriorCase:
    """A closed case the bank already decided, retrieved as memory."""

    case_id: str
    card_id: str
    outcome: str
    pattern: str
    opened_at: datetime
    exposure_usd: float
    n_txns: int
    report_filed: bool
    actions_taken: str
    notes: str
    relation: str
    """How it reaches this investigation: same_card, sibling_card, connected or similar."""

    @property
    def confirmed_fraud(self) -> bool:
        return self.outcome == "confirmed_fraud"


@dataclass(frozen=True)
class SharedOrigin:
    """An element several cards have in common inside a window (Policy R6)."""

    kind: str
    value: str
    cards: tuple[str, ...]
    txn_ids: tuple[str, ...] = ()
    prior_fraud_cases: tuple[str, ...] = ()
    new_device_share: float = 0.0
    proxy_share: float = 0.0
    fully_specified: bool = False
    """The profile names a device, an operating system, a browser and a screen. Partial
    profiles are software fingerprints shared by hundreds of unrelated customers."""

    @property
    def card_count(self) -> int:
        return len(self.cards)


@dataclass(frozen=True)
class CaseEvidence:
    """Everything gathered for one investigation, before any judgement is made."""

    flagged: Txn
    card_id: str
    customer_id: str
    opened_at: datetime
    window: tuple[Txn, ...] = ()
    baseline: CardBaseline | None = None
    device_reach: DeviceReach | None = None
    regions: tuple[RegionSpan, ...] = ()
    prior_cases: tuple[PriorCase, ...] = ()
    shared_origins: tuple[SharedOrigin, ...] = ()
    customer_cards: tuple[str, ...] = ()
    recurring_candidates: tuple[Txn, ...] = ()
    """Earlier charges on the card close to the flagged amount (R7)."""

    similar_cases: tuple[PriorCase, ...] = ()
    """Closed cases retrieved because they read like this one."""

    def txns_before_alert(self) -> tuple[Txn, ...]:
        return tuple(t for t in self.window if t.ts <= self.opened_at)

    def amount_gap_days(self) -> float:
        if self.baseline is None or self.baseline.last_ts is None:
            return math.inf
        return (self.flagged.ts - self.baseline.last_ts).total_seconds() / 86_400
